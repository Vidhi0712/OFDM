"""
train_model.py - Dataset generation, offline labels, Decision-Tree selector
===========================================================================
Pipeline
  1. Simulate bursts (SNR x delay profile x Doppler); on EACH burst run all six
     strategies on the identical received signal and record per-strategy bit errors,
     symbol errors and symbol squared error (offline ground truth).
  2. Record receiver-observable features (receiver.extract_features_burst).
  3. Labels (offline only), documented objective:
        cost_s = bit_errors_s / n_bits + LAMBDA_COMPLEXITY * (C_s / C_max + SE_loss_s)
        label  = argmin_s cost_s          (np.argmin -> lowest strategy id on exact ties)
     BER is the principal objective.  LAMBDA_COMPLEXITY (fixed BEFORE final testing) is
     far smaller than one bit error (1/n_bits), so it only breaks ties: among equally
     accurate strategies pick the cheapest one (compute proxy + training overhead).
  4. Train / validation / test come from three independent RNG streams (different
     seeds => different channel realisations).  Disjointness is checked by hashing the
     channel taps (`assert_disjoint`).
  5. Hyper-parameters (depth, min leaf, sample weighting) are tuned on VALIDATION data
     only, by the receiver-level BER of the selected strategy.  The test split is used
     once, after the model is frozen (experiments.py).
"""

import os, json, pickle, platform, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ofdm
import receiver as R

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_PAY = 8
N_TRAIN = ofdm.N_TRAIN_DEFAULT
N_STRAT = len(R.STRATEGY_IDS)
LAMBDA_COMPLEXITY = 1e-4          # fixed before any test evaluation

SEED_TRAIN, SEED_VAL, SEED_TEST = 101, 202, 303

DEFAULT_GRID = dict(
    snr_range=list(range(0, 32, 2)),
    tap_configs=[1, 2, 3, 5, 8, 12],
    fd_configs=[0.0, 0.02],
)


# --------------------------------------------------------------------------- labels
def strategy_costs(bit_errors, n_bits, n_pay=N_PAY, n_train=N_TRAIN):
    """Objective used for labelling: BER + tiny complexity/overhead penalty."""
    bit_errors = np.asarray(bit_errors, dtype=float)
    cm = np.array([R.complexity_cm_per_frame(s, n_pay + n_train) for s in R.STRATEGY_IDS])
    se = np.array([R.net_spectral_efficiency(s, n_pay, n_train) for s in R.STRATEGY_IDS])
    se_loss = 1.0 - se / se.max()
    penalty = LAMBDA_COMPLEXITY * (cm / cm.max() + se_loss)
    return bit_errors / n_bits + penalty


def make_labels(bit_err, n_bits, n_pay=N_PAY, n_train=N_TRAIN):
    """Deterministic labels for a (n_samples, 6) matrix of per-strategy bit errors."""
    bit_err = np.atleast_2d(bit_err)
    n_bits = np.broadcast_to(np.asarray(n_bits, dtype=float), (bit_err.shape[0],))
    costs = np.array([strategy_costs(e, nb, n_pay, n_train) for e, nb in zip(bit_err, n_bits)])
    return np.argmin(costs, axis=1)


# --------------------------------------------------------------------------- dataset
def _simulate_config(args):
    snr_db, n_taps, fd, n_bursts, seed_seq_state, n_pay, n_train = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_seq_state))
    rows = []
    for _ in range(n_bursts):
        b = ofdm.transmit_burst_ext(n_pay, snr_db, n_taps, fd, n_train=n_train, rng=rng)
        fe = R.FrontEnd(b.rx_frames)
        feats = R.extract_features_burst(fe)
        be, se, sq = [], [], []
        for s in R.STRATEGY_IDS:
            m = R.burst_metrics(b, R.run_strategy(s, b, fe))
            be.append(m["bit_err"]); se.append(m["sym_err"]); sq.append(m["sq_err"])
        rows.append((feats, be, se, sq, len(b.tx_bits), len(b.tx_syms),
                     snr_db, n_taps, fd, b.channel_id))
    return rows


def generate_dataset(seed, n_per_config, grid=None, n_pay=N_PAY, n_train=N_TRAIN, n_jobs=1):
    """Simulate a dataset. Every (snr, taps, fd) config gets its own child seed derived
    from `seed`, so results are reproducible and independent of n_jobs."""
    grid = grid or DEFAULT_GRID
    cfgs = [(snr, t, fd) for fd in grid["fd_configs"] for t in grid["tap_configs"]
            for snr in grid["snr_range"]]
    jobs = [(snr, t, fd, n_per_config, [seed, i], n_pay, n_train)
            for i, (snr, t, fd) in enumerate(cfgs)]
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            res = list(ex.map(_simulate_config, jobs, chunksize=4))
    else:
        res = [_simulate_config(j) for j in jobs]
    rows = [r for chunk in res for r in chunk]
    ds = dict(
        X=np.array([r[0] for r in rows]),
        bit_err=np.array([r[1] for r in rows], dtype=int),
        sym_err=np.array([r[2] for r in rows], dtype=int),
        sq_err=np.array([r[3] for r in rows], dtype=float),
        n_bits=np.array([r[4] for r in rows]),
        n_syms=np.array([r[5] for r in rows]),
        snr=np.array([r[6] for r in rows], dtype=float),
        taps=np.array([r[7] for r in rows]),
        fd=np.array([r[8] for r in rows], dtype=float),
        channel_id=np.array([r[9] for r in rows], dtype=np.int64),
        scenario_id=np.arange(len(rows)),
    )
    ds["y"] = make_labels(ds["bit_err"], ds["n_bits"], n_pay, n_train)
    ds["meta"] = dict(seed=seed, n_per_config=n_per_config, n_pay=n_pay, n_train=n_train, grid=grid)
    return ds


def save_dataset(ds, path):
    arrays = {k: v for k, v in ds.items() if k != "meta"}
    np.savez_compressed(path, meta=json.dumps(ds["meta"]), **arrays)


def load_dataset(path):
    z = np.load(path, allow_pickle=False)
    ds = {k: z[k] for k in z.files if k != "meta"}
    ds["meta"] = json.loads(str(z["meta"]))
    return ds


def assert_disjoint(*datasets):
    """Raise if two datasets share any channel realisation (leakage guard)."""
    seen = set()
    for i, ds in enumerate(datasets):
        ids = set(ds["channel_id"].tolist())
        if len(ids) != len(ds["channel_id"]):
            raise AssertionError("duplicate channel realisations inside dataset %d" % i)
        if seen & ids:
            raise AssertionError("train/val/test share channel realisations")
        seen |= ids
    return True


# --------------------------------------------------------------------------- training
def selected_ber(ds, choice):
    idx = np.arange(len(choice))
    return float(ds["bit_err"][idx, choice].sum() / ds["n_bits"].sum())


def regret_weights(bit_err, cap=32):
    """Sample weights = 1 + (worst - best bit errors), capped: bursts where the choice
    matters count more (derived from training labels only)."""
    return 1.0 + np.minimum(bit_err.max(axis=1) - bit_err.min(axis=1), cap)


def train_selector(train_ds, val_ds, seed=42, depths=(3, 4, 6, 8, 12), leaves=(5, 10, 20, 40),
                   weightings=("none", "regret")):
    """Grid-search on the validation split; return (fitted clf, best-params dict, grid rows).
    Selection criterion: receiver-level BER of the selected strategy on validation data
    (ties -> higher validation macro-F1 -> smaller tree)."""
    rows, best = [], None
    for wname in weightings:
        sw = regret_weights(train_ds["bit_err"]) if wname == "regret" else None
        for d in depths:
            for leaf in leaves:
                clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=seed)
                clf.fit(train_ds["X"], train_ds["y"], sample_weight=sw)
                pred = clf.predict(val_ds["X"])
                ber = selected_ber(val_ds, pred)
                f1 = f1_score(val_ds["y"], pred, average="macro", zero_division=0)
                acc = accuracy_score(val_ds["y"], pred)
                rows.append(dict(weighting=wname, max_depth=d, min_samples_leaf=leaf,
                                 val_ber=ber, val_macro_f1=f1, val_accuracy=acc,
                                 n_leaves=int(clf.get_n_leaves())))
                key = (round(ber, 12), -f1, clf.get_n_leaves())
                if best is None or key < best[0]:
                    best = (key, clf, dict(weighting=wname, max_depth=d, min_samples_leaf=leaf))
    return best[1], best[2], rows


def classification_metrics(clf, ds):
    pred = clf.predict(ds["X"])
    labels = sorted(set(ds["y"].tolist()) | set(pred.tolist()))
    names = [R.STRATEGY_NAMES[i] for i in labels]
    rep = classification_report(ds["y"], pred, labels=labels, target_names=names,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(ds["y"], pred, labels=list(range(N_STRAT)))
    return dict(pred=pred, accuracy=float(accuracy_score(ds["y"], pred)),
                macro_f1=float(f1_score(ds["y"], pred, labels=labels, average="macro", zero_division=0)),
                report=rep, confusion=cm)


# --------------------------------------------------------------------------- artifacts
def selector_meta(best_params, extra=None):
    meta = dict(
        feature_names=R.FEATURE_NAMES,
        feature_docs=R.FEATURE_DOCS,
        strategy_names={str(k): v for k, v in R.STRATEGY_NAMES.items()},
        hyperparameters=best_params,
        label_objective="argmin( BER + %g*(C/Cmax + SE_loss) ); ties -> lowest id" % LAMBDA_COMPLEXITY,
        n_pay=N_PAY, n_train=N_TRAIN,
        seeds=dict(train=SEED_TRAIN, val=SEED_VAL, test=SEED_TEST),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                      sklearn=sklearn.__version__, platform=platform.platform()),
    )
    if extra:
        meta.update(extra)
    return meta


def save_selector(clf, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "strategy_selector.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(out_dir, "selector_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_selector(out_dir):
    """Reload the fitted tree + metadata and verify they are consistent."""
    with open(os.path.join(out_dir, "strategy_selector.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(out_dir, "selector_meta.json")) as f:
        meta = json.load(f)
    if meta["feature_names"] != R.FEATURE_NAMES or clf.n_features_in_ != len(R.FEATURE_NAMES):
        raise ValueError("selector artefact does not match current feature definition")
    return clf, meta


def save_training_artifacts(clf, best_params, grid_rows, val_cls, out_dir):
    """Rules, validation grid, validation classification report/confusion/importances."""
    os.makedirs(out_dir, exist_ok=True)
    save_selector(clf, selector_meta(best_params), out_dir)
    with open(os.path.join(out_dir, "decision_tree_rules.txt"), "w") as f:
        f.write(export_text(clf, feature_names=R.FEATURE_NAMES, max_depth=12))
    with open(os.path.join(out_dir, "validation_grid.csv"), "w") as f:
        keys = list(grid_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in grid_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(R.FEATURE_NAMES, clf.feature_importances_, color="steelblue")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Decision-tree feature importances")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=150); plt.close(fig)


def save_test_classification(cls, out_dir):
    """Held-out TEST classification report + confusion matrix (accuracy alone is not evidence)."""
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(dict(accuracy=cls["accuracy"], macro_f1=cls["macro_f1"], report=cls["report"],
                       confusion_matrix=cls["confusion"].tolist()), f, indent=2)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cls["confusion"], display_labels=[R.STRATEGY_LETTERS[i] for i in range(N_STRAT)]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("TEST confusion matrix (acc %.1f%%, macro-F1 %.2f)\nrows: offline-best label, cols: ML prediction"
                 % (100 * cls["accuracy"], cls["macro_f1"]), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    tr = generate_dataset(SEED_TRAIN, 10)
    va = generate_dataset(SEED_VAL, 5)
    assert_disjoint(tr, va)
    clf, best, rows = train_selector(tr, va)
    print("best", best)
    save_training_artifacts(clf, best, rows, classification_metrics(clf, va), RESULTS_DIR)
