"""
main.py - Entry point
======================
Runs the full pipeline:
  1. Generate train / validation / test datasets (independent seeds & channel realisations)
  2. Tune + fit the Decision-Tree strategy selector (validation split only)
  3. Freeze the model, evaluate ONCE on the held-out test split (fixed vs rule vs ML)
  4. Channel-estimation MSE, LMS/NLMS/RLS convergence, runtime/complexity/SE tables
  5. Save every plot, table, model artefact and a JSON run summary to results/

Usage:
    python main.py                 # full run (a few minutes on one CPU core)
    python main.py --quick         # small smoke-size run
    python main.py --out DIR --jobs 4
"""

import os, sys, time, json, argparse
import numpy as np

import train_model as T
import experiments as E

FULL = dict(n_train=20, n_val=10, n_test=60, grid=T.DEFAULT_GRID,
            ce_bursts=12, conv_trials=60, rt_bursts=30, tune_rep=8)
QUICK = dict(n_train=3, n_val=2, n_test=3,
             grid=dict(snr_range=[0, 10, 20, 30], tap_configs=[2, 8], fd_configs=[0.0, 0.02]),
             ce_bursts=2, conv_trials=4, rt_bursts=4, tune_rep=1)


def run_pipeline(out_dir, quick=False, n_jobs=1, verbose=True):
    cfg = QUICK if quick else FULL
    os.makedirs(out_dir, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()

    log("[1/5] Generating datasets (train/val/test seeds %d/%d/%d) ..." % (T.SEED_TRAIN, T.SEED_VAL, T.SEED_TEST))
    tr = T.generate_dataset(T.SEED_TRAIN, cfg["n_train"], cfg["grid"], n_jobs=n_jobs)
    va = T.generate_dataset(T.SEED_VAL, cfg["n_val"], cfg["grid"], n_jobs=n_jobs)
    te = T.generate_dataset(T.SEED_TEST, cfg["n_test"], cfg["grid"], n_jobs=n_jobs)
    T.assert_disjoint(tr, va, te)
    T.save_dataset(te, os.path.join(out_dir, "test_dataset.npz"))
    log("      bursts: train=%d val=%d test=%d; train label counts %s" %
        (len(tr["y"]), len(va["y"]), len(te["y"]), np.bincount(tr["y"], minlength=6).tolist()))

    log("[2/5] Tuning + training Decision Tree on validation split ...")
    clf, best, grid_rows = T.train_selector(tr, va)
    val_cls = T.classification_metrics(clf, va)
    T.save_training_artifacts(clf, best, grid_rows, val_cls, out_dir)
    log("      best hyper-parameters: %s ; validation accuracy %.3f, macro-F1 %.3f" %
        (best, val_cls["accuracy"], val_cls["macro_f1"]))

    log("[3/5] Frozen-model evaluation on held-out test split ...")
    summary, test_cls, sel = E.evaluate_test_set(te, clf, out_dir)
    log(summary)

    log("[4/5] Channel-estimation, convergence, runtime/complexity experiments ...")
    E.channel_estimation_experiment(out_dir, snr_range=cfg["grid"]["snr_range"], n_bursts=cfg["ce_bursts"])
    E.convergence_experiment(out_dir, n_trials=cfg["conv_trials"])
    E.runtime_complexity_table(out_dir, n_bursts=cfg["rt_bursts"])
    E.tune_adaptive_params(out_dir, n_rep=cfg["tune_rep"])

    elapsed = time.time() - t0
    run_summary = dict(
        mode="quick" if quick else "full", elapsed_seconds=round(elapsed, 1),
        n_bursts=dict(train=len(tr["y"]), val=len(va["y"]), test=len(te["y"])),
        bits_per_burst=int(te["n_bits"][0]), test_bits=int(te["n_bits"].sum()),
        grid=cfg["grid"], hyperparameters=best,
        test_accuracy=test_cls["accuracy"], test_macro_f1=test_cls["macro_f1"],
        test_ber={k: float(E.agg(te, v)[0]) for k, v in sel.items()},
        seeds=dict(train=T.SEED_TRAIN, val=T.SEED_VAL, test=T.SEED_TEST))
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)
    log("[5/5] Done in %.1f s. Artefacts in %s" % (elapsed, os.path.abspath(out_dir)))
    for fn in sorted(os.listdir(out_dir)):
        log("      - %-32s %.1f KB" % (fn, os.path.getsize(os.path.join(out_dir, fn)) / 1024))
    return run_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="ML-assisted adaptive OFDM receiver pipeline")
    ap.add_argument("--quick", action="store_true", help="small, fast smoke run")
    ap.add_argument("--out", default=T.RESULTS_DIR, help="output directory")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for dataset generation")
    a = ap.parse_args(argv)
    print("=" * 66)
    print("  ML-Assisted Adaptive Channel Estimation & Equalization for OFDM")
    print("=" * 66)
    run_pipeline(a.out, quick=a.quick, n_jobs=a.jobs)


if __name__ == "__main__":
    main()
