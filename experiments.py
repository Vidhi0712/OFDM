"""
experiments.py - Receiver-level evaluation, convergence, channel-estimation MSE, cost
=====================================================================================
All receiver-level comparisons use the SAME held-out test bursts: every fixed strategy,
the rule-based selector, the ML selector and an (unrealisable) offline oracle are
evaluated on identical transmitted data, channel and noise realisations.

Outputs (in out_dir): ber_vs_snr.png, ser_vs_snr.png, symbol_mse_vs_snr.png,
channel_est_mse_vs_snr.png, convergence.png, ml_decisions.png, performance_table.txt,
summary_receiver_level.txt, convergence_table.txt, complexity_runtime_table.txt,
adaptive_hparam_sweep.txt, channel_est_table.txt
"""

import os, sys, time, platform, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ofdm, adaptive
import receiver as R
import train_model as T

RESULTS_DIR = T.RESULTS_DIR
COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#9467bd", 3: "#8c564b", 4: "#17becf", 5: "#bcbd22"}
SEL_NAMES = {"rule": "Rule-based selector", "ml": "ML selector (Decision Tree)", "oracle": "Oracle (offline best, not realisable)"}


# ============================================================================ selections
def build_selections(ds, clf):
    n = len(ds["y"])
    sel = {R.STRATEGY_NAMES[s]: np.full(n, s) for s in R.STRATEGY_IDS}
    sel[SEL_NAMES["rule"]] = np.array([R.rule_based_selector(x) for x in ds["X"]])
    sel[SEL_NAMES["ml"]] = clf.predict(ds["X"])
    sel[SEL_NAMES["oracle"]] = ds["y"]
    return sel


def agg(ds, choice, mask=None):
    """(BER, SER, symbol MSE) of a per-burst strategy choice over the masked bursts."""
    idx = np.arange(len(choice))
    m = np.ones(len(choice), bool) if mask is None else mask
    if not m.any():
        return np.nan, np.nan, np.nan
    be = ds["bit_err"][idx, choice][m].sum()
    se = ds["sym_err"][idx, choice][m].sum()
    sq = ds["sq_err"][idx, choice][m].sum()
    return be / ds["n_bits"][m].sum(), se / ds["n_syms"][m].sum(), sq / ds["n_syms"][m].sum()


def per_snr(ds, choice, mask=None):
    out = []
    for s in np.unique(ds["snr"]):
        m = ds["snr"] == s
        if mask is not None:
            m = m & mask
        out.append(agg(ds, choice, m))
    return np.array(out)       # (n_snr, 3)


def bootstrap_ci(ds, choice, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(choice))
    e = ds["bit_err"][idx, choice].astype(float)
    b = ds["n_bits"].astype(float)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(e), len(e))
        vals.append(e[k].sum() / b[k].sum())
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap_diff(ds, choice_a, choice_b, n_boot=500, seed=1):
    """95% CI of BER(a) - BER(b) with bursts resampled jointly (paired, same bursts)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(choice_a))
    ea = ds["bit_err"][idx, choice_a].astype(float); eb = ds["bit_err"][idx, choice_b].astype(float)
    b = ds["n_bits"].astype(float)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ea), len(ea))
        d.append((ea[k].sum() - eb[k].sum()) / b[k].sum())
    return np.percentile(d, [2.5, 97.5])


# ============================================================================ plots/tables
def _style(name):
    if name == SEL_NAMES["ml"]:
        return dict(color="#2ca02c", marker="D", ls="-", lw=2.4, ms=6, zorder=5)
    if name == SEL_NAMES["rule"]:
        return dict(color="#d62728", marker="v", ls="-", lw=1.6, ms=5, zorder=4)
    if name == SEL_NAMES["oracle"]:
        return dict(color="k", marker="", ls=":", lw=1.6, zorder=3)
    s = [k for k, v in R.STRATEGY_NAMES.items() if v == name][0]
    return dict(color=COLORS[s], marker="os^sx*"[s % 6], ls="--", lw=1.1, ms=4, alpha=0.85)


def plot_metric(ds, sel, which, fname, ylabel, out_dir, db=False):
    panels = [("All test bursts", None), ("Static channel (fd*Ts = 0)", ds["fd"] == 0.0),
              ("Doppler (fd*Ts = 0.02)", ds["fd"] > 0.0)]
    snrs = np.unique(ds["snr"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, (title, mask) in zip(axes, panels):
        for name, ch in sel.items():
            y = per_snr(ds, ch, mask)[:, which]
            if db:
                ax.plot(snrs, 10 * np.log10(np.maximum(y, 1e-10)), label=name, **_style(name))
            else:
                ok = y > 0
                if ok.any():
                    ax.semilogy(snrs[ok], y[ok], label=name, **_style(name))
        ax.set_title(title); ax.set_xlabel("True SNR (dB) [simulation axis]"); ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=140); plt.close(fig)


def write_performance_tables(ds, sel, out_dir):
    names = list(sel.keys())
    snrs = np.unique(ds["snr"])
    lines = []
    for title, which, fmt in [("BER", 0, "{:9.5f}"), ("SER", 1, "{:9.5f}"), ("Post-equalisation symbol MSE (dB)", 2, "{:9.2f}")]:
        for ptitle, mask in [("ALL test bursts", None), ("STATIC (fd*Ts=0)", ds["fd"] == 0.0), ("DOPPLER (fd*Ts=0.02)", ds["fd"] > 0.0)]:
            lines.append("== %s -- %s ==" % (title, ptitle))
            hdr = "SNR(dB) | " + " | ".join("%-9s" % (n.split(":")[0] if ":" in n else
                                                       {"Rule-based selector": "Rule", "ML selector (Decision Tree)": "ML",
                                                        "Oracle (offline best, not realisable)": "Oracle"}[n]) for n in names)
            lines.append(hdr); lines.append("-" * len(hdr))
            tabs = {n: per_snr(ds, sel[n], mask)[:, which] for n in names}
            for i, s in enumerate(snrs):
                vals = []
                for n in names:
                    v = tabs[n][i]
                    if which == 2:
                        v = 10 * np.log10(max(v, 1e-10))
                    vals.append(fmt.format(v))
                lines.append("%7.0f | " % s + " | ".join(vals))
            lines.append("")
    lines.append("Legend: A=LS+ZF B=DFT-LS+MMSE C=LMMSE+MMSE D=LS-init+LMS E=LS-init+NLMS F=LS-init+RLS; "
                 "Rule/ML/Oracle = selectors. Oracle uses ground truth to pick the best strategy per burst "
                 "(upper bound only).")
    with open(os.path.join(out_dir, "performance_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_summary(ds, sel, cls, out_dir, n_pay=T.N_PAY, n_train=T.N_TRAIN):
    n = len(ds["y"])
    lines = ["Receiver-level summary on %d held-out test bursts (%d payload bits each, SNR 0-30 dB uniform,"
             % (n, int(ds["n_bits"][0])),
             "delay profiles %s taps, fd*Ts in %s). BER 95%% CI = bootstrap over bursts." %
             (sorted(set(ds["taps"].tolist())), sorted(set(ds["fd"].tolist()))), ""]
    hdr = "%-40s | %9s | %-19s | %9s | %9s | %8s | %6s" % ("Method", "BER", "95% CI", "SER", "MSE(dB)", "SE b/s/Hz", "CM/frm")
    lines += [hdr, "-" * len(hdr)]
    for name, ch in sel.items():
        ber, ser, mse = agg(ds, ch)
        lo, hi = bootstrap_ci(ds, ch)
        if name in R.STRATEGY_NAMES.values():
            s = [k for k, v in R.STRATEGY_NAMES.items() if v == name][0]
            se_ = R.net_spectral_efficiency(s, n_pay, n_train); cm = R.complexity_cm_per_frame(s, n_pay + n_train)
        else:       # selectors: usage-weighted
            usage = np.bincount(ch, minlength=6) / len(ch)
            se_ = sum(usage[s] * R.net_spectral_efficiency(s, n_pay, n_train) for s in range(6))
            cm = sum(usage[s] * R.complexity_cm_per_frame(s, n_pay + n_train) for s in range(6))
        lines.append("%-40s | %9.5f | [%8.5f,%8.5f] | %9.5f | %9.2f | %8.3f | %6.0f" %
                     (name, ber, lo, hi, ser, 10 * np.log10(max(mse, 1e-10)), se_, cm))
    lines += ["", "ML selector classification on the same TEST bursts (labels = offline-best strategy per burst):",
              "  accuracy = %.3f, macro-F1 = %.3f  (accuracy alone does NOT prove receiver improvement; see BER rows above)" %
              (cls["accuracy"], cls["macro_f1"]),
              "  ML strategy usage: " + ", ".join("%s=%.1f%%" % (R.STRATEGY_LETTERS[s], 100 * np.mean(sel[SEL_NAMES["ml"]] == s)) for s in range(6)),
              "  Oracle label usage: " + ", ".join("%s=%.1f%%" % (R.STRATEGY_LETTERS[s], 100 * np.mean(ds["y"] == s)) for s in range(6)),
              "", "SE = net payload bits per time sample (pilot + CP overhead; D-F also pay %d training symbols per %d-symbol payload burst)." % (n_train, n_pay),
              "Selectors: SE/CM are usage-weighted averages. CM = complexity proxy (see receiver.complexity_cm_per_frame)."]
    best_fixed = min(R.STRATEGY_NAMES.values(), key=lambda nm: agg(ds, sel[nm])[0])
    ml_ber = agg(ds, sel[SEL_NAMES["ml"]])[0]; bf_ber = agg(ds, sel[best_fixed])[0]
    lines.append("")
    lo, hi = paired_bootstrap_diff(ds, sel[SEL_NAMES["ml"]], sel[best_fixed])
    verdict = ("ML is significantly BETTER than the best fixed strategy" if hi < 0 else
               "ML is significantly WORSE than the best fixed strategy" if lo > 0 else
               "no statistically significant difference between ML and the best fixed strategy (inconclusive)")
    lines.append("Best fixed strategy: %s (BER %.5f). ML selector BER %.5f. Paired-bootstrap 95%% CI of BER(ML)-BER(best fixed): [%.5f, %.5f] -> %s." %
                 (best_fixed, bf_ber, ml_ber, lo, hi, verdict))
    rl, rh = paired_bootstrap_diff(ds, sel[SEL_NAMES["ml"]], sel[SEL_NAMES["rule"]])
    lines.append("ML vs rule-based selector: CI of BER(ML)-BER(rule) = [%.5f, %.5f]." % (rl, rh))
    gap = bf_ber - agg(ds, sel[SEL_NAMES["oracle"]])[0]
    lines.append("Headroom: best fixed minus oracle = %.5f BER (the oracle exploits per-burst noise luck, so it is not an attainable target)." % gap)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary_receiver_level.txt"), "w") as f:
        f.write(txt)
    return txt


def plot_ml_decisions(ds, ml_choice, out_dir):
    snrs = np.unique(ds["snr"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (title, mask) in zip(axes, [("Static (fd*Ts=0)", ds["fd"] == 0.0), ("Doppler (fd*Ts=0.02)", ds["fd"] > 0.0)]):
        bottom = np.zeros(len(snrs))
        for s in R.STRATEGY_IDS:
            frac = np.array([100 * np.mean(ml_choice[(ds["snr"] == v) & mask] == s) if ((ds["snr"] == v) & mask).any() else 0 for v in snrs])
            ax.bar(snrs, frac, bottom=bottom, width=1.6, color=COLORS[s], label=R.STRATEGY_NAMES[s])
            bottom += frac
        ax.set_title(title); ax.set_xlabel("True SNR (dB)")
    axes[0].set_ylabel("ML selection share (%)"); axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ml_decisions.png"), dpi=140); plt.close(fig)


# ============================================================================ channel estimation MSE
def channel_estimation_experiment(out_dir, snr_range=None, tap_configs=(2, 5, 8), n_bursts=12, n_frames=6, seed=777):
    """Channel-estimation MSE vs SNR (static channels). The TRUE channel is used only here,
    offline, to score the estimators.  LMMSE(operational) uses the assumed prior PDP and
    the receiver's estimated noise variance; LMMSE(oracle stats) uses the true PDP and true
    noise variance and is shown only to quantify the cost of unknown statistics."""
    snr_range = list(range(0, 32, 2)) if snr_range is None else list(snr_range)
    rng = np.random.default_rng(seed)
    names = ["LS (linear interp.)", "DFT-filtered LS", "LMMSE (operational)", "LMMSE (oracle statistics)"]
    out = np.zeros((len(snr_range), 4))
    for i, snr in enumerate(snr_range):
        acc = np.zeros(4); cnt = 0
        for t in tap_configs:
            for _ in range(n_bursts):
                b = ofdm.transmit_burst_ext(n_frames, snr, t, 0.0, n_train=0, rng=rng)
                fe = R.FrontEnd(b.rx_frames)
                W_op = ofdm.lmmse_weights(fe.noise_var)
                p = np.exp(-np.arange(t) / 2.0); p /= p.sum()
                W_or = ofdm.lmmse_weights(b.noise_var_true, pdp=p)
                for n in range(n_frames):
                    Ht = b.H_true[n]
                    est = [fe.H_ls(n), ofdm.dft_filter_estimate(fe.H_ls(n)), W_op @ fe.Hp[n], W_or @ fe.Hp[n]]
                    acc += [np.mean(np.abs(e - Ht) ** 2) for e in est]
                    cnt += 1
        out[i] = acc / cnt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for j, nm in enumerate(names):
        ax.semilogy(snr_range, out[:, j], marker="osv^"[j], label=nm)
    ax.set_xlabel("True SNR (dB)"); ax.set_ylabel("Channel-estimation MSE  E|H_hat - H|^2")
    ax.set_title("Channel estimation error (static multipath, taps %s)" % (list(tap_configs),))
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "channel_est_mse_vs_snr.png"), dpi=140); plt.close(fig)
    lines = ["SNR(dB) | " + " | ".join("%-26s" % n for n in names)]
    lines += ["%7d | " % s + " | ".join("%-26.6f" % v for v in row) for s, row in zip(snr_range, out)]
    lines += ["", "Frames per SNR point: %d; seed %d. LMMSE operational: prior PDP exp(-n/2) over %d taps, sigma2 estimated from pilots." %
              (len(tap_configs) * n_bursts * n_frames, seed, ofdm.LMMSE_ASSUMED_TAPS)]
    with open(os.path.join(out_dir, "channel_est_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return snr_range, out


# ============================================================================ convergence
def convergence_experiment(out_dir, snrs=(10, 20), n_taps=5, n_iter=80, n_trials=60, seed=555):
    """Learning curves of LMS/NLMS/RLS on a static channel with an all-known training
    sequence (supervised mode), averaged over trials and subcarriers.  Two initialisations:
    'LS-init' (as in strategies D-F) and 'cold start' (w0 = 1).  Convergence time = number of
    OFDM symbols until the curve stays within 1 dB of its steady state (mean of the last 20%).
    Execution time (us per OFDM symbol) is measured separately."""
    rng = np.random.default_rng(seed)
    results, lines = {}, []
    fig, axes = plt.subplots(len(snrs), 2, figsize=(11, 3.8 * len(snrs)), squeeze=False)
    lines.append("%-6s | %-10s | %-5s | %-9s | %-17s | %-12s | %-18s" %
                 ("SNR", "init", "algo", "steady dB", "conv. (OFDM sym)", "conv. samples", "exec us/OFDM sym"))
    for ri, snr in enumerate(snrs):
        curves = {(init, k): np.zeros(n_iter) for init in ("LS-init", "cold") for k in adaptive.KINDS}
        for _ in range(n_trials):
            h = ofdm.multipath_channel(n_taps, rng=rng)
            ys, ds_ = [], []
            for _i in range(n_iter):
                bits = ofdm.random_bits(2 * ofdm.N_DATA, rng=rng)
                rx, sy = ofdm.transmit_ofdm_symbol(bits, h, snr, rng=rng)
                ys.append(rx); ds_.append(sy)
            fe = R.FrontEnd(ys[:4])
            H0 = ofdm.ls_channel_estimate(ys[0])[ofdm.DATA_IDX]
            w_ls = np.conj(H0) / (np.abs(H0) ** 2 + fe.noise_var)
            yd = [f[ofdm.DATA_IDX] for f in ys]
            for init, w0 in (("LS-init", w_ls), ("cold", np.ones(ofdm.N_DATA, complex))):
                for k in adaptive.KINDS:
                    c, _ = adaptive.learning_curve(k, yd, ds_, w0)
                    curves[(init, k)] += c / n_trials
        for ci, init in enumerate(("LS-init", "cold")):
            ax = axes[ri, ci]
            for k in adaptive.KINDS:
                c = curves[(init, k)]
                ax.plot(10 * np.log10(c), label=k.upper())
                nconv, steady = adaptive.convergence_time(c)
                # execution time per OFDM symbol (separate from convergence time)
                bank = adaptive.AdaptiveEqualizerBank(k, np.ones(ofdm.N_DATA, complex))
                y0, d0 = yd[0], ds_[0]
                t0 = time.perf_counter()
                for _ in range(300):
                    bank.step(y0, d0)
                us = 1e6 * (time.perf_counter() - t0) / 300
                results[(snr, init, k)] = (nconv, steady, us)
                lines.append("%-6d | %-10s | %-5s | %9.2f | %17d | %12d | %18.1f" %
                             (snr, init, k.upper(), 10 * np.log10(steady), nconv, nconv * (ofdm.N_SC + ofdm.CP_LEN), us))
            ax.set_title("%s, SNR %d dB, %d taps" % (init, snr, n_taps))
            ax.set_xlabel("Iteration (OFDM symbol)"); ax.set_ylabel("A-priori MSE  E|e|^2 (dB)")
            ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=140); plt.close(fig)
    lines += ["", "Static channel, %d trials x %d training symbols, 48 one-tap filters per trial, seed %d." % (n_trials, n_iter, seed),
              "Convergence time (symbols to reach 1 dB of steady state) is an algorithmic property; execution time",
              "is wall-clock per OFDM symbol on this machine and is environment dependent."]
    with open(os.path.join(out_dir, "convergence_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return results


# ============================================================================ runtime / complexity / SE
def runtime_complexity_table(out_dir, n_bursts=30, seed=4242):
    rng = np.random.default_rng(seed)
    bursts = [ofdm.transmit_burst_ext(T.N_PAY, 15, 5, 0.0, rng=rng) for _ in range(n_bursts)]
    lines = ["Runtime measured with time.perf_counter over %d bursts x %d frames (SNR 15 dB, 5 taps), median of 3 repeats;" %
             (n_bursts, bursts[0].n_frames),
             "includes each strategy's own front-end work (pilot LS, noise estimate). Environment: Python %s, NumPy %s, %s, %s." %
             (sys.version.split()[0], np.__version__, platform.platform(), platform.processor() or platform.machine()),
             "Runtimes are environment dependent (single-threaded NumPy; Python-loop overhead dominates the adaptive filters).", ""]
    hdr = "%-20s | %12s | %10s | %9s | %-9s" % ("Strategy", "us / frame", "CM / frame", "SE b/s/Hz", "Training")
    lines += [hdr, "-" * len(hdr)]
    rows = {}
    for s in R.STRATEGY_IDS:
        us = R.measure_runtime(s, bursts)
        cm = R.complexity_cm_per_frame(s, T.N_PAY + T.N_TRAIN)
        se = R.net_spectral_efficiency(s, T.N_PAY, T.N_TRAIN)
        rows[s] = (us, cm, se)
        lines.append("%-20s | %12.1f | %10.0f | %9.3f | %-9s" % (R.STRATEGY_NAMES[s], us, cm, se,
                     "%d symbols" % T.N_TRAIN if R.REQUIRES_TRAINING[s] else "none"))
    lines += ["", "CM = complex-multiplication-equivalent proxy per received OFDM symbol (receiver.complexity_cm_per_frame).",
              "SE = payload bits / transmitted time samples: pilots (16/64) and CP (16/64 samples) are already excluded from",
              "payload; D-F additionally spend %d of every %d OFDM symbols on known training." % (T.N_TRAIN, T.N_PAY + T.N_TRAIN),
              "A-C BER does not depend on the training symbols, so their SE is computed as if those symbols were payload."]
    with open(os.path.join(out_dir, "complexity_runtime_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


# ============================================================================ adaptive hyper-parameter sweep
def tune_adaptive_params(out_dir, seed=999, n_rep=8):
    """Small grid for LMS mu / NLMS mu / RLS (lambda, delta) on a TUNING seed that is
    disjoint from the train/val/test seeds. Reports the sweep that motivated the defaults."""
    rng = np.random.default_rng(seed)
    bursts = [ofdm.transmit_burst_ext(T.N_PAY, snr, t, fd, rng=rng)
              for snr in (6, 12, 20, 28) for fd in (0.0, 0.02) for t in (2, 5, 8) for _ in range(n_rep)]
    fes = [R.FrontEnd(b.rx_frames) for b in bursts]

    def ber(sid, ap):
        e = n = 0
        for b, fe in zip(bursts, fes):
            r = R.run_strategy(sid, b, fe, adaptive_params=ap)
            e += np.sum(b.tx_bits != r.bits); n += len(b.tx_bits)
        return e / n
    lines = ["Adaptive-equalizer hyper-parameter sweep on tuning seed %d (%d bursts, SNR {6,12,20,28}, fd {0,0.02}, taps {2,5,8})." % (seed, len(bursts)),
             "Reference: A (LS+ZF) BER %.4f, C (LMMSE+MMSE) BER %.4f" % (ber(0, {}), ber(2, {})), ""]
    for mu in (0.05, 0.15, 0.3):
        lines.append("LMS  mu=%-5s BER %.4f" % (mu, ber(3, dict(mu=mu))))
    for mu in (0.1, 0.25, 0.5, 0.9):
        lines.append("NLMS mu=%-5s BER %.4f" % (mu, ber(4, dict(mu=mu))))
    for lam, dl in itertools.product((0.9, 0.98, 1.0), (0.1, 0.5, 2.0)):
        lines.append("RLS  lambda=%-5s delta=%-4s BER %.4f" % (lam, dl, ber(5, dict(lam=lam, delta=dl))))
    lines += ["", "Defaults used: LMS mu=%s, NLMS mu=%s (eps=%s), RLS lambda=%s delta=%s." %
              (adaptive.LMS_MU, adaptive.NLMS_MU, adaptive.NLMS_EPS, adaptive.RLS_LAMBDA, adaptive.RLS_DELTA)]
    with open(os.path.join(out_dir, "adaptive_hparam_sweep.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================ top level
def evaluate_test_set(test_ds, clf, out_dir):
    """Full receiver-level evaluation + all plots/tables built from the test dataset."""
    os.makedirs(out_dir, exist_ok=True)
    sel = build_selections(test_ds, clf)
    cls = T.classification_metrics(clf, test_ds)
    T.save_test_classification(cls, out_dir)
    plot_metric(test_ds, sel, 0, "ber_vs_snr.png", "Bit error rate", out_dir)
    plot_metric(test_ds, sel, 1, "ser_vs_snr.png", "Symbol error rate", out_dir)
    plot_metric(test_ds, sel, 2, "symbol_mse_vs_snr.png", "Post-equalisation symbol MSE (dB)", out_dir, db=True)
    write_performance_tables(test_ds, sel, out_dir)
    plot_ml_decisions(test_ds, sel[SEL_NAMES["ml"]], out_dir)
    return write_summary(test_ds, sel, cls, out_dir), cls, sel


if __name__ == "__main__":
    import main
    main.main()
