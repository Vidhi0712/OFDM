"""
experiments.py - BER-vs-SNR Comparison Experiment
===================================================
Compares three approaches across a sweep of SNR values:
  1. Fixed Strategy A  (LS channel estimation + Zero-Forcing equalization)
  2. Fixed Strategy B  (Enhanced DFT-filtered channel estimation + MMSE equalization)
  3. ML-selected strategy  (Decision Tree dynamically selects A or B per burst)

Evaluates:
  - Bit Error Rate (BER)
  - Post-equalization Mean Squared Error (MSE in dB) / EVM
  - Strategy selection distribution across SNR

Outputs saved to results/:
  - ber_vs_snr.png          -- BER & Equalization MSE curves
  - performance_table.txt   -- Tabulated numeric results
  - ml_decisions.png        -- Strategy selection distribution
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm import transmit_burst, N_DATA
from receiver import (
    strategy_a_burst_syms, strategy_b_burst_syms,
    compute_ber, compute_evm, extract_features_burst,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_model():
    path = os.path.join(RESULTS_DIR, "strategy_selector.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def run_experiment(snr_range=None, n_bursts=40, burst_len=10,
                   tap_configs=None, seed=1234):
    """Run the full BER and EVM experiment.

    For each (SNR, tap config), transmit n_bursts bursts of burst_len
    OFDM symbols.  The ML selector picks a strategy per burst.
    """
    if snr_range is None:
        snr_range = np.arange(0, 32, 2)
    if tap_configs is None:
        tap_configs = [1, 2, 3, 5, 8, 12]

    clf = load_model()

    n_snr = len(snr_range)
    ber_a   = np.zeros(n_snr)
    ber_b   = np.zeros(n_snr)
    ber_ml  = np.zeros(n_snr)

    mse_a   = np.zeros(n_snr)
    mse_b   = np.zeros(n_snr)
    mse_ml  = np.zeros(n_snr)

    ml_choices = {float(snr): [] for snr in snr_range}

    rng = np.random.default_rng(seed)

    for si, snr_db in enumerate(snr_range):
        errs_a = errs_b = errs_ml = 0
        total_bits = 0

        sum_mse_a = sum_mse_b = sum_mse_ml = 0.0
        total_bursts = 0

        for n_taps in tap_configs:
            for _ in range(n_bursts):
                rx_frames, all_bits, h, all_data_syms = transmit_burst(
                    burst_len, snr_db, n_taps=n_taps, rng=rng
                )
                true_syms_concat = np.concatenate(all_data_syms)

                rx_a, syms_a = strategy_a_burst_syms(rx_frames, snr_db)
                rx_b, syms_b = strategy_b_burst_syms(rx_frames, snr_db)

                # ML decision (one decision per burst based only on observable features)
                feats = extract_features_burst(rx_frames, snr_db).reshape(1, -1)
                choice = clf.predict(feats)[0]
                rx_ml = rx_a if choice == 0 else rx_b
                syms_ml = syms_a if choice == 0 else syms_b
                ml_choices[float(snr_db)].append(int(choice))

                n_bits = len(all_bits)
                errs_a  += np.sum(all_bits != rx_a)
                errs_b  += np.sum(all_bits != rx_b)
                errs_ml += np.sum(all_bits != rx_ml)
                total_bits += n_bits

                m_a, _  = compute_evm(true_syms_concat, syms_a)
                m_b, _  = compute_evm(true_syms_concat, syms_b)
                m_ml, _ = compute_evm(true_syms_concat, syms_ml)

                sum_mse_a  += m_a
                sum_mse_b  += m_b
                sum_mse_ml += m_ml
                total_bursts += 1

        ber_a[si]  = errs_a  / total_bits
        ber_b[si]  = errs_b  / total_bits
        ber_ml[si] = errs_ml / total_bits

        mse_a[si]  = sum_mse_a  / total_bursts
        mse_b[si]  = sum_mse_b  / total_bursts
        mse_ml[si] = sum_mse_ml / total_bursts

    return dict(
        snr_range=snr_range,
        ber_a=ber_a,
        ber_b=ber_b,
        ber_ml=ber_ml,
        mse_a=mse_a,
        mse_b=mse_b,
        mse_ml=mse_ml,
        ml_choices=ml_choices,
        n_bursts=n_bursts,
        burst_len=burst_len,
    )


def plot_ber(results):
    """Generate dual-panel plot: BER vs SNR (top) and Equalization MSE vs SNR (bottom)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    snr = results["snr_range"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Panel 1: BER
    def safe_semilog(ax, x, y, *args, **kwargs):
        mask = y > 0
        if np.any(mask):
            ax.semilogy(x[mask], y[mask], *args, **kwargs)

    safe_semilog(ax1, snr, results["ber_a"],  "s--", color="#1f77b4", label="Strategy A (LS + ZF)", markersize=5)
    safe_semilog(ax1, snr, results["ber_b"],  "o--", color="#ff7f0e", label="Strategy B (DFT-LS + MMSE)", markersize=5)
    safe_semilog(ax1, snr, results["ber_ml"], "D-",  color="#2ca02c", label="ML-Selected Strategy",
                 markersize=6, linewidth=2)

    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.set_title("OFDM Adaptive Receiver Performance Comparison")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=1e-5)

    # Panel 2: Post-Equalization MSE (dB)
    mse_db_a  = 10 * np.log10(np.maximum(results["mse_a"], 1e-10))
    mse_db_b  = 10 * np.log10(np.maximum(results["mse_b"], 1e-10))
    mse_db_ml = 10 * np.log10(np.maximum(results["mse_ml"], 1e-10))

    ax2.plot(snr, mse_db_a,  "s--", color="#1f77b4", label="Strategy A (LS + ZF)", markersize=5)
    ax2.plot(snr, mse_db_b,  "o--", color="#ff7f0e", label="Strategy B (DFT-LS + MMSE)", markersize=5)
    ax2.plot(snr, mse_db_ml, "D-",  color="#2ca02c", label="ML-Selected Strategy",
             markersize=6, linewidth=2)

    ax2.set_xlabel("Signal-to-Noise Ratio (SNR in dB)")
    ax2.set_ylabel("Equalization MSE (dB)")
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ber_vs_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] BER and MSE plot saved -> {}".format(path))


def save_table(results):
    """Write a formatted performance table with BER, MSE, and selection statistics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    snr = results["snr_range"]

    lines = []
    header = "{:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
        "SNR(dB)", "BER A(ZF)", "BER B(MMSE)", "BER ML", "MSE A(dB)", "MSE B(dB)", "ML % MMSE"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for i, s in enumerate(snr):
        choices = results["ml_choices"][float(s)]
        pct_mmse = 100.0 * np.mean(choices) if choices else 0
        mse_a_db = 10 * np.log10(max(results["mse_a"][i], 1e-10))
        mse_b_db = 10 * np.log10(max(results["mse_b"][i], 1e-10))

        lines.append(
            "{:8.0f} | {:10.5f} | {:11.5f} | {:10.5f} | {:10.2f} | {:10.2f} | {:9.1f}%".format(
                s,
                results["ber_a"][i],
                results["ber_b"][i],
                results["ber_ml"][i],
                mse_a_db,
                mse_b_db,
                pct_mmse,
            )
        )

    table = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "performance_table.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print("[OK] Table saved -> {}".format(path))
    print("")
    print(table)


def plot_ml_decisions(results):
    """Bar chart showing ML strategy selection distribution across SNRs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    snr = results["snr_range"]
    pct_mmse = [100 * np.mean(results["ml_choices"][float(s)]) for s in snr]
    pct_zf   = [100 - p for p in pct_mmse]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(snr, pct_zf,   width=1.6, label="Strategy A (ZF/LS)", color="#1f77b4")
    ax.bar(snr, pct_mmse, width=1.6, bottom=pct_zf, label="Strategy B (MMSE/DFT)", color="#ff7f0e")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Selection Percentage (%)")
    ax.set_title("ML Strategy Selection Distribution Across SNR")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ml_decisions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[OK] ML-decision chart saved -> {}".format(path))


# -- Standalone execution --
if __name__ == "__main__":
    print("Running BER experiment (this may take a few minutes) ...")
    results = run_experiment(n_bursts=40, burst_len=10, seed=1234)
    plot_ber(results)
    save_table(results)
    plot_ml_decisions(results)
    print("\nDone.")
