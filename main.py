"""
main.py - Entry Point
======================
Runs the full pipeline:
  1. Generate labelled training data
  2. Train the Decision-Tree strategy selector
  3. Run BER-vs-SNR experiments
  4. Save all plots and results to results/

Usage:
    python main.py
"""

import os, sys, time
import numpy as np


def main():
    print("=" * 65)
    print("  ML-Assisted Adaptive Channel Estimation & Equalization")
    print("  for OFDM Systems -- Python Prototype")
    print("=" * 65)
    start = time.time()

    # -- Step 1: Train the ML model --
    print("")
    print("[1/3] Training ML strategy selector ...")
    from train_model import generate_dataset, train_classifier, save_artifacts

    X, y = generate_dataset(n_bursts_per_config=25, burst_len=10, seed=42)
    print("      Dataset: {} samples x {} features".format(X.shape[0], X.shape[1]))
    print("      Labels : Strategy A(ZF/LS)={}, Strategy B(MMSE/DFT)={}".format(int(np.sum(y==0)), int(np.sum(y==1))))

    clf, X_test, y_test, y_pred, acc, report, cm = train_classifier(X, y, seed=42)
    save_artifacts(clf, X_test, y_test, y_pred, acc, report, cm)

    # -- Step 2: Run experiments --
    print("")
    print("[2/3] Running BER-vs-SNR experiment ...")
    from experiments import run_experiment, plot_ber, save_table, plot_ml_decisions

    results = run_experiment(n_bursts=35, burst_len=10, seed=1234)
    plot_ber(results)
    save_table(results)
    plot_ml_decisions(results)

    # -- Step 3: Summary --
    elapsed = time.time() - start
    print("")
    print("[3/3] All done in {:.1f} s".format(elapsed))
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    print("")
    print("      Output directory: {}".format(os.path.abspath(results_dir)))
    print("      Files generated:")
    for f in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print("        - {:35s}  ({:.1f} KB)".format(f, size_kb))

    print("")
    print("=" * 65)
    print("  Ready for demonstration.")
    print("=" * 65)


if __name__ == "__main__":
    main()
