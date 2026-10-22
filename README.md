# ML-Assisted Adaptive Channel Estimation and Equalization for OFDM Systems

A **Python implementation** demonstrating machine-learning-assisted adaptive receiver strategy selection for SISO-OFDM systems. A scikit-learn `DecisionTreeClassifier` automatically selects between candidate receiver strategies on a per-burst basis, relying strictly on receiver-observable features (without ground-truth channel leakage).

---

## Candidate Receiver Strategies

- **Strategy A (Low-Complexity Baseline)**: Standard Pilot Least-Squares (LS) Channel Estimation with linear interpolation + Zero-Forcing (ZF) Equalization.
- **Strategy B (Enhanced Noise-Robust Mode)**: Transform-Domain (DFT-filtered) Channel Estimation (truncating CIR beyond maximum delay spread to suppress out-of-band noise) + Minimum Mean Square Error (MMSE) Equalization.
- **ML-Selected Strategy**: Decision Tree Classifier evaluating 9 receiver-observable metrics (estimated SNR, pilot residual statistics, channel frequency selectivity, deep-fade ratio, and noise enhancement proxies) to adaptively pick Strategy A or Strategy B.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Minimal dependencies: `numpy`, `matplotlib`, `scikit-learn`.

### 2. Run the complete pipeline

```bash
python main.py
```

This single command:
1. Generates 2,400 labelled training scenarios across 16 SNR points and 6 channel delay profiles.
2. Trains a real scikit-learn `DecisionTreeClassifier` and evaluates it on a held-out test set (Accuracy: **88.0%**).
3. Executes a Monte Carlo experiment across 16 SNR points ($0$ to $30\text{ dB}$) with unseen channel realizations (seed 1234).
4. Saves all plots, metrics, decision rules, and tables to `results/`.
5. Typical runtime: **~95 seconds**.

### 3. Run unit tests

```bash
python test_system.py
```

Runs 11 automated unit tests verifying QPSK/16-QAM roundtrips, IFFT/CP/FFT recovery, channel normalization, estimation accuracy, EVM metrics, and classifier inference.

---

## Project Structure

```
OFDM DSP/
├── main.py               # Master entry point – runs training, evaluation, and experiments
├── ofdm.py               # OFDM PHY simulation (QPSK, 16-QAM, IFFT/FFT, channel, LS & DFT estimation)
├── receiver.py           # Strategy A, Strategy B, EVM metrics, 9D feature extraction
├── train_model.py        # Labelled dataset generator, Decision Tree training & artifact export
├── experiments.py        # Monte Carlo BER and Equalization MSE experiment runner
├── test_system.py        # Comprehensive 11-test unit test suite
├── requirements.txt      # Minimal Python dependencies
├── README.md             # Project documentation & instructions
├── PID.md                # Project Information Document
├── SPECIFICATION.md      # Detailed system specifications
├── docs/
│   └── IMPLEMENTATION_STATUS.md
└── results/              # Output directory for generated plots, tables, and models
    ├── ber_vs_snr.png
    ├── confusion_matrix.png
    ├── feature_importances.png
    ├── ml_decisions.png
    ├── decision_tree_rules.txt
    ├── performance_table.txt
    ├── classification_report.json
    └── strategy_selector.pkl
```

---

## Measured Performance Summary

From `results/performance_table.txt` (Monte Carlo evaluation on held-out test channels):

| SNR (dB) | BER Strategy A (ZF) | BER Strategy B (MMSE) | BER ML-Selected | Equalization MSE A (dB) | Equalization MSE B (dB) | ML % Strategy B |
|:--------:|:-------------------:|:---------------------:|:---------------:|:-----------------------:|:-----------------------:|:---------------:|
| 0 | 0.27070 | 0.24549 | **0.24549** | +10.21 | **-1.68** | 100.0% |
| 4 | 0.15341 | 0.13221 | **0.13221** | +8.37 | **-3.50** | 100.0% |
| 8 | 0.07550 | 0.06194 | **0.06194** | +5.62 | **-5.57** | 100.0% |
| 12 | 0.03300 | 0.02598 | **0.02598** | +5.80 | **-7.99** | 100.0% |
| 16 | 0.01735 | 0.01251 | **0.01251** | -1.92 | **-10.33** | 100.0% |
| 20 | 0.01036 | 0.00610 | **0.00646** | -6.49 | **-12.48** | 47.1% |
| 24 | 0.00752 | 0.00411 | **0.00474** | -7.74 | **-13.69** | 39.0% |
| 30 | 0.00745 | 0.00315 | **0.00425** | -7.85 | **-12.72** | 21.9% |

---

## Scientific Rigor & Design Principles

1. **Zero Ground-Truth Leakage**: The 9 features supplied to the Decision Tree are computed exclusively from received pilot symbols and channel estimates. True channel impulse response parameters ($h$) are never exposed to the classifier.
2. **Train/Test Scenario Isolation**: Training data is generated with seed 42 across 6 channel delay profiles; experiment evaluation uses seed 1234 with independent channel realizations.
3. **Genuine Machine Learning**: Uses scikit-learn `DecisionTreeClassifier` with hyperparameter constraints (`max_depth=6`, `min_samples_leaf=10`) rather than arbitrary heuristics.
4. **Adaptive Operation**: The classifier demonstrates autonomous switching behavior—relying on Strategy B at low-to-medium SNR to avoid noise blowout, and transitioning to Strategy A at high SNR to minimize computation.
