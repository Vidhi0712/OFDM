# Project Information Document (PID)

## Project Title
Machine Learning-Assisted Adaptive Channel Estimation and Equalization
for OFDM Systems

## Implementation Language
**Python 3** (NumPy, Matplotlib, scikit-learn)

## Objective
Design and implement a prototype OFDM receiver that uses a trained
Decision Tree classifier to adaptively select between Zero-Forcing (ZF)
and MMSE equalization strategies on a per-frame basis, based on
receiver-observable channel conditions.

## System Overview

```
Random Bits → QPSK Mod → OFDM (IFFT + CP) → Multipath Channel → AWGN
                                                                  ↓
BER ← Demod ← Equalization ← Strategy Selector (ML) ← LS Estimation ← FFT (CP removed)
```

## Key Components

| Module | Responsibility |
|--------|---------------|
| `ofdm.py` | SISO-OFDM PHY: QPSK, 64-pt FFT, CP, multipath channel, AWGN, LS estimation |
| `receiver.py` | ZF and MMSE equalizers, BER computation, feature extraction |
| `train_model.py` | Labelled dataset generation, Decision Tree training |
| `experiments.py` | BER-vs-SNR comparison experiments, plots, tables |
| `main.py` | Single-command entry point |

## OFDM Parameters

| Parameter | Value |
|-----------|-------|
| Subcarriers | 64 |
| Cyclic prefix | 16 samples |
| Pilot spacing | Every 4th subcarrier (16 pilots) |
| Data subcarriers | 48 |
| Modulation | QPSK (2 bits/symbol) |
| Channel | Rayleigh multipath, exponential PDP, 1–12 taps |
| Channel estimation | Least Squares (LS) with linear interpolation |

## ML Model

- **Algorithm**: scikit-learn `DecisionTreeClassifier`
- **Max depth**: 6
- **Features**: 7 receiver-observable metrics (estimated SNR, pilot residuals,
  channel variation, power spread, mean magnitude, peak-to-average)
- **Labels**: Binary – which strategy (ZF or MMSE) achieves fewer bit errors
- **Training/test split**: 75% / 25%, stratified, seed 42
- **Experiment seed**: 1234 (no overlap with training)

## Data Integrity

- **No data leakage**: Features are computed solely from the received signal
  and LS channel estimate. The true channel impulse response `h` is never
  used as a feature.
- **Separate seeds**: Training data (seed 42) and experimental evaluation
  (seed 1234) use different random number generators.

## Deliverables

1. Runnable Python code (`python main.py`)
2. BER-vs-SNR comparison plot
3. Confusion matrix and classification report
4. Decision tree rules (text) and feature importances (plot)
5. Performance summary table
6. Documentation (README, PID, implementation status)

## Team
Individual project.

## Date
October 2026
