# Project Information Document (PID)

## Project Title
Machine Learning-Assisted Adaptive Channel Estimation and Equalization
for OFDM Systems

## Implementation Language
**Python 3** (NumPy, Matplotlib, scikit-learn)

## Objective
Design and implement a prototype OFDM receiver that uses a trained
Decision Tree classifier to adaptively select, per burst, among six receiver
strategies (LS+ZF, DFT-LS+MMSE, LMMSE+MMSE, and LS-initialised LMS / NLMS / RLS
adaptive equalizers) based on receiver-observable channel conditions.

## System Overview

```
Random Bits → QPSK Mod → OFDM (IFFT + CP) → Multipath Channel → AWGN
                                                                  ↓
BER ← Demod ← Equalization (A-F) ← Strategy Selector (ML) ← pilot features / LS ← FFT (CP removed)
```

## Key Components

| Module | Responsibility |
|--------|---------------|
| `ofdm.py` | SISO-OFDM PHY: QPSK/16-QAM, 64-pt FFT, CP, multipath + Doppler channel, AWGN, LS / DFT-LS / LMMSE estimation, burst simulation |
| `adaptive.py` | LMS / NLMS / RLS adaptive equalizer bank |
| `receiver.py` | Strategies A-F, pilot noise/SNR estimation, 10 features, rule-based selector, metrics, complexity |
| `train_model.py` | Labelled dataset generation, validation tuning, Decision Tree training |
| `experiments.py` | Receiver-level comparison, channel-est MSE, convergence, runtime, plots, tables |
| `main.py` | Single-command entry point |

## OFDM Parameters

| Parameter | Value |
|-----------|-------|
| Subcarriers | 64 |
| Cyclic prefix | 16 samples |
| Pilot spacing | Every 4th subcarrier (16 pilots) |
| Data subcarriers | 48 |
| Burst | 2 known training + 8 payload OFDM symbols |
| Modulation | QPSK (2 bits/symbol); 16-QAM implemented/tested |
| Channel | Rayleigh multipath, exponential PDP, 1–12 taps; optional Doppler (block fading per symbol, f_D T = 0.02 used) |
| Channel estimation | LS (linear interp.), DFT-filtered LS, LMMSE (prior PDP + pilot-estimated noise variance) |

## ML Model

- **Algorithm**: scikit-learn `DecisionTreeClassifier`, 6 classes (strategies A-F)
- **Features**: 10 receiver-observable metrics (pilot-estimated SNR, pilot residual power, delay-spread proxy,
  channel variation, power spread, peak-to-average, deep-fade ratio, ZF noise-enhancement, Doppler proxy, estimate correlation)
- **Labels**: per burst, lowest bit-error strategy; deterministic complexity/training-overhead tie-break (coefficient 1e-4, fixed before testing)
- **Splits**: independent seeds for train (101) / validation (202) / test (303); hyper-parameters tuned on validation only
- **Evaluation**: fixed strategies vs rule-based vs ML vs oracle on identical held-out bursts

## Data Integrity

- **No data leakage**: features come only from received frames/pilots. The simulator SNR is NOT passed to the
  classifier (the previous version did), and a unit test scrambles all simulator-only fields to prove it.
- Train/validation/test share no channel realisations (checked by hashing channel taps).

## Deliverables

1. Runnable Python code (`python main.py`)
2. BER-vs-SNR comparison plot
3. Confusion matrix and classification report (held-out test)
4. Decision tree rules (text) and feature importances (plot)
5. Performance summary table
6. Documentation (README, PID, implementation status)

## Team
Individual project.

## Date
October 2026
