# ML-Assisted Adaptive Channel Estimation and Equalization for OFDM Systems

Software-only SISO-OFDM simulation (Python 3, NumPy, Matplotlib, scikit-learn). A scikit-learn
`DecisionTreeClassifier` selects, **per burst**, one of **six** classical receiver strategies using only
receiver-observable features. Everything below was measured by running the code in this repository
(`python main.py`, full run ≈ 82 s on one CPU core).

---

## Quick start

```bash
pip install -r requirements.txt
python test_system.py        # 53 automated tests (~10 s, includes a small end-to-end smoke run)
python main.py               # full pipeline -> results/   (python main.py --quick for a tiny run)
```
Options: `python main.py --out DIR --jobs N` (N worker processes for dataset generation).

## Project structure

```
main.py            entry point: datasets -> tune/train -> frozen test evaluation -> experiments -> results/
ofdm.py            PHY: QPSK/16-QAM, IFFT/FFT, CP, Rayleigh multipath, Doppler, AWGN, LS / DFT-LS / LMMSE estimation, Burst simulation
adaptive.py        NEW  LMS / NLMS / RLS adaptive equalizer bank (training + decision-directed)
receiver.py        ZF/MMSE, pilot-based noise/SNR estimation, Strategies A-F (common interface), 10 features,
                   rule-based selector, metrics, complexity proxy, spectral efficiency, runtime
train_model.py     dataset generation, offline labels, validation tuning, Decision Tree, artefacts
experiments.py     receiver-level evaluation, channel-estimation MSE, convergence, runtime/complexity
test_system.py     53 tests            requirements.txt   numpy, matplotlib, scikit-learn
PID.md, SPECIFICATION.md, docs/IMPLEMENTATION_STATUS.md, results/ (generated)
```

## System model and parameters (unchanged baseline)

`Y[k] = H[k] X[k] + N[k]`. 64 subcarriers, CP 16, pilots every 4th subcarrier (16 pilots, value 1), 48 data
subcarriers, QPSK (16-QAM implemented/tested but not used in the experiments), Rayleigh multipath with
exponential PDP `p_n ∝ exp(-n/2)`, 1–12 taps, unit-energy static channel, AWGN at the stated SNR.

**Burst structure (new):** 2 known training OFDM symbols followed by 8 payload OFDM symbols
(768 payload bits). BER/SER/MSE are measured on the payload only, for every strategy, on identical received
signals. **Doppler (new):** optional time-varying channel, per-tap Gauss–Markov process with
`ρ = J0(2π f_D T)`, `f_D T = 0.02` in the experiments; the channel is constant inside an OFDM symbol
(block fading), so **inter-carrier interference is not modelled**. **Interference: not implemented.**

## Channel estimation

| Estimator | Definition |
|---|---|
| **LS** | `H_p = Y_p / X_p` at pilots, linear interpolation to all subcarriers (existing, retained) |
| **DFT-filtered LS** | IFFT of the LS estimate, keep first `CP_LEN` taps, FFT (existing, retained) |
| **LMMSE (new)** | `Ĥ = R_hp (R_pp + σ² I/|X_p|²)^-1 H_p`, `R_hh[k,l] = Σ_n p_n e^{-j2πn(k-l)/N}` (64×64), `R_pp = R_hh[P,P]` (16×16), `R_hp = R_hh[:,P]` (64×16) |

*Statistics available to the operational receiver:* the PDP is a **fixed design-time prior** (`exp(-n/2)` over
8 taps — the same family the simulator draws from, but **not** the scenario's true tap count, so mismatch
occurs, e.g. 1-tap or 12-tap channels). The noise variance is **estimated from the pilots**: the 16 pilot LS
values form a 16-point spectrum; its IFFT taps `n ≥ 12` contain only noise, so
`σ̂² = 16·mean|h16[n≥12]|²`. Implementation: linear solve (no explicit inverse) with diagonal loading `1e-9`.
*Offline only:* `channel_est_mse_vs_snr.png` also shows an "oracle statistics" LMMSE (true PDP and true σ²)
to quantify the cost of unknown statistics; the true channel is used only to score MSE.
Limitation: `σ̂²` is biased for delay spreads > 12 taps.

## Equalization

Retained: **ZF** `X̂ = Y/Ĥ`, **MMSE** `X̂ = Ĥ* Y /(|Ĥ|² + 1/SNR)` (SNR now = `1/σ̂²`, estimated, not the simulator value).

**Adaptive equalizers (new, `adaptive.py`) — architecture.** A bank of 48 independent **one-tap complex
adaptive filters, one per data subcarrier, adapted across OFDM symbols** (these are not channel estimators):

| | |
|---|---|
| input | `y_k[n]` FFT output on data subcarrier k, symbol n |
| output | `x̂_k[n] = w_k[n-1] y_k[n]` (a-priori) |
| desired | known training symbol for the first 2 symbols; afterwards the receiver's own QPSK **decision** of `x̂` (decision-directed). Payload symbols are never used as inputs |
| error | `e = d − w y` |
| init | `w_k = Ĥ*/(\|Ĥ\|² + σ̂²)` from the LS estimate of the first symbol (the "LS-based" part) |

Updates (complex, gradient w.r.t. `w*` of `|e|²`):
**LMS** `w ← w + μ e y*` (μ = 0.15; stable iff μ|y|² < 2);
**NLMS** `w ← w + μ e y*/(ε+|y|²)` (μ = 0.25, ε = 1e-3, stable for 0<μ<2);
**RLS** `g = P y*/(λ+|y|²P)`, `w ← w + g e`, `P ← P/(λ+|y|²P)` (λ = 0.98, `P0 = 1/δ`, δ = 0.5).
Filter length is 1 tap per subcarrier. Hyper-parameters come from a small sweep on a dedicated tuning seed
(999, disjoint from train/val/test) — see `results/adaptive_hparam_sweep.txt`. Training overhead (2 of 10 OFDM
symbols carry no payload) is charged to D–F in the spectral-efficiency numbers.

## Receiver strategies (common interface `run_strategy(id, burst)`)

| ID | Strategy | Training needed |
|---|---|---|
| A | LS + ZF | no |
| B | DFT-filtered LS + MMSE | no |
| C | LMMSE + MMSE | no |
| D | LS-initialised **LMS** | 2 symbols |
| E | LS-initialised **NLMS** | 2 symbols |
| F | LS-initialised **RLS** | 2 symbols |

## ML selector

* **Model:** `DecisionTreeClassifier` (interpretable, kept as the primary model), 6 classes.
* **Features (10, all receiver-observable; equations/units in `receiver.FEATURE_DOCS` and `results/selector_meta.json`):**
  `est_snr_db` (pilot-based, *not* the simulator SNR — the old pipeline leaked it), `pilot_res_power`
  (residual of the pilots w.r.t. a short-CIR fit — the old residual was identically zero because the estimate was
  interpolated *through* the pilots), `delay_spread_proxy`, `chan_variation`, `chan_power_spread`,
  `chan_est_peak_to_avg`, `deep_fade_ratio`, `zf_noise_enhance_log`, `doppler_proxy`, `chan_est_corr`.
  A test proves features/strategies are unchanged when every simulator-only field (true channel, payload, true SNR) is scrambled.
* **Labels (offline):** `argmin_s [ BER_s + 1e-4·(C_s/C_max + SE_loss_s) ]` per burst on identical signals.
  The penalty is far below one bit error (1/768), so it only breaks ties toward the cheaper / training-free strategy; exact ties go to the lowest ID (deterministic).
* **Splits:** train / validation / test from independent seeds (101/202/303) = different channel realisations
  (verified by hashing channel taps). 3,840 / 1,920 / 11,520 bursts over 16 SNRs × 6 delay profiles × 2 Doppler settings.
* **Tuning:** depth ∈ {3,4,6,8,12}, min-leaf ∈ {5,10,20,40}, sample weighting ∈ {none, regret}, chosen on the **validation**
  receiver-level BER → `max_depth=4, min_samples_leaf=20, regret weighting`. The test split is evaluated once with the frozen model.
* **Baselines:** all 6 fixed strategies, a documented **rule-based selector** (`est_snr<14 dB → B; else doppler_proxy>0.02 → C; else A`), and an **oracle** (upper bound, not realisable).

## Measured results (held-out test: 11,520 bursts, 8,847,360 bits; SNR 0–30 dB; taps {1,2,3,5,8,12}; f_D T ∈ {0, 0.02})

| Method | BER | 95% CI | SER | Sym. MSE (dB) | Net SE (b/s/Hz) | CM/frame (proxy) |
|---|---|---|---|---|---|---|
| A LS+ZF | 0.06202 | [0.06066, 0.06369] | 0.11021 | 4.03 | 1.200 | 176 |
| B DFT-LS+MMSE | 0.05258 | [0.05131, 0.05403] | 0.09378 | −6.89 | 1.200 | 608 |
| **C LMMSE+MMSE** | **0.04985** | [0.04865, 0.05127] | 0.08906 | −7.17 | 1.200 | 2959 |
| D LS+LMS | 0.06591 | [0.06449, 0.06753] | 0.11681 | −5.49 | 0.960 | 154 |
| E LS+NLMS | 0.06387 | [0.06239, 0.06558] | 0.11234 | −4.05 | 0.960 | 202 |
| F LS+RLS | 0.06000 | [0.05867, 0.06156] | 0.10640 | −5.99 | 0.960 | 250 |
| Rule-based selector | 0.05457 | [0.05330, 0.05599] | 0.09740 | −4.17 | 1.200 | 497 |
| **ML selector** | **0.04982** | [0.04862, 0.05123] | 0.08903 | −7.05 | 1.200 | 2032 |
| Oracle (offline) | 0.04691 | [0.04576, 0.04828] | 0.08409 | −6.78 | 1.135 | 1002 |

**What this does and does not show (honest reading):**

* **The ML selector is statistically indistinguishable from the best fixed strategy (C).** Paired-bootstrap 95% CI of
  BER(ML)−BER(C) = [−0.00008, +0.00003]. It is significantly better than the rule-based selector
  (CI [−0.00491, −0.00459]) and than A, B, D, E, F. It chooses C for 64% of bursts and A/B for the rest, so the
  *usage-weighted complexity proxy* is ≈31% lower than always running C at the same BER; this is a proxy, and the measured
  wall-clock times (below) are dominated by Python overhead and do not show that saving.
* **Classification accuracy is low and is not the success measure:** test accuracy 46.5%, macro-F1 0.263. Per-burst
  "best strategy" labels are noisy (768 bits per burst, many near-ties), and the tree **never predicts D, E or F**.
* **The adaptive equalizers (D–F) did not win on average**, even before charging their 20% training overhead. RLS is the
  best of them (BER 0.0600 vs A 0.0620); they are clearly worse than per-symbol re-estimation under Doppler
  (e.g. at 20 dB with f_D T = 0.02: C 0.0062, E 0.0233, F 0.0261, D 0.0315) because decision-directed adaptation lags the channel.
  In static channels at 26–30 dB NLMS is comparable to B (slightly lower BER). These are results of *this* architecture and parameters, not a general statement about adaptive filters.
* The oracle gap (0.0029 BER) is partly per-burst noise luck and is not an attainable target.
* LS+ZF has an error floor (≈0.007–0.008 BER at 30 dB) from linear-interpolation error in frequency-selective channels.

Per-SNR tables for BER/SER/MSE (all, static, Doppler) are in `results/performance_table.txt`.

### Channel-estimation MSE (static channels, taps {2,5,8}; `results/channel_est_table.txt`)

| SNR | LS | DFT-LS | LMMSE (operational) | LMMSE (oracle stats) |
|---|---|---|---|---|
| 10 dB | 0.0890 | 0.0490 | 0.0426 | 0.0296 |
| 20 dB | 0.0171 | 0.0099 | 0.0049 | 0.0031 |
| 30 dB | 0.0172 | 0.0102 | 0.0005 | 0.0003 |

LMMSE (operational) beats DFT-LS and LS at every SNR; the remaining gap to oracle statistics is the price of the prior-PDP mismatch and the estimated σ².

### Convergence (static channel, 5 taps; `results/convergence_table.txt`, `convergence.png`)

Symbols until the a-priori MSE stays within 1 dB of steady state (1 symbol = 80 samples). LS-init / cold-start (w₀=1):

| SNR | LMS | NLMS | RLS |
|---|---|---|---|
| 10 dB | 4 / 20 | 0 / 5 | 2 / 4 |
| 20 dB | 20 / 49 | 3 / 9 | 12 / 19 |

"0" means the LS initialisation is already within 1 dB of steady state. Convergence time is separate from execution time.

### Runtime and complexity (`results/complexity_runtime_table.txt`)

Median wall-clock per received OFDM frame (30 bursts, 3 repeats; Python 3.12, NumPy 2.4, single core, includes each
strategy's own front end): A 27.5 µs, B 39.9, C 35.8, D 39.3, E 42.2, F 43.7. **These are environment dependent and
Python-loop bound; use the CM proxy (complex-multiplication equivalents/frame) for algorithmic cost.**
Per-iteration adaptive-filter execution: LMS ≈ 5, NLMS ≈ 7, RLS ≈ 8–10 µs per OFDM symbol (48 filters).
Net spectral efficiency (payload bits per time sample, pilots and CP accounted): A–C 1.200 (as if no training were needed),
D–F 0.960 (2 training symbols per 10).

## Limitations

* Doppler is block-fading per OFDM symbol (no ICI); interference is **not** implemented; SISO only; QPSK in experiments.
* LMMSE uses a fixed prior PDP; the noise estimator assumes delay spread ≤ 12 taps (< CP).
* Adaptive equalizers are one-tap-per-subcarrier, decision-directed after 2 training symbols; other architectures/parameters might perform differently.
* Labels come from single bursts, so they are noisy; selection gain over always-C is not demonstrated in BER.
* Complexity is an arithmetic proxy; runtimes depend on the machine.
* The numbers in the earlier README/slides (e.g. "88% accuracy", the old BER table) came from the previous 2-class pipeline,
  which fed the simulator's SNR to the tree as a feature and used a degenerate pilot residual; they are **superseded** and must not be quoted with this code.

## Reproducibility

Seeds: train 101, validation 202, test 303, channel-estimation 777, convergence 555, adaptive tuning 999, runtime 4242.
Dataset generation derives per-configuration child seeds (`SeedSequence`), so results do not depend on `--jobs`.
Artefacts: `strategy_selector.pkl` + `selector_meta.json` (feature definitions, strategy map, hyper-parameters, library versions);
`train_model.load_selector()` verifies they match the current feature definition.
