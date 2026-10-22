# Technical Specification: OFDM Adaptive Receiver

## 1. System model
A single-antenna OFDM transmitter maps bits to QAM symbols, places data and known pilots on subcarriers, applies an IFFT, and prepends a cyclic prefix. A simulated channel applies multipath fading and optional time variation, followed by noise and optional interference. The receiver removes the prefix, applies an FFT, estimates the channel from pilots, equalizes data subcarriers, demodulates, and calculates metrics.

For a frequency-domain subcarrier model:
`Y[k] = H[k] X[k] + N[k]`
where `X[k]` is the transmitted subcarrier symbol, `H[k]` is the channel frequency response, `N[k]` is noise/interference, and `Y[k]` is received.

## 2. Blocks and responsibilities
1. **Bit source:** reproducible random bit generation.
2. **QAM mapper:** QPSK first; support 16-QAM after the baseline works. Document Gray mapping and normalization.
3. **Resource-grid builder:** allocate active, pilot, and data subcarriers; document DC/guard treatment.
4. **IFFT:** convert frequency-domain symbols to time domain; define normalization convention.
5. **Cyclic prefix:** copy the final `Ncp` samples to the front; ensure CP length is tested against channel delay spread.
6. **Channel:** first AWGN-only; then static frequency-selective multipath; then optional Doppler/time variation and interference.
7. **Receiver front end:** remove CP and FFT using the same normalization convention.
8. **Pilot-based channel estimation:** LS and LMMSE.
9. **Equalization:** one-tap ZF/MMSE frequency-domain equalizer for the baseline; adaptive LMS/NLMS/RLS equalizers as additional strategies when the formulation is well-defined.
10. **Demapper:** recover bits and count bit/symbol errors.
11. **Feature extractor:** compute estimates only from known pilots and received samples.
12. **Label generator:** run candidate strategies offline and select the best according to a documented objective.
13. **Classifier:** Decision Tree maps features to a strategy ID.
14. **Evaluation:** compare fixed, rule-based, and ML selector on identical held-out scenarios.

## 3. Candidate algorithms
### LS channel estimation
At pilot subcarriers, with known pilot `X_p` and received pilot `Y_p`:
`H_LS = Y_p / X_p` for nonzero pilots.
Interpolation is required to estimate `H[k]` on data subcarriers. Specify linear interpolation initially and test alternatives only if needed.

### LMMSE channel estimation
Uses channel/noise statistical assumptions to reduce estimation error compared with raw LS in suitable conditions. It requires a covariance/statistical model and noise-variance estimate. Document how those quantities are obtained; do not silently assume perfect knowledge without stating it.

### Equalization
- **ZF one-tap:** `X_hat[k] = Y[k] / H_hat[k]`; can amplify noise when `|H_hat[k]|` is small.
- **MMSE one-tap:** regularizes the division using an estimated noise-to-symbol-power ratio.
- **LMS:** iterative adaptive filter update based on error.
- **NLMS:** LMS update normalized by input power, often less sensitive to input scale.
- **RLS:** recursive least-squares update, generally more computationally expensive but can converge quickly.

LMS/NLMS/RLS are adaptive-filter algorithms, not direct replacements for LS/LMMSE channel estimation. If used as equalizers, define filter inputs, desired/training symbols, update equations, initialization, step size/forgetting factor, training period, and decision-directed phase.

## 4. ML model and data
### Primary model
`sklearn.tree.DecisionTreeClassifier`.

### Feature candidates
- `snr_est_db`: pilot-based estimate, not simulator ground truth.
- `pilot_residual_power`: residual power after fitting/interpolating the channel.
- `delay_spread_proxy`: derived from an estimated impulse response.
- `doppler_proxy`: change/correlation in channel estimates across OFDM symbols.
- `channel_estimate_correlation`: correlation between estimates from nearby pilot-bearing symbols.
- `interference_power_proxy`: only if it can be estimated from receiver observations.

Every feature must have a documented equation, units, valid range, and implementation test. If a feature cannot be estimated reliably in the chosen simulation, omit it and document why.

### Label-generation protocol
For each training scenario:
1. Generate the received signal and record receiver-observable features.
2. Run each valid candidate receiver strategy on the same signal/channel realization where practical.
3. Measure BER and any other objective metrics.
4. Choose the lowest-BER strategy, using a deterministic tie-break. Optionally add a complexity penalty, but fix its coefficient before final testing.
5. Store features, chosen strategy ID, per-strategy metrics, scenario ID, and configuration.

The label generator may use simulation ground truth for analysis, but **the online classifier input must not contain true channel parameters or the strategy's test-set performance**.

### Data split and training
Split by scenario/seed/channel configuration to reduce leakage. Fit preprocessing only on training data. Tune tree depth/leaf size on validation data. Evaluate the test set once after choices are frozen. Report accuracy, macro-F1, confusion matrix, and receiver-level BER/SER/MSE; classifier accuracy alone does not prove communication improvement.

## 5. Required metrics
- **BER:** bit errors divided by transmitted bits.
- **SER:** symbol errors divided by transmitted symbols.
- **Channel-estimation MSE:** mean squared difference between estimated and true channel response; true channel is used only for offline evaluation.
- **Runtime:** measured per frame with repeated runs and stated hardware/software; treat it as environment-dependent.
- **Complexity proxy:** operation count or algorithm-specific cost estimate if reliable runtime comparisons are not possible.
- **Spectral efficiency:** report a defined metric accounting for pilot and cyclic-prefix overhead. Do not simply equate modulation bits/subcarrier with net spectral efficiency.

## 6. Test plan
- QAM mapper/demapper round-trip tests for every supported modulation.
- IFFT/FFT round-trip within numerical tolerance.
- CP insertion/removal shape and recovery tests.
- AWGN-only sanity check: BER should generally decrease as SNR increases.
- Flat-channel estimate test with known channel.
- Multipath/CP test: verify behavior when CP is adequate and when it is too short.
- LS/LMMSE shape, finite-value, and error-metric tests.
- Adaptive equalizer convergence test on a controlled known sequence.
- Feature tests: finite values, documented units, no hidden ground-truth fields.
- Label-generation repeatability with fixed seed.
- Train/test scenario separation test.
- Full experiment smoke test that regenerates at least one plot.

## 7. Reporting rules
Always include simulation parameters, random seeds, number of frames/bits, train/validation/test split, algorithm settings, and software versions. Plot axes and units must be labelled. Report negative or inconclusive results honestly. Distinguish simulated evidence from theoretical expectations.
