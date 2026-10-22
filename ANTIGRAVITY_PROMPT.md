# Antigravity Master Prompt

You are implementing a university DSP research project. Work incrementally and preserve tokens. The project is **Machine Learning-Assisted Adaptive Channel Estimation and Equalization for OFDM Systems**.

## First action: one-time repository audit
1. Inspect the repository structure and relevant source/config/test/documentation files **once**.
2. Read `README.md`, `PID.md`, and `SPECIFICATION.md` if present.
3. Produce a concise audit in `docs/IMPLEMENTATION_STATUS.md`: existing files, working features, missing features, current run/test commands, and the next smallest useful milestone.
4. Do not repeatedly dump or reread the entire codebase. Maintain `docs/IMPLEMENTATION_STATUS.md` as the source of truth. On later turns, inspect only files directly related to the requested change plus this status document.
5. Do not take screenshots unless I explicitly ask or a visual bug cannot be diagnosed any other way. Prefer logs, tests, small targeted code excerpts, and programmatic checks.

## Scientific definition (do not change silently)
Build a software OFDM link, not a hardware radio or end-to-end deep-learning receiver. The receiver pipeline is: bits → QAM → resource-grid/pilots → IFFT → cyclic prefix → simulated channel → CP removal → FFT → channel estimation → equalization → demodulation → metrics.

Use **MATLAB as the primary simulator**, as in the proposal, unless the existing repository is clearly a Python implementation or the user explicitly chooses Python. Do not create a second parallel implementation. If using Python for ML, use `scikit-learn` and define a clear interface (e.g. CSV/JSON feature rows and strategy IDs) between the simulator and classifier.

Primary ML model: `sklearn.tree.DecisionTreeClassifier`, with fixed random seed and validation-based depth/leaf tuning. Optional Random Forest comparison is allowed only after the end-to-end baseline works. Do not introduce CNN/LSTM or unnecessary dependencies.

Important algorithm distinction:
- LS and LMMSE are channel estimators.
- LMS, NLMS, and RLS are adaptive-filter algorithms; in this project they should be implemented as equalizers if their signal model and training procedure are explicitly defined.
- Do not present all five as interchangeable estimators.
- Start with two valid receiver strategies (e.g. LS + ZF and LMMSE + MMSE). Add LMS/NLMS/RLS strategies only after the baseline works and tests pass.

The ML classifier's features must be receiver-observable estimates from pilots, such as estimated SNR, pilot residual power, delay-spread proxy, Doppler/time-variation proxy, and channel-estimate correlation. Never use true simulator SNR, true Doppler, true channel taps, or per-strategy test BER as online classifier inputs. Generate training labels offline by running candidate strategies on training scenarios and selecting the best according to a documented objective. Compare the classifier with a fixed strategy and a simple rule-based selector.

## Token and workflow rules
- Make the smallest coherent change that advances the current milestone.
- Before editing, state in at most 5 bullets: files to change, reason, and tests to run.
- Do not print entire files or giant directory listings. Summarize only relevant findings.
- Do not regenerate or rewrite docs unless requirements changed; edit only affected sections.
- Do not take screenshots for routine progress.
- Do not install dependencies without checking existing environment and explaining why the dependency is necessary.
- Do not make unrelated refactors, rename files, change styling, or reformat the entire project.
- Do not claim a feature works unless you ran a relevant test or clearly label it unverified.
- After changes, report: changed files, commands/tests run and outcomes, remaining issues, and the next smallest step.
- Keep `docs/IMPLEMENTATION_STATUS.md` current after every meaningful milestone. Do not reread all source files on every turn; use the status document and targeted inspection.

## Implementation order
Milestone 1: Audit and runnable baseline.
Milestone 2: Unit-tested QPSK/QAM, IFFT/FFT, cyclic prefix, and AWGN-only OFDM chain.
Milestone 3: Pilot placement and channel-estimation baseline.
Milestone 4: Multipath channel and LS/LMMSE estimates.
Milestone 5: Valid equalizer strategies; LMS/NLMS/RLS only with documented equations/settings.
Milestone 6: Feature extraction and offline candidate benchmarking/label generation.
Milestone 7: Decision Tree training, validation, held-out testing.
Milestone 8: Fixed-vs-rule-based-vs-ML experiments, plots, report.
Milestone 9: Clean run instructions and final reproducibility check.

## Definition of done
A reproducible software simulation; valid classical DSP baselines; a lightweight Decision Tree using receiver-observable features; fair held-out comparisons using BER/SER/channel-estimation MSE and runtime/complexity; tests and plots; honest limitations. Do not fabricate results or claim the ML receiver wins unless the experiments show it.

Begin now with the one-time audit and `docs/IMPLEMENTATION_STATUS.md`. Do not implement everything in one giant untested change. After the audit, implement only the smallest missing milestone and report what you found.
