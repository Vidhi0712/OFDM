"""
receiver.py - Estimation, Equalization Strategies A-F, Features, Metrics
=========================================================================
Strategy A : LS channel estimation            + ZF   equalization
Strategy B : DFT-filtered LS estimation        + MMSE equalization
Strategy C : LMMSE channel estimation          + MMSE equalization
Strategy D : LS-initialised adaptive equalizer : LMS   (training + decision-directed)
Strategy E : LS-initialised adaptive equalizer : NLMS  (training + decision-directed)
Strategy F : LS-initialised adaptive equalizer : RLS   (training + decision-directed)

Every strategy has the same interface:
    result = run_strategy(strategy_id, burst, front_end=None)
and returns the recovered PAYLOAD bits / equalized symbols (+ side metrics).

Operational vs. offline information
-----------------------------------
* The receiver only sees: received frames, pilot values, and the known training
  sequence.  Noise variance and SNR are ESTIMATED from the pilots (see
  `estimate_noise_var`) - the simulator's true SNR/channel never reaches a strategy
  or the feature extractor.
* `burst.H_true`, `burst.tx_bits`, ... are used ONLY by the offline metric helpers.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from ofdm import (
    ls_channel_estimate, dft_channel_estimate, dft_filter_estimate,
    lmmse_weights, qpsk_demodulate, DATA_IDX, PILOT_IDX, PILOT_VALUE,
    N_SC, N_DATA, N_PILOTS, CP_LEN,
)
import adaptive


# -- Equalization --

def zf_equalize(rx_frame, H_est):
    """Zero-Forcing equalization: X_hat = Y / H_est  at data subcarriers."""
    H_data = H_est[DATA_IDX]
    H_data = np.where(np.abs(H_data) < 1e-10, 1e-10, H_data)
    X_hat = rx_frame[DATA_IDX] / H_data
    return X_hat


def mmse_equalize(rx_frame, H_est, snr_linear):
    """MMSE equalization at data subcarriers.
    X_hat = conj(H) / (|H|^2 + 1/SNR) * Y
    """
    H_data = H_est[DATA_IDX]
    X_hat = (np.conj(H_data) / (np.abs(H_data)**2 + 1.0 / max(snr_linear, 1e-10))) * rx_frame[DATA_IDX]
    return X_hat


# -- Strategy wrappers (single frame) --

def equalize_frame_a(rx_frame, snr_db):
    """Strategy A: Standard LS channel estimation + ZF equalization.
    Returns (demod bits, equalized symbols)."""
    H_est = ls_channel_estimate(rx_frame)
    X_hat = zf_equalize(rx_frame, H_est)
    return qpsk_demodulate(X_hat), X_hat


def equalize_frame_b(rx_frame, snr_db):
    """Strategy B: Enhanced DFT-filtered channel estimation + MMSE equalization.
    Returns (demod bits, equalized symbols)."""
    snr_lin = 10 ** (snr_db / 10)
    H_est = dft_channel_estimate(rx_frame)
    X_hat = mmse_equalize(rx_frame, H_est, snr_lin)
    return qpsk_demodulate(X_hat), X_hat


# -- Burst-level strategies --

def strategy_a_burst(rx_frames, snr_db):
    """Apply ZF to all frames in a burst, return concatenated bits."""
    all_bits = []
    for frame in rx_frames:
        bits, _ = equalize_frame_a(frame, snr_db)
        all_bits.append(bits)
    return np.concatenate(all_bits)


def strategy_b_burst(rx_frames, snr_db):
    """Apply MMSE to all frames in a burst, return concatenated bits."""
    all_bits = []
    for frame in rx_frames:
        bits, _ = equalize_frame_b(frame, snr_db)
        all_bits.append(bits)
    return np.concatenate(all_bits)


def strategy_a_burst_syms(rx_frames, snr_db):
    """Apply Strategy A, returning both demodulated bits and all equalized symbols."""
    all_bits, all_syms = [], []
    for frame in rx_frames:
        bits, syms = equalize_frame_a(frame, snr_db)
        all_bits.append(bits)
        all_syms.append(syms)
    return np.concatenate(all_bits), np.concatenate(all_syms)


def strategy_b_burst_syms(rx_frames, snr_db):
    """Apply Strategy B, returning both demodulated bits and all equalized symbols."""
    all_bits, all_syms = [], []
    for frame in rx_frames:
        bits, syms = equalize_frame_b(frame, snr_db)
        all_bits.append(bits)
        all_syms.append(syms)
    return np.concatenate(all_bits), np.concatenate(all_syms)


# -- BER and EVM helpers --

def compute_ber(tx_bits, rx_bits):
    """Bit Error Rate between two equal-length bit arrays."""
    assert len(tx_bits) == len(rx_bits)
    return np.mean(tx_bits != rx_bits)


def compute_evm(tx_syms, rx_syms):
    """Error Vector Magnitude (in %) and Mean Squared Error."""
    assert len(tx_syms) == len(rx_syms)
    mse = np.mean(np.abs(rx_syms - tx_syms)**2)
    ref_power = np.mean(np.abs(tx_syms)**2)
    evm_pct = np.sqrt(mse / max(ref_power, 1e-12)) * 100.0
    return mse, evm_pct



# =============================================================================
# Receiver front end: pilot-based noise / SNR estimation (receiver-observable)
# =============================================================================
NOISE_TAIL_START = 12   # CIR taps >= 12 assumed channel-free (simulated delay spread <= 12 < CP)


def pilot_observations(rx_frame):
    """LS estimates at the pilot subcarriers only: H_LS,p = Y_p / X_p   (length N_PILOTS)."""
    return rx_frame[PILOT_IDX] / PILOT_VALUE


def estimate_noise_var(rx_frames):
    """Frequency-domain noise variance sigma^2 estimated from the pilots.

    The 16 pilot LS values (spacing 4 -> 16-point sampled spectrum) are IFFT'd to a
    16-tap CIR.  Channel taps occupy n < NOISE_TAIL_START; taps n >= NOISE_TAIL_START
    contain only noise, each with variance sigma^2 / N_PILOTS (|X_p| = 1).  Hence
        sigma2_hat = N_PILOTS * mean_{frames, n>=12} |h16[n]|^2 / |X_p|^2 .
    Limitation: biased if the true delay spread exceeds NOISE_TAIL_START taps.
    """
    Hp = np.array([pilot_observations(f) for f in rx_frames])
    cir = np.fft.ifft(Hp, axis=1)
    tail = np.abs(cir[:, NOISE_TAIL_START:]) ** 2
    return float(max(N_PILOTS * np.mean(tail) * abs(PILOT_VALUE) ** 2, 1e-6))


def estimate_snr_db(rx_frames, noise_var=None):
    """Receiver-side SNR estimate: (P_y - sigma2_hat) / sigma2_hat, with P_y the mean
    received power over all subcarriers (E|X|^2 = 1, channel energy normalised to 1)."""
    if noise_var is None:
        noise_var = estimate_noise_var(rx_frames)
    p_y = float(np.mean([np.mean(np.abs(f) ** 2) for f in rx_frames]))
    sig = max(p_y - noise_var, 1e-3 * p_y)
    return float(np.clip(10 * np.log10(sig / noise_var), -10.0, 50.0))


class FrontEnd:
    """Per-burst cache of receiver-observable quantities shared by all strategies."""

    def __init__(self, rx_frames):
        self.rx_frames = rx_frames
        self.Hp = np.array([pilot_observations(f) for f in rx_frames])
        self.noise_var = estimate_noise_var(rx_frames)
        self.snr_db = estimate_snr_db(rx_frames, self.noise_var)
        self._H_ls = {}

    def H_ls(self, n):
        if n not in self._H_ls:
            self._H_ls[n] = ls_channel_estimate(self.rx_frames[n])
        return self._H_ls[n]


# =============================================================================
# Strategies A-F
# =============================================================================
STRATEGY_IDS = [0, 1, 2, 3, 4, 5]
STRATEGY_LETTERS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
STRATEGY_NAMES = {
    0: "A: LS + ZF",
    1: "B: DFT-LS + MMSE",
    2: "C: LMMSE + MMSE",
    3: "D: LS-init + LMS",
    4: "E: LS-init + NLMS",
    5: "F: LS-init + RLS",
}
REQUIRES_TRAINING = {0: False, 1: False, 2: False, 3: True, 4: True, 5: True}


@dataclass
class StrategyResult:
    bits: np.ndarray            # recovered payload bits
    syms: np.ndarray            # equalized payload symbols (pre-decision)
    error_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))   # adaptive |e|^2 per frame
    H_est: list = field(default_factory=list)                              # channel estimates (A-C)


def run_strategy(strategy_id, burst, front_end=None, adaptive_params=None):
    """Run strategy `strategy_id` on a Burst. Only receiver-observable data are used:
    burst.rx_frames, burst.train_syms and burst.n_train."""
    fe = front_end or FrontEnd(burst.rx_frames)
    n0 = burst.n_train
    frames = burst.rx_frames
    pay = range(n0, len(frames))
    snr_lin = 1.0 / fe.noise_var
    ap = adaptive_params or {}

    if strategy_id == 0:                                   # A: LS + ZF
        H = [fe.H_ls(n) for n in pay]
        syms = np.concatenate([zf_equalize(frames[n], Hn) for n, Hn in zip(pay, H)])
        return StrategyResult(qpsk_demodulate(syms), syms, H_est=H)

    if strategy_id == 1:                                   # B: DFT-LS + MMSE
        H = [dft_filter_estimate(fe.H_ls(n)) for n in pay]
        syms = np.concatenate([mmse_equalize(frames[n], Hn, snr_lin) for n, Hn in zip(pay, H)])
        return StrategyResult(qpsk_demodulate(syms), syms, H_est=H)

    if strategy_id == 2:                                   # C: LMMSE + MMSE
        W = lmmse_weights(fe.noise_var)
        H = [W @ fe.Hp[n] for n in pay]
        syms = np.concatenate([mmse_equalize(frames[n], Hn, snr_lin) for n, Hn in zip(pay, H)])
        return StrategyResult(qpsk_demodulate(syms), syms, H_est=H)

    if strategy_id in (3, 4, 5):                           # D/E/F adaptive equalizers
        kind = {3: "lms", 4: "nlms", 5: "rls"}[strategy_id]
        H0 = fe.H_ls(0)[DATA_IDX]
        w0 = np.conj(H0) / (np.abs(H0) ** 2 + fe.noise_var)   # regularised inverse of LS estimate
        y = np.array([f[DATA_IDX] for f in frames])
        outs, curve, _ = adaptive.run_adaptive_burst(kind, y, burst.train_syms, w0, **ap)
        syms = outs.reshape(-1)
        return StrategyResult(qpsk_demodulate(syms), syms, error_curve=curve)

    raise ValueError("unknown strategy id %r" % (strategy_id,))


def burst_metrics(burst, result):
    """Offline metrics of one strategy on one burst (uses simulator ground truth).
    Returns dict: bit_err, sym_err, sq_err (sum |x^-x|^2), n_bits, n_syms, ch_sq_err, ch_n."""
    bit_err = int(np.sum(burst.tx_bits != result.bits))
    sym_err = int(np.sum((burst.tx_bits != result.bits).reshape(-1, 2).any(axis=1)))
    sq_err = float(np.sum(np.abs(result.syms - burst.tx_syms) ** 2))
    ch_sq, ch_n = 0.0, 0
    for n, Hn in zip(range(burst.n_train, burst.n_frames), result.H_est):
        ch_sq += float(np.sum(np.abs(Hn - burst.H_true[n]) ** 2))
        ch_n += len(Hn)
    return dict(bit_err=bit_err, sym_err=sym_err, sq_err=sq_err,
                n_bits=len(burst.tx_bits), n_syms=len(burst.tx_syms),
                ch_sq_err=ch_sq, ch_n=ch_n)


# =============================================================================
# Complexity proxy, spectral efficiency, runtime
# =============================================================================
def complexity_cm_per_frame(strategy_id, n_frames=10):
    """Complexity PROXY: complex-multiplication equivalents (CM) per received OFDM
    symbol, amortising per-burst work over `n_frames`.  Conventions: complex x complex
    multiply = 1 CM, real x complex = 0.5, complex division = 2, FFT = (N/2)log2 N,
    slicer = 0.25 per subcarrier.  This is an arithmetic count only - it ignores memory
    traffic, and the training overhead is reported separately (spectral efficiency)."""
    N, Np, Nd = N_SC, N_PILOTS, N_DATA
    ls = Np + N
    fft = (N / 2) * np.log2(N)
    zf, mmse = 2 * Nd, 3 * Nd
    init = (ls + 3 * Nd) / n_frames          # LS estimate + weight init, once per burst
    table = {
        0: ls + zf,
        1: ls + 2 * fft + mmse,
        2: Np + N * Np + (N * Np ** 2 + Np ** 3 / 3.0) / n_frames + mmse,
        3: 2.75 * Nd + init,
        4: 3.75 * Nd + init,
        5: 4.75 * Nd + init,
    }
    return float(table[strategy_id])


def net_spectral_efficiency(strategy_id, n_pay, n_train, n_bits_per_sc=2):
    """Net payload bits per transmitted time-domain sample (= bit/s/Hz at Nyquist rate).
    Overhead accounted: pilots (N_PILOTS of N_SC subcarriers), cyclic prefix (CP_LEN per
    symbol) and, for strategies that need it, n_train training symbols per burst."""
    train = n_train if REQUIRES_TRAINING[strategy_id] else 0
    payload_bits = n_pay * N_DATA * n_bits_per_sc
    samples = (n_pay + train) * (N_SC + CP_LEN)
    return payload_bits / samples


def measure_runtime(strategy_id, bursts, repeats=3):
    """Median wall-clock time per received OFDM frame (microseconds), including the
    strategy's own front-end work (a fresh FrontEnd per call). Environment dependent."""
    times = []
    n_frames = sum(b.n_frames for b in bursts)
    for _ in range(repeats):
        t0 = time.perf_counter()
        for b in bursts:
            run_strategy(strategy_id, b)
        times.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(times)) / n_frames


# =============================================================================
# Feature extraction (receiver-observable only)
# =============================================================================
FEATURE_NAMES = [
    "est_snr_db",
    "pilot_res_power",
    "delay_spread_proxy",
    "chan_variation",
    "chan_power_spread",
    "chan_est_peak_to_avg",
    "deep_fade_ratio",
    "zf_noise_enhance_log",
    "doppler_proxy",
    "chan_est_corr",
]

FEATURE_DOCS = {
    "est_snr_db": "dB, [-10,50]. 10log10((P_y - s2)/s2); s2 = 16*mean|h16[n>=12]|^2 (pilot-CIR tail noise estimate); P_y = mean received power.",
    "pilot_res_power": "unitless, >=0. mean|H_p - LP(H_p)|^2 / mean|H_p|^2, LP = pilot-domain CIR truncated to CP_LEN/2 taps. Non-trivial residual (noise + long delay tail); averaged over frames.",
    "delay_spread_proxy": "samples, [0,~5]. RMS delay spread of the noise-floor-corrected 16-tap pilot CIR power profile (taps<12), averaged over frames.",
    "chan_variation": "unitless, >=0. mean|diff_k H_LS[k]| over subcarriers (frequency selectivity / noise roughness), averaged over frames.",
    "chan_power_spread": "unitless, >=0. std(|H_LS|^2)/mean(|H_LS|^2) over subcarriers.",
    "chan_est_peak_to_avg": "unitless, >=1. max|H_LS| / mean|H_LS|.",
    "deep_fade_ratio": "fraction in [0,1]. share of subcarriers with |H_LS| < 0.3.",
    "zf_noise_enhance_log": "log10 of mean(1/max(|H_LS|^2,1e-6)): proxy for ZF noise boost.",
    "doppler_proxy": "unitless, >=0. max(0, mean|H_p[n]-H_p[n-1]|^2 - 2 s2) / mean|H_p|^2 : frame-to-frame channel change beyond noise (0 for single-frame bursts).",
    "chan_est_corr": "[-1,1]. mean over consecutive frames of Re<H_p[n],H_p[n-1]>/(|H_p[n]||H_p[n-1]|) (1 for single-frame bursts).",
}

_FEATURE_LP_TAPS = CP_LEN // 2


def extract_features_burst(rx_frames, snr_db=None):
    """Compute the 10 receiver-observable features from a burst of received frames.

    `snr_db` is accepted only for backward compatibility and is IGNORED: the SNR feature
    is estimated from the pilots, never taken from the simulator configuration.
    Nothing here uses the true channel, true SNR, true Doppler or transmitted payload.
    """
    fe = rx_frames if isinstance(rx_frames, FrontEnd) else FrontEnd(rx_frames)
    frames = fe.rx_frames
    s2 = fe.noise_var
    Hp = fe.Hp                                              # (n_frames, 16)
    cir = np.fft.ifft(Hp, axis=1)

    # pilot residual after low-pass (short-CIR) fit
    cir_lp = cir.copy()
    cir_lp[:, _FEATURE_LP_TAPS:] = 0.0
    Hp_lp = np.fft.fft(cir_lp, axis=1)
    pr = np.mean(np.abs(Hp - Hp_lp) ** 2) / (np.mean(np.abs(Hp) ** 2) + 1e-12)

    # delay-spread proxy from the noise-corrected pilot CIR power profile
    p = np.mean(np.abs(cir) ** 2, axis=0) - s2 / N_PILOTS
    p = np.clip(p, 0.0, None)
    p[NOISE_TAIL_START:] = 0.0
    if p.sum() > 1e-12:
        n = np.arange(N_PILOTS)
        m1 = np.sum(n * p) / p.sum()
        ds = float(np.sqrt(max(np.sum(n ** 2 * p) / p.sum() - m1 ** 2, 0.0)))
    else:
        ds = 0.0

    per_frame = []
    for i in range(len(frames)):
        H = fe.H_ls(i)
        mag = np.abs(H)
        pw = mag ** 2
        per_frame.append([
            np.mean(np.abs(np.diff(H))),
            np.std(pw) / (np.mean(pw) + 1e-12),
            np.max(mag) / (np.mean(mag) + 1e-12),
            np.mean(mag < 0.3),
            np.log10(np.mean(1.0 / np.maximum(pw, 1e-6))),
        ])
    pf = np.mean(per_frame, axis=0)

    if len(frames) > 1:
        d = Hp[1:] - Hp[:-1]
        dop = max(0.0, np.mean(np.abs(d) ** 2) - 2 * s2) / (np.mean(np.abs(Hp) ** 2) + 1e-12)
        num = np.real(np.sum(Hp[1:] * np.conj(Hp[:-1]), axis=1))
        den = np.linalg.norm(Hp[1:], axis=1) * np.linalg.norm(Hp[:-1], axis=1) + 1e-12
        corr = float(np.mean(num / den))
    else:
        dop, corr = 0.0, 1.0

    return np.array([fe.snr_db, pr, ds, pf[0], pf[1], pf[2], pf[3], pf[4], dop, corr], dtype=float)


def extract_features_single(rx_frame, snr_db=None):
    """Feature vector from a single frame (wraps the burst version)."""
    return extract_features_burst([rx_frame])


# =============================================================================
# Documented rule-based selector (baseline for the ML selector)
# =============================================================================
RULE_SNR_LOW_DB = 14.0     # below: noise-robust DFT-LS + MMSE (B)
RULE_DOPPLER_HIGH = 0.02   # above (at high SNR): per-frame LMMSE re-estimation (C); else cheap LS + ZF (A)


def rule_based_selector(features):
    """Hand-written threshold rules on the SAME receiver-observable features.
    1. est_snr_db < RULE_SNR_LOW_DB            -> B (DFT-LS + MMSE)
    2. else doppler_proxy > RULE_DOPPLER_HIGH  -> C (LMMSE + MMSE, re-estimates every frame)
    3. else                                    -> A (LS + ZF, cheapest)"""
    snr = features[FEATURE_NAMES.index("est_snr_db")]
    dop = features[FEATURE_NAMES.index("doppler_proxy")]
    if snr < RULE_SNR_LOW_DB:
        return 1
    if dop > RULE_DOPPLER_HIGH:
        return 2
    return 0
