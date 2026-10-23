"""
ofdm.py - SISO-OFDM Physical Layer Simulation
================================================
Implements QPSK modulation, IFFT/FFT, cyclic prefix insertion/removal,
multipath Rayleigh fading channel, AWGN noise, and pilot-based LS
channel estimation with linear interpolation.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# -- OFDM System Parameters --
N_SC = 64              # Total subcarriers
CP_LEN = 16            # Cyclic prefix length
PILOT_SPACING = 4      # One pilot every 4 subcarriers
PILOT_VALUE = 1 + 0j   # Known pilot symbol (unit energy)

# Derived indices
PILOT_IDX = np.arange(0, N_SC, PILOT_SPACING)
DATA_IDX = np.array([i for i in range(N_SC) if i not in PILOT_IDX])
N_DATA = len(DATA_IDX)   # 48 data subcarriers
N_PILOTS = len(PILOT_IDX)

# -- QPSK Constellation --
QPSK_TABLE = np.array([
    ( 1 + 1j) / np.sqrt(2),   # 00
    (-1 + 1j) / np.sqrt(2),   # 01
    ( 1 - 1j) / np.sqrt(2),   # 10
    (-1 - 1j) / np.sqrt(2),   # 11
])


def random_bits(n_bits, rng=None):
    """Generate *n_bits* random bits."""
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=n_bits)


def qpsk_modulate(bits):
    """Map bit array (length must be even) to QPSK symbols."""
    assert len(bits) % 2 == 0
    idx = bits[0::2] * 2 + bits[1::2]   # 2-bit index per symbol
    return QPSK_TABLE[idx]


def qpsk_demodulate(symbols):
    """Hard-decision QPSK demodulation -> bit array (vectorised).

    For the Gray-style table above: MSB = 1 when Im < 0, LSB = 1 when Re < 0,
    which is identical to the minimum-distance decision."""
    symbols = np.asarray(symbols)
    bits = np.zeros(2 * len(symbols), dtype=int)
    bits[0::2] = (np.imag(symbols) < 0).astype(int)
    bits[1::2] = (np.real(symbols) < 0).astype(int)
    return bits


def build_ofdm_symbol(data_symbols):
    """Insert pilots and data into an OFDM frequency-domain frame."""
    assert len(data_symbols) == N_DATA
    frame = np.zeros(N_SC, dtype=complex)
    frame[PILOT_IDX] = PILOT_VALUE
    frame[DATA_IDX] = data_symbols
    return frame


def add_cyclic_prefix(time_signal):
    """Prepend cyclic prefix."""
    return np.concatenate([time_signal[-CP_LEN:], time_signal])


def remove_cyclic_prefix(rx_signal):
    """Remove cyclic prefix."""
    return rx_signal[CP_LEN:]


def multipath_channel(n_taps, rng=None):
    """Generate a random Rayleigh fading multipath channel impulse response
    with an exponentially decaying power-delay profile."""
    rng = rng or np.random.default_rng()
    pdp = np.exp(-np.arange(n_taps) / 2.0)
    h = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2)
    h *= np.sqrt(pdp)
    h /= np.sqrt(np.sum(np.abs(h)**2))
    return h


def apply_channel(tx_signal, h):
    """Convolve transmit signal with channel impulse response."""
    return np.convolve(tx_signal, h, mode='full')[:len(tx_signal)]


def add_awgn(signal, snr_db, rng=None):
    """Add AWGN noise for a given SNR (dB)."""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


def ls_channel_estimate(rx_frame):
    """Least-Squares channel estimation at pilot positions,
    linearly interpolated to all subcarriers."""
    H_pilot = rx_frame[PILOT_IDX] / PILOT_VALUE
    H_est = np.interp(np.arange(N_SC),
                      PILOT_IDX.astype(float),
                      np.real(H_pilot)) \
           + 1j * np.interp(np.arange(N_SC),
                            PILOT_IDX.astype(float),
                            np.imag(H_pilot))
    return H_est


def dft_filter_estimate(H_ls, max_delay_taps=CP_LEN):
    """DFT-domain filtering of an (interpolated) LS estimate: IFFT -> keep only
    the first *max_delay_taps* CIR taps -> FFT."""
    h_cir = np.fft.ifft(H_ls, N_SC)
    h_cir[max_delay_taps:] = 0.0
    return np.fft.fft(h_cir, N_SC)


def dft_channel_estimate(rx_frame, max_delay_taps=CP_LEN):
    """Transform-domain (DFT-filtered) channel estimation:
    1. Compute linearly interpolated LS estimate H_ls.
    2. Convert to time domain via IFFT (channel impulse response).
    3. Truncate / zero-out CIR taps beyond the maximum expected channel
       delay spread (<= CP_LEN), eliminating out-of-band noise.
    4. Convert back to frequency domain via FFT.
    """
    return dft_filter_estimate(ls_channel_estimate(rx_frame), max_delay_taps)


# -- LMMSE channel estimation (pilot domain) -----------------------------------
# Estimator (Wiener filter over the pilot observations):
#     H_hat = R_hp (R_pp + sigma2 / |X_p|^2 * I)^-1 * H_LS,p
# with  H_LS,p = Y_p / X_p                      (16 x 1, noisy pilot observations)
#       R_hh[k,l] = sum_n p_n exp(-j 2 pi n (k-l) / N)   (64 x 64 frequency covariance
#                   of a WSSUS channel with power-delay profile p_n, sum p_n = 1)
#       R_pp = R_hh[pilots, pilots]  (16 x 16),   R_hp = R_hh[:, pilots]  (64 x 16)
# Statistics used by the *operational receiver* (never the true per-scenario values):
#   * PDP prior : exponential decay exp(-n/2), n = 0..LMMSE_ASSUMED_TAPS-1 -- the same
#                 *family* the simulator draws from (a design-time prior), but NOT the
#                 scenario's actual tap count. A mismatch is therefore possible/expected.
#   * sigma2    : estimated by the receiver from pilot-domain CIR tail taps (receiver.py).
# Offline analysis may additionally use an "oracle" variant (true PDP + true sigma2) to
# quantify how much performance is lost to the unknown statistics.
LMMSE_ASSUMED_TAPS = 8
LMMSE_PDP_DECAY = 2.0
_R_CACHE = {}


def assumed_pdp(n_taps=LMMSE_ASSUMED_TAPS, decay=LMMSE_PDP_DECAY):
    """Normalised exponential power-delay profile (sum = 1)."""
    p = np.exp(-np.arange(n_taps) / decay)
    return p / p.sum()


def frequency_covariance(pdp):
    """64x64 Hermitian PSD frequency-domain channel covariance for a given PDP."""
    key = tuple(np.round(pdp, 12))
    if key not in _R_CACHE:
        k = np.arange(N_SC)
        d = k[:, None] - k[None, :]
        delays = np.arange(len(pdp))
        _R_CACHE[key] = np.sum(
            pdp[None, None, :] * np.exp(-2j * np.pi * d[:, :, None] * delays / N_SC),
            axis=-1)
    return _R_CACHE[key]


def lmmse_weights(noise_var, pdp=None):
    """LMMSE interpolation/denoising matrix W (N_SC x N_PILOTS) so that H_hat = W @ H_LS,p.
    Computed with a linear solve (no explicit inverse) and a tiny diagonal loading
    for numerical stability."""
    pdp = assumed_pdp() if pdp is None else np.asarray(pdp, dtype=float)
    R = frequency_covariance(pdp)
    R_pp = R[np.ix_(PILOT_IDX, PILOT_IDX)]
    R_hp = R[:, PILOT_IDX]
    A = R_pp + (max(noise_var, 1e-9) / abs(PILOT_VALUE) ** 2 + 1e-9) * np.eye(N_PILOTS)
    # W = R_hp A^-1  <=>  A^T W^T = R_hp^T
    return np.linalg.solve(A.T, R_hp.T).T


def lmmse_channel_estimate(rx_frame, noise_var, pdp=None, W=None):
    """LMMSE channel estimate on all N_SC subcarriers from one received frame."""
    if W is None:
        W = lmmse_weights(noise_var, pdp)
    return W @ (rx_frame[PILOT_IDX] / PILOT_VALUE)


# -- 16-QAM Modulation and Demodulation (Normalized, Gray-coded) --
QAM16_LEVELS = np.array([-3, -1, 3, 1]) / np.sqrt(10)


def qam16_modulate(bits):
    """Map bit array (length must be multiple of 4) to normalized 16-QAM symbols."""
    assert len(bits) % 4 == 0, "Bit length must be a multiple of 4 for 16-QAM"
    b_I = bits[0::4] * 2 + bits[1::4]
    b_Q = bits[2::4] * 2 + bits[3::4]
    return QAM16_LEVELS[b_I] + 1j * QAM16_LEVELS[b_Q]


def qam16_demodulate(symbols):
    """Hard-decision 16-QAM demodulation -> bit array (vectorised)."""
    symbols = np.asarray(symbols)
    idx_I = np.argmin(np.abs(np.real(symbols)[:, None] - QAM16_LEVELS[None, :]), axis=1)
    idx_Q = np.argmin(np.abs(np.imag(symbols)[:, None] - QAM16_LEVELS[None, :]), axis=1)
    bits = np.zeros(4 * len(symbols), dtype=int)
    bits[0::4] = idx_I >> 1
    bits[1::4] = idx_I & 1
    bits[2::4] = idx_Q >> 1
    bits[3::4] = idx_Q & 1
    return bits


def transmit_ofdm_symbol(bits, h, snr_db, rng=None):
    """Transmit a single OFDM symbol through a *given* channel h.

    Returns
    -------
    rx_frame : ndarray  - frequency-domain received frame
    tx_data  : ndarray  - transmitted QPSK data symbols (for MSE)
    """
    rng = rng or np.random.default_rng()
    data_syms = qpsk_modulate(bits)
    frame_tx = build_ofdm_symbol(data_syms)
    tx_time = np.fft.ifft(frame_tx, N_SC)
    tx_cp = add_cyclic_prefix(tx_time)
    rx_cp = apply_channel(tx_cp, h)
    rx_cp = add_awgn(rx_cp, snr_db, rng=rng)
    rx_time = remove_cyclic_prefix(rx_cp)
    rx_frame = np.fft.fft(rx_time, N_SC)
    return rx_frame, data_syms


def transmit_burst(n_symbols, snr_db, n_taps=5, rng=None):
    """Transmit a burst of *n_symbols* OFDM symbols through the SAME
    channel realisation.  This gives enough bits for BER differences
    between equalizers to emerge.

    Returns
    -------
    rx_frames : list of ndarray  - received frequency-domain frames
    all_bits  : ndarray          - all transmitted bits concatenated
    h         : ndarray          - channel impulse response
    all_data_syms : list         - transmitted data symbols per frame
    """
    rng = rng or np.random.default_rng()
    h = multipath_channel(n_taps, rng=rng)

    rx_frames = []
    all_bits = []
    all_data_syms = []
    for _ in range(n_symbols):
        bits = random_bits(2 * N_DATA, rng=rng)
        rx_frame, data_syms = transmit_ofdm_symbol(bits, h, snr_db, rng=rng)
        rx_frames.append(rx_frame)
        all_bits.append(bits)
        all_data_syms.append(data_syms)

    return rx_frames, np.concatenate(all_bits), h, all_data_syms



# =============================================================================
# Extended burst simulation: training symbols + optional Doppler (time variation)
# =============================================================================
N_TRAIN_DEFAULT = 2          # known (training) OFDM symbols at the start of a burst
TRAINING_SEED = 20260101     # fixes the *known* training sequence (shared by Tx and Rx)
_TRAIN_ROWS = 64


def known_training_bits(n_train):
    """Deterministic training bit matrix (n_train, 2*N_DATA), known to the receiver."""
    rng = np.random.default_rng(TRAINING_SEED)
    allbits = rng.integers(0, 2, size=(_TRAIN_ROWS, 2 * N_DATA))
    return allbits[:n_train]


def bessel_j0(x):
    """Bessel function J0 via its power series (accurate for |x| < ~8)."""
    x = float(x)
    return sum((-1) ** k * (x / 2.0) ** (2 * k) / math.factorial(k) ** 2 for k in range(40))


def time_varying_channel(n_taps, n_frames, fd_ts=0.0, rng=None):
    """Channel taps per OFDM symbol, shape (n_frames, n_taps).

    fd_ts = f_D * T_sym is the normalised Doppler. fd_ts == 0 reproduces the static
    baseline channel exactly (one `multipath_channel` draw, unit energy).
    For fd_ts > 0 every tap follows a Gauss-Markov (AR(1)) process
        g_n = rho g_{n-1} + sqrt(1-rho^2) w_n,   rho = J0(2 pi fd_ts),
    with the same exponential PDP as the baseline, scaled so that the EXPECTED channel
    energy is 1 (E||h||^2 = 1; instantaneous energy fluctuates like a Rayleigh channel,
    unlike the static baseline which normalises every realisation to exactly 1).
    Model limitation: the channel is constant *within* an
    OFDM symbol (block fading per symbol), so inter-carrier interference is NOT modelled.
    """
    rng = rng or np.random.default_rng()
    if fd_ts == 0.0:
        h = multipath_channel(n_taps, rng=rng)
        return np.tile(h, (n_frames, 1))
    rho = bessel_j0(2 * np.pi * fd_ts)
    amp = np.sqrt(np.exp(-np.arange(n_taps) / 2.0))

    def draw():
        return (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2) * amp

    g = draw()
    scale = 1.0 / np.sqrt(np.sum(np.exp(-np.arange(n_taps) / 2.0)))
    out = np.zeros((n_frames, n_taps), dtype=complex)
    for n in range(n_frames):
        if n > 0:
            g = rho * g + np.sqrt(max(1 - rho ** 2, 0.0)) * draw()
        out[n] = g * scale
    return out


@dataclass
class Burst:
    """One simulated burst: n_train known training symbols followed by payload symbols.

    Receiver-visible fields : rx_frames, train_syms, n_train.
    Simulator-only fields   : tx_bits, tx_syms (payload ground truth), h, H_true,
                              snr_db, noise_var_true, n_taps, fd_ts, channel_id.
    These must never reach online feature extraction / strategy selection.
    """
    rx_frames: list
    train_syms: np.ndarray
    n_train: int
    tx_bits: np.ndarray
    tx_syms: np.ndarray
    h: np.ndarray
    H_true: np.ndarray
    snr_db: float
    noise_var_true: float
    n_taps: int
    fd_ts: float
    channel_id: int

    @property
    def n_frames(self):
        return len(self.rx_frames)

    @property
    def n_pay(self):
        return len(self.rx_frames) - self.n_train


def transmit_burst_ext(n_pay, snr_db, n_taps=5, fd_ts=0.0, n_train=N_TRAIN_DEFAULT, rng=None):
    """Simulate a burst with `n_train` known training symbols + `n_pay` payload symbols."""
    rng = rng or np.random.default_rng()
    n_frames = n_train + n_pay
    hs = time_varying_channel(n_taps, n_frames, fd_ts, rng=rng)
    train_bits = known_training_bits(n_train)
    rx_frames, pay_bits, pay_syms = [], [], []
    for i in range(n_frames):
        bits = train_bits[i] if i < n_train else random_bits(2 * N_DATA, rng=rng)
        rx, syms = transmit_ofdm_symbol(bits, hs[i], snr_db, rng=rng)
        rx_frames.append(rx)
        if i >= n_train:
            pay_bits.append(bits)
            pay_syms.append(syms)
    train_syms = (np.array([qpsk_modulate(b) for b in train_bits]) if n_train > 0
                  else np.zeros((0, N_DATA), dtype=complex))
    cid = int.from_bytes(hashlib.md5(np.ascontiguousarray(hs[0]).tobytes()).digest()[:7], "little")
    return Burst(
        rx_frames=rx_frames, train_syms=train_syms, n_train=n_train,
        tx_bits=np.concatenate(pay_bits), tx_syms=np.concatenate(pay_syms),
        h=hs, H_true=np.array([np.fft.fft(h, N_SC) for h in hs]),
        snr_db=float(snr_db), noise_var_true=float(10 ** (-snr_db / 10)),
        n_taps=int(n_taps), fd_ts=float(fd_ts), channel_id=cid)
