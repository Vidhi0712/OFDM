"""
ofdm.py - SISO-OFDM Physical Layer Simulation
================================================
Implements QPSK modulation, IFFT/FFT, cyclic prefix insertion/removal,
multipath Rayleigh fading channel, AWGN noise, and pilot-based LS
channel estimation with linear interpolation.
"""

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
    """Hard-decision QPSK demodulation -> bit array."""
    bits = np.zeros(2 * len(symbols), dtype=int)
    for i, s in enumerate(symbols):
        dists = np.abs(s - QPSK_TABLE)
        nearest = np.argmin(dists)
        bits[2*i]   = nearest >> 1       # MSB
        bits[2*i+1] = nearest & 1        # LSB
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


def dft_channel_estimate(rx_frame, max_delay_taps=CP_LEN):
    """Transform-domain (DFT-filtered) channel estimation:
    1. Compute linearly interpolated LS estimate H_ls.
    2. Convert to time domain via IFFT (channel impulse response).
    3. Truncate / zero-out CIR taps beyond the maximum expected channel
       delay spread (<= CP_LEN), eliminating out-of-band noise.
    4. Convert back to frequency domain via FFT.
    """
    H_ls = ls_channel_estimate(rx_frame)
    h_cir = np.fft.ifft(H_ls, N_SC)
    h_cir[max_delay_taps:] = 0.0
    return np.fft.fft(h_cir, N_SC)


# -- 16-QAM Modulation and Demodulation (Normalized, Gray-coded) --
QAM16_LEVELS = np.array([-3, -1, 3, 1]) / np.sqrt(10)


def qam16_modulate(bits):
    """Map bit array (length must be multiple of 4) to normalized 16-QAM symbols."""
    assert len(bits) % 4 == 0, "Bit length must be a multiple of 4 for 16-QAM"
    b_I = bits[0::4] * 2 + bits[1::4]
    b_Q = bits[2::4] * 2 + bits[3::4]
    return QAM16_LEVELS[b_I] + 1j * QAM16_LEVELS[b_Q]


def qam16_demodulate(symbols):
    """Hard-decision 16-QAM demodulation -> bit array."""
    bits = np.zeros(4 * len(symbols), dtype=int)
    for i, s in enumerate(symbols):
        dist_I = np.abs(np.real(s) - QAM16_LEVELS)
        idx_I = np.argmin(dist_I)
        dist_Q = np.abs(np.imag(s) - QAM16_LEVELS)
        idx_Q = np.argmin(dist_Q)
        bits[4*i]   = idx_I >> 1
        bits[4*i+1] = idx_I & 1
        bits[4*i+2] = idx_Q >> 1
        bits[4*i+3] = idx_Q & 1
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

