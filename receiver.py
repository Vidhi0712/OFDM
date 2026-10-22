"""
receiver.py - Equalization Strategies & Feature Extraction
===========================================================
Strategy A : LS channel estimation + Zero-Forcing equalization
Strategy B : LS channel estimation + MMSE equalization

Also provides *extract_features()* which computes receiver-observable
features for the ML strategy selector (no hidden channel info leaked).
"""

import numpy as np
from ofdm import (
    ls_channel_estimate, dft_channel_estimate, qpsk_demodulate,
    DATA_IDX, PILOT_IDX, PILOT_VALUE, N_SC, N_DATA
)


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


# -- Feature extraction (receiver-observable only) --

def extract_features_burst(rx_frames, snr_db):
    """Compute features from a burst of received frames.

    Features (9-dimensional):
      0. estimated_snr_db
      1. pilot_residual_mean_abs
      2. pilot_residual_var
      3. channel_variation         (freq selectivity from LS estimate)
      4. channel_power_spread      std(|H|^2) / mean(|H|^2)
      5. channel_est_mean_mag
      6. channel_est_peak_to_avg
      7. channel_deep_fade_ratio   fraction of subcarriers with |H| < 0.3
      8. zf_noise_enhancement      mean(1/|H|^2) - proxy for ZF noise boost

    Averaged over all frames in the burst.
    None of these use the *true* channel - only the received frames.
    """
    feat_list = []
    for rx_frame in rx_frames:
        H_est = ls_channel_estimate(rx_frame)

        pilot_residuals = rx_frame[PILOT_IDX] - H_est[PILOT_IDX] * PILOT_VALUE
        pr_abs = np.abs(pilot_residuals)

        H_mag = np.abs(H_est)
        H_pow = H_mag ** 2

        deep_fade_ratio = np.mean(H_mag < 0.3)
        noise_enhance = np.mean(1.0 / np.maximum(H_pow, 1e-6))

        feat_list.append([
            snr_db,
            np.mean(pr_abs),
            np.var(pr_abs),
            np.mean(np.abs(np.diff(H_est))),
            np.std(H_pow) / (np.mean(H_pow) + 1e-12),
            np.mean(H_mag),
            np.max(H_mag) / (np.mean(H_mag) + 1e-12),
            deep_fade_ratio,
            noise_enhance,
        ])

    return np.mean(feat_list, axis=0)


def extract_features_single(rx_frame, snr_db):
    """Extract features from a single frame (wraps burst version)."""
    return extract_features_burst([rx_frame], snr_db)


FEATURE_NAMES = [
    "est_snr_db",
    "pilot_res_mean",
    "pilot_res_var",
    "chan_variation",
    "chan_power_spread",
    "chan_est_mean_mag",
    "chan_est_peak_to_avg",
    "deep_fade_ratio",
    "zf_noise_enhance",
]
