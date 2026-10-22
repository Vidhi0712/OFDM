"""
test_system.py - Verification and Unit Tests for OFDM DSP System
=================================================================
Validates:
  1. QPSK modulation and demodulation round-trip (noise-free)
  2. 16-QAM Gray modulation and demodulation round-trip (noise-free)
  3. OFDM frame construction, IFFT, CP, and FFT round-trip (ideal channel)
  4. Multipath Rayleigh channel energy normalization
  5. LS and DFT-filtered channel estimation shapes and numerical validity
  6. DFT channel estimation noise reduction vs raw LS
  7. Equalization outputs (Strategy A: LS+ZF, Strategy B: DFT-LS+MMSE)
  8. EVM and MSE metric computation
  9. Feature extraction: shape (9,), finite values, observable-only
 10. ML Model artifact: loadable, valid predict output in {0, 1}
 11. End-to-end transmit-receive burst integrity
"""

import os
import unittest
import numpy as np

from ofdm import (
    random_bits, qpsk_modulate, qpsk_demodulate,
    qam16_modulate, qam16_demodulate,
    build_ofdm_symbol, add_cyclic_prefix, remove_cyclic_prefix,
    multipath_channel, apply_channel, add_awgn,
    ls_channel_estimate, dft_channel_estimate,
    transmit_ofdm_symbol, transmit_burst,
    N_SC, CP_LEN, N_DATA, PILOT_IDX, PILOT_VALUE,
)
from receiver import (
    zf_equalize, mmse_equalize,
    equalize_frame_a, equalize_frame_b,
    strategy_a_burst, strategy_b_burst,
    strategy_a_burst_syms, strategy_b_burst_syms,
    compute_ber, compute_evm,
    extract_features_burst, extract_features_single,
    FEATURE_NAMES,
)


class TestOFDMPhysicalLayer(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_qpsk_roundtrip(self):
        """Bits modulated and demodulated without noise must match exactly."""
        n_bits = 2 * N_DATA
        bits = random_bits(n_bits, rng=self.rng)
        symbols = qpsk_modulate(bits)
        self.assertEqual(len(symbols), N_DATA)
        demod_bits = qpsk_demodulate(symbols)
        np.testing.assert_array_equal(bits, demod_bits)

    def test_16qam_roundtrip(self):
        """16-QAM Gray-mapped modulation and demodulation round-trip."""
        n_bits = 4 * N_DATA
        bits = random_bits(n_bits, rng=self.rng)
        symbols = qam16_modulate(bits)
        self.assertEqual(len(symbols), N_DATA)
        demod_bits = qam16_demodulate(symbols)
        np.testing.assert_array_equal(bits, demod_bits)

    def test_ofdm_ideal_channel_roundtrip(self):
        """In an ideal channel (h=[1], no noise), received bits must match transmitted bits."""
        bits = random_bits(2 * N_DATA, rng=self.rng)
        syms = qpsk_modulate(bits)
        frame = build_ofdm_symbol(syms)

        # IFFT -> CP -> no channel -> remove CP -> FFT
        time_sig = np.fft.ifft(frame, N_SC)
        tx_cp = add_cyclic_prefix(time_sig)
        self.assertEqual(len(tx_cp), N_SC + CP_LEN)

        rx_time = remove_cyclic_prefix(tx_cp)
        self.assertEqual(len(rx_time), N_SC)
        rx_frame = np.fft.fft(rx_time, N_SC)

        np.testing.assert_allclose(frame, rx_frame, atol=1e-12)

    def test_multipath_channel_energy(self):
        """Channel impulse response must be normalised to unit energy."""
        h = multipath_channel(5, rng=self.rng)
        energy = np.sum(np.abs(h)**2)
        self.assertAlmostEqual(energy, 1.0, places=6)

    def test_ls_and_dft_channel_estimation(self):
        """LS and DFT channel estimates must have N_SC subcarriers and finite values."""
        bits = random_bits(2 * N_DATA, rng=self.rng)
        h = multipath_channel(4, rng=self.rng)
        rx_frame, _ = transmit_ofdm_symbol(bits, h, snr_db=15.0, rng=self.rng)

        H_ls = ls_channel_estimate(rx_frame)
        H_dft = dft_channel_estimate(rx_frame)

        self.assertEqual(len(H_ls), N_SC)
        self.assertEqual(len(H_dft), N_SC)
        self.assertTrue(np.all(np.isfinite(H_ls)))
        self.assertTrue(np.all(np.isfinite(H_dft)))

    def test_dft_channel_estimation_noise_filtering(self):
        """DFT-filtered channel estimate should have lower error than raw LS at moderate SNR."""
        h = multipath_channel(5, rng=self.rng)
        H_true = np.fft.fft(h, N_SC)
        bits = random_bits(2 * N_DATA, rng=self.rng)
        rx_frame, _ = transmit_ofdm_symbol(bits, h, snr_db=10.0, rng=self.rng)

        H_ls = ls_channel_estimate(rx_frame)
        H_dft = dft_channel_estimate(rx_frame)

        mse_ls = np.mean(np.abs(H_ls - H_true)**2)
        mse_dft = np.mean(np.abs(H_dft - H_true)**2)
        self.assertLess(mse_dft, mse_ls)

    def test_burst_transmission(self):
        """Burst transmission returns correct number of frames and bit lengths."""
        n_symbols = 5
        rx_frames, all_bits, h, all_syms = transmit_burst(
            n_symbols, snr_db=15.0, n_taps=4, rng=self.rng
        )
        self.assertEqual(len(rx_frames), n_symbols)
        self.assertEqual(len(all_bits), n_symbols * 2 * N_DATA)
        self.assertEqual(len(all_syms), n_symbols)


class TestReceiverAndEqualization(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(100)

    def test_equalizers_finite_output(self):
        """Both Strategy A and Strategy B must return finite complex arrays."""
        bits = random_bits(2 * N_DATA, rng=self.rng)
        h = multipath_channel(5, rng=self.rng)
        rx_frame, _ = transmit_ofdm_symbol(bits, h, snr_db=10.0, rng=self.rng)

        bits_a, x_a = equalize_frame_a(rx_frame, snr_db=10.0)
        bits_b, x_b = equalize_frame_b(rx_frame, snr_db=10.0)

        self.assertEqual(len(bits_a), 2 * N_DATA)
        self.assertEqual(len(bits_b), 2 * N_DATA)
        self.assertTrue(np.all(np.isfinite(x_a)))
        self.assertTrue(np.all(np.isfinite(x_b)))

    def test_ber_and_evm_computation(self):
        """BER and EVM computations match analytical ground truth."""
        b1 = np.array([0, 1, 1, 0])
        b2 = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(compute_ber(b1, b2), 0.5)

        s_true = np.array([1.0 + 0j, -1.0 + 0j])
        s_rx = np.array([1.1 + 0j, -0.9 + 0j])
        mse, evm = compute_evm(s_true, s_rx)
        self.assertAlmostEqual(mse, 0.01, places=5)
        self.assertAlmostEqual(evm, 10.0, places=4)

    def test_feature_extraction(self):
        """Feature extraction must produce 9 finite values without NaN/Inf."""
        n_symbols = 4
        rx_frames, _, _, _ = transmit_burst(
            n_symbols, snr_db=12.0, n_taps=5, rng=self.rng
        )
        feats = extract_features_burst(rx_frames, snr_db=12.0)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(len(feats), 9)
        self.assertTrue(np.all(np.isfinite(feats)))
        self.assertEqual(feats[0], 12.0)


class TestMLModelArtifacts(unittest.TestCase):

    def test_trained_model_inference(self):
        """Verify saved model loads and predicts valid classes."""
        import pickle
        model_path = os.path.join(os.path.dirname(__file__), "results", "strategy_selector.pkl")
        if not os.path.exists(model_path):
            self.skipTest("strategy_selector.pkl not found; run main.py first.")

        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        dummy_feats = np.array([[15.0, 0.2, 0.05, 0.08, 0.3, 0.9, 1.2, 0.0, 1.1]])
        pred = clf.predict(dummy_feats)
        self.assertIn(pred[0], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
