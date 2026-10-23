"""
test_system.py - Verification and Unit Tests for the ML-assisted adaptive OFDM receiver
=========================================================================================
Run:  python test_system.py          (add -v style output is already on)

Groups
  1. OFDM physical layer          : QPSK/16-QAM round trip, IFFT/CP/FFT, channel normalisation,
                                    CP adequacy, Doppler model
  2. Channel estimation           : LS, DFT-LS, LMMSE (shape, covariance, error behaviour),
                                    pilot-based noise/SNR estimation
  3. Equalizers                   : ZF, MMSE, LMS/NLMS/RLS updates, stability, convergence
  4. Strategies A-F               : interface, no ground-truth leakage, end-to-end integrity
  5. Features                     : shape, finiteness, observability, non-trivial residual
  6. Labels / splits / selector   : deterministic tie-breaking, separation, multi-class, serialisation
  7. Metrics                      : BER / SER / MSE / EVM / spectral efficiency / complexity
  8. Pipeline smoke test          : regenerates the expected output artefacts

NOTE on two legacy tests that were REPLACED (not weakened):
  * test_feature_extraction used to assert feats[0] == configured snr_db. That encoded the
    invalid assumption that the "estimated SNR" feature may be the simulator's true SNR.
    It is replaced by estimator-accuracy and "SNR argument is ignored" tests.
  * test_trained_model_inference used a 9-feature, binary {0,1} model. It now checks the
    10-feature, 6-class artefact (and metadata consistency).
"""

import dataclasses
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from ofdm import (
    random_bits, qpsk_modulate, qpsk_demodulate,
    qam16_modulate, qam16_demodulate,
    build_ofdm_symbol, add_cyclic_prefix, remove_cyclic_prefix,
    multipath_channel, apply_channel, add_awgn,
    ls_channel_estimate, dft_channel_estimate,
    lmmse_weights, lmmse_channel_estimate, frequency_covariance, assumed_pdp,
    transmit_ofdm_symbol, transmit_burst, transmit_burst_ext,
    time_varying_channel, bessel_j0, known_training_bits,
    N_SC, CP_LEN, N_DATA, N_PILOTS, PILOT_IDX, DATA_IDX, PILOT_VALUE,
)
from receiver import (
    zf_equalize, mmse_equalize,
    equalize_frame_a, equalize_frame_b,
    strategy_a_burst, strategy_b_burst,
    strategy_a_burst_syms, strategy_b_burst_syms,
    compute_ber, compute_evm,
    extract_features_burst, extract_features_single,
    FEATURE_NAMES, FrontEnd, estimate_noise_var, estimate_snr_db,
    run_strategy, burst_metrics, StrategyResult, STRATEGY_IDS, STRATEGY_NAMES,
    complexity_cm_per_frame, net_spectral_efficiency, rule_based_selector,
)
import adaptive as A
import train_model as T


# =============================================================================
class TestOFDMPhysicalLayer(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_qpsk_roundtrip(self):
        """Bits modulated and demodulated without noise must match exactly."""
        bits = random_bits(2 * N_DATA, rng=self.rng)
        symbols = qpsk_modulate(bits)
        self.assertEqual(len(symbols), N_DATA)
        np.testing.assert_array_equal(bits, qpsk_demodulate(symbols))

    def test_16qam_roundtrip(self):
        """16-QAM Gray-mapped modulation and demodulation round-trip."""
        bits = random_bits(4 * N_DATA, rng=self.rng)
        symbols = qam16_modulate(bits)
        self.assertEqual(len(symbols), N_DATA)
        np.testing.assert_array_equal(bits, qam16_demodulate(symbols))

    def test_constellation_energy_and_gray(self):
        """QPSK and 16-QAM are unit-average-energy; 16-QAM neighbours differ in exactly 1 bit."""
        all_q = qpsk_modulate(np.array([b for v in range(4) for b in (v >> 1, v & 1)]))
        self.assertAlmostEqual(np.mean(np.abs(all_q) ** 2), 1.0)
        bits16 = np.array([[(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1] for v in range(16)]).reshape(-1)
        s = qam16_modulate(bits16)
        self.assertAlmostEqual(np.mean(np.abs(s) ** 2), 1.0)
        d_min = np.min([abs(a - b) for i, a in enumerate(s) for b in s[i + 1:]])
        b16 = bits16.reshape(16, 4)
        for i in range(16):
            for j in range(16):
                if i != j and abs(s[i] - s[j]) < d_min * 1.01:
                    self.assertEqual(int(np.sum(b16[i] != b16[j])), 1)

    def test_ofdm_ideal_channel_roundtrip(self):
        """In an ideal channel (h=[1], no noise), received frame equals transmitted frame."""
        syms = qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng))
        frame = build_ofdm_symbol(syms)
        tx_cp = add_cyclic_prefix(np.fft.ifft(frame, N_SC))
        self.assertEqual(len(tx_cp), N_SC + CP_LEN)
        rx_time = remove_cyclic_prefix(tx_cp)
        self.assertEqual(len(rx_time), N_SC)
        np.testing.assert_allclose(frame, np.fft.fft(rx_time, N_SC), atol=1e-12)

    def test_multipath_channel_energy(self):
        """Channel impulse response must be normalised to unit energy."""
        h = multipath_channel(5, rng=self.rng)
        self.assertAlmostEqual(np.sum(np.abs(h) ** 2), 1.0, places=6)

    def test_cp_adequate_gives_one_tap_model(self):
        """With CP >= delay spread, Y[k] = H[k] X[k] exactly (noise-free); with a channel
        longer than the CP the model breaks (ISI/ICI)."""
        frame = build_ofdm_symbol(qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng)))
        tx_cp = add_cyclic_prefix(np.fft.ifft(frame, N_SC))
        h = multipath_channel(8, rng=self.rng)
        rx = np.fft.fft(remove_cyclic_prefix(apply_channel(tx_cp, h)), N_SC)
        np.testing.assert_allclose(rx, np.fft.fft(h, N_SC) * frame, atol=1e-10)
        h_long = (self.rng.standard_normal(30) + 1j * self.rng.standard_normal(30)) / np.sqrt(60)
        rx_long = np.fft.fft(remove_cyclic_prefix(apply_channel(tx_cp, h_long)), N_SC)
        err = np.max(np.abs(rx_long - np.fft.fft(h_long, N_SC) * frame))
        self.assertGreater(err, 1e-3)

    def test_awgn_matches_requested_snr(self):
        sig = np.exp(2j * np.pi * np.arange(20000) / 37.0)
        for snr in (0.0, 10.0, 20.0):
            noisy = add_awgn(sig, snr, rng=self.rng)
            meas = 10 * np.log10(np.mean(np.abs(sig) ** 2) / np.mean(np.abs(noisy - sig) ** 2))
            self.assertAlmostEqual(meas, snr, delta=0.3)

    def test_bessel_j0_and_doppler_correlation(self):
        self.assertAlmostEqual(bessel_j0(0.0), 1.0)
        self.assertAlmostEqual(bessel_j0(2.404825557695773), 0.0, places=6)
        # static: constant taps with unit energy
        hs = time_varying_channel(5, 6, 0.0, rng=self.rng)
        np.testing.assert_allclose(hs, np.tile(hs[0], (6, 1)))
        self.assertAlmostEqual(np.sum(np.abs(hs[0]) ** 2), 1.0)
        # time-varying: lag-1 tap correlation ~ J0(2 pi fd)
        fd = 0.05
        num = den = 0.0
        for _ in range(400):
            h = time_varying_channel(1, 2, fd, rng=self.rng)
            num += np.real(h[1, 0] * np.conj(h[0, 0])); den += abs(h[0, 0]) ** 2
        self.assertAlmostEqual(num / den, bessel_j0(2 * np.pi * fd), delta=0.08)

    def test_ls_and_dft_channel_estimation(self):
        """LS and DFT channel estimates must have N_SC subcarriers and finite values."""
        h = multipath_channel(4, rng=self.rng)
        rx_frame, _ = transmit_ofdm_symbol(random_bits(2 * N_DATA, rng=self.rng), h, 15.0, rng=self.rng)
        for est in (ls_channel_estimate(rx_frame), dft_channel_estimate(rx_frame)):
            self.assertEqual(len(est), N_SC)
            self.assertTrue(np.all(np.isfinite(est)))

    def test_ls_exact_on_flat_channel_noise_free(self):
        """Flat known channel, no noise: LS recovers it on every subcarrier."""
        frame = build_ofdm_symbol(qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng)))
        H = 0.6 * np.exp(1j * 0.4)
        np.testing.assert_allclose(ls_channel_estimate(H * frame), H * np.ones(N_SC), atol=1e-12)

    def test_dft_channel_estimation_noise_filtering(self):
        """DFT-filtered channel estimate should have lower error than raw LS at moderate SNR."""
        h = multipath_channel(5, rng=self.rng)
        H_true = np.fft.fft(h, N_SC)
        rx_frame, _ = transmit_ofdm_symbol(random_bits(2 * N_DATA, rng=self.rng), h, 10.0, rng=self.rng)
        mse_ls = np.mean(np.abs(ls_channel_estimate(rx_frame) - H_true) ** 2)
        mse_dft = np.mean(np.abs(dft_channel_estimate(rx_frame) - H_true) ** 2)
        self.assertLess(mse_dft, mse_ls)

    def test_burst_transmission(self):
        """Burst transmission returns correct number of frames and bit lengths."""
        rx_frames, all_bits, h, all_syms = transmit_burst(5, 15.0, n_taps=4, rng=self.rng)
        self.assertEqual(len(rx_frames), 5)
        self.assertEqual(len(all_bits), 5 * 2 * N_DATA)
        self.assertEqual(len(all_syms), 5)

    def test_extended_burst_structure(self):
        b = transmit_burst_ext(6, 12.0, 4, 0.02, n_train=2, rng=self.rng)
        self.assertEqual(b.n_frames, 8)
        self.assertEqual(len(b.tx_bits), 6 * 2 * N_DATA)
        self.assertEqual(b.H_true.shape, (8, N_SC))
        np.testing.assert_array_equal(
            qpsk_demodulate(b.train_syms[0]), known_training_bits(2)[0])   # training sequence is known


# =============================================================================
class TestChannelEstimation(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_covariance_hermitian_psd_unit_diag(self):
        R = frequency_covariance(assumed_pdp())
        self.assertEqual(R.shape, (N_SC, N_SC))
        np.testing.assert_allclose(R, R.conj().T, atol=1e-12)
        self.assertGreater(np.min(np.linalg.eigvalsh(R)), -1e-9)
        np.testing.assert_allclose(np.diag(R).real, 1.0, atol=1e-12)

    def test_lmmse_shapes_and_finite(self):
        W = lmmse_weights(0.1)
        self.assertEqual(W.shape, (N_SC, N_PILOTS))
        b = transmit_burst_ext(1, 10.0, 5, n_train=0, rng=self.rng)
        H = lmmse_channel_estimate(b.rx_frames[0], 0.1)
        self.assertEqual(H.shape, (N_SC,))
        self.assertTrue(np.all(np.isfinite(H)) and np.all(np.isfinite(W)))

    def test_lmmse_noise_free_interpolates_pilots(self):
        """sigma2 -> 0: the estimate must reproduce the pilot observations."""
        h = multipath_channel(4, rng=self.rng)
        frame = build_ofdm_symbol(qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng)))
        rx = np.fft.fft(remove_cyclic_prefix(apply_channel(add_cyclic_prefix(np.fft.ifft(frame, N_SC)), h)), N_SC)
        H = lmmse_channel_estimate(rx, 1e-10)
        np.testing.assert_allclose(H[PILOT_IDX], np.fft.fft(h, N_SC)[PILOT_IDX], atol=1e-3)

    def test_lmmse_shrinks_with_noise(self):
        """Large assumed noise variance -> estimate magnitude shrinks toward 0 (Wiener behaviour)."""
        Hp = np.ones(N_PILOTS, dtype=complex)
        small = np.linalg.norm(lmmse_weights(1e-3) @ Hp)
        large = np.linalg.norm(lmmse_weights(10.0) @ Hp)
        self.assertLess(large, 0.5 * small)

    def test_lmmse_beats_ls_with_matched_statistics(self):
        """Oracle-statistics LMMSE and operational LMMSE both beat LS at low/moderate SNR."""
        mse = {"ls": 0.0, "op": 0.0, "or": 0.0}
        n = 0
        for _ in range(60):
            b = transmit_burst_ext(2, 5.0, 8, n_train=0, rng=self.rng)
            fe = FrontEnd(b.rx_frames)
            p = np.exp(-np.arange(8) / 2.0); p /= p.sum()
            W_or = lmmse_weights(b.noise_var_true, pdp=p)
            W_op = lmmse_weights(fe.noise_var)
            for i in range(2):
                Ht = b.H_true[i]
                mse["ls"] += np.mean(np.abs(fe.H_ls(i) - Ht) ** 2)
                mse["op"] += np.mean(np.abs(W_op @ fe.Hp[i] - Ht) ** 2)
                mse["or"] += np.mean(np.abs(W_or @ fe.Hp[i] - Ht) ** 2)
                n += 1
        self.assertLess(mse["or"], 0.6 * mse["ls"])
        self.assertLess(mse["op"], 0.8 * mse["ls"])
        self.assertLessEqual(mse["or"], mse["op"] * 1.05)    # oracle statistics never meaningfully worse

    def test_estimation_error_decreases_with_snr(self):
        errs = []
        for snr in (0.0, 15.0, 30.0):
            e = 0.0
            for _ in range(20):
                b = transmit_burst_ext(1, snr, 5, n_train=0, rng=self.rng)
                fe = FrontEnd(b.rx_frames)
                e += np.mean(np.abs(lmmse_weights(fe.noise_var) @ fe.Hp[0] - b.H_true[0]) ** 2)
            errs.append(e)
        self.assertGreater(errs[0], errs[1]); self.assertGreater(errs[1], errs[2])

    def test_noise_and_snr_estimators_are_accurate(self):
        """Receiver-side (pilot-only) SNR estimate tracks the true SNR (replaces the old
        'feature 0 equals the configured SNR' assumption)."""
        for snr in (5.0, 15.0, 25.0):
            est = []
            for _ in range(30):
                b = transmit_burst_ext(8, snr, int(self.rng.choice([2, 5, 8])), rng=self.rng)
                est.append(estimate_snr_db(b.rx_frames))
            self.assertAlmostEqual(float(np.mean(est)), snr, delta=1.5)
        self.assertGreater(estimate_noise_var([build_ofdm_symbol(np.ones(N_DATA, complex))]), 0)


# =============================================================================
class TestEqualizers(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(100)

    def test_zf_inverts_known_channel(self):
        X = qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng))
        H = np.fft.fft(multipath_channel(5, rng=self.rng), N_SC)
        Y = H * build_ofdm_symbol(X)                      # full 64-subcarrier received frame
        np.testing.assert_allclose(zf_equalize(Y, H), X, atol=1e-10)

    def test_mmse_limits(self):
        """High SNR -> MMSE == ZF; low SNR -> output shrinks (regularised)."""
        H = np.fft.fft(multipath_channel(5, rng=self.rng), N_SC)
        Y = np.zeros(N_SC, dtype=complex); Y[DATA_IDX] = H[DATA_IDX] * qpsk_modulate(random_bits(2 * N_DATA, rng=self.rng))
        np.testing.assert_allclose(mmse_equalize(Y, H, 1e8), zf_equalize(Y, H), atol=1e-5)
        self.assertLess(np.linalg.norm(mmse_equalize(Y, H, 0.1)), np.linalg.norm(zf_equalize(Y, H)))

    def test_equalizers_finite_output(self):
        bits = random_bits(2 * N_DATA, rng=self.rng)
        h = multipath_channel(5, rng=self.rng)
        rx_frame, _ = transmit_ofdm_symbol(bits, h, 10.0, rng=self.rng)
        bits_a, x_a = equalize_frame_a(rx_frame, snr_db=10.0)
        bits_b, x_b = equalize_frame_b(rx_frame, snr_db=10.0)
        self.assertEqual(len(bits_a), 2 * N_DATA); self.assertEqual(len(bits_b), 2 * N_DATA)
        self.assertTrue(np.all(np.isfinite(x_a)) and np.all(np.isfinite(x_b)))

    # ---- adaptive filters ----------------------------------------------------
    def test_lms_update_formula(self):
        w, y, d, mu = 0.5 + 0.2j, 1 - 1j, 1 + 0j, 0.1
        e = d - w * y
        w_new, e_ret = A.lms_update(np.array([w]), np.array([y]), np.array([d]), mu)
        self.assertAlmostEqual(e_ret[0], e)
        self.assertAlmostEqual(w_new[0], w + mu * e * np.conj(y))

    def test_nlms_update_and_aposteriori_zero_error(self):
        """NLMS with mu=1, eps->0 forces the a-posteriori error to zero (classic property),
        and the step is invariant to input scale."""
        w, y, d = np.array([0.3 - 0.1j]), np.array([2.0 + 1j]), np.array([1 - 1j])
        w_new, _ = A.nlms_update(w, y, d, mu=1.0, eps=1e-12)
        self.assertAlmostEqual(abs(d[0] - w_new[0] * y[0]), 0.0, places=8)
        # scale invariance of the output correction: |dw|*|y| = mu*|e| for any input scale (w0 = 0 -> e = d)
        z = np.zeros(1, complex)
        w1, _ = A.nlms_update(z, y, d, 0.5, 1e-12)
        w2, _ = A.nlms_update(z, 10 * y, d, 0.5, 1e-12)
        self.assertAlmostEqual(abs(w1[0]) * abs(y[0]), 0.5 * abs(d[0]), places=8)
        self.assertAlmostEqual(abs(w2[0]) * abs(10 * y[0]), 0.5 * abs(d[0]), places=8)

    def test_rls_matches_batch_least_squares(self):
        """lam=1, w0=0, tiny delta: recursive solution == batch LS  w = sum(conj(y)d)/sum|y|^2."""
        y = self.rng.standard_normal(50) + 1j * self.rng.standard_normal(50)
        d = (0.8 - 0.3j) * y + 0.05 * (self.rng.standard_normal(50) + 1j * self.rng.standard_normal(50))
        bank = A.AdaptiveEqualizerBank("rls", np.zeros(1, complex), lam=1.0, delta=1e-9)
        for yi, di in zip(y, d):
            bank.step(np.array([yi]), np.array([di]))
        w_batch = np.sum(np.conj(y) * d) / np.sum(np.abs(y) ** 2)
        self.assertAlmostEqual(abs(bank.w[0] - w_batch), 0.0, places=6)
        self.assertGreater(bank.P[0], 0)

    def test_convergence_on_known_sequence_complex_channel(self):
        """Noise-free y = H d with complex H: every algorithm converges to w = 1/H.
        (Fails if the conjugate in the update is wrong.)"""
        H = 0.7 * np.exp(1j * 0.9)
        d = qpsk_modulate(random_bits(600, rng=self.rng))
        for kind, n_it in (("lms", 400), ("nlms", 150), ("rls", 10)):
            curve, w = A.learning_curve(kind, (H * d[:n_it]).reshape(-1, 1), d[:n_it].reshape(-1, 1),
                                        np.array([1.0 + 0j]), **({"lam": 1.0, "delta": 1e-6} if kind == "rls" else {}))
            self.assertLess(abs(w[0] - 1 / H), 1e-3, kind)
            self.assertLess(np.mean(curve[-5:]), 0.01 * np.mean(curve[:3]), kind)

    def test_noisy_convergence_reaches_noise_floor(self):
        H, sigma2 = 0.9 * np.exp(-0.5j), 0.01
        n = 400
        d = qpsk_modulate(random_bits(2 * n, rng=self.rng))
        noise = np.sqrt(sigma2 / 2) * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        y = H * d + noise
        for kind in A.KINDS:
            curve, _ = A.learning_curve(kind, y.reshape(-1, 1), d.reshape(-1, 1), np.array([1.0 + 0j]))
            steady = np.mean(curve[-100:])
            self.assertLess(steady, 4 * sigma2 / abs(H) ** 2, kind)       # near the Wiener floor
            self.assertLess(steady, 0.3 * np.mean(curve[:5]), kind)

    def test_stability_bounds(self):
        d = np.ones(200, dtype=complex)
        # LMS: stable for mu|y|^2 < 2, diverges beyond
        y = 2.0 * np.ones(200, dtype=complex)         # |y|^2 = 4 -> limit mu < 0.5
        c_ok, w_ok = A.learning_curve("lms", y.reshape(-1, 1), d.reshape(-1, 1), np.zeros(1, complex), mu=0.4)
        c_bad, _ = A.learning_curve("lms", y.reshape(-1, 1), d.reshape(-1, 1), np.zeros(1, complex), mu=0.6)
        self.assertLess(c_ok[-1], 1e-6); self.assertGreater(c_bad[-1], 1e3)
        # NLMS: stable for mu in (0,2) whatever the input scale
        for scale in (1e-2, 1.0, 1e2):
            c, w = A.learning_curve("nlms", (scale * y).reshape(-1, 1), d.reshape(-1, 1), np.zeros(1, complex), mu=1.9)
            self.assertTrue(np.all(np.isfinite(c)) and c[-1] < 1e-4)
        # RLS with forgetting: finite, positive P over a long noisy run
        yy = self.rng.standard_normal(2000) + 1j * self.rng.standard_normal(2000)
        dd = (0.5 + 0.5j) * yy + 0.1 * self.rng.standard_normal(2000)
        bank = A.AdaptiveEqualizerBank("rls", np.zeros(1, complex), lam=0.9, delta=0.5)
        for yi, di in zip(yy, dd):
            bank.step(np.array([yi]), np.array([di]))
        self.assertTrue(np.all(np.isfinite(bank.w)) and bank.P[0] > 0 and np.isfinite(bank.P[0]))

    def test_slicer_and_convergence_time(self):
        np.testing.assert_allclose(A.qpsk_slicer(np.array([0.1 + 2j, -3 - 0.1j])),
                                   np.array([1 + 1j, -1 - 1j]) / np.sqrt(2))
        curve = np.concatenate([np.linspace(1, 0.01, 50), 0.01 * np.ones(150)])
        n, steady = A.convergence_time(curve)
        self.assertAlmostEqual(steady, 0.01)
        self.assertTrue(40 <= n <= 50)


# =============================================================================
class TestStrategies(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def test_interface_lengths_and_finiteness(self):
        b = transmit_burst_ext(8, 12.0, 5, 0.01, rng=self.rng)
        fe = FrontEnd(b.rx_frames)
        for s in STRATEGY_IDS:
            r = run_strategy(s, b, fe)
            self.assertEqual(len(r.bits), len(b.tx_bits), STRATEGY_NAMES[s])
            self.assertEqual(len(r.syms), len(b.tx_syms), STRATEGY_NAMES[s])
            self.assertTrue(np.all(np.isfinite(r.syms)), STRATEGY_NAMES[s])

    def test_no_ground_truth_leakage(self):
        """Outputs must be identical if every simulator-only field (true channel, payload,
        true SNR, ...) is replaced by garbage: strategies only use rx frames + training."""
        b = transmit_burst_ext(8, 12.0, 5, 0.01, rng=self.rng)
        scrambled = dataclasses.replace(
            b, tx_bits=1 - b.tx_bits, tx_syms=-b.tx_syms, h=b.h * 0, H_true=b.H_true * 0,
            snr_db=-99.0, noise_var_true=123.0, n_taps=99, fd_ts=9.9)
        for s in STRATEGY_IDS:
            r1, r2 = run_strategy(s, b), run_strategy(s, scrambled)
            np.testing.assert_array_equal(r1.bits, r2.bits)
            np.testing.assert_allclose(r1.syms, r2.syms)

    def test_all_strategies_work_at_high_snr(self):
        errs = {s: 0 for s in STRATEGY_IDS}; n = 0
        for _ in range(10):
            b = transmit_burst_ext(8, 40.0, 3, 0.0, rng=self.rng)
            fe = FrontEnd(b.rx_frames)
            for s in STRATEGY_IDS:
                errs[s] += burst_metrics(b, run_strategy(s, b, fe))["bit_err"]
            n += len(b.tx_bits)
        for s in STRATEGY_IDS:
            self.assertLess(errs[s] / n, 0.02, STRATEGY_NAMES[s])

    def test_ber_decreases_with_snr(self):
        for s in (0, 2, 5):
            ber = []
            for snr in (0.0, 15.0, 30.0):
                e = n = 0
                for _ in range(15):
                    b = transmit_burst_ext(8, snr, 5, rng=self.rng)
                    e += burst_metrics(b, run_strategy(s, b))["bit_err"]; n += len(b.tx_bits)
                ber.append(e / n)
            self.assertGreater(ber[0], ber[1]); self.assertGreater(ber[1], ber[2])

    def test_adaptive_works_without_training_symbols(self):
        """n_train = 0 -> purely decision-directed mode still produces finite output."""
        b = transmit_burst_ext(6, 20.0, 3, n_train=0, rng=self.rng)
        r = run_strategy(5, b)
        self.assertEqual(len(r.syms), 6 * N_DATA); self.assertTrue(np.all(np.isfinite(r.syms)))

    def test_legacy_burst_api(self):
        rx_frames, all_bits, h, syms = transmit_burst(6, 20.0, n_taps=4, rng=self.rng)
        for fn in (strategy_a_burst, strategy_b_burst):
            self.assertEqual(len(fn(rx_frames, 20.0)), len(all_bits))
        bits, s = strategy_a_burst_syms(rx_frames, 20.0)
        self.assertEqual(len(s), 6 * N_DATA)


# =============================================================================
class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_feature_shape_finite_ranges(self):
        for snr in (0.0, 15.0, 30.0):
            b = transmit_burst_ext(6, snr, 5, 0.02, rng=self.rng)
            f = extract_features_burst(b.rx_frames)
            self.assertEqual(len(f), len(FEATURE_NAMES)); self.assertEqual(len(f), 10)
            self.assertTrue(np.all(np.isfinite(f)))
            d = dict(zip(FEATURE_NAMES, f))
            self.assertTrue(-10 <= d["est_snr_db"] <= 50)
            self.assertTrue(0 <= d["deep_fade_ratio"] <= 1)
            self.assertTrue(-1.0 <= d["chan_est_corr"] <= 1.0)
            self.assertGreaterEqual(d["doppler_proxy"], 0); self.assertGreaterEqual(d["delay_spread_proxy"], 0)
        self.assertEqual(len(extract_features_single(b.rx_frames[0])), 10)

    def test_snr_argument_is_ignored_and_not_ground_truth(self):
        """Regression test for the old leak: the SNR feature is estimated from pilots."""
        b = transmit_burst_ext(8, 12.0, 5, rng=self.rng)
        f0 = extract_features_burst(b.rx_frames, snr_db=0.0)
        f1 = extract_features_burst(b.rx_frames, snr_db=30.0)
        np.testing.assert_array_equal(f0, f1)
        self.assertNotEqual(f0[0], 12.0)
        self.assertAlmostEqual(f0[0], 12.0, delta=3.0)

    def test_features_depend_only_on_received_frames(self):
        b = transmit_burst_ext(8, 12.0, 5, rng=self.rng)
        f = extract_features_burst(b.rx_frames)
        b2 = dataclasses.replace(b, tx_bits=1 - b.tx_bits, H_true=b.H_true * 0, snr_db=-50.0, h=b.h * 0)
        np.testing.assert_array_equal(f, extract_features_burst(b2.rx_frames))

    def test_pilot_residual_is_meaningful(self):
        """Old definition (residual of the INTERPOLATED estimate at the pilots) is identically
        zero; the new feature is non-trivial and larger at low SNR."""
        b = transmit_burst_ext(4, 10.0, 5, rng=self.rng)
        fr = b.rx_frames[0]
        old = fr[PILOT_IDX] - ls_channel_estimate(fr)[PILOT_IDX] * PILOT_VALUE
        self.assertLess(np.max(np.abs(old)), 1e-12)
        idx = FEATURE_NAMES.index("pilot_res_power")
        lo = np.mean([extract_features_burst(transmit_burst_ext(4, 0.0, 5, rng=self.rng).rx_frames)[idx] for _ in range(20)])
        hi = np.mean([extract_features_burst(transmit_burst_ext(4, 30.0, 5, rng=self.rng).rx_frames)[idx] for _ in range(20)])
        self.assertGreater(lo, 0); self.assertGreater(lo, 2 * hi)

    def test_snr_feature_monotone_and_doppler_proxy_responds(self):
        mean_snr = [np.mean([extract_features_burst(transmit_burst_ext(6, s, 5, rng=self.rng).rx_frames)[0]
                             for _ in range(15)]) for s in (0.0, 10.0, 20.0, 30.0)]
        self.assertTrue(all(a < b for a, b in zip(mean_snr, mean_snr[1:])))
        di = FEATURE_NAMES.index("doppler_proxy"); ci = FEATURE_NAMES.index("chan_est_corr")
        st = [extract_features_burst(transmit_burst_ext(8, 30.0, 5, 0.0, rng=self.rng).rx_frames) for _ in range(15)]
        dy = [extract_features_burst(transmit_burst_ext(8, 30.0, 5, 0.05, rng=self.rng).rx_frames) for _ in range(15)]
        self.assertGreater(np.mean([f[di] for f in dy]), 5 * np.mean([f[di] for f in st]) + 1e-4)
        self.assertLess(np.mean([f[ci] for f in dy]), np.mean([f[ci] for f in st]))


# =============================================================================
class TestLabelsSplitsAndSelector(unittest.TestCase):

    GRID = dict(snr_range=[0, 12, 24], tap_configs=[2, 8], fd_configs=[0.0, 0.02])

    def test_labels_lowest_ber_then_cheapest(self):
        n = 768
        self.assertEqual(T.make_labels(np.array([[5, 5, 5, 5, 5, 5]]), n)[0], 0)      # all tied -> cheapest (A)
        self.assertEqual(T.make_labels(np.array([[2, 3, 0, 5, 5, 5]]), n)[0], 2)      # lowest BER wins, even if costly
        self.assertEqual(T.make_labels(np.array([[9, 9, 9, 3, 9, 3]]), n)[0], 3)      # D vs F tied -> cheaper D
        self.assertEqual(T.make_labels(np.array([[4, 2, 2, 9, 9, 9]]), n)[0], 1)      # B vs C tied -> cheaper B
        self.assertEqual(T.make_labels(np.array([[4, 2, 2, 9, 9, 9]]), n)[0],
                         T.make_labels(np.array([[4, 2, 2, 9, 9, 9]]), n)[0])         # deterministic
        # complexity penalty can never override a single bit error
        costs = T.strategy_costs([1, 0, 1, 1, 1, 1], n)
        self.assertEqual(int(np.argmin(costs)), 1)
        self.assertLess(T.LAMBDA_COMPLEXITY * 2.5, 1.0 / n)

    def test_dataset_repeatable_and_labels_consistent(self):
        d1 = T.generate_dataset(5, 2, self.GRID); d2 = T.generate_dataset(5, 2, self.GRID)
        for k in ("X", "bit_err", "y", "channel_id"):
            np.testing.assert_array_equal(d1[k], d2[k])
        np.testing.assert_array_equal(d1["y"], T.make_labels(d1["bit_err"], d1["n_bits"]))
        self.assertEqual(d1["X"].shape[1], len(FEATURE_NAMES))
        self.assertEqual(d1["bit_err"].shape[1], len(STRATEGY_IDS))

    def test_train_val_test_separation(self):
        tr = T.generate_dataset(T.SEED_TRAIN, 3, self.GRID)
        va = T.generate_dataset(T.SEED_VAL, 3, self.GRID)
        te = T.generate_dataset(T.SEED_TEST, 3, self.GRID)
        self.assertTrue(T.assert_disjoint(tr, va, te))
        with self.assertRaises(AssertionError):
            T.assert_disjoint(tr, tr)
        self.assertEqual(len({T.SEED_TRAIN, T.SEED_VAL, T.SEED_TEST}), 3)

    def test_tuning_uses_validation_and_returns_valid_multiclass_model(self):
        tr = T.generate_dataset(1, 6, self.GRID); va = T.generate_dataset(2, 4, self.GRID)
        clf, best, rows = T.train_selector(tr, va, depths=(3, 6), leaves=(5, 20))
        self.assertEqual(len(rows), 2 * 2 * 2)
        self.assertAlmostEqual(min(r["val_ber"] for r in rows),
                               T.selected_ber(va, clf.predict(va["X"])), places=12)
        self.assertTrue(set(clf.predict(va["X"])).issubset(set(range(6))))
        self.assertGreaterEqual(len(set(tr["y"])), 3)                       # genuinely multi-class labels
        self.assertEqual(clf.n_features_in_, len(FEATURE_NAMES))

    def test_serialisation_roundtrip_and_consistency_check(self):
        tr = T.generate_dataset(1, 4, self.GRID); va = T.generate_dataset(2, 3, self.GRID)
        clf, best, _ = T.train_selector(tr, va, depths=(4,), leaves=(10,), weightings=("none",))
        d = tempfile.mkdtemp()
        try:
            T.save_selector(clf, T.selector_meta(best), d)
            clf2, meta = T.load_selector(d)
            np.testing.assert_array_equal(clf.predict(va["X"]), clf2.predict(va["X"]))
            self.assertEqual(meta["feature_names"], FEATURE_NAMES)
            self.assertEqual(set(meta["strategy_names"]), {str(i) for i in STRATEGY_IDS})
            with open(os.path.join(d, "selector_meta.json")) as f:
                m = json.load(f)
            m["feature_names"] = m["feature_names"][:-1]
            with open(os.path.join(d, "selector_meta.json"), "w") as f:
                json.dump(m, f)
            with self.assertRaises(ValueError):
                T.load_selector(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_rule_based_selector(self):
        f = np.zeros(len(FEATURE_NAMES))
        f[FEATURE_NAMES.index("est_snr_db")] = 5.0
        self.assertEqual(rule_based_selector(f), 1)
        f[FEATURE_NAMES.index("est_snr_db")] = 25.0
        self.assertEqual(rule_based_selector(f), 0)
        f[FEATURE_NAMES.index("doppler_proxy")] = 0.1
        self.assertEqual(rule_based_selector(f), 2)

    def test_trained_model_inference(self):
        """If results/strategy_selector.pkl exists it must be a 10-feature multi-class tree."""
        res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        if not os.path.exists(os.path.join(res, "strategy_selector.pkl")):
            self.skipTest("strategy_selector.pkl not found; run main.py first.")
        clf, meta = T.load_selector(res)
        dummy = np.array([[15.0, 0.2, 2.0, 0.3, 0.8, 2.0, 0.03, 0.4, 0.01, 0.9]])
        self.assertIn(int(clf.predict(dummy)[0]), STRATEGY_IDS)
        self.assertEqual(clf.n_features_in_, 10)


# =============================================================================
class TestMetrics(unittest.TestCase):

    def test_ber_and_evm_computation(self):
        self.assertAlmostEqual(compute_ber(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])), 0.5)
        mse, evm = compute_evm(np.array([1.0 + 0j, -1.0 + 0j]), np.array([1.1 + 0j, -0.9 + 0j]))
        self.assertAlmostEqual(mse, 0.01, places=5); self.assertAlmostEqual(evm, 10.0, places=4)

    def test_ber_ser_mse_from_burst_metrics(self):
        rng = np.random.default_rng(3)
        b = transmit_burst_ext(4, 20.0, 3, rng=rng)
        bits = b.tx_bits.copy(); bits[0] ^= 1; bits[2] ^= 1; bits[3] ^= 1     # symbols 0 and 1 wrong; 3 bit errors
        res = StrategyResult(bits=bits, syms=b.tx_syms + 0.1 * (1 + 1j))
        m = burst_metrics(b, res)
        self.assertEqual(m["bit_err"], 3); self.assertEqual(m["sym_err"], 2)
        self.assertEqual(m["n_bits"], 4 * 96); self.assertEqual(m["n_syms"], 4 * 48)
        self.assertAlmostEqual(m["sq_err"] / m["n_syms"], 0.02, places=9)

    def test_channel_mse_metric_uses_true_channel_offline_only(self):
        rng = np.random.default_rng(4)
        b = transmit_burst_ext(3, 20.0, 3, n_train=1, rng=rng)
        res = StrategyResult(bits=b.tx_bits, syms=b.tx_syms, H_est=[b.H_true[n] + 0.1 * (1 + 1j) for n in range(1, 4)])
        m = burst_metrics(b, res)
        self.assertAlmostEqual(m["ch_sq_err"] / m["ch_n"], 0.02, places=9)

    def test_spectral_efficiency_and_complexity(self):
        self.assertAlmostEqual(net_spectral_efficiency(0, 8, 2), 8 * 96 / (8 * 80))
        self.assertAlmostEqual(net_spectral_efficiency(3, 8, 2), 8 * 96 / (10 * 80))
        self.assertLess(net_spectral_efficiency(5, 8, 2), net_spectral_efficiency(0, 8, 2))
        cm = [complexity_cm_per_frame(s, 10) for s in STRATEGY_IDS]
        self.assertLess(cm[0], cm[1]); self.assertLess(cm[1], cm[2])
        self.assertLess(cm[3], cm[4]); self.assertLess(cm[4], cm[5])


# =============================================================================
class TestPipelineSmoke(unittest.TestCase):

    def test_end_to_end_pipeline_creates_artifacts(self):
        import main as M
        d = tempfile.mkdtemp()
        try:
            summary = M.run_pipeline(d, quick=True, verbose=False)
            expected = ["ber_vs_snr.png", "ser_vs_snr.png", "symbol_mse_vs_snr.png", "channel_est_mse_vs_snr.png",
                        "convergence.png", "ml_decisions.png", "confusion_matrix.png", "feature_importances.png",
                        "performance_table.txt", "summary_receiver_level.txt", "convergence_table.txt",
                        "complexity_runtime_table.txt", "channel_est_table.txt", "adaptive_hparam_sweep.txt",
                        "decision_tree_rules.txt", "classification_report.json", "validation_grid.csv",
                        "strategy_selector.pkl", "selector_meta.json", "run_summary.json", "test_dataset.npz"]
            for fn in expected:
                p = os.path.join(d, fn)
                self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 0, fn)
            clf, meta = T.load_selector(d)
            self.assertEqual(clf.n_features_in_, 10)
            for k in ("ML selector (Decision Tree)", "Rule-based selector"):
                self.assertTrue(0.0 <= summary["test_ber"][k] <= 0.5)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
