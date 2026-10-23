"""
adaptive.py - Adaptive equalizers (LMS, NLMS, RLS) for SISO-OFDM
=================================================================
ARCHITECTURE (documented design decision)
-----------------------------------------
A *bank of N_DATA independent one-tap complex adaptive filters*, one per data
subcarrier k, adapted ACROSS successive OFDM symbols n (the standard frequency-domain
adaptive equalizer for OFDM).  This is NOT a channel estimator: the weights w_k are
trained directly to minimise the symbol error.

    input   y_k[n]   received FFT output on data subcarrier k, OFDM symbol n
    output  x^_k[n] = w_k[n-1] * y_k[n]                     (a-priori output)
    desired d_k[n]   = known training symbol          for n <  N_TRAIN   (training mode)
                     = hard QPSK decision of x^_k[n]   for n >= N_TRAIN   (decision-directed)
    error   e_k[n]  = d_k[n] - x^_k[n]
    The payload symbols are NEVER used as filter inputs for training; in decision-
    directed mode the "desired" signal is the receiver's own decision, not the
    transmitted symbol (no ground-truth leakage).

Complex conventions (J = |e|^2, gradient w.r.t. conj(w) is  -e * conj(y)):
    LMS  : w <- w + mu * e * conj(y)
    NLMS : w <- w + mu * e * conj(y) / (eps + |y|^2)
    RLS  : g = P conj(y) / (lam + |y|^2 P);  w <- w + g e;  P <- P / (lam + |y|^2 P)
           (scalar-regressor RLS; P is real, P0 = 1/delta; lam = forgetting factor)

Initialisation: w_k[-1] = conj(H_LS,k) / (|H_LS,k|^2 + sigma2_hat) from the LS channel
estimate of the first training symbol (this is the "LS-based" part of strategies D-F).
Training overhead: N_TRAIN OFDM symbols per burst carry no payload (see
receiver.net_spectral_efficiency).
"""

import numpy as np

# Default hyper-parameters (see README "Adaptive equalizer settings")
LMS_MU = 0.15          # stable while mu*|y|^2 < 2 for every subcarrier
NLMS_MU = 0.25         # stable for 0 < mu < 2 (chosen by tune_adaptive_params on a dedicated tuning seed)
NLMS_EPS = 1e-3        # regularisation of |y|^2 normalisation
RLS_LAMBDA = 0.98      # forgetting factor (1.0 = infinite memory)
RLS_DELTA = 0.5        # P0 = 1/delta (regularisation of the initial inverse correlation)

KINDS = ("lms", "nlms", "rls")


def qpsk_slicer(x):
    """Hard decision to the unit-energy QPSK constellation (vectorised)."""
    x = np.asarray(x)
    return (np.where(np.real(x) < 0, -1.0, 1.0) + 1j * np.where(np.imag(x) < 0, -1.0, 1.0)) / np.sqrt(2)


def lms_update(w, y, d, mu=LMS_MU):
    e = d - w * y
    return w + mu * e * np.conj(y), e


def nlms_update(w, y, d, mu=NLMS_MU, eps=NLMS_EPS):
    e = d - w * y
    return w + mu * e * np.conj(y) / (eps + np.abs(y) ** 2), e


def rls_update(w, P, y, d, lam=RLS_LAMBDA):
    e = d - w * y
    denom = lam + np.abs(y) ** 2 * P
    g = P * np.conj(y) / denom
    return w + g * e, P / denom, e


class AdaptiveEqualizerBank:
    """Bank of independent one-tap adaptive filters (vector state, any length)."""

    def __init__(self, kind, w0, mu=None, eps=NLMS_EPS, lam=RLS_LAMBDA, delta=RLS_DELTA):
        if kind not in KINDS:
            raise ValueError("kind must be one of %s" % (KINDS,))
        self.kind = kind
        self.w = np.array(w0, dtype=complex)
        self.mu = mu if mu is not None else (LMS_MU if kind == "lms" else NLMS_MU)
        self.eps, self.lam, self.delta = eps, lam, delta
        self.P = np.full(self.w.shape, 1.0 / delta)

    def output(self, y):
        return self.w * y

    def step(self, y, d):
        """Process one OFDM symbol. Returns (a-priori output, error); updates weights."""
        y = np.asarray(y, dtype=complex)
        out = self.w * y
        if self.kind == "lms":
            self.w, e = lms_update(self.w, y, d, self.mu)
        elif self.kind == "nlms":
            self.w, e = nlms_update(self.w, y, d, self.mu, self.eps)
        else:
            self.w, self.P, e = rls_update(self.w, self.P, y, d, self.lam)
        return out, e


def run_adaptive_burst(kind, y_frames, train_syms, w0, **kw):
    """Run the equalizer over a burst of data-subcarrier observations.

    y_frames  : (n_frames, N_DATA) received data-subcarrier values
    train_syms: (n_train, N_DATA) known training symbols for the first n_train frames
    Returns payload equalized symbols (n_pay, N_DATA), per-frame mean |e|^2 curve,
    and the final weights.
    """
    bank = AdaptiveEqualizerBank(kind, w0, **kw)
    n_train = len(train_syms)
    outs, curve = [], []
    for n, y in enumerate(y_frames):
        if n < n_train:
            d = train_syms[n]
        else:
            d = qpsk_slicer(bank.output(y))
        out, e = bank.step(y, d)
        curve.append(float(np.mean(np.abs(e) ** 2)))
        if n >= n_train:
            outs.append(out)
    return np.array(outs), np.array(curve), bank.w


def learning_curve(kind, y_frames, d_frames, w0, **kw):
    """Fully supervised learning curve (all symbols known): per-iteration mean |e|^2 over
    subcarriers. Used for convergence experiments and unit tests."""
    bank = AdaptiveEqualizerBank(kind, w0, **kw)
    curve = []
    for y, d in zip(y_frames, d_frames):
        _, e = bank.step(y, d)
        curve.append(float(np.mean(np.abs(e) ** 2)))
    return np.array(curve), bank.w


def convergence_time(curve, tol_db=1.0, tail_frac=0.2):
    """First iteration at which the curve is within `tol_db` of its steady state
    (mean of the last `tail_frac` of the curve) and stays there. Returns (n_iter, steady)."""
    curve = np.asarray(curve, dtype=float)
    n_tail = max(1, int(len(curve) * tail_frac))
    steady = float(np.mean(curve[-n_tail:]))
    thr = steady * 10 ** (tol_db / 10)
    above = np.where(curve > thr)[0]
    n_conv = 0 if len(above) == 0 else int(above[-1]) + 1
    return n_conv, steady
