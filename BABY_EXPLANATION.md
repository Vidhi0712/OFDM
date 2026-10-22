# Explain the Project Like I'm a Beginner

## The big picture
Imagine sending a message through a noisy, echo-filled room. The receiver hears a mixture of the original sound, echoes, and noise. It must estimate what happened to the signal and undo as much damage as possible.

Wireless signals have a similar problem. Buildings, walls, movement, and interference alter the radio signal. Our project simulates a wireless communication method called OFDM and tests whether a small ML model can help choose a suitable receiver strategy.

## What is OFDM?
OFDM splits a high-speed data stream across many closely spaced subcarriers. Instead of sending everything on one carrier, it sends pieces in parallel. This is useful in modern wireless systems. An IFFT turns the subcarrier symbols into a time-domain waveform; the receiver uses an FFT to recover the subcarriers.

## Why add a cyclic prefix?
Multipath means copies of the signal arrive at different times. The cyclic prefix copies a small tail of the OFDM symbol to its front. If it is long enough for the channel delay spread and timing is correct, it helps prevent adjacent OFDM symbols from interfering and preserves the simple per-subcarrier model.

## What is channel estimation?
The receiver needs an estimate of how the channel changed each subcarrier. Known pilot symbols act like reference markers. The receiver compares what it expected to send with what it received to estimate the channel.

- **LS (Least Squares):** a simple estimate from received pilots. It is cheap but can be noisy.
- **LMMSE (Linear Minimum Mean-Square Error):** uses statistical information about channel/noise to reduce expected estimation error when its assumptions are suitable.

## What is equalization?
After estimating the channel, the receiver tries to undo its effect on the data symbols.
- **ZF (Zero Forcing):** divides by the estimated channel. Simple, but can magnify noise on a deeply faded subcarrier.
- **MMSE equalization:** balances undoing channel distortion with not amplifying noise too aggressively.
- **LMS/NLMS/RLS:** adaptive filtering methods that update filter coefficients from errors/training observations. They need a carefully defined input, desired signal, and update setup.

## Where does machine learning fit?
We do not ask ML to decode every radio sample from scratch. We first build several conventional receiver strategies. In simulated training experiments, we observe which strategy performs best for each scenario. A small Decision Tree learns patterns from measurable pilot features and predicts which strategy to use.

At runtime, the classifier does not know the true channel settings or which strategy would win after trying all of them. It only sees estimated features. That is why the final test must use unseen scenarios.

## What does the model learn?
For example, it may learn a rule like: “when the estimated pilot SNR is low and pilot residuals are high, prefer strategy B.” The actual rules must come from training; do not invent them in the report.

## How will we know if it works?
We compare:
1. A receiver that always uses strategy A.
2. A receiver that always uses strategy B.
3. A hand-designed threshold/rule-based selector.
4. The ML selector.

We measure:
- **BER:** fraction of transmitted bits recovered incorrectly.
- **SER:** fraction of transmitted symbols recovered incorrectly.
- **MSE:** average squared channel-estimation error.
- **Runtime/complexity:** the cost of processing a frame.
If ML is useful, it should improve a meaningful metric or trade-off on held-out scenarios. It may not win everywhere, and that is still a valid research finding if reported honestly.

## What is the novelty?
The novelty is not inventing OFDM, LS, LMMSE, or Decision Trees. It is implementing and experimentally evaluating an ML-based strategy-selection layer that combines pilot-derived channel information with classical DSP receiver options, and measuring when that adaptation helps compared with fixed and threshold-based choices.

## What to tell the professor
“We are building a software-only SISO-OFDM simulation. We will first validate the conventional transmitter/receiver chain, then implement LS/LMMSE channel estimation and defined equalization strategies. We will generate pilot-derived features and offline performance labels across controlled SNR, multipath, and time-varying channel scenarios. A scikit-learn Decision Tree will predict the receiver strategy from receiver-observable features. We will compare fixed, rule-based, and ML-selected receivers on held-out channel scenarios using BER, SER, channel-estimation MSE, and runtime/complexity. The project’s research question is whether this interpretable selection layer improves the performance/complexity trade-off; improvement is a hypothesis, not an assumed result.”
