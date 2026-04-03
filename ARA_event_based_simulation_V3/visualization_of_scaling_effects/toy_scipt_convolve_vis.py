import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import fftconvolve
import time

# --------------------------------
# Pulse shape
# --------------------------------
def burst_waveform(time, amp):
    A = amp
    B = 10.0
    C = 10.0
    omega = 1.2
    k = 1.0

    cosine_term = A * math.cos(omega * time)
    heaviside_term = 1 / (1 + math.exp(-2 * k * time))
    exponent_term = math.exp(-(time - B) / C)

    return cosine_term * heaviside_term * exponent_term


# --------------------------------
# FFT-based normalized xcorr
# --------------------------------
def normalized_xcorr_fft(x, y):
    """
    Normalized cross-correlation via FFT.
    Returns (corr, lags).
    """
    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        lags = np.arange(-len(y) + 1, len(x))
        return np.zeros_like(lags, dtype=float), lags

    # cross-correlation = convolution with reversed second signal
    r = fftconvolve(x, y[::-1], mode='full') / denom
    lags = np.arange(-len(y) + 1, len(x))
    return r, lags


# --------------------------------
# Manual normalized xcorr
# --------------------------------
def normalized_xcorr_manual(x, y):
    """
    Manual cross-correlation by iterating over all possible lags.
    Uses the same global normalization as the FFT version so they match closely.
    """
    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)
    lags = np.arange(-len(y) + 1, len(x))

    if denom == 0:
        return np.zeros_like(lags, dtype=float), lags

    corr = []

    for lag in lags:
        s = 0.0

        # r[lag] = sum_n x[n] y[n-lag]
        for n in range(len(x)):
            m = n - lag
            if 0 <= m < len(y):
                s += x[n] * y[m]

        corr.append(s / denom)

    return np.array(corr), lags


# --------------------------------
# Build two toy channels
# --------------------------------
num_samples = 600
sigma_noise = 1.0
pulse_center_1 = 150
pulse_shift = 0   # second channel pulse is delayed by 50 samples
pulse_center_2 = pulse_center_1 + pulse_shift

rng = np.random.default_rng(5)
t = np.arange(num_samples)

# build clean template centered in each channel
local_t1 = t - pulse_center_1
local_t2 = t - pulse_center_2

raw_template = np.array([burst_waveform(tt, 1.0) for tt in local_t1])
raw_half_p2p = 0.5 * (raw_template.max() - raw_template.min())

# choose amplitude so the clean pulse has SNR=1 by P2P/(2*RMS)
amp_scale = sigma_noise / raw_half_p2p

signal_1 = np.array([burst_waveform(tt, amp_scale*6) for tt in local_t1])
signal_2 = np.array([burst_waveform(tt, amp_scale*6) for tt in local_t2])

noise_1 = rng.normal(0, sigma_noise, num_samples)
noise_2 = rng.normal(0, sigma_noise, num_samples)

ch1 = signal_1 + noise_1
ch2 = signal_2 + noise_2

# --------------------------------
# Compute cross-correlations
# --------------------------------
corr_manual, lags_manual = normalized_xcorr_manual(ch1, ch2)
corr_fft, lags_fft = normalized_xcorr_fft(ch1, ch2)


# --------------------------------
# Benchmark cross-correlations
# --------------------------------
start_manual = time.time()
best_lag_manual = lags_manual[np.argmax(corr_manual)]
time_manual = time.time() - start_manual

start_fft = time.time()
best_lag_fft = lags_fft[np.argmax(corr_fft)]
time_fft = time.time() - start_fft

print(f"Manual xcorr time: {time_manual*1000:.4f} ms")
print(f"FFT xcorr time: {time_fft*1000:.4f} ms")
print(f"Speedup: {time_manual/time_fft:.2f}x")

# --------------------------------
# Plot
# --------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(lags_manual, corr_manual, linewidth=1.8)
axes[0].axvline(best_lag_manual, linestyle='--', linewidth=1.5)
axes[0].set_title("Manual Cross-Correlation", fontsize=16)
axes[0].set_ylabel("Correlation", fontsize=13)
axes[0].grid(True, alpha=0.35)
axes[0].text(
    0.02, 0.95,
    f"Peak lag = {best_lag_manual} samples",
    transform=axes[0].transAxes,
    va="top",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.7)
)

axes[1].plot(lags_fft, corr_fft, linewidth=1.8)
axes[1].axvline(best_lag_fft, linestyle='--', linewidth=1.5)
axes[1].set_title("FFT / scipy.signal.fftconvolve Cross-Correlation", fontsize=16)
axes[1].set_xlabel("Lag (samples)", fontsize=13)
axes[1].set_ylabel("Correlation", fontsize=13)
axes[1].grid(True, alpha=0.35)
axes[1].text(
    0.02, 0.95,
    f"Peak lag = {best_lag_fft} samples",
    transform=axes[1].transAxes,
    va="top",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.7)
)

plt.tight_layout()
plt.savefig("xcorr_comparison.png", dpi=300)
plt.show()