import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

# -----------------------------
# User-provided burst waveform
# -----------------------------
def burst_waveform(time, amp):
    A = amp
    B = 10.0
    C = 10.0
    omega = 1.2
    k = 1.0

    cosine_term = A * math.cos(omega * time)              # oscillation
    heaviside_term = 1 / (1 + math.exp(-2 * k * time))   # smooth turn-on
    exponent_term = math.exp(-(time - B) / C)            # attenuation

    voltage = cosine_term * heaviside_term * exponent_term
    return voltage


# -----------------------------
# Parameters
# -----------------------------
num_samples = 600
num_channels = 20
sigma_noise = 1.0
pulse_center = 310  # put the pulse near the middle of the 600-sample window
rng = np.random.default_rng(seed=42)  # for reproducibility

# Time axis in "nanoseconds" (1 sample = 1 ns)
t = np.arange(num_samples)
local_t = t - pulse_center

# Build a unit-amplitude template, then rescale so that
# SNR = P2P / (2 * RMS_noise) = 1 for the clean signal.
raw_template = np.array([burst_waveform(tt, 1.0) for tt in local_t])
raw_half_p2p = 0.5 * (raw_template.max() - raw_template.min())

# Choose amplitude scaling so that half the peak-to-peak equals the noise RMS
amp_scale = sigma_noise / raw_half_p2p
signal_template = amp_scale * raw_template

# Verify target clean-signal SNR
target_snr = (signal_template.max() - signal_template.min()) / (2 * sigma_noise)

# Generate 20 uncorrelated Gaussian-noise channels and add the same signal to each
noise = rng.normal(0.0, sigma_noise, size=(num_channels, num_samples))
channels = noise + signal_template

# Create the three requested waveforms
wf_1 = channels[0] ** 2
noise_1 = noise[0] ** 2

wf_10 = channels[:10].sum(axis=0) ** 2
noise_10 = noise[:10].sum(axis=0) ** 2

wf_20 = channels[:20].sum(axis=0) ** 2
noise_20 = noise[:20].sum(axis=0) ** 2

def measured_snr(waveform, noise_only):
    p2p = waveform.max() - waveform.min()
    rms = np.sqrt(np.mean(noise_only**2))
    return p2p / (2 * rms), p2p, rms

snr_1, p2p_1, rms_1 = measured_snr(wf_1, noise_1)
snr_10, p2p_10, rms_10 = measured_snr(wf_10, noise_10)
snr_20, p2p_20, rms_20 = measured_snr(wf_20, noise_20)

# -----------------------------
# Segment averaging helper
# -----------------------------
def plot_segment_averages(ax, x, y, n_segments=10, linewidth=2.0):
    segment_length = len(y) // n_segments

    # First pass: find segment averages and their maximum
    segment_avgs = []
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < n_segments - 1 else len(y)
        segment_avg = np.mean(y[start:end])
        segment_avgs.append(segment_avg)
    
    max_segment_avg = np.max(segment_avgs)
    signal_max = np.max(y)
    scale_factor = signal_max / max_segment_avg if max_segment_avg > 0 else max_segment_avg#
    
    # Second pass: plot scaled segment averages
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < n_segments - 1 else len(y)
        
        scaled_avg = segment_avgs[i] * scale_factor
        
        # Plot horizontal line over that segment
        ax.hlines(scaled_avg, x[start], x[end - 1], linewidth=linewidth, color='red', alpha=0.7, linestyle='--', label='segment average (scaled)' if i == 0 else '')
    
    if n_segments > 0:
        ax.legend(loc='upper right', fontsize=20)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

plot_specs = [
    (axes[0], wf_1, 1, snr_1, p2p_1, rms_1),
    (axes[1], wf_10, 10, snr_10, p2p_10, rms_10),
    (axes[2], wf_20, 20, snr_20, p2p_20, rms_20),
]

for ax, wf, N, snr_val, p2p_val, rms_val in plot_specs:
    ax.plot(t, wf, linewidth=1.5)
    plot_segment_averages(ax, t, wf, n_segments=10, linewidth=2.5)

    ax.set_ylabel("Voltage²", fontsize=24)
    ax.grid(True, alpha=0.35)

    annotation = (
        f"N = {N}\n"
        f"Measured SNR = {snr_val:.2f}\n"
        f"Original signal SNR = {target_snr:.2f}"
    )
    ax.text(
        0.02, 0.96, annotation,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.45", alpha=0.7)
    )

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=18)
axes[0].set_title("Waveform²", fontsize=24, pad=14)
axes[-1].set_xlabel("Time (ns)", fontsize=24)

plt.tight_layout()

fig.savefig('Hello4', dpi=200, bbox_inches="tight")
#plt.show()
# plt.close(fig)