import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import math
import time
import random
import json
from pathlib import Path
from sim_functions import *
from scipy.signal import firwin, lfilter

# =========================================================
# user knobs
UPSAMPLE_FACTOR       = 4
N_CH                  = 4
FS_GHZ                = 0.472                 # 0.472 GS/s
DT_NS                 = 1.0 / FS_GHZ          # ~2.1186 ns
N_SAMPLES             = 1024
F_SIGNAL_MHZ          = 20.0                  # test sine frequency
AMP_PER_CH            = 50.0                  # amplitude per channel
NOISE_RMS             = 0.0                   # set >0 to add noise
TEST_BEAM_ANGLE_DEG   = 14.0                  # try 0.0 and 30.0
# PA windowing (like your firmware notes)
POWER_WINDOW_SIZE     = 24                    # in upsampled samples
POWER_WINDOW_STEP     = 4                     # in upsampled samples
POWER_DIVISION_FACTOR = 32
PHASED_THRESHOLD      = 75.0                 # will fire when aligned
# =========================================================

# --- make a poorly sampled sine (few samples per cycle) ---
fs = 50.0                  # original sample rate (Hz)
dt = 1.0 / fs
f_sig = 10.0                # sine frequency (Hz) -> ~2.86 samples per cycle
T = 1.0                    # duration (s)
t = np.arange(0, T, dt)
x = np.sin(2*np.pi*f_sig*t)
# full samples of the signal
x_full = np.sin(2*np.pi*f_sig*np.arange(0, T, 1.0/(fs*UPSAMPLE_FACTOR)))
t_full = np.arange(0, T, 1.0/(fs*UPSAMPLE_FACTOR))

# --- zero-stuff upsampling + FIR low-pass ---
L = 8                      # upsample factor
fs_up = fs * L
dt_up = 1.0 / fs_up

# zero stuffing
up = np.zeros(len(x) * L, dtype=float)
up[::L] = x

# low-pass to reconstruct between samples (cutoff at original Nyquist)
numtaps = 45
taps = firwin(numtaps, cutoff=fs*0.5, fs=fs_up, pass_zero='lowpass')
y = lfilter(taps, [1.0], up)

# compensate the FIR group delay for visual alignment
gd = (numtaps - 1) // 2
y_aligned = np.roll(y, -gd)

t_up = np.arange(len(up)) * dt_up

# --- plots ---
plt.figure(figsize=(10, 5))
plt.title("Zero-stuff upsampling example")
# original samples

plt.plot(t, x, "o", label="original samples", zorder=3)
# raw zero-stuffed (impulses with zeros between)
plt.step(t_up, up, where="mid", alpha=0.4, label="zero-stuffed (pre-filter)")
plt.plot(t_full, x_full, ":", label="full samples (ideal)", zorder=1)  # ideal full
# filtered, aligned
plt.plot(t_up, y_aligned* L, label="upsampled + LPF", linewidth=2)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("upsampling_example.png", dpi=300)
