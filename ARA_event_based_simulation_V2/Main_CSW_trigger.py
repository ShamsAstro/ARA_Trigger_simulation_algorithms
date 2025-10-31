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
from scipy.optimize import curve_fit
from sim_functions import *
from trig_functions import *


SAMPLING_RATE       = 3.2                      # GHz
TIME_STEP           = 1.0 / SAMPLING_RATE      # ns
NOISE_EQUALIZE      = 100                      # ADC (use as noise_rms)
MAX_SIGNAL          = 4095                     # ADC
WINDOW_SIZE         = 5.88*1e6                 # MHz (name kept from your script)
n_of_windows        = 1
SIMULATION_DURATION_NS = n_of_windows/(WINDOW_SIZE) * 1e9  # ns
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)
N_of_channels       = 8
N_REQ               = 1                        # not needed for CSW, but kept
COINC_NS            = SIMULATION_DURATION_NS
SCAN_RATE           = 50

# ---- define a single CSW trigger value (you can change this) ----
CSW_THRESHOLD =150   # <- “range of trigger of 5” interpreted as trigger value = 5

PULSE_AMPLITUDES = np.concatenate([
    np.arange(120, 200, 30),
    np.arange(200, 400, 30),
    np.arange(400, 550, 30)
])
#PULSE_AMPLITUDES = np.array([0])

# ---------------- Load pulse and impulse response ----------------
pulse_json_path = Path("../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)

impulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]  # Start from 0 ns

# ---------------- Scan over amplitudes and build efficiency ----------------
pass_fraction = []
SNR_values = []

for run, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES):
    channel_signals = [[] for _ in range(N_of_channels)]
    time_start = run * SIMULATION_DURATION_NS
    COINC = 0

    for SCAN in range(SCAN_RATE):
        start_seed = random.uniform(0, TIME_STEP)

        for ch in range(N_of_channels):
            t, channel_signals[ch] = make_full_signal(
                impulse_json_path=impulse_response_path,
                SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                SAMPLING_RATE=SAMPLING_RATE,
                NOISE_EQUALIZE=NOISE_EQUALIZE,
                pulse_voltage=pulse_voltage,
                pulse_time=pulse_time,
                time_step=TIME_STEP,
                simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                amplitude_scale=run_pulse_amplitude,
                max_signal=MAX_SIGNAL,
                start_time=start_seed
            )

        time_axis = t + time_start

        # SNR definition consistent with your code
        SNR = run_pulse_amplitude / NOISE_EQUALIZE

        # ---------- CSW trigger decision ----------
        triggers = ARA_CSW_trigger(
            channel_signals,
            time_axis,
            threshold=CSW_THRESHOLD,
            noise_rms=NOISE_EQUALIZE,     # use your noise scale as RMS
        )

        if len(triggers) > 0:
            COINC += 1
    
    pass_fraction.append(COINC / SCAN_RATE)
    SNR_values.append(SNR)
    print(f"\r Progress: {run+1}/{len(PULSE_AMPLITUDES)} completed", end='')

# ---------------- Sigmoid fit and plot ----------------
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


plt.figure(figsize=(10, 6))
plt.plot(SNR_values, pass_fraction, marker='o', label='Pass Fraction vs SNR')

#"""
params, _ = curve_fit(sigmoid, SNR_values, pass_fraction, p0=[1, np.mean(SNR_values)])
a, b = params
pass_fraction_sigmoid = sigmoid(np.array(SNR_values), a, b)

plt.plot(SNR_values, pass_fraction_sigmoid, marker='x', linestyle='--', label='Sigmoid Fit')
plt.axhline(y=0.5, linestyle='--', label='50% Pass Threshold')
plt.axvline(x=b, linestyle='--', label='50% eff SNR at {:.2f}'.format(b))

plt.title(f'ARA CSW Trigger Efficiency Scan (threshold={CSW_THRESHOLD})')
plt.xlabel('SNR')
plt.ylabel('Pass Fraction')
plt.grid()
plt.legend()
plt.savefig(f"Trigger_eff_scan_CSW_threshold_{CSW_THRESHOLD:.2f}.png")

print(f"\nSigmoid parameters: a = {a}, b = {b} (50% efficiency SNR), CSW threshold = {CSW_THRESHOLD}")