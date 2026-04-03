import numpy as np
import matplotlib
matplotlib.use('TkAgg')   
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
from trig_functions_cop import *
from scipy.signal import fftconvolve


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
CSW_THRESHOLD =15.71   # <- “range of trigger of 5” interpreted as trigger value = 5

PULSE_AMPLITUDES = np.concatenate([
    np.arange(60, 200, 15),
    np.arange(200, 400, 10),
    np.arange(400, 550, 25)
])
#PULSE_AMPLITUDES= np.arange(100, 501,10)
#PULSE_AMPLITUDES = np.array([300]*8)

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

#start benchmarking time
benchmark_times = []

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
        #start benchmarking time
        time0 = time.time()
        # ---------- CSW trigger decision ----------
        triggers = ARA_CSW_trigger_FFT_optimized(
                channel_signals,
                time_axis,
                threshold=float(CSW_THRESHOLD),
                noise_rms=float(NOISE_EQUALIZE),
                N_segments=int(1),
            )
        
        
        #_no_shifting  
        if len(triggers) > 0:
            COINC += 1

        time1 = time.time()
        benchmark_times.append(time1 - time0)
    pass_fraction.append(COINC / SCAN_RATE)
    SNR_values.append(SNR)
    print(f"\r Progress: {run+1}/{len(PULSE_AMPLITUDES)} completed", end='')


#benchmarking analysis
benchmark_times = np.array(benchmark_times)
#mean, std of the mean and mean per event and std of the mean per event
mean_benchmark_time = np.mean(benchmark_times)
std_benchmark_time = np.std(benchmark_times)
mean_benchmark_time_per_event = mean_benchmark_time / (len(PULSE_AMPLITUDES) * SCAN_RATE)
std_benchmark_time_per_event = std_benchmark_time / (len(PULSE_AMPLITUDES) * SCAN_RATE)
print(f"\nBenchmarking results:")
print(f"for a total of {len(PULSE_AMPLITUDES)} runs and {SCAN_RATE} scans, total events: {len(PULSE_AMPLITUDES) * SCAN_RATE}")
print(f"Mean benchmarking time: {mean_benchmark_time:.6e} s")
print(f"Std of benchmarking time: {std_benchmark_time:.6e} s")
print(f"Mean benchmarking time per event: {mean_benchmark_time_per_event:.6e} s")
print(f"Std of benchmarking time per event: {std_benchmark_time_per_event:.6e} s")




"""
"""
# ---------------- Sigmoid fit and plot ----------------
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


plt.figure(figsize=(10, 6))
plt.plot(SNR_values, pass_fraction, marker='o', label='Pass Fraction vs SNR')

 #####"#"#"#
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
plt.savefig(f"Trigger_eff_scan_CSW_threshold_{CSW_THRESHOLD:.2f}_FFT_5HzRate_10N.png")
#plt.show()


print(f"\nSigmoid parameters: a = {a}, b = {b} (50% efficiency SNR), CSW threshold = {CSW_THRESHOLD}")
