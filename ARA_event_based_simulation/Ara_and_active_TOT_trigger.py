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


#parameters
SAMPLING_RATE   =  3.2            # GHz  
TIME_STEP       = 1.0 / SAMPLING_RATE   # ns
NOISE_EQUALIZE = 100 #ADC
MAX_SIGNAL = 4095 #ADC
WINDOW_SIZE = 5.88*1e6 #MHz
n_of_windows = 1
SIMULATION_DURATION_NS= n_of_windows/(WINDOW_SIZE) *1e9 #ns
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)  # Number of samples in the simulation duration
N_of_channels = 8
THRESHOLD_V= [78761]*N_of_channels  # ADC^2 counts
N_REQ = 3  # Number of channels required for a trigger
COINC_NS = SIMULATION_DURATION_NS
SCAN_RATE = 300 
MIN_ALLOWED_TOT= 10 # in samples (ns / TIME_STEP_NS), minimum TOT to consider a trigger valid

PULSE_AMPLITUDES = np.concatenate([
    np.arange(100, 200, 10),   
    np.arange(200, 400, 10),  
    np.arange(400, 600, 25)   
])  
#PULSE_AMPLITUDES= np.arange(100, 600,10)

"""
#preparring the sample pulse
jsons_folder = Path(__file__).parent / "jsons"
pulse_json_path = jsons_folder / "upsampled_2filter_pulse_example.json"
impulse_response_path = jsons_folder / "impulse_response_Freauency_35_240.json"

with open(pulse_json_path) as f:
    pulse_data = json.load(f)
"""
with open('/home/shams/ARA_simulation_algorithms/ARA_Trigger_simulation_algorithms/RNOG_sim_copy/jsons/upsampled_2filter_pulse_example.json') as f:
    pulse_data = json.load(f)
impulse_response_path = Path("../RNOG_sim_copy/jsons/impulse_response_Freauency_35_240.json").resolve()


pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)  # Normalized
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]  # Start from 0 ns


pass_fraction = []  # Example fraction of runs where a pulse is present
SNR_values = []  # Example SNR values for each run
tot_SNR_values = []  # Example SNR values for each run
TOT_values = []  # Example TOT values for each run

#starting benchmarking
#benchmarking over the loop
time0 = time.time()

for run, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES):
    channel_signals= [[] for _ in range(N_of_channels)]
    time_start= run * SIMULATION_DURATION_NS
    COINC=0
    for SCAN in range(SCAN_RATE):
        start_seed=random.uniform(0, TIME_STEP)  

        for ch in range(N_of_channels):
            t, channel_signals[ch] = make_full_signal(impulse_json_path=impulse_response_path,
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
        time_axis = t + time_start  # Adjust time axis for the current run
        #finding if channels exceed the threshold
        #plot_channels_signals(time_axis, channel_signals, title=f"Run {run+1}, Scan {SCAN+1}, Pulse Amplitude {run_pulse_amplitude} ADC")
        SNR = run_pulse_amplitude / NOISE_EQUALIZE
        triggers = find_ARA_env_triggers(channel_signals, time_axis, threshold=THRESHOLD_V, n_channels_required=N_REQ)
        if len(triggers) > 0:
            COINC += 1
            TOT, n_triggered_channels = TOT_finder(channel_signals, time_axis, threshold=THRESHOLD_V, n_channels_required=N_REQ)
            if TOT > MIN_ALLOWED_TOT:  # Only consider events with TOT greater than 5 nsamples
                TOT_values.append(TOT)
                tot_SNR_values.append(SNR)
    pass_fraction.append(COINC / SCAN_RATE)
    SNR_values.append(SNR)
    print(f"\r Progress: {run+1}/{len(PULSE_AMPLITUDES)} completed", end='')

time1 = time.time()
print(f"\nTotal benchmarking time: {time1 - time0:.2f} seconds")



# Fit the sigmoid function to the data
params, _ = curve_fit(sigmoid, SNR_values, pass_fraction, p0=[1, np.mean(SNR_values)])
a, b = params
# Generate sigmoid values for plotting
pass_fraction_sigmoid = sigmoid(np.array(SNR_values), a, b)


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(SNR_values, pass_fraction, marker='o', label='Pass Fraction vs SNR')
plt.plot(SNR_values, pass_fraction_sigmoid, marker='x', linestyle='--', label='Sigmoid Fit')
plt.axhline(y=0.5, color='r', linestyle='--', label='50% Pass Threshold')
plt.axvline(x=b, color='g', linestyle='--', label='50% eff SNR at {:.2f}'.format(b))
plt.title('Trigger_efficiency_scan_at_5Hz_target_threshold_w_TOT_trigger_eliminate_10tot_long.png')
plt.xlabel('SNR')
plt.ylabel('Pass Fraction')
plt.grid()
plt.legend()
plt.savefig("Trigger_efficiency_scan_rate_5Hz_target_10TOT_trigger.png")



# Plot TOT vs SNR
plt.figure(figsize=(10, 6))
plt.scatter(tot_SNR_values, TOT_values, alpha=0.7)  
plt.title('Time Over Threshold (TOT) vs SNR for Triggered Events_ 5Hz target threshold_eliminate_tot10.png')
plt.xlabel('SNR')
plt.ylabel('Time Over Threshold (ns)')
plt.grid()
plt.savefig("TOT_scan_rate_5Hz_target_10TOT_trigger.png")

"""
"""

