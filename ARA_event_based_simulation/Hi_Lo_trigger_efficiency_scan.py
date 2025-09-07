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


#parameters
SAMPLING_RATE   =  0.472            # GHz   (0.472 GS/s)
TIME_STEP       = 1.0 / SAMPLING_RATE   # ns
NOISE_EQUALIZE = 5 #ADC
MAX_SIGNAL = 4095 #ADC
WINDOW_SIZE = 18.75*1e6 #MHz
n_of_windows = 3
SIMULATION_DURATION_NS= n_of_windows/(WINDOW_SIZE) *1e9 #ns
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)+1  # Number of samples in the simulation duration
N_of_channels = 4
THRESHOLD_V= [18,21,19,21]  # ADC counts
N_REQ = 2  # Number of channels required for a trigger
COINC_NS = SIMULATION_DURATION_NS
SCAN_RATE = 50 
PULSE_AMPLITUDES = np.concatenate([
    np.arange(6, 12, 3),   
    np.arange(12, 21, 1),  
    np.arange(22, 38, 3)   
])  

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
impulse_response_path   = Path("/home/shams/ARA_simulation_algorithms/ARA_Trigger_simulation_algorithms/RNOG_sim_copy/jsons/impulse_response_Freauency_35_240.json")



pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)  # Normalized
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]  # Start from 0 ns


pass_fraction = []  # Example fraction of runs where a pulse is present
SNR_values = []  # Example SNR values for each run

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
        SNR = run_pulse_amplitude / NOISE_EQUALIZE
        triggers = find_triggers(channel_signals, time_axis, threshold=THRESHOLD_V, coincidence_ns=COINC_NS, n_channels_required=N_REQ)
        if len(triggers) > 0:
            COINC += 1
    pass_fraction.append(COINC / SCAN_RATE)
    SNR_values.append(SNR)
    print(f"\r Progress: {run+1}/{len(PULSE_AMPLITUDES)} completed", end='')



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
plt.title('Hi-Lo Trigger Efficiency Scan')
plt.xlabel('SNR')
plt.ylabel('Pass Fraction')
plt.grid()
plt.legend()
plt.savefig("TESTNOW_SCAN.png")


