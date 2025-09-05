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
from sim_functions import make_band_limited_noise, generate_pulse, digitize_signal, make_full_signal, plot_4_channels_signals, sigmoid, find_phased_triggers
from scipy.optimize import curve_fit

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
SCAN_RATE = 1000 
PULSE_AMPLITUDES = np.array([0,1])


# Trigger configuration
PHASED_THRESHOLD     = 30   # per-channel thresholds in ADC
UPSAMPLE_FACTOR      = 4 # upsampling factor for the pulse
PHASED_BEAMS = [
    -60.0, -45.11838005, -33.44299614, -23.18167437, -13.66170567,
    -4.51554582, 4.51554582, 13.66170567, 23.18167437, 33.44299614,
    45.11838005, 60.0
] # Phased beam angles in degrees
POWER_WINDOW_SIZE = 24 #Samples (upsampled sampling)
POWER_WINDOOW_STEP = 4 #Samples (upsampled sampling)
POWER_DIVISION_FACTOR = 32
PHASED_TRIGGER_PARAMETERS = [PHASED_THRESHOLD, UPSAMPLE_FACTOR, PHASED_BEAMS,POWER_WINDOW_SIZE, POWER_WINDOOW_STEP, POWER_DIVISION_FACTOR]


#preparring the sample pulse
with open('/home/shamshassiki/Shams_Analyzing_scripts/Trigger_simulation_and_tests/jsons/upsampled_2filter_pulse_example.json') as f:
    pulse_data = json.load(f)
impulse_response_path   = Path("/home/shamshassiki/Shams_Analyzing_scripts/Trigger_simulation_and_tests/jsons/impulse_response_Freauency_35_240.json")

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
        triggers = find_phased_triggers(channel_signals, time_axis, PHASED_TRIGGER_PARAMETERS)
        if triggers:
            COINC += 1
    pass_fraction.append(COINC / SCAN_RATE)
    print(COINC, COINC / SCAN_RATE)
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
plt.title('Phased Trigger Efficiency Scan')
plt.xlabel('SNR')
plt.ylabel('Pass Fraction')
plt.grid()
plt.legend()
#plt.savefig("Phased_test_0.png")


