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
THRESHOLD_V= [85000]*N_of_channels  # ADC^2 counts
N_REQ = 3  # Number of channels required for a trigger
COINC_NS = SIMULATION_DURATION_NS
SCAN_RATE = 100 


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
            t, channel_signals[ch]= make_band_limited_noise(
                                    json_path=impulse_response_path,
                                    channel_key="ch2_2x_amp",
                                    window_ns=SIMULATION_DURATION_NS,
                                    adc_rate_ghz=SAMPLING_RATE,
                                    target_rms_mV=NOISE_EQUALIZE
                                    )

        time_axis = t + time_start  # Adjust time axis for the current run
        #finding if channels exceed the threshold

        triggers = find_ARA_env_triggers(channel_signals, time_axis, threshold=THRESHOLD_V, n_channels_required=N_REQ)
        if len(triggers) > 0:
            COINC += 1
            TOT, n_triggered_channels = TOT_finder(channel_signals, time_axis, threshold=THRESHOLD_V, n_channels_required=N_REQ)
            TOT_values.append(TOT)

    pass_fraction.append(COINC / SCAN_RATE)

time1 = time.time()
print(f"\nTotal benchmarking time: {time1 - time0:.2f} seconds")




