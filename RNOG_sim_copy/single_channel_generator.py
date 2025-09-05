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
from sim_functions import make_band_limited_noise, generate_pulse, digitize_signal, make_full_signal

#parameters
SAMPLING_RATE   =  0.472            # GHz   (0.472 GS/s)
TIME_STEP       = 1.0 / SAMPLING_RATE   # ns
NOISE_EQUALIZE = 5 #ADC
MAX_SIGNAL = 4095 #ADC
WINDOW_SIZE = 18.75*1e6 #MHz
n_of_windows = 3
SIMULATION_DURATION_NS= n_of_windows/(WINDOW_SIZE) *1e9 #ns
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)+1  # Number of samples in the simulation duration
print(TIME_STEP, SIMULATION_DURATION_NS, SIMULATION_DURATION_SAMPLES)


#preparring the sample pulse
with open('/home/shamshassiki/Shams_Analyzing_scripts/Trigger_simulation_and_tests/jsons/upsampled_2filter_pulse_example.json') as f:
    pulse_data = json.load(f)

pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)  # Normalized
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]  # Start from 0 ns


#Noise makier
impulse_response_path   = Path("/home/shamshassiki/Shams_Analyzing_scripts/Trigger_simulation_and_tests/jsons/impulse_response_Freauency_35_240.json")


start_seed=random.uniform(0, TIME_STEP)

t, ch_signal = make_full_signal(impulse_json_path=impulse_response_path,
                            SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                            SAMPLING_RATE=SAMPLING_RATE,
                            NOISE_EQUALIZE=NOISE_EQUALIZE,
                            pulse_voltage=pulse_voltage,
                            pulse_time=pulse_time,
                            time_step=TIME_STEP,
                            simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                            amplitude_scale=40,
                            max_signal=MAX_SIGNAL,
                            start_time=start_seed
                            ) 




plt.figure(figsize=(12, 6))

plt.plot(t, ch_signal, label='Signal with Noise', color='blue')
plt.title('Signal with Noise and Pulse')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid()
plt.savefig("full_signal.png")