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
from sim_functions import make_band_limited_noise, generate_pulse, digitize_signal, make_full_signal, plot_4_channels_signals, find_triggers


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
THRESHOLD_V= [18, 18, 18, 18]  # ADC counts
N_REQ = 2  # Number of channels required for a trigger
COINC_NS = SIMULATION_DURATION_NS


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


SIMULATION_TOTAL_DURATION_NS= SIMULATION_DURATION_NS * 1000 #1e3
NUMBER_OF_RUNS = int(SIMULATION_TOTAL_DURATION_NS // SIMULATION_DURATION_NS)


n_trigs = 0
for run in range(NUMBER_OF_RUNS):
    channel_signals= [[] for _ in range(N_of_channels)]
    time_start= run * SIMULATION_DURATION_NS
    no_pulses = 0 
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
                                amplitude_scale=no_pulses,
                                max_signal=MAX_SIGNAL,
                                start_time=start_seed
                                )
    time_axis = t + time_start  # Adjust time axis for the current run

    triggers = find_triggers(channel_signals, time_axis,
                             threshold=THRESHOLD_V,
                             coincidence_ns=COINC_NS,
                             n_channels_required=N_REQ)
    if len(triggers) > 0:
        n_trigs += 1



    #print(f"Run {run}: {len(triggers)} trigger(s) found")
    #for trig in triggers:
    #    print(f"  • t = {trig['t_trigger']:.1f} ns on channels {trig['channels']}")

    #plot_4_channels_signals(time_axis, channel_signals, title=f"Run {run+1} - 4 Channels Signals")
    #print(all_triggers.extend(triggers))
    
    #print progresion
    if run % 10 == 0 or run == NUMBER_OF_RUNS - 1:
        print(f"\r Progress: {run+1}/{NUMBER_OF_RUNS} completed", end='')
    
print("\n ",n_trigs)

"""
plt.figure(figsize=(12, 6))
plt.plot(t, ch_signal, label='Signal with Noise', color='blue')
plt.title('Signal with Noise and Pulse')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid()
plt.savefig("full_signal.png")


"""