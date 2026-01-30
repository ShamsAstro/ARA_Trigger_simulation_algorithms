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
THRESHOLD_V= [90000]*N_of_channels  # ADC^2 counts
N_REQ = 3  # Number of channels required for a trigger
COINC_NS = SIMULATION_DURATION_NS
SCAN_RATE = 200 
PULSE_AMPLITUDES = np.concatenate([
    np.arange(100, 400, 10),   
    np.arange(12, 22, 0.5),  
    np.arange(22, 28, 0.5)   
])  
PULSE_AMPLITUDES= np.arange(100, 500,10)


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



print(len(pulse_time), len(pulse_voltage))
#plot the pulse
plt.plot(pulse_time, pulse_voltage)
plt.xlabel("Time (ns)")
plt.ylabel("Normalized Voltage")
plt.title("Sample Pulse Waveform")
plt.grid()
plt.savefig("sample_pulse_waveform.png")
plt.close()

print("Sample pulse waveform saved as 'sample_pulse_waveform.png'")

#open the impulse response and show the details and keys
with open(impulse_response_path) as f:
    impulse_data = json.load(f)
print("Impulse response data keys:", impulse_data.keys())

#more details about the keys 'freq_GHz', 'ch2_2x_amp'
print("Frequency (GHz):", impulse_data['freq_GHz'][:10], "...")
print("Channel 2 2x Amplified Impulse Response:", impulse_data['ch2_2x_amp'][:10], "...")
#plot the impulse response
plt.plot(impulse_data['freq_GHz'], impulse_data['ch2_2x_amp'])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Amplitude")
plt.xlim(0, 1)
plt.title("Impulse Response for Channel 2 (2x Amplified)")
plt.grid()
plt.savefig("impulse_response_channel2_2x.png")


#open new data
new_impulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/ARA_impulse_response.txt").resolve()

with open(new_impulse_response_path) as f:
    lines = f.readlines()

#details about the new impulse response file
print("First 5 lines of the new impulse response file:")

print(lines[5], lines[5][1])
sample=np.array([float(lines[i].split()[0]) for i in range(len(lines))])
amplitudes=np.array([float(lines[i].split()[1]) for i in range(len(lines))])
freqs= (sample/sample[-1])

#save the data in a json file
new_impulse_response_json_path = Path("new_impulse_response_ARA_event_based_simulation_V2.json").resolve()
with open(new_impulse_response_json_path, 'w') as f:
    json.dump({'freq_GHz': freqs.tolist(), 'ch2_2x_amp': amplitudes.tolist()}, f)


plt.plot(freqs, amplitudes)
plt.xlabel("Frequency (GHz)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.xlim(0, 1)
plt.title("New Impulse Response from ARA_event_based_simulation_V2")
plt.grid()
plt.savefig("new_impulse_response_ARA_event_based_simulation_V22.png", dpi=300)
plt.close()
print("done!")


"""

#FFT the times to freauency 





new_pulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/normalized_neutrino_waveform.txt").resolve()
with open(new_pulse_response_path) as f:
    lines = f.readlines()
print("First 5 lines of the new pulse waveform file:")
print(lines[5], lines[5][1])
time_ns=np.array([float(lines[i].split()[0]) for i in range(1,len(lines))])
voltages=np.array([float(lines[i].split()[1]) for i in range(1,len(lines))])

# create a boolean mask for the time window and apply it to both arrays
mask = (time_ns >= 235) & (time_ns <= 235 + 121)
time_ns = time_ns[mask]
voltages = voltages[mask]
time_ns = time_ns - time_ns[0] + 450
# save times and amplitudes in a json file as t_axis_ns and avg_wave

#new_pulse_json_path = Path("new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
#with open(new_pulse_json_path, 'w') as f:
#    json.dump({'t_axis_ns': time_ns.tolist(), 'avg_wave': voltages.tolist()}, f)



plt.plot(time_ns, voltages)
plt.xlabel("Time (ns)", fontsize=14)
plt.ylabel("Normalized Voltage", fontsize=14)
plt.title("New Pulse Waveform from ARA_event_based_simulation_V2")
plt.grid()
plt.savefig("new_pulse_waveform_ARA_event_based_simulation_V22.png", dpi=300)




