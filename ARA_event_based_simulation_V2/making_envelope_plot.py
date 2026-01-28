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
from trig_functions import *
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
SCAN_RATE           = 1

# ---- define a single CSW trigger value (you can change this) ----
CSW_THRESHOLD =51.4   # <- “range of trigger of 5” interpreted as trigger value = 5


PULSE_AMPLITUDES = np.array([500])
ENVELOPES=[10,8,4,2]

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
start_time_benchmark = time.time()

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

        #time_axis = t + time_start
        #make a grid of 4 plots with the signal and envelope for each envelope option
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f'Pulse signal with different envelopes (SNR={PULSE_AMPLITUDES[run]//NOISE_EQUALIZE})')
        axes = axes.flatten()
        
        for idx, env in enumerate(ENVELOPES):
            axes[idx].plot(t, channel_signals[0]**2, label='(Noise + Pulse)²', linewidth=1.7, alpha=0.7)
            axes[idx].plot(t, envelope_with_edge_rules(channel_signals[0]**2, env), label=f'Envelope samples = {env}', linewidth=1.1)
            if idx == 2 or idx == 3:
                axes[idx].set_xlabel('Time (ns)', fontsize=14)
            if idx == 0 or idx == 2:
                axes[idx].set_ylabel('ADC² Counts', fontsize=14)
            if idx == 0 or idx == 1:
                #no x ticks
                axes[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if idx == 1 or idx == 3:
                #no y ticks
                axes[idx].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'envelope_plots_SNR_{PULSE_AMPLITUDES[run]//NOISE_EQUALIZE}.png', dpi=300)
        #plt.show()

