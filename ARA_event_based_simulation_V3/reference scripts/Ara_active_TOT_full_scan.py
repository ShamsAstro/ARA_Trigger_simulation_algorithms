import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time
from pathlib import Path
from scipy.optimize import curve_fit
from sim_functions import *
from trig_functions import *

# ─────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE   = 3.2   # GHz  
TIME_STEP       = 1.0 / SAMPLING_RATE
NOISE_EQUALIZE  = 100   # ADC
MAX_SIGNAL      = 4095  # ADC
WINDOW_SIZE     = 5.88e6 # MHz
n_of_windows    = 1
SIM_DURATION_NS = n_of_windows / WINDOW_SIZE * 1e9
SIM_DURATION_SAMPLES = int(SIM_DURATION_NS / TIME_STEP)
N_of_channels   = 8
N_REQ           = 3
SCAN_RATE       = 500
envelope_samples = 4 # Number of samples for envelope calculation

PULSE_AMPLITUDES = np.concatenate([
    np.arange(100, 200, 25),
    np.arange(200, 311, 10),
    np.arange(340, 521, 40)
])

# TOT elimination thresholds list
TOT_thresholds = {
    0: 71983,
    1: 71566,
    2: 72196,
    3: 71689,
    4: 68810,
    5: 64620,
    6: 52881,
    7: 41093,
    8: 35058,
    9: 31800,
    10: 29500
}

# ─────────────────────────────────────────────────────────────
# Pulse template
# ─────────────────────────────────────────────────────────────
pulse_json_path = Path("../ARA_event_based_simulation_V3/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)
   
impulse_response_path = Path("../ARA_event_based_simulation_V3/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()
pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-(x - b) / a))

OUT_DIR = Path("TOT_efficiency_plots_new_ARA_pulse_4env")
OUT_DIR.mkdir(exist_ok=True)

efficiency_summary = {}

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
for min_tot, thr_value in TOT_thresholds.items():
    print(f"\nRunning scan for TOT≥{min_tot}, Threshold={thr_value} ADC²")

    pass_fraction = []
    SNR_values = []
    tot_SNR_values = []
    TOT_values = []

    time0 = time.time()

    for run, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES):
        channel_signals = [[] for _ in range(N_of_channels)]
        time_start = run * SIM_DURATION_NS
        COINC = 0

        for SCAN in range(SCAN_RATE):
            start_seed = random.uniform(0, TIME_STEP)

            for ch in range(N_of_channels):
                t, channel_signals[ch] = make_full_signal(
                    impulse_json_path=impulse_response_path,
                    SIMULATION_DURATION_NS=SIM_DURATION_NS,
                    SAMPLING_RATE=SAMPLING_RATE,
                    NOISE_EQUALIZE=NOISE_EQUALIZE,
                    pulse_voltage=pulse_voltage,
                    pulse_time=pulse_time,
                    time_step=TIME_STEP,
                    simulation_duration_samples=SIM_DURATION_SAMPLES,
                    amplitude_scale=run_pulse_amplitude,
                    max_signal=MAX_SIGNAL,
                    start_time=start_seed
                )
            time_axis = t + time_start

            SNR = run_pulse_amplitude / NOISE_EQUALIZE
            triggers = find_ARA_env_triggers(channel_signals, time_axis,
                                             threshold=[thr_value]*N_of_channels,
                                             n_channels_required=N_REQ,
                                             envelope_window_points=envelope_samples)
            if triggers:
                TOT, n_triggered_channels = TOT_finder_mod(channel_signals, time_axis,
                                                       threshold=[thr_value]*N_of_channels,
                                                       n_channels_required=N_REQ,
                                                        env_parameter=envelope_samples )
                if TOT > min_tot:
                    COINC += 1
                    TOT_values.append(TOT)
                    tot_SNR_values.append(SNR)

        pass_fraction.append(COINC / SCAN_RATE)
        SNR_values.append(SNR)
        print(f"\r Progress: {run+1}/{len(PULSE_AMPLITUDES)}", end='')

    time1 = time.time()
    print(f"\nCompleted in {time1 - time0:.2f} seconds")

    # Fit sigmoid
    try:
        params, _ = curve_fit(sigmoid, SNR_values, pass_fraction, p0=[1, np.mean(SNR_values)])
        a, b = params
    except Exception:
        a, b = np.nan, np.nan

    efficiency_summary[min_tot] = b  # 50% efficiency SNR

    pass_fraction_sigmoid = sigmoid(np.array(SNR_values), a, b) if not np.isnan(b) else None

    # Plot 1: Pass fraction vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_values, pass_fraction, marker='o', label='Pass Fraction')
    if pass_fraction_sigmoid is not None:
        plt.plot(SNR_values, pass_fraction_sigmoid, 'x--', label='Sigmoid Fit')
        plt.axhline(0.5, color='r', linestyle='--', label='50% Threshold')
        plt.axvline(b, color='g', linestyle='--', label=f'50% eff SNR = {b:.2f}')
    plt.title(f"Trigger Efficiency — TOT≥{min_tot}, Thr={thr_value}")
    plt.xlabel('SNR')
    plt.ylabel('Pass Fraction')
    plt.grid()
    plt.legend()
    plt.savefig(OUT_DIR / f"Efficiency_TOT{min_tot}_Thr{thr_value}.png")
    plt.close()

    # Plot 2: TOT vs SNR
    plt.figure(figsize=(10, 6))
    plt.scatter(tot_SNR_values, TOT_values, alpha=0.7)
    plt.title(f"TOT vs SNR — TOT≥{min_tot}, Thr={thr_value}")
    plt.xlabel('SNR')
    plt.ylabel('TOT (samples)')
    plt.grid()
    plt.savefig(OUT_DIR / f"TOT_vs_SNR_TOT{min_tot}_Thr{thr_value}.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────
print("\n50% efficiency SNR summary:")
for tot, snr50 in efficiency_summary.items():
    print(f"TOT≥{tot}: 50% SNR ≈ {snr50:.2f}")
