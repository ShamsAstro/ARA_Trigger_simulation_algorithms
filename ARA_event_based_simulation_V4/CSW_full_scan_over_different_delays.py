import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import json
import time
import math
import re
from pathlib import Path

from sim_functions import *
from trig_functions_cop import *

# ============================================================
# User settings
# ============================================================
SAMPLING_RATE = 3.2                     # GHz
TIME_STEP = 1.0 / SAMPLING_RATE         # ns
NOISE_EQUALIZE = 100                    # ADC
MAX_SIGNAL = 4095                       # ADC
WINDOW_SIZE = 5.88 * 1e6                # MHz
N_OF_WINDOWS = 1
SIMULATION_DURATION_NS = N_OF_WINDOWS / WINDOW_SIZE * 1e9
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)

N_OF_CHANNELS = 8
SCAN_RATE = 150                         # number of events per angle setting
CSW_THRESHOLD = 15.71

PULSE_AMPLITUDES = np.concatenate([
    np.arange(120, 150, 10),
    np.arange(150, 225, 6),
    np.arange(225, 270, 15)
]) # np.array([300]) 

ANGLE_PERCENT_TO_USE = 100             # choose 10 for 10%, 100 for full scan
RANDOM_SEED = 12347

PLOT_FIRST_EVENT = False                # helpful for troubleshooting
PRINT_EVERY_N_EVENTS = 45               # progress print frequency

DELAY_JSON_PATH = Path("delay_list_full.json").resolve()
OUTPUT_JSON_PATH = Path("efficiency_scan_100per_results.json").resolve()

PULSE_JSON_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json"
).resolve()

IMPULSE_RESPONSE_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json"
).resolve()

PULSE_START_TIME = 450                  # ns
PULSE_END_TIME = 570                    # ns


# ============================================================
# Helpers
# ============================================================
def parse_theta_phi_from_key(key):
    """
    Parse keys of the form 'theta_90_phi_150'
    """
    match = re.match(r"theta_([-+]?\d+\.?\d*)_phi_([-+]?\d+\.?\d*)", key)
    if match is None:
        raise ValueError("Could not parse theta/phi from key: {}".format(key))
    theta = float(match.group(1))
    phi = float(match.group(2))
    return theta, phi


def load_delay_settings(delay_json_path):
    """
    Returns a list of dictionaries:
    [
        {
            "key": "theta_90_phi_150",
            "theta": 90.0,
            "phi": 150.0,
            "delays_ns": [...]
        },
        ...
    ]
    """
    with open(delay_json_path, "r") as f:
        raw = json.load(f)

    settings = []
    for key, delays in raw.items():
        theta, phi = parse_theta_phi_from_key(key)
        settings.append({
            "key": key,
            "theta": theta,
            "phi": phi,
            "delays_ns": delays
        })

    return settings


def select_angle_subset(settings, percent_to_use, rng):
    """
    Select a random subset of angle settings.
    """
    if percent_to_use <= 0:
        raise ValueError("ANGLE_PERCENT_TO_USE must be > 0")
    if percent_to_use > 100:
        raise ValueError("ANGLE_PERCENT_TO_USE must be <= 100")

    n_total = len(settings)
    n_keep = max(1, int(np.ceil((percent_to_use / 100.0) * n_total)))

    indices = np.arange(n_total)
    rng.shuffle(indices)
    selected_indices = np.sort(indices[:n_keep])

    return [settings[i] for i in selected_indices]


def load_and_prepare_pulse(pulse_json_path, pulse_start_time, pulse_end_time):
    with open(pulse_json_path, "r") as f:
        pulse_data = json.load(f)

    pulse_voltage = np.array(pulse_data["avg_wave"])
    pulse_time = np.array(pulse_data["t_axis_ns"])

    mask = (pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)
    pulse_voltage = pulse_voltage[mask]
    pulse_time = pulse_time[mask]

    pulse_voltage = pulse_voltage / np.max(np.abs(pulse_voltage))
    pulse_time = pulse_time - pulse_time[0]

    return pulse_voltage, pulse_time


def compute_snr(amplitude_scale, noise_equalize):
    """
    Simple amplitude/noise definition.
    Adjust here if you want your exact thesis SNR convention.
    """
    return float(amplitude_scale / noise_equalize)


# ============================================================
# Main scan
# ============================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

print("Loading delay settings from:")
print(DELAY_JSON_PATH)
all_angle_settings = load_delay_settings(DELAY_JSON_PATH)
selected_angle_settings = select_angle_subset(all_angle_settings, ANGLE_PERCENT_TO_USE, rng)

print("Total angle settings in file: {}".format(len(all_angle_settings)))
print("Selected angle settings: {} ({:.2f}%)".format(
    len(selected_angle_settings),
    100.0 * len(selected_angle_settings) / len(all_angle_settings)
))

print("Loading pulse waveform...")
pulse_voltage, pulse_time = load_and_prepare_pulse(
    PULSE_JSON_PATH,
    PULSE_START_TIME,
    PULSE_END_TIME
)

results = {
    "settings": {
        "sampling_rate_ghz": SAMPLING_RATE,
        "time_step_ns": TIME_STEP,
        "noise_equalize_adc": NOISE_EQUALIZE,
        "max_signal_adc": MAX_SIGNAL,
        "window_size_mhz": WINDOW_SIZE,
        "n_of_windows": N_OF_WINDOWS,
        "simulation_duration_ns": SIMULATION_DURATION_NS,
        "simulation_duration_samples": SIMULATION_DURATION_SAMPLES,
        "n_of_channels": N_OF_CHANNELS,
        "scan_rate": SCAN_RATE,
        "csw_threshold": CSW_THRESHOLD,
        "pulse_amplitudes": PULSE_AMPLITUDES.tolist(),
        "angle_percent_to_use": ANGLE_PERCENT_TO_USE,
        "random_seed": RANDOM_SEED,
        "pulse_start_time_ns": PULSE_START_TIME,
        "pulse_end_time_ns": PULSE_END_TIME,
        "delay_json_path": str(DELAY_JSON_PATH),
        "pulse_json_path": str(PULSE_JSON_PATH),
        "impulse_response_path": str(IMPULSE_RESPONSE_PATH)
    },
    "results": []
}

global_start = time.time()
n_total_jobs = len(selected_angle_settings) * len(PULSE_AMPLITUDES)
job_counter = 0

for angle_index, angle_setting in enumerate(selected_angle_settings, start=1):
    theta = angle_setting["theta"]
    phi = angle_setting["phi"]
    delay_list = np.array(angle_setting["delays_ns"], dtype=float)

    print("\n============================================================")
    print("Angle setting {}/{}".format(angle_index, len(selected_angle_settings)))
    print("theta = {}, phi = {}".format(theta, phi))
    print("delays [ns] = {}".format(np.round(delay_list, 3)))
    print("============================================================")

    for amp_index, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES, start=1):
        job_counter += 1
        job_start = time.time()

        COINC = 0
        event_times = []
        event_triggered = []

        print("\nStarting amplitude {}/{} at this angle: {}".format(
            amp_index, len(PULSE_AMPLITUDES), run_pulse_amplitude
        ))
        print("Global job {}/{}".format(job_counter, n_total_jobs))

        for event_idx in range(SCAN_RATE):
            start_seed = random.uniform(0, TIME_STEP)
            channel_signals = [[] for _ in range(N_OF_CHANNELS)]

            for ch in range(N_OF_CHANNELS):
                t_axis, channel_signals[ch] = make_full_signal_with_delay(
                    impulse_json_path=IMPULSE_RESPONSE_PATH,
                    SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                    SAMPLING_RATE=SAMPLING_RATE,
                    NOISE_EQUALIZE=NOISE_EQUALIZE,
                    pulse_voltage=pulse_voltage,
                    pulse_time=pulse_time,
                    time_step=TIME_STEP,
                    simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                    amplitude_scale=run_pulse_amplitude,
                    max_signal=MAX_SIGNAL,
                    start_time=start_seed,
                    pulse_delay_ns=delay_list[ch]
                )

            if PLOT_FIRST_EVENT and event_idx == 0:
                plot_channels_signals(t_axis, channel_signals, run_pulse_amplitude)

            triggers = ARA_CSW_trigger_FFT_optimized(
                channel_signals,
                t_axis,
                threshold=float(CSW_THRESHOLD),
                noise_rms=float(NOISE_EQUALIZE),
                N_segments=int(1),
            )

            triggered = len(triggers) > 0
            if triggered:
                COINC += 1

            event_triggered.append(bool(triggered))
            event_times.append(float(start_seed))

            if ((event_idx + 1) % PRINT_EVERY_N_EVENTS == 0) or (event_idx == SCAN_RATE - 1):
                print(
                    "  Event {}/{} | current pass fraction = {:.4f}".format(
                        event_idx + 1,
                        SCAN_RATE,
                        COINC / float(event_idx + 1)
                    )
                )

        pass_fraction = COINC / float(SCAN_RATE)
        snr_value = compute_snr(run_pulse_amplitude, NOISE_EQUALIZE)
        job_elapsed = time.time() - job_start

        print("Finished amplitude {} at theta={}, phi={}".format(
            run_pulse_amplitude, theta, phi
        ))
        print("SNR = {:.4f}".format(snr_value))
        print("Pass fraction = {:.4f}".format(pass_fraction))
        print("Elapsed time for this job = {:.2f} s".format(job_elapsed))

        results["results"].append({
            "key": angle_setting["key"],
            "theta_deg": theta,
            "phi_deg": phi,
            "delays_ns": delay_list.tolist(),
            "amplitude_scale": float(run_pulse_amplitude),
            "snr": float(snr_value),
            "scan_rate": int(SCAN_RATE),
            "n_pass": int(COINC),
            "pass_fraction": float(pass_fraction),
            "event_start_times_ns": event_times,
            "event_triggered": event_triggered
        })

total_elapsed = time.time() - global_start
results["summary"] = {
    "n_total_angle_settings_in_file": len(all_angle_settings),
    "n_selected_angle_settings": len(selected_angle_settings),
    "n_total_jobs": n_total_jobs,
    "total_elapsed_seconds": total_elapsed
}

print("\n============================================================")
print("Full scan finished.")
print("Total elapsed time = {:.2f} s".format(total_elapsed))
print("Saving results to:")
print(OUTPUT_JSON_PATH)
print("============================================================")

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Done.")
