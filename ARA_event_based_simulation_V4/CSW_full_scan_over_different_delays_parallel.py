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
from multiprocessing import Pool, cpu_count

from sim_functions import *
from trig_functions_cop import *

# ============================================================
# User settings
# ============================================================
SAMPLING_RATE = 3.2
TIME_STEP = 1.0 / SAMPLING_RATE
NOISE_EQUALIZE = 100
MAX_SIGNAL = 4095
WINDOW_SIZE = 5.88 * 1e6
N_OF_WINDOWS = 2
SIMULATION_DURATION_NS = N_OF_WINDOWS / WINDOW_SIZE * 1e9
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)

N_OF_CHANNELS = 8
SCAN_RATE = 80
CSW_THRESHOLD = 13.32

# Parallel settings
N_CORES = 20

PULSE_AMPLITUDES = np.concatenate([
    np.arange(90, 130, 20),
    np.arange(130, 200, 10),
    np.arange(200, 270, 20)
])

ANGLE_PERCENT_TO_USE = 100
RANDOM_SEED = 12350

PLOT_FIRST_EVENT = False
PRINT_EVERY_N_EVENTS = 1000

DELAY_JSON_PATH = Path("Event_sources_for_CSW_2500.json").resolve()
OUTPUT_JSON_PATH = Path("efficiency_scan_100per_results_2500beams_FULL_all_angles.json").resolve()

PULSE_JSON_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json"
).resolve()

IMPULSE_RESPONSE_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json"
).resolve()

PULSE_START_TIME = 450
PULSE_END_TIME = 570


# ============================================================
# Helpers
# ============================================================
def parse_theta_phi_from_key(key):
    match = re.match(r"theta_([-+]?\d+\.?\d*)_phi_([-+]?\d+\.?\d*)", key)
    if match is None:
        raise ValueError("Could not parse theta/phi from key: {}".format(key))
    theta = float(match.group(1))
    phi = float(match.group(2))
    return theta, phi


def load_delay_settings(delay_json_path):
    with open(delay_json_path, "r") as f:
        raw = json.load(f)

    settings = []

    for key, val in raw.items():

        # New format:
        # "zen_30.00_az_0.00": {
        #     "zenith_deg": 30.0,
        #     "azimuth_deg": 0.0,
        #     "delays_ns": [...]
        # }
        if isinstance(val, dict) and "delays_ns" in val:
            theta = float(val["zenith_deg"])
            phi = float(val["azimuth_deg"])
            delays = val["delays_ns"]

        # Old format:
        # "theta_90_phi_150": [...]
        elif isinstance(val, list):
            theta, phi = parse_theta_phi_from_key(key)
            delays = val

        else:
            raise ValueError(f"Unrecognized delay format for key: {key}")

        settings.append({
            "key": key,
            "theta": theta,
            "phi": phi,
            "delays_ns": delays
        })

    return settings


def select_angle_subset(settings, percent_to_use, rng):
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
    return float(amplitude_scale / noise_equalize)


def run_one_event(args):
    """
    One independent event simulation.
    This is what gets distributed across CPU cores.
    """

    (
        event_idx,
        run_pulse_amplitude,
        delay_list,
        pulse_voltage,
        pulse_time,
        seed
    ) = args

    # Independent random generator per event
    rng_local = np.random.default_rng(seed)
    start_seed = float(rng_local.uniform(0, TIME_STEP))

    channel_signals = [[] for _ in range(N_OF_CHANNELS)]
    t_axis = None

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

    triggers = ARA_CSW_trigger_FFT_optimized(
        channel_signals,
        t_axis,
        threshold=float(CSW_THRESHOLD),
        noise_rms=float(NOISE_EQUALIZE),
        N_segments=int(1),
    )

    triggered = len(triggers) > 0

    return {
        "event_idx": int(event_idx),
        "start_seed": float(start_seed),
        "triggered": bool(triggered),
    }


# ============================================================
# Main scan
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading delay settings from:")
    print(DELAY_JSON_PATH)
    all_angle_settings = load_delay_settings(DELAY_JSON_PATH)
    selected_angle_settings = select_angle_subset(
        all_angle_settings,
        ANGLE_PERCENT_TO_USE,
        rng
    )

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
            "impulse_response_path": str(IMPULSE_RESPONSE_PATH),
            "n_parallel_cores": min(N_CORES, cpu_count())
        },
        "results": []
    }

    global_start = time.time()
    n_total_jobs = len(selected_angle_settings) * len(PULSE_AMPLITUDES)
    job_counter = 0

    n_workers = min(N_CORES, cpu_count())
    print("Using {} CPU cores".format(n_workers))

    with Pool(processes=n_workers) as pool:

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

                print("\nStarting amplitude {}/{} at this angle: {}".format(
                    amp_index, len(PULSE_AMPLITUDES), run_pulse_amplitude
                ))
                print("Global job {}/{}".format(job_counter, n_total_jobs))

                # Make deterministic but independent seeds for all events
                event_seeds = rng.integers(
                    low=0,
                    high=2**32 - 1,
                    size=SCAN_RATE,
                    dtype=np.uint32
                )

                event_args = [
                    (
                        event_idx,
                        float(run_pulse_amplitude),
                        delay_list,
                        pulse_voltage,
                        pulse_time,
                        int(event_seeds[event_idx])
                    )
                    for event_idx in range(SCAN_RATE)
                ]

                event_results = pool.map(run_one_event, event_args)

                # Sort to preserve original event order in the output JSON
                event_results = sorted(event_results, key=lambda x: x["event_idx"])

                event_times = [r["start_seed"] for r in event_results]
                event_triggered = [r["triggered"] for r in event_results]
                COINC = int(np.sum(event_triggered))

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


if __name__ == "__main__":
    main()