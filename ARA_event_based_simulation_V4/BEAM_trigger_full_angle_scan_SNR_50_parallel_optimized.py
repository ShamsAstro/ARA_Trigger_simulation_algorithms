# ============================================================
# Thread-control must happen before importing numpy/scipy
# This prevents NumPy/OpenBLAS/MKL from oversubscribing CPU threads
# while multiprocessing is already using many processes.
# ============================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
SCAN_RATE = 12

# Beam trigger threshold
BEAM_THRESHOLD = 1.26e8

# Parallel settings
N_CORES = 20

"""
# Keep same pulse amplitude range
PULSE_AMPLITUDES = np.concatenate([
    np.arange(100, 350, 35),
    np.arange(200, 280, 20),
    np.arange(280, 340, 20),

])
"""
PULSE_AMPLITUDES = np.arange(130, 350, 35)

ANGLE_PERCENT_TO_USE = 100
RANDOM_SEED = 12350

PLOT_FIRST_EVENT = False
PRINT_EVERY_N_EVENTS = 1000

# Event source delays and predefined beam delays
EVENT_DELAY_JSON = Path("Event_sources_origins_25000.json").resolve()
BEAM_DELAY_JSON = Path("Trigger_beams_1000_+-60.json").resolve()

OUTPUT_JSON_PATH = Path("efficiency_scan_BEAM_100per_results_25000events_1000beams_FULL_all_angles.json").resolve()

PULSE_JSON_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json"
).resolve()

IMPULSE_RESPONSE_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json"
).resolve()

PULSE_START_TIME = 450
PULSE_END_TIME = 570

# Save partial progress every N completed angle/amplitude jobs
SAVE_EVERY_N_JOBS = 500


# ============================================================
# Helpers
# ============================================================
def parse_theta_phi_from_key(key):
    """
    Supports old keys like:
      theta_90_phi_150

    And newer keys like:
      zen_30.00_az_0.00
    """

    match = re.match(r"theta_([-+]?\d+\.?\d*)_phi_([-+]?\d+\.?\d*)", key)
    if match is not None:
        theta = float(match.group(1))
        phi = float(match.group(2))
        return theta, phi

    match = re.match(r"zen_([-+]?\d+\.?\d*)_az_([-+]?\d+\.?\d*)", key)
    if match is not None:
        theta = float(match.group(1))
        phi = float(match.group(2))
        return theta, phi

    raise ValueError("Could not parse theta/phi from key: {}".format(key))


def load_delay_settings(delay_json_path):
    """
    Supports both formats:

    New format:
    {
      "zen_30.00_az_0.00": {
        "zenith_deg": 30.0,
        "azimuth_deg": 0.0,
        "delays_ns": [...]
      }
    }

    Old format:
    {
      "theta_90_phi_150": [...]
    }
    """

    with open(delay_json_path, "r") as f:
        raw = json.load(f)

    settings = []

    for key, val in raw.items():

        if isinstance(val, dict) and "delays_ns" in val:
            theta = float(val["zenith_deg"])
            phi = float(val["azimuth_deg"])
            delays = val["delays_ns"]

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


def build_beam_inputs_from_settings(beam_settings):
    """
    Converts loaded beam settings into:

      beam_angles = [(theta, phi), ...]
      beam_delays[(theta, phi)] = np.array([...])
    """

    beam_angles = []
    beam_delays = {}

    for s in beam_settings:
        angle_key = (float(s["theta"]), float(s["phi"]))
        beam_angles.append(angle_key)
        beam_delays[angle_key] = np.array(s["delays_ns"], dtype=float)

    return beam_angles, beam_delays


def save_results_json(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


# ============================================================
# Worker globals
# ============================================================
GLOBAL_BEAM_ANGLES = None
GLOBAL_BEAM_DELAYS = None
GLOBAL_PULSE_VOLTAGE = None
GLOBAL_PULSE_TIME = None


def init_worker(beam_angles, beam_delays, pulse_voltage, pulse_time):
    """
    Store constant objects inside each worker once.

    This avoids sending the beam bank and pulse arrays with every job.
    """

    global GLOBAL_BEAM_ANGLES
    global GLOBAL_BEAM_DELAYS
    global GLOBAL_PULSE_VOLTAGE
    global GLOBAL_PULSE_TIME

    GLOBAL_BEAM_ANGLES = beam_angles
    GLOBAL_BEAM_DELAYS = beam_delays
    GLOBAL_PULSE_VOLTAGE = pulse_voltage
    GLOBAL_PULSE_TIME = pulse_time


def run_one_angle_amplitude_job(args):
    """
    One independent job:
      one event angle + one pulse amplitude + SCAN_RATE events

    This is more efficient than parallelizing only the SCAN_RATE events,
    because SCAN_RATE is only 12 while the machine has 20 cores.
    """

    (
        angle_index,
        amp_index,
        angle_setting,
        run_pulse_amplitude,
        event_seeds
    ) = args

    theta = angle_setting["theta"]
    phi = angle_setting["phi"]
    delay_list = np.array(angle_setting["delays_ns"], dtype=float)

    event_times = []
    event_triggered = []

    for event_idx in range(SCAN_RATE):
        rng_local = np.random.default_rng(int(event_seeds[event_idx]))
        start_seed = float(rng_local.uniform(0, TIME_STEP))

        channel_signals = [[] for _ in range(N_OF_CHANNELS)]
        t_axis = None

        for ch in range(N_OF_CHANNELS):
            t_axis, channel_signals[ch] = make_full_signal_with_delay(
                impulse_json_path=IMPULSE_RESPONSE_PATH,
                SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                SAMPLING_RATE=SAMPLING_RATE,
                NOISE_EQUALIZE=NOISE_EQUALIZE,
                pulse_voltage=GLOBAL_PULSE_VOLTAGE,
                pulse_time=GLOBAL_PULSE_TIME,
                time_step=TIME_STEP,
                simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                amplitude_scale=run_pulse_amplitude,
                max_signal=MAX_SIGNAL,
                start_time=start_seed,
                pulse_delay_ns=delay_list[ch]
            )

        channel_signals = np.asarray(channel_signals, dtype=float)

        # ========================================================
        # BEAM TRIGGER CALL
        # Keep this as your working function call.
        # ========================================================
        triggers = ARA_beam_trigger(
            channel_signals,
            t_axis,
            threshold=float(BEAM_THRESHOLD),
            sampling_rate=float(SAMPLING_RATE),
            beam_angles=GLOBAL_BEAM_ANGLES,
            beam_delays=GLOBAL_BEAM_DELAYS
        )

        # Supports both styles:
        #   - function returns bool
        #   - function returns list/array of trigger objects
        if isinstance(triggers, (bool, np.bool_)):
            triggered = bool(triggers)
        else:
            triggered = len(triggers) > 0

        event_times.append(float(start_seed))
        event_triggered.append(bool(triggered))

    COINC = int(np.sum(event_triggered))
    pass_fraction = COINC / float(SCAN_RATE)
    snr_value = compute_snr(run_pulse_amplitude, NOISE_EQUALIZE)

    result_row = {
        "key": angle_setting["key"],
        "theta_deg": float(theta),
        "phi_deg": float(phi),
        "delays_ns": delay_list.tolist(),
        "amplitude_scale": float(run_pulse_amplitude),
        "snr": float(snr_value),
        "scan_rate": int(SCAN_RATE),
        "n_pass": int(COINC),
        "pass_fraction": float(pass_fraction),
        "event_start_times_ns": event_times,
        "event_triggered": event_triggered
    }

    return {
        "angle_index": int(angle_index),
        "amp_index": int(amp_index),
        "result_row": result_row
    }


# ============================================================
# Main scan
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading event delay settings from:")
    print(EVENT_DELAY_JSON)

    all_angle_settings = load_delay_settings(EVENT_DELAY_JSON)
    selected_angle_settings = select_angle_subset(
        all_angle_settings,
        ANGLE_PERCENT_TO_USE,
        rng
    )

    print("Total event angle settings in file: {}".format(len(all_angle_settings)))
    print("Selected event angle settings: {} ({:.2f}%)".format(
        len(selected_angle_settings),
        100.0 * len(selected_angle_settings) / len(all_angle_settings)
    ))

    print("\nLoading beam delay settings from:")
    print(BEAM_DELAY_JSON)

    beam_settings = load_delay_settings(BEAM_DELAY_JSON)
    beam_angles, beam_delays = build_beam_inputs_from_settings(beam_settings)

    print("Loaded {} predefined beams.".format(len(beam_angles)))

    print("\nLoading pulse waveform...")
    pulse_voltage, pulse_time = load_and_prepare_pulse(
        PULSE_JSON_PATH,
        PULSE_START_TIME,
        PULSE_END_TIME
    )

    results = {
        "settings": {
            "algorithm": "BEAM",
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
            "beam_threshold": BEAM_THRESHOLD,
            "n_beams": len(beam_angles),
            "pulse_amplitudes": PULSE_AMPLITUDES.tolist(),
            "angle_percent_to_use": ANGLE_PERCENT_TO_USE,
            "random_seed": RANDOM_SEED,
            "pulse_start_time_ns": PULSE_START_TIME,
            "pulse_end_time_ns": PULSE_END_TIME,
            "event_delay_json_path": str(EVENT_DELAY_JSON),
            "beam_delay_json_path": str(BEAM_DELAY_JSON),
            "pulse_json_path": str(PULSE_JSON_PATH),
            "impulse_response_path": str(IMPULSE_RESPONSE_PATH),
            "n_parallel_cores": min(N_CORES, cpu_count()),
            "parallelization_mode": "angle_amplitude_jobs"
        },
        "results": []
    }

    global_start = time.time()

    n_workers = min(N_CORES, cpu_count())
    print("\nUsing {} CPU cores".format(n_workers))

    # ------------------------------------------------------------
    # Build independent angle/amplitude jobs.
    # This preserves the same logical output rows as the old script:
    # one result row per angle + amplitude combination.
    # ------------------------------------------------------------
    job_args = []

    for angle_index, angle_setting in enumerate(selected_angle_settings, start=1):
        for amp_index, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES, start=1):

            event_seeds = rng.integers(
                low=0,
                high=2**32 - 1,
                size=SCAN_RATE,
                dtype=np.uint32
            )

            job_args.append((
                angle_index,
                amp_index,
                angle_setting,
                float(run_pulse_amplitude),
                event_seeds
            ))

    n_total_jobs = len(job_args)

    print("Total angle/amplitude jobs: {}".format(n_total_jobs))
    print("Each job runs SCAN_RATE = {} events.".format(SCAN_RATE))

    completed_jobs = 0
    partial_rows = []

    try:
        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(beam_angles, beam_delays, pulse_voltage, pulse_time)
        ) as pool:

            # chunksize > 1 reduces multiprocessing overhead.
            # 1 is safest for uneven job times, 2-5 may be faster.
            chunksize = 1

            for job_result in pool.imap_unordered(
                run_one_angle_amplitude_job,
                job_args,
                chunksize=chunksize
            ):
                completed_jobs += 1
                partial_rows.append(job_result)

                row = job_result["result_row"]

                if (
                    completed_jobs % 25 == 0
                    or completed_jobs == 1
                    or completed_jobs == n_total_jobs
                ):
                    elapsed = time.time() - global_start
                    rate_jobs = completed_jobs / elapsed if elapsed > 0 else 0.0
                    remaining = n_total_jobs - completed_jobs
                    eta_sec = remaining / rate_jobs if rate_jobs > 0 else float("nan")

                    print(
                        "Completed job {}/{} | theta={:.3f}, phi={:.3f}, "
                        "amp={:.1f}, pass_fraction={:.4f} | elapsed={:.1f}s | ETA={:.1f}s".format(
                            completed_jobs,
                            n_total_jobs,
                            row["theta_deg"],
                            row["phi_deg"],
                            row["amplitude_scale"],
                            row["pass_fraction"],
                            elapsed,
                            eta_sec
                        )
                    )

                # Save partial progress every SAVE_EVERY_N_JOBS.
                # The final file keeps the same format and sorted order.
                if completed_jobs % SAVE_EVERY_N_JOBS == 0:
                    sorted_partial = sorted(
                        partial_rows,
                        key=lambda x: (x["angle_index"], x["amp_index"])
                    )

                    results["results"] = [x["result_row"] for x in sorted_partial]

                    results["summary"] = {
                        "algorithm": "BEAM",
                        "n_total_event_angle_settings_in_file": len(all_angle_settings),
                        "n_selected_event_angle_settings": len(selected_angle_settings),
                        "n_beams": len(beam_angles),
                        "n_total_jobs": n_total_jobs,
                        "completed_jobs_so_far": completed_jobs,
                        "total_elapsed_seconds_so_far": time.time() - global_start,
                        "partial_output": True
                    }

                    save_results_json(results, OUTPUT_JSON_PATH)
                    print("Partial results saved to {}".format(OUTPUT_JSON_PATH))

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")

    finally:
        # Sort rows so the output order matches the original nested-loop order:
        # angle_index first, then amp_index.
        sorted_rows = sorted(
            partial_rows,
            key=lambda x: (x["angle_index"], x["amp_index"])
        )

        total_elapsed = time.time() - global_start

        results["results"] = [x["result_row"] for x in sorted_rows]

        results["summary"] = {
            "algorithm": "BEAM",
            "n_total_event_angle_settings_in_file": len(all_angle_settings),
            "n_selected_event_angle_settings": len(selected_angle_settings),
            "n_beams": len(beam_angles),
            "n_total_jobs": n_total_jobs,
            "completed_jobs": len(sorted_rows),
            "total_elapsed_seconds": total_elapsed,
            "partial_output": len(sorted_rows) < n_total_jobs
        }

        print("\n============================================================")
        print("BEAM scan finished.")
        print("Completed jobs: {}/{}".format(len(sorted_rows), n_total_jobs))
        print("Total elapsed time = {:.2f} s".format(total_elapsed))
        print("Saving results to:")
        print(OUTPUT_JSON_PATH)
        print("============================================================")

        save_results_json(results, OUTPUT_JSON_PATH)

        print("Done.")


if __name__ == "__main__":
    main()