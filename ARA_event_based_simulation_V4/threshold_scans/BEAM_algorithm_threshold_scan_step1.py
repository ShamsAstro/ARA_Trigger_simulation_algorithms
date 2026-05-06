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
from multiprocessing import Pool, cpu_count

from sim_functions import *
from trig_functions_cop import *

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_GHZ         = 3.2
TIME_STEP_NS              = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC             = 100
MAX_SIGNAL_ADC            = 4095
WINDOW_SIZE_MHZ           = 5.88e6
N_WINDOWS                 = 2
SIM_DURATION_NS           = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9
SIM_DURATION_SAMPLES      = int(SIM_DURATION_NS / TIME_STEP_NS)
N_CHANNELS                = 8
SCAN_TIME_LIMIT_SEC       = 60 * 30

# Beam-power threshold scan settings
START_THRESHOLD           = 8.0e7
THRESHOLD_STEP            = 3.50e6
TRIGGERS_PER_THRESHOLD    = 12

# Parallel settings
N_CORES = 20
BATCH_SIZE = 20

# Input JSONs
impulse_response_path = Path(
    "../../ARA_event_based_simulation_V3/jsons/new_impulse_response_ARA_event_based_simulation_V2.json"
).resolve()

BEAM_DELAY_JSON = Path("../Trigger_beams_1000_+-60.json").resolve()

# Output file
OUT_JSON = Path("Test_threshold_scan_BEAM_algorithm_1000beams.json")


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────
def save_results(results_list, out_path: Path):
    out_path.write_text(json.dumps(results_list, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Beam delay JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_delay_settings_json(json_path):
    """
    Supports beam-delay JSON format like:

    {
      "zen_30.00_az_0.00": {
        "zenith_deg": 30.0,
        "azimuth_deg": 0.0,
        "delays_ns": [...]
      }
    }

    Returns
    -------
    settings : list of dict
        [
            {
                "theta": zenith_deg,
                "phi": azimuth_deg,
                "delays_ns": [...]
            },
            ...
        ]
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    settings = []

    for key, entry in raw.items():

        if isinstance(entry, dict) and "delays_ns" in entry:
            settings.append({
                "key": key,
                "theta": float(entry["zenith_deg"]),
                "phi": float(entry["azimuth_deg"]),
                "delays_ns": list(entry["delays_ns"])
            })

        elif isinstance(entry, list):
            # Fallback for older format:
            # "theta_90_phi_150": [delays...]
            theta, phi = parse_theta_phi_from_key(key)
            settings.append({
                "key": key,
                "theta": float(theta),
                "phi": float(phi),
                "delays_ns": list(entry)
            })

        else:
            raise ValueError(f"Unrecognized beam-delay JSON format for key: {key}")

    settings.sort(key=lambda d: (d["theta"], d["phi"]))
    return settings


def parse_theta_phi_from_key(key):
    """
    Fallback parser for older beam files.
    Accepts:
      theta_90_phi_150
      zen_30.00_az_0.00
    """
    import re

    match = re.match(r"theta_([-+]?\d+\.?\d*)_phi_([-+]?\d+\.?\d*)", key)
    if match is not None:
        return float(match.group(1)), float(match.group(2))

    match = re.match(r"zen_([-+]?\d+\.?\d*)_az_([-+]?\d+\.?\d*)", key)
    if match is not None:
        return float(match.group(1)), float(match.group(2))

    raise ValueError(f"Could not parse theta/phi from key: {key}")


def build_beam_inputs_from_settings(settings):
    """
    Returns
    -------
    beam_angles : list of tuple
        [(theta, phi), ...]

    beam_delays : dict
        beam_delays[(theta, phi)] = delays_ns
    """
    beam_angles = []
    beam_delays = {}

    for s in settings:
        angle_key = (float(s["theta"]), float(s["phi"]))
        beam_angles.append(angle_key)
        beam_delays[angle_key] = np.array(s["delays_ns"], dtype=float)

    return beam_angles, beam_delays




# ─────────────────────────────────────────────────────────────────────────────
# Worker setup
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_BEAM_ANGLES = None
GLOBAL_BEAM_DELAYS = None


def init_worker(beam_angles, beam_delays):
    """
    Store beam bank globally inside each worker process.
    This avoids passing the beam list repeatedly for every single event.
    """
    global GLOBAL_BEAM_ANGLES
    global GLOBAL_BEAM_DELAYS

    GLOBAL_BEAM_ANGLES = beam_angles
    GLOBAL_BEAM_DELAYS = beam_delays


def run_one_noise_event(threshold):
    """
    Runs one pure-noise event and returns True/False for whether it triggered.
    This function is independent, so it can run safely in parallel.
    """

    channel_signals = []
    t_axis = None

    for ch in range(N_CHANNELS):
        t_axis, noise = make_band_limited_noise_digitized(
            json_path=impulse_response_path,
            channel_key="ch2_2x_amp",
            window_ns=SIM_DURATION_NS,
            adc_rate_ghz=SAMPLING_RATE_GHZ,
            target_rms_mV=NOISE_RMS_ADC,
            max_signal=MAX_SIGNAL_ADC,
        )
        channel_signals.append(noise)

    channel_signals = np.asarray(channel_signals, dtype=float)

    triggered = ARA_beam_trigger(
        channel_signals=channel_signals,
        time_axis=t_axis,
        threshold=threshold,
        sampling_rate=SAMPLING_RATE_GHZ,
        beam_angles=GLOBAL_BEAM_ANGLES,
        beam_delays=GLOBAL_BEAM_DELAYS,
        return_power=False
    )

    return bool(triggered)


# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = []

    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text())
        except Exception:
            results = []

    print("Loading beam delays from:")
    print(BEAM_DELAY_JSON)

    beam_settings = load_delay_settings_json(BEAM_DELAY_JSON)
    beam_angles, beam_delays = build_beam_inputs_from_settings(beam_settings)

    print(f"Loaded {len(beam_angles)} beams.")

    t_start = time.time()
    threshold = START_THRESHOLD
    thresholds_completed = 0

    n_workers = min(N_CORES, cpu_count())
    print(f"Using {n_workers} CPU cores")

    try:
        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(beam_angles, beam_delays)
        ) as pool:

            while True:
                if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                    print("\nTime limit reached. Stopping scan.")
                    break

                num_triggers = 0
                num_events_scanned = 0

                print(f"\n=== Threshold {threshold:.6e} beam-power units ===")

                while num_triggers < TRIGGERS_PER_THRESHOLD:
                    if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                        print("\nTime limit reached mid-threshold. Stopping.")
                        break

                    batch_thresholds = [threshold] * BATCH_SIZE
                    batch_results = pool.map(run_one_noise_event, batch_thresholds)

                    for triggered in batch_results:
                        num_events_scanned += 1

                        if triggered:
                            num_triggers += 1

                            if num_triggers % 3 == 0:
                                rate = num_triggers / num_events_scanned
                                print(
                                    f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                                    f"Events: {num_events_scanned} | Rate: {rate:.4f}"
                                )

                        if num_triggers >= TRIGGERS_PER_THRESHOLD:
                            break

                if num_triggers < TRIGGERS_PER_THRESHOLD:
                    break

                trigger_rate = num_triggers / num_events_scanned if num_events_scanned else 0.0

                record = {
                    "threshold": float(threshold),
                    "num_triggers": int(num_triggers),
                    "num_events_scanned": int(num_events_scanned),
                    "trigger_rate": float(trigger_rate),
                }

                results.append(record)
                save_results(results, OUT_JSON)
                thresholds_completed += 1

                print(
                    f"Completed threshold {threshold:.6e}: "
                    f"{num_triggers}/{num_events_scanned} → rate={trigger_rate:.6f}. "
                    f"Saved to '{OUT_JSON.name}'."
                )

                threshold += THRESHOLD_STEP

                if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                    print("\nTime limit reached after completing this threshold. Stopping.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results…")

    finally:
        save_results(results, OUT_JSON)
        elapsed = time.time() - t_start
        print(
            f"\nScan finished. Thresholds completed: {thresholds_completed}. "
            f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'."
        )


if __name__ == "__main__":
    main()