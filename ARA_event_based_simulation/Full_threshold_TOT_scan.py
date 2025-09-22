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

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_GHZ         = 3.2      # GHz
TIME_STEP_NS              = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC             = 100      # ADC (amplitude rms for noise generator)
MAX_SIGNAL_ADC            = 4095     
WINDOW_SIZE_MHZ           = 5.88e6   # MHz
N_WINDOWS                 = 1
SIM_DURATION_NS           = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9  # ns
SIM_DURATION_SAMPLES      = int(SIM_DURATION_NS / TIME_STEP_NS)
N_CHANNELS                = 8
N_REQ_COINC               = 3        # channels required for a trigger
SCAN_TIME_LIMIT_SEC       = 60 #3600*2   # 2 hours per TOT elimination setting
START_THRESHOLD           = 10000    # in POWER units (ADC^2)
THRESHOLD_STEP            = 1500      # increment per completed threshold
TRIGGERS_PER_THRESHOLD    = 10       # stop each threshold at n triggers
starting_MIN_ALLOWED_TOT  = 3        # in samples (ns / TIME_STEP_NS), minimum TOT to consider a trigger valid
ending_MIN_ALLOWED_TOT    = 13       # in samples (ns / TIME_STEP_NS
# Impulse-response JSON path
impulse_response_path = Path("../RNOG_sim_copy/jsons/impulse_response_Freauency_35_240.json").resolve()

# Output file
OUT_JSON = Path("Full_threshold_scan_for_TOT_range.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helper to save results incrementally
# ─────────────────────────────────────────────────────────────────────────────
def save_results(results_dict, out_path: Path):
    out_path.write_text(json.dumps(results_dict, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────
def run_scan(min_tot: int, results_dict: dict):
    t_start = time.time()
    threshold = START_THRESHOLD
    thresholds_completed = 0

    results_for_this_tot = []

    try:
        while True:
            if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                print(f"\nTime limit (2 hours) reached for TOT={min_tot}. Moving on.")
                break

            num_triggers = 0
            num_events_scanned = 0
            tot_samples = []

            THRESHOLD_V = [threshold] * N_CHANNELS

            print(f"\n=== TOT {min_tot}, Threshold {threshold} (power units) ===")

            while num_triggers < TRIGGERS_PER_THRESHOLD:
                if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                    print(f"\nTime limit reached mid-threshold for TOT={min_tot}.")
                    break

                channel_signals = []
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

                triggers = find_ARA_env_triggers(
                    channel_signals,
                    t_axis,
                    threshold=THRESHOLD_V,
                    n_channels_required=N_REQ_COINC
                )

                num_events_scanned += 1

                if triggers:
                    tot, n_ch_trig = TOT_finder(
                        channel_signals,
                        t_axis,
                        threshold=THRESHOLD_V,
                        n_channels_required=N_REQ_COINC
                    )
                    if tot > min_tot:
                        tot_samples.append(int(tot))
                        num_triggers += 1
                        rate = num_triggers / num_events_scanned
                        print(f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                              f"Events: {num_events_scanned} | "
                              f"Rate: {rate:.4f}")

            if num_triggers < TRIGGERS_PER_THRESHOLD:
                # save partial data
                record = {
                    "threshold": int(threshold),
                    "num_triggers": int(num_triggers),
                    "num_events_scanned": int(num_events_scanned),
                    "trigger_rate": float(num_triggers / num_events_scanned) if num_events_scanned else 0.0,
                    "tot_samples": [int(x) for x in tot_samples],
                }
                results_for_this_tot.append(record)
                break

            record = {
                "threshold": int(threshold),
                "num_triggers": int(num_triggers),
                "num_events_scanned": int(num_events_scanned),
                "trigger_rate": float(num_triggers / num_events_scanned),
                "tot_samples": [int(x) for x in tot_samples],
            }
            results_for_this_tot.append(record)
            results_dict[str(min_tot)] = results_for_this_tot
            save_results(results_dict, OUT_JSON)

            thresholds_completed += 1
            print(f"Completed threshold {threshold} for TOT={min_tot}: "
                  f"{num_triggers}/{num_events_scanned} → rate={record['trigger_rate']:.6f}")

            threshold += THRESHOLD_STEP

    except KeyboardInterrupt:
        print(f"\nInterrupted by user during TOT={min_tot}. Saving partial results…")
    finally:
        results_dict[str(min_tot)] = results_for_this_tot
        save_results(results_dict, OUT_JSON)
        elapsed = time.time() - t_start
        print(f"\nScan finished for TOT={min_tot}. "
              f"Thresholds completed: {thresholds_completed}. "
              f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'.")

# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results_dict = {}
    if OUT_JSON.exists():
        try:
            results_dict = json.loads(OUT_JSON.read_text())
        except Exception:
            results_dict = {}

    for min_tot in range(starting_MIN_ALLOWED_TOT, ending_MIN_ALLOWED_TOT + 1):
        run_scan(min_tot, results_dict)

if __name__ == "__main__":
    main()
