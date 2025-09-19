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
# Parameters (tune as you like)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_GHZ         = 3.2      # GHz
TIME_STEP_NS              = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC             = 100      # ADC (amplitude rms for noise generator)
MAX_SIGNAL_ADC            = 4095     
WINDOW_SIZE_MHZ           = 5.88e6   # MHz (your prior definition)
N_WINDOWS                 = 1
SIM_DURATION_NS           = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9  # ns
SIM_DURATION_SAMPLES      = int(SIM_DURATION_NS / TIME_STEP_NS)
N_CHANNELS                = 8
N_REQ_COINC               = 3        # channels required for a trigger
SCAN_TIME_LIMIT_SEC       = 3600         #3600     # one hour
START_THRESHOLD           = 2000     # in POWER units (ADC^2), by your spec
THRESHOLD_STEP            = 3000     # increment per completed threshold
TRIGGERS_PER_THRESHOLD    = 15       # stop each threshold at n triggers
MIN_ALLOWED_TOT          = 4        # in samples (ns / TIME_STEP_NS), minimum TOT to consider a trigger valid

# Impulse-response JSON path (adjust if needed)
# If you implemented caching in sim_functions, it will be used automatically.
impulse_response_path = Path("../RNOG_sim_copy/jsons/impulse_response_Freauency_35_240.json").resolve()

# Output file
OUT_JSON = Path("threshold_scan_rates_Eliminate_low_tot_long.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helper to save results incrementally
# ─────────────────────────────────────────────────────────────────────────────
def save_results(results_list, out_path: Path):
    out_path.write_text(json.dumps(results_list, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng()  # optional (not strictly required)
    results = []

    # If file exists, resume and continue appending
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text())
        except Exception:
            # If corrupted, start fresh (or handle otherwise)
            results = []

    t_start = time.time()
    threshold = START_THRESHOLD
    thresholds_completed = 0

    try:
        while True:
            # Time check (stop if 1 hour passed)
            if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                print("\nTime limit reached (1 hour). Stopping scan.")
                break

            # Prepare per-threshold counters
            num_triggers = 0
            num_events_scanned = 0
            tot_samples = []

            THRESHOLD_V = [threshold] * N_CHANNELS  # per-channel (POWER units)

            print(f"\n=== Threshold {threshold} (power units) ===")
            # Inner loop: keep generating events until we reach 15 triggers
            while num_triggers < TRIGGERS_PER_THRESHOLD:
                # Time check inside too, to avoid overruns
                if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                    print("\nTime limit reached (1 hour) mid-threshold. Stopping.")
                    break

                # Generate one pure-noise event across all channels
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

                # Analyze for trigger using your envelope-based trigger
                triggers = find_ARA_env_triggers(
                    channel_signals,
                    t_axis,  # local event time axis is fine
                    threshold=THRESHOLD_V,
                    n_channels_required=N_REQ_COINC
                )

                num_events_scanned += 1

                if triggers:
                    # Only compute TOT if triggered (as requested)
                    tot, n_ch_trig = TOT_finder(
                        channel_signals,
                        t_axis,
                        threshold=THRESHOLD_V,
                        n_channels_required=N_REQ_COINC
                    )
                    if tot> MIN_ALLOWED_TOT:  # Only consider events with TOT greater than 5 nsamples
                        tot_samples.append(int(tot))
                        num_triggers += 1

                    # Optional: brief progress line
                    if num_triggers % 2 == 0:  # print every few triggers
                        rate = num_triggers / num_events_scanned
                        print(f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                              f"Events: {num_events_scanned} | "
                              f"Current rate: {rate:.4f}")

            # If we broke due to time, and didn't finish this threshold, don't record it
            if num_triggers < TRIGGERS_PER_THRESHOLD:
                break

            # Compute and record stats for this threshold
            trigger_rate = num_triggers / num_events_scanned if num_events_scanned else 0.0
            record = {
                "threshold": int(threshold),
                "num_triggers": int(num_triggers),                 # should be 15
                "num_events_scanned": int(num_events_scanned),
                "trigger_rate": float(trigger_rate),               # 15 / events
                "tot_samples": [int(x) for x in tot_samples],      # list of 15 values
            }
            results.append(record)
            save_results(results, OUT_JSON)
            thresholds_completed += 1

            print(f"Completed threshold {threshold}: "
                  f"{num_triggers}/{num_events_scanned} → rate={trigger_rate:.6f}. "
                  f"Saved to '{OUT_JSON.name}'.")

            # Next threshold
            threshold += THRESHOLD_STEP

            # Check time again before next threshold
            if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                print("\nTime limit reached after completing this threshold. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results…")
    finally:
        # Always write whatever we have
        save_results(results, OUT_JSON)
        elapsed = time.time() - t_start
        print(f"\nScan finished. Thresholds completed: {thresholds_completed}. "
              f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'.")

if __name__ == "__main__":
    main()
