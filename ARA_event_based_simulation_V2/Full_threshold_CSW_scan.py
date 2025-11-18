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
#this version uses the CSW trigger logic 
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
SCAN_TIME_LIMIT_SEC       = 60*5  #3600*3   # 2 hours per  
START_THRESHOLD           = 50    # in POWER units (ADC^2)
THRESHOLD_STEP            = 25      # increment per completed threshold
TRIGGERS_PER_THRESHOLD    = 12       # stop each threshold at n triggers
CSW_corr_scan_step         = 3       # step for correlation scan in CSW trigger


# Impulse-response JSON path (adjust if needed)
# If you implemented caching in sim_functions, it will be used automatically.
pulse_json_path = Path("../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)
    
impulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

# Output file
OUT_JSON = Path("Full_threshold_scan_CSW_test_no_shifting.json")



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
                
                triggers = ARA_CSW_trigger_no_shifting(
                    channel_signals,
                    t_axis,
                    threshold=threshold,
                    noise_rms=NOISE_RMS_ADC, # use your noise scale as RMS
                )

                num_events_scanned += 1
                if len(triggers) > 0:
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
