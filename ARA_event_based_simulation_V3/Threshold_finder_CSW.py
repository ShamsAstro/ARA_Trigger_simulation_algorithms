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
#from trig_functions import *
from trig_functions_cop import *  # include CSW trigger definitions

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_GHZ         = 3.2
TIME_STEP_NS              = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC             = 100
MAX_SIGNAL_ADC            = 4095
WINDOW_SIZE_MHZ           = 5.88e6
N_WINDOWS                 = 1
SIM_DURATION_NS           = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9
SIM_DURATION_SAMPLES      = int(SIM_DURATION_NS / TIME_STEP_NS)
N_CHANNELS                = 8
SCAN_TIME_LIMIT_SEC       = 60   #*60   # 10 hours
START_THRESHOLD            = 4        # CSW trigger threshold
THRESHOLD_STEP             = 0.5
TRIGGERS_PER_THRESHOLD     = 12

# Input JSONs
pulse_json_path = Path("../ARA_event_based_simulation_V3/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)

impulse_response_path = Path("../ARA_event_based_simulation_V3/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

# Output file
OUT_JSON = Path("threshold_scan_CSWFFT_8channels.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helper to save results incrementally
# ─────────────────────────────────────────────────────────────────────────────
def save_results(results_list, out_path: Path):
    out_path.write_text(json.dumps(results_list, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng()
    results = []

    # Resume if existing file
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text())
        except Exception:
            results = []

    t_start = time.time()
    threshold = START_THRESHOLD
    thresholds_completed = 0

    try:
        while True:
            if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                print("\nTime limit reached. Stopping scan.")
                break

            num_triggers = 0
            num_events_scanned = 0

            print(f"\n=== Threshold {threshold} (CSW units) ===")

            while num_triggers < TRIGGERS_PER_THRESHOLD:
                if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                    print("\nTime limit reached mid-threshold. Stopping.")
                    break

                # Generate pure noise for all channels
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

                # Apply CSW FFT trigger
                triggers = ARA_CSW_trigger_FFT_optimized(
                    channel_signals,
                    t_axis,
                    threshold=threshold,
                    noise_rms=NOISE_RMS_ADC,
                    N_segments=10
                )

                num_events_scanned += 1

                if triggers:
                    num_triggers += 1
                    if num_triggers % 3 == 0:
                        rate = num_triggers / num_events_scanned
                        print(f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                              f"Events: {num_events_scanned} | Rate: {rate:.4f}")

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

            print(f"Completed threshold {threshold}: "
                  f"{num_triggers}/{num_events_scanned} → rate={trigger_rate:.6f}. "
                  f"Saved to '{OUT_JSON.name}'.")

            threshold += THRESHOLD_STEP

            if time.time() - t_start >= SCAN_TIME_LIMIT_SEC:
                print("\nTime limit reached after completing this threshold. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results…")
    finally:
        save_results(results, OUT_JSON)
        elapsed = time.time() - t_start
        print(f"\nScan finished. Thresholds completed: {thresholds_completed}. "
              f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'.")

if __name__ == "__main__":
    main()
