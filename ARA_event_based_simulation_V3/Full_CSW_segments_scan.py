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
# from trig_functions import *
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
SCAN_TIME_LIMIT_SEC       = 60*60*10   #*60   # 10 hours
START_THRESHOLD            = 4        # CSW trigger threshold
THRESHOLD_STEP             = 1
TRIGGERS_PER_THRESHOLD     = 15

# Input JSONs
pulse_json_path = Path("../ARA_event_based_simulation_V3/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)

impulse_response_path = Path("../ARA_event_based_simulation_V3/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

# Output file
OUT_JSON = Path("threshold_CSW_segments_N_1segment_scan.json")

# N_segments scan: 1,2,4, then 6..20 step 2
N_SEGMENTS_LIST =[1] #[1, 2, 4] + list(range(6, 22, 2))

# ─────────────────────────────────────────────────────────────────────────────
# Output structure helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_metadata():
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
        "simulation_parameters": {
            "SAMPLING_RATE_GHZ": SAMPLING_RATE_GHZ,
            "TIME_STEP_NS": TIME_STEP_NS,
            "NOISE_RMS_ADC": NOISE_RMS_ADC,
            "MAX_SIGNAL_ADC": MAX_SIGNAL_ADC,
            "WINDOW_SIZE_MHZ": WINDOW_SIZE_MHZ,
            "N_WINDOWS": N_WINDOWS,
            "SIM_DURATION_NS": SIM_DURATION_NS,
            "SIM_DURATION_SAMPLES": SIM_DURATION_SAMPLES,
            "N_CHANNELS": N_CHANNELS,
            "SCAN_TIME_LIMIT_SEC": SCAN_TIME_LIMIT_SEC,
            "START_THRESHOLD": START_THRESHOLD,
            "THRESHOLD_STEP": THRESHOLD_STEP,
            "TRIGGERS_PER_THRESHOLD": TRIGGERS_PER_THRESHOLD,
            "N_SEGMENTS_LIST": N_SEGMENTS_LIST,
        },
        "inputs": {
            "pulse_json_path": str(pulse_json_path),
            "impulse_response_path": str(impulse_response_path),
            "impulse_channel_key": "ch2_2x_amp",
        },
        "trigger": {
            "function": "ARA_CSW_trigger_FFT_optimized",
            "notes": "Noise-only scans; each record reports trigger_rate = num_triggers/num_events_scanned."
        }
    }

def load_or_init_output(out_path: Path):
    """
    JSON layout:
    {
      "meta": {...},
      "results": [ {record}, {record}, ... ]
    }
    """
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
            if isinstance(data, dict) and "meta" in data and "results" in data:
                return data
        except Exception:
            pass

    return {"meta": build_metadata(), "results": []}

def save_output(data: dict, out_path: Path):
    out_path.write_text(json.dumps(data, indent=2))

def completed_pairs(results_list):
    """Return set of (N_segments, threshold) already done."""
    done = set()
    for r in results_list:
        try:
            done.add((int(r["N_segments"]), float(r["threshold"])))
        except Exception:
            continue
    return done

# ─────────────────────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng()

    out = load_or_init_output(OUT_JSON)
    results = out["results"]
    done = completed_pairs(results)

    t_start = time.time()
    segments_completed = 0

    t_global_start = time.time()  # optional: for reporting total runtime

    try:
        for N_segments in N_SEGMENTS_LIST:
            print(f"\n==============================")
            print(f"Scanning N_segments = {N_segments}")
            print(f"==============================")

            t_seg_start = time.time()   # <--- RESET timer per N_segments
            threshold = START_THRESHOLD

            # Run thresholds until this N_segments budget is exhausted
            while True:
                seg_elapsed = time.time() - t_seg_start
                if seg_elapsed >= SCAN_TIME_LIMIT_SEC:
                    print(f"\nPer-segment time limit reached for N_segments={N_segments}. "
                        f"Elapsed: {seg_elapsed:.2f}s. Moving to next N_segments.")
                    break  # move to next N_segments

                # If resuming, skip already-completed (N_segments, threshold)
                if (N_segments, float(threshold)) in done:
                    print(f"  Skipping (N_segments={N_segments}, threshold={threshold}) [already in JSON]")
                    threshold += THRESHOLD_STEP
                    continue

                num_triggers = 0
                num_events_scanned = 0

                print(f"\n=== N_segments {N_segments} | Threshold {threshold} (CSW units) ===")

                while num_triggers < TRIGGERS_PER_THRESHOLD:
                    seg_elapsed = time.time() - t_seg_start
                    if seg_elapsed >= SCAN_TIME_LIMIT_SEC:
                        print(f"\nPer-segment time limit hit mid-threshold "
                            f"(N_segments={N_segments}, threshold={threshold}). "
                            f"Elapsed: {seg_elapsed:.2f}s. Moving to next N_segments.")
                        # IMPORTANT: break out of trigger-collection loop,
                        # then break out of threshold loop, then continue to next N_segments
                        num_triggers = -1  # sentinel so we know we timed out
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
                        N_segments=N_segments
                    )

                    num_events_scanned += 1

                    if triggers:
                        num_triggers += 1
                        if num_triggers % 3 == 0:
                            rate = num_triggers / num_events_scanned
                            print(f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                                  f"Events: {num_events_scanned} | Rate: {rate:.4f}")

                if num_triggers < TRIGGERS_PER_THRESHOLD:
                    print(f"Could not finish threshold={threshold} for N_segments={N_segments}. "
                      f"Moving to next N_segments.")
                    # we hit time limit mid-threshold (or broke early)
                    break

                trigger_rate = num_triggers / num_events_scanned if num_events_scanned else 0.0
                record = {
                    "N_segments": int(N_segments),
                    "threshold": float(threshold),
                    "num_triggers": int(num_triggers),
                    "num_events_scanned": int(num_events_scanned),
                    "trigger_rate": float(trigger_rate),
                }

                results.append(record)
                done.add((int(N_segments), float(threshold)))
                out["results"] = results
                save_output(out, OUT_JSON)

                print(f"Completed N_segments={N_segments}, threshold={threshold}: "
                      f"{num_triggers}/{num_events_scanned} → rate={trigger_rate:.6f}. "
                      f"Saved to '{OUT_JSON.name}'.")

                threshold += THRESHOLD_STEP

            segments_completed += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results…")
    finally:
        out["results"] = results
        save_output(out, OUT_JSON)
        elapsed = time.time() - t_start
        print(f"\nScan finished. N_segments completed: {segments_completed}/{len(N_SEGMENTS_LIST)}. "
              f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'.")

if __name__ == "__main__":
    main()
