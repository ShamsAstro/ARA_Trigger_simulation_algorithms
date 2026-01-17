import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from sim_functions import *
from trig_functions import *

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE_GHZ         = 3.2      # GHz
TIME_STEP_NS              = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC             = 100      # ADC (noise RMS)
MAX_SIGNAL_ADC            = 4095
WINDOW_SIZE_MHZ           = 5.88e6   # MHz
N_WINDOWS                 = 1
SIM_DURATION_NS           = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9  # ns
SIM_DURATION_SAMPLES      = int(SIM_DURATION_NS / TIME_STEP_NS)

N_CHANNELS                = 8
N_REQ_COINC               = 3        # channels required for a trigger

# Per-TOT scan time limit
SCAN_TIME_LIMIT_SEC       = 3600     #3600 * 1.5

START_THRESHOLD           = 15000    # POWER units (ADC^2)
THRESHOLD_STEP            = 1500
TRIGGERS_PER_THRESHOLD    = 15

starting_MIN_ALLOWED_TOT  = 0        # samples
ending_MIN_ALLOWED_TOT    = 10       # samples (inclusive)

envelope_samples          = 4       # real experiment uses 10

# Input JSONs
pulse_json_path = Path("../ARA_event_based_simulation_V3_TOT/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
impulse_response_path = Path("../ARA_event_based_simulation_V3_TOT/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

# Output
OUT_JSON = Path("Full_threshold_scan_long_4env.json")


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_metadata():
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
        "description": "ARA envelope+TOT noise-only threshold scan. Results keyed by min_allowed_tot (samples).",
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
            "N_REQ_COINC": N_REQ_COINC,
            "SCAN_TIME_LIMIT_SEC_per_TOT": SCAN_TIME_LIMIT_SEC,
            "START_THRESHOLD": START_THRESHOLD,
            "THRESHOLD_STEP": THRESHOLD_STEP,
            "TRIGGERS_PER_THRESHOLD": TRIGGERS_PER_THRESHOLD,
            "starting_MIN_ALLOWED_TOT": starting_MIN_ALLOWED_TOT,
            "ending_MIN_ALLOWED_TOT": ending_MIN_ALLOWED_TOT,
            "envelope_samples": envelope_samples,
        },
        "inputs": {
            "pulse_json_path": str(pulse_json_path),
            "impulse_response_path": str(impulse_response_path),
            "impulse_channel_key": "ch2_2x_amp",
        },
        "trigger_functions": {
            "pretrigger": "find_ARA_env_triggers",
            "tot_measure": "TOT_finder_mod",
            "notes": "Counts a trigger only if TOT_finder_mod tot > min_allowed_tot."
        }
    }


def load_or_init(out_path: Path):
    """
    Expected format:
    {
      "meta": {...},
      "results": { "0": [ {record}, ... ], "1": [...], ... }
    }
    """
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
            if isinstance(data, dict) and "meta" in data and "results" in data:
                if not isinstance(data["results"], dict):
                    data["results"] = {}
                return data
        except Exception:
            pass

    return {"meta": build_metadata(), "results": {}}


def save_output(data: dict, out_path: Path):
    out_path.write_text(json.dumps(data, indent=2))


def completed_thresholds_for_tot(results_dict: dict, min_tot: int):
    """
    Returns a set of thresholds already recorded for this min_tot.
    """
    done = set()
    arr = results_dict.get(str(min_tot), [])
    if not isinstance(arr, list):
        return done
    for r in arr:
        try:
            done.add(int(r["threshold"]))
        except Exception:
            continue
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Scan core
# ─────────────────────────────────────────────────────────────────────────────
def run_scan_for_min_tot(min_tot: int, out_data: dict):
    """
    Runs per-TOT scan for up to SCAN_TIME_LIMIT_SEC, saving incrementally.
    """
    t_tot_start = time.time()
    threshold = START_THRESHOLD
    thresholds_completed = 0

    results_dict = out_data["results"]
    results_for_this_tot = results_dict.get(str(min_tot), [])
    if not isinstance(results_for_this_tot, list):
        results_for_this_tot = []

    done_thresholds = completed_thresholds_for_tot(results_dict, min_tot)

    try:
        while True:
            tot_elapsed = time.time() - t_tot_start
            if tot_elapsed >= SCAN_TIME_LIMIT_SEC:
                print(f"\nTime limit reached for TOT(min)={min_tot}. Moving to next TOT.")
                break

            # Skip thresholds already present (resume-friendly)
            if int(threshold) in done_thresholds:
                threshold += THRESHOLD_STEP
                continue

            num_triggers = 0
            num_events_scanned = 0
            tot_samples = []

            THRESHOLD_V = [int(threshold)] * N_CHANNELS

            print(f"\n=== MIN_TOT {min_tot} | Threshold {threshold} (power units) ===")

            while num_triggers < TRIGGERS_PER_THRESHOLD:
                tot_elapsed = time.time() - t_tot_start
                if tot_elapsed >= SCAN_TIME_LIMIT_SEC:
                    print(f"\nTime limit hit mid-threshold for MIN_TOT={min_tot}.")
                    break

                # Generate pure noise for all channels
                channel_signals = []
                for _ch in range(N_CHANNELS):
                    t_axis, noise = make_band_limited_noise_digitized(
                        json_path=impulse_response_path,
                        channel_key="ch2_2x_amp",
                        window_ns=SIM_DURATION_NS,
                        adc_rate_ghz=SAMPLING_RATE_GHZ,
                        target_rms_mV=NOISE_RMS_ADC,
                        max_signal=MAX_SIGNAL_ADC,
                    )
                    channel_signals.append(noise)

                # Envelope pre-trigger (fast)
                triggers = find_ARA_env_triggers(
                    channel_signals,
                    t_axis,
                    threshold=THRESHOLD_V,
                    n_channels_required=N_REQ_COINC,
                    envelope_window_points=envelope_samples,
                )

                num_events_scanned += 1

                if triggers:
                    tot, n_ch_trig = TOT_finder_mod(
                        channel_signals,
                        t_axis,
                        threshold=THRESHOLD_V,
                        n_channels_required=N_REQ_COINC,
                        env_parameter=envelope_samples,
                    )
                    if tot > min_tot:
                        tot_samples.append(int(tot))
                        num_triggers += 1
                        if num_triggers % 3 == 0:
                            rate = num_triggers / num_events_scanned
                            print(f"  Triggers: {num_triggers}/{TRIGGERS_PER_THRESHOLD} | "
                                  f"Events: {num_events_scanned} | Rate: {rate:.4f}")

            # Record what happened at this threshold (even if partial)
            trigger_rate = (num_triggers / num_events_scanned) if num_events_scanned else 0.0
            record = {
                "min_allowed_tot": int(min_tot),
                "threshold": int(threshold),
                "num_triggers": int(num_triggers),
                "num_events_scanned": int(num_events_scanned),
                "trigger_rate": float(trigger_rate),
                "tot_samples": [int(x) for x in tot_samples],
                "completed": bool(num_triggers >= TRIGGERS_PER_THRESHOLD),
                "elapsed_sec_in_this_min_tot": float(time.time() - t_tot_start),
            }

            results_for_this_tot.append(record)
            results_dict[str(min_tot)] = results_for_this_tot
            done_thresholds.add(int(threshold))
            save_output(out_data, OUT_JSON)

            thresholds_completed += 1

            print(f"Recorded threshold {threshold} for MIN_TOT={min_tot}: "
                  f"{num_triggers}/{num_events_scanned} → rate={trigger_rate:.6f} "
                  f"(completed={record['completed']}).")

            # If time ran out mid-threshold, stop this min_tot
            if (time.time() - t_tot_start) >= SCAN_TIME_LIMIT_SEC:
                break

            # If you want to stop the min_tot scan as soon as you fail to reach triggers,
            # keep this. If you'd rather keep stepping thresholds anyway, delete this block.
            if num_triggers < TRIGGERS_PER_THRESHOLD:
                print(f"Could not reach {TRIGGERS_PER_THRESHOLD} triggers at threshold={threshold} "
                      f"for MIN_TOT={min_tot}. Stopping this MIN_TOT scan.")
                break

            threshold += THRESHOLD_STEP

    except KeyboardInterrupt:
        print(f"\nInterrupted during MIN_TOT={min_tot}. Saving partial results…")
    finally:
        results_dict[str(min_tot)] = results_for_this_tot
        save_output(out_data, OUT_JSON)
        elapsed = time.time() - t_tot_start
        print(f"\nScan finished for MIN_TOT={min_tot}. "
              f"Thresholds recorded: {thresholds_completed}. "
              f"Elapsed: {elapsed:.2f} s. Results in '{OUT_JSON.name}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # keep pulse_data load (even if not used) to match your current workflow
    # and ensure file paths are valid early
    with open(pulse_json_path, "r") as f:
        _pulse_data = json.load(f)

    out_data = load_or_init(OUT_JSON)

    for min_tot in range(starting_MIN_ALLOWED_TOT, ending_MIN_ALLOWED_TOT + 1):
        run_scan_for_min_tot(min_tot, out_data)


if __name__ == "__main__":
    main()
