import json
import time
import random
import numpy as np
from pathlib import Path

from sim_functions import *
from trig_functions import *

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG: input summary (from your TOT analysis) + output results (efficiency scans)
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_JSON = Path("TOT_threshold_analysis_outputs_10env/summary_TOT_threshold_analysis_10env.json")
OUT_JSON = Path("performance_TOT_trigger_results_10env.json")

# Efficiency scan settings (template-style)
SAMPLING_RATE_GHZ = 3.2
TIME_STEP_NS = 1.0 / SAMPLING_RATE_GHZ
NOISE_EQUALIZE = 100
MAX_SIGNAL = 4095
WINDOW_SIZE_MHZ = 5.88e6
N_WINDOWS = 1
SIM_DURATION_NS = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9
SIM_DURATION_SAMPLES = int(SIM_DURATION_NS / TIME_STEP_NS)

N_CHANNELS = 8
N_REQ = 3
SCAN_RATE = 600
envelope_samples = 10

PULSE_AMPLITUDES = np.concatenate([
    np.arange(100, 176, 25),
    np.arange(185, 331, 10),
    np.arange(355, 450, 40)
])

# Pulse + impulse response (use your TOT-specific ones)
pulse_json_path = Path("../ARA_event_based_simulation_V3_TOT/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
impulse_response_path = Path("../ARA_event_based_simulation_V3_TOT/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

pulse_start_time_ns, pulse_end_time_ns = 450, 570

GLOBAL_SEED = 24680  # change if you want deterministic different scans


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def load_and_prepare_pulse():
    pulse_data = load_json(pulse_json_path)
    pulse_voltage = np.asarray(pulse_data["avg_wave"], dtype=float)
    pulse_time = np.asarray(pulse_data["t_axis_ns"], dtype=float)

    mask = (pulse_time >= pulse_start_time_ns) & (pulse_time <= pulse_end_time_ns)
    pulse_voltage = pulse_voltage[mask]
    pulse_time = pulse_time[mask]

    pulse_voltage = pulse_voltage / np.max(pulse_voltage)
    pulse_time = pulse_time - pulse_time[0]
    return pulse_voltage, pulse_time


def run_efficiency_scan_for_tot(*, min_tot: int, thr_value: float, rng: np.random.Generator,
                                pulse_voltage, pulse_time):
    """
    For a fixed (min_tot, threshold), scan over pulse amplitudes and estimate pass fraction.
    Returns: SNR_values, pass_fraction, coincidences, events_scanned, tot_values, tot_snr_values
    """
    pass_fraction = []
    SNR_values = []
    coincidences = []
    events_scanned = []

    TOT_values = []
    tot_SNR_values = []

    thr_vec = [float(thr_value)] * N_CHANNELS

    for run_amp in PULSE_AMPLITUDES:
        coinc = 0
        snr = float(run_amp) / float(NOISE_EQUALIZE)

        for _ in range(SCAN_RATE):
            start_seed = float(rng.uniform(0.0, TIME_STEP_NS))

            channel_signals = []
            for _ch in range(N_CHANNELS):
                t, sig = make_full_signal(
                    impulse_json_path=impulse_response_path,
                    SIMULATION_DURATION_NS=SIM_DURATION_NS,
                    SAMPLING_RATE=SAMPLING_RATE_GHZ,
                    NOISE_EQUALIZE=NOISE_EQUALIZE,
                    pulse_voltage=pulse_voltage,
                    pulse_time=pulse_time,
                    time_step=TIME_STEP_NS,
                    simulation_duration_samples=SIM_DURATION_SAMPLES,
                    amplitude_scale=float(run_amp),
                    max_signal=MAX_SIGNAL,
                    start_time=start_seed
                )
                channel_signals.append(sig)

            time_axis = t  # time_start not needed for trigger logic

            triggers = find_ARA_env_triggers(
                channel_signals,
                time_axis,
                threshold=thr_vec,
                n_channels_required=N_REQ,
                envelope_window_points=envelope_samples
            )

            if triggers:
                tot_val, n_triggered_channels = TOT_finder_mod(
                    channel_signals,
                    time_axis,
                    threshold=thr_vec,
                    n_channels_required=N_REQ,
                    env_parameter=envelope_samples
                )
                if tot_val > min_tot:
                    coinc += 1
                    TOT_values.append(float(tot_val))
                    tot_SNR_values.append(float(snr))

        pass_fraction.append(float(coinc / SCAN_RATE))
        SNR_values.append(float(snr))
        coincidences.append(int(coinc))
        events_scanned.append(int(SCAN_RATE))

    return {
        "SNR_values": SNR_values,
        "pass_fraction": pass_fraction,
        "coincidences": coincidences,
        "events_scanned": events_scanned,
        "TOT_values": TOT_values,
        "tot_SNR_values": tot_SNR_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    summary = load_json(SUMMARY_JSON)
    results = summary.get("results", {})
    if not isinstance(results, dict) or len(results) == 0:
        raise RuntimeError("Summary JSON missing or empty: summary['results'].")

    pulse_voltage, pulse_time = load_and_prepare_pulse()

    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "summary_json": str(SUMMARY_JSON),
            "pulse_json_path": str(pulse_json_path),
            "impulse_response_path": str(impulse_response_path),
        },
        "efficiency_scan_settings": {
            "SAMPLING_RATE_GHZ": SAMPLING_RATE_GHZ,
            "TIME_STEP_NS": TIME_STEP_NS,
            "NOISE_EQUALIZE": NOISE_EQUALIZE,
            "MAX_SIGNAL": MAX_SIGNAL,
            "WINDOW_SIZE_MHZ": WINDOW_SIZE_MHZ,
            "N_WINDOWS": N_WINDOWS,
            "SIM_DURATION_NS": SIM_DURATION_NS,
            "SIM_DURATION_SAMPLES": SIM_DURATION_SAMPLES,
            "N_CHANNELS": N_CHANNELS,
            "N_REQ": N_REQ,
            "SCAN_RATE": SCAN_RATE,
            "envelope_samples": envelope_samples,
            "PULSE_AMPLITUDES": [float(x) for x in PULSE_AMPLITUDES],
            "pulse_crop_ns": [float(pulse_start_time_ns), float(pulse_end_time_ns)],
            "seed": GLOBAL_SEED,
        },
        "results": {},
    }

    t0 = time.time()

    # For each TOT key in the summary: do two scans at thr_pred ± thr_err
    for tot_key in sorted(results.keys(), key=lambda k: int(k)):
        r = results[tot_key]
        if r.get("status") != "ok":
            out["results"][str(tot_key)] = {
                "status": "skipped",
                "reason": f"summary status != ok (status={r.get('status')})",
                "summary_entry": r,
                "efficiency_scans": [],
            }
            continue

        thr_pred = r.get("threshold_at_target_hz", None)
        thr_err = r.get("threshold_at_target_hz_err", None)

        if thr_pred is None or not np.isfinite(thr_pred):
            out["results"][str(tot_key)] = {
                "status": "skipped",
                "reason": "threshold_at_target_hz missing or non-finite",
                "summary_entry": r,
                "efficiency_scans": [],
            }
            continue

        thr_pred = float(thr_pred)
        thr_err_val = float(thr_err) if (thr_err is not None and np.isfinite(thr_err)) else None

        # Two thresholds to test: predicted - err and predicted + err
        # If err missing, run the same threshold twice (still two scans as requested)
        if thr_err_val is None or thr_err_val <= 0:
            thresholds_to_test = [thr_pred, thr_pred]
            labels = ["pred_minus_err_unavailable", "pred_plus_err_unavailable"]
        else:
            thresholds_to_test = [thr_pred - thr_err_val, thr_pred + thr_err_val]
            labels = ["pred_minus_err", "pred_plus_err"]

        tot_entry = {
            "status": "ok",
            "summary_entry_used": r,
            "threshold_prediction": {
                "threshold_at_target_hz": thr_pred,
                "threshold_at_target_hz_err": thr_err_val,
            },
            "efficiency_scans": [],
        }

        min_tot = int(tot_key)

        for label, thr_value in zip(labels, thresholds_to_test):
            # deterministic per (TOT, +/-)
            scan_seed = (GLOBAL_SEED + 100000 * min_tot + (1 if "plus" in label else 0)) % (2**32 - 1)
            rng = np.random.default_rng(scan_seed)

            t_scan0 = time.time()
            scan_res = run_efficiency_scan_for_tot(
                min_tot=min_tot,
                thr_value=float(thr_value),
                rng=rng,
                pulse_voltage=pulse_voltage,
                pulse_time=pulse_time
            )
            t_scan1 = time.time()

            tot_entry["efficiency_scans"].append({
                "label": label,
                "min_tot": min_tot,
                "threshold_used": float(thr_value),
                "scan_seed": int(scan_seed),
                "runtime_sec": float(t_scan1 - t_scan0),
                **scan_res
            })

            print(f"TOT≥{min_tot} [{label}]: thr={float(thr_value):.2f} done")

        out["results"][str(tot_key)] = tot_entry

    out["total_runtime_sec"] = float(time.time() - t0)

    save_json(out, OUT_JSON)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
