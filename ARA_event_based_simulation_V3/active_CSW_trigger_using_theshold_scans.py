import json
import numpy as np
import time
import random
from pathlib import Path

from sim_functions import *
from trig_functions_cop import *  # contains ARA_CSW_trigger_FFT_optimized


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT:
# This must point to the *analysis summary* JSON where:
#   data["results"] is a dict keyed by N_segments ("1","2",...)
# and each entry contains:
#   "threshold_at_target_hz" and "threshold_at_target_hz_err"
#
# Example snippet you showed:
# "results": { "1": { ... "threshold_at_target_hz": ..., "threshold_at_target_hz_err": ... }, ... }
THRESH_SCAN_SUMMARY_JSON = Path("summary_nsegments_analysis.json")

# Output JSON (no plots)
OUT_JSON = Path("performance_CSW_trigger_results.json")

# Efficiency-scan settings (adapted from your script)
SAMPLING_RATE_GHZ = 3.2
TIME_STEP_NS = 1.0 / SAMPLING_RATE_GHZ
NOISE_RMS_ADC = 100
MAX_SIGNAL_ADC = 4095

WINDOW_SIZE_MHZ = 5.88e6
N_WINDOWS = 1
SIM_DURATION_NS = N_WINDOWS / WINDOW_SIZE_MHZ * 1e9
SIM_DURATION_SAMPLES = int(SIM_DURATION_NS / TIME_STEP_NS)

N_CHANNELS = 8
SCAN_RATE = 800  # number of events per amplitude

# Must match what you used when converting pass_fraction -> Hz in the analysis summary
EVENT_NS = 170.0
TARGET_HZ = 5.0

# Amplitude scan (same as your script)
PULSE_AMPLITUDES = np.concatenate([
    np.arange(60, 200, 15),
    np.arange(200, 400, 10),
    np.arange(400, 550, 25)
])

# Input pulse / impulse response
pulse_json_path = Path("../ARA_event_based_simulation_V3/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
impulse_response_path = Path("../ARA_event_based_simulation_V3/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

# Pulse cropping (same as your script)
pulse_start_time_ns, pulse_end_time_ns = 450, 570  # ns

# Randomness control
GLOBAL_SEED = 12345  # change if you want
# ─────────────────────────────────────────────────────────────────────────────


def load_summary_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    if not (isinstance(data, dict) and isinstance(data.get("results"), dict)):
        raise RuntimeError(
            f"{path} is not in the expected summary format.\n"
            "Expected a JSON dict with a 'results' dict keyed by N_segments.\n"
            "Example: data['results'] = {'1': {...}, '2': {...}, ...}"
        )
    return data


def load_and_prepare_pulse():
    with open(pulse_json_path, "r") as f:
        pulse_data = json.load(f)

    pulse_voltage = np.array(pulse_data["avg_wave"], dtype=float)
    pulse_time = np.array(pulse_data["t_axis_ns"], dtype=float)

    mask = (pulse_time >= pulse_start_time_ns) & (pulse_time <= pulse_end_time_ns)
    pulse_voltage = pulse_voltage[mask]
    pulse_time = pulse_time[mask]

    # normalize and start from 0 ns
    pulse_voltage = pulse_voltage / np.max(pulse_voltage)
    pulse_time = pulse_time - pulse_time[0]
    return pulse_voltage, pulse_time


def run_efficiency_scan_for_threshold(*, csw_threshold, N_segments, pulse_voltage, pulse_time, rng):
    """
    Returns:
      SNR_values, pass_fraction, coincidences, events_scanned
    """
    pass_fraction = []
    SNR_values = []
    coincidences = []
    events_scanned = []

    for run_amp in PULSE_AMPLITUDES:
        coinc = 0

        # SNR definition consistent with your script
        snr = float(run_amp) / float(NOISE_RMS_ADC)

        for _ in range(SCAN_RATE):
            start_seed = rng.uniform(0.0, TIME_STEP_NS)

            channel_signals = []
            for _ch in range(N_CHANNELS):
                t, sig = make_full_signal(
                    impulse_json_path=impulse_response_path,
                    SIMULATION_DURATION_NS=SIM_DURATION_NS,
                    SAMPLING_RATE=SAMPLING_RATE_GHZ,
                    NOISE_EQUALIZE=NOISE_RMS_ADC,
                    pulse_voltage=pulse_voltage,
                    pulse_time=pulse_time,
                    time_step=TIME_STEP_NS,
                    simulation_duration_samples=SIM_DURATION_SAMPLES,
                    amplitude_scale=float(run_amp),
                    max_signal=MAX_SIGNAL_ADC,
                    start_time=float(start_seed),
                )
                channel_signals.append(sig)

            time_axis = t  # no need to offset for trigger logic

            triggers = ARA_CSW_trigger_FFT_optimized(
                channel_signals,
                time_axis,
                threshold=float(csw_threshold),
                noise_rms=float(NOISE_RMS_ADC),
                N_segments=int(N_segments),
            )

            if triggers:
                coinc += 1

        pf = coinc / float(SCAN_RATE)
        pass_fraction.append(float(pf))
        SNR_values.append(float(snr))
        coincidences.append(int(coinc))
        events_scanned.append(int(SCAN_RATE))

    return SNR_values, pass_fraction, coincidences, events_scanned


def main():
    # load summary results keyed by N_segments
    summary = load_summary_json(THRESH_SCAN_SUMMARY_JSON)
    results_by_nseg = summary["results"]

    nseg_list = sorted(int(k) for k in results_by_nseg.keys())
    if not nseg_list:
        raise RuntimeError("No N_segments keys found in summary['results'].")

    pulse_voltage, pulse_time = load_and_prepare_pulse()

    out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "threshold_scan_summary_json": str(THRESH_SCAN_SUMMARY_JSON),
            "pulse_json_path": str(pulse_json_path),
            "impulse_response_path": str(impulse_response_path),
        },
        "analysis_context_from_summary": {
            "event_ns_for_rate": float(summary.get("event_ns", summary.get("event_ns_for_rate", EVENT_NS))) if isinstance(summary, dict) else EVENT_NS,
            "target_hz": float(summary.get("target_hz", TARGET_HZ)) if isinstance(summary, dict) else TARGET_HZ,
        },
        "efficiency_scan_settings": {
            "sampling_rate_ghz": SAMPLING_RATE_GHZ,
            "time_step_ns": TIME_STEP_NS,
            "noise_rms_adc": NOISE_RMS_ADC,
            "max_signal_adc": MAX_SIGNAL_ADC,
            "sim_duration_ns": SIM_DURATION_NS,
            "sim_duration_samples": SIM_DURATION_SAMPLES,
            "n_channels": N_CHANNELS,
            "scan_rate_events_per_amplitude": SCAN_RATE,
            "pulse_start_time_ns": pulse_start_time_ns,
            "pulse_end_time_ns": pulse_end_time_ns,
            "pulse_amplitudes": [float(x) for x in PULSE_AMPLITUDES],
            "seed": GLOBAL_SEED,
        },
        "results": {},
    }

    t0 = time.time()

    for nseg in nseg_list:
        s = results_by_nseg.get(str(nseg), {})
        status = s.get("status", None)

        if status != "ok":
            out["results"][str(nseg)] = {
                "status": "skipped",
                "reason": f"summary status != ok (status={status})",
                "summary_entry": s,
                "efficiency_scans": [],
            }
            continue

        thr_pred = s.get("threshold_at_target_hz", None)
        thr_err = s.get("threshold_at_target_hz_err", None)

        if thr_pred is None or not np.isfinite(thr_pred):
            out["results"][str(nseg)] = {
                "status": "skipped",
                "reason": "threshold_at_target_hz missing or non-finite",
                "summary_entry": s,
                "efficiency_scans": [],
            }
            continue

        # two scans: (pred - err) and (pred + err)
        if thr_err is None or (not np.isfinite(thr_err)) or thr_err <= 0:
            thresholds_to_test = [float(thr_pred), float(thr_pred)]
            labels = ["pred_minus_err_unavailable", "pred_plus_err_unavailable"]
        else:
            thresholds_to_test = [float(thr_pred - thr_err), float(thr_pred + thr_err)]
            labels = ["pred_minus_err", "pred_plus_err"]

        nseg_entry = {
            "status": "ok",
            "summary_entry_used": s,
            "threshold_prediction": {
                "target_hz": float(TARGET_HZ),
                "threshold_at_target": float(thr_pred),
                "threshold_at_target_err": float(thr_err) if thr_err is not None else None,
            },
            "efficiency_scans": [],
        }

        for label, csw_thr in zip(labels, thresholds_to_test):
            # deterministic per (N_segments, +/-) scan seed
            scan_seed = (GLOBAL_SEED + 100000 * int(nseg) + (1 if "plus" in label else 0)) % (2**32 - 1)
            scan_rng = np.random.default_rng(scan_seed)

            t_scan0 = time.time()
            snr_vals, pass_fracs, coincs, n_events = run_efficiency_scan_for_threshold(
                csw_threshold=csw_thr,
                N_segments=nseg,
                pulse_voltage=pulse_voltage,
                pulse_time=pulse_time,
                rng=scan_rng,
            )
            t_scan1 = time.time()

            nseg_entry["efficiency_scans"].append({
                "label": label,
                "csw_threshold_used": float(csw_thr),
                "scan_seed": int(scan_seed),
                "runtime_sec": float(t_scan1 - t_scan0),
                "SNR_values": snr_vals,
                "pass_fraction": pass_fracs,
                "coincidences": coincs,
                "events_scanned": n_events,
            })

        out["results"][str(nseg)] = nseg_entry
        print(f"Done N_segments={nseg}: thr@{TARGET_HZ}Hz={float(thr_pred):.4f} ± {float(thr_err) if thr_err is not None else float('nan'):.4f}")

    out["total_runtime_sec"] = float(time.time() - t0)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
