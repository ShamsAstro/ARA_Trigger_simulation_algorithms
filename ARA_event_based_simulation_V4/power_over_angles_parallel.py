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
from trig_functions_cop import *
from scipy.signal import fftconvolve
from multiprocessing import Pool, cpu_count


SAMPLING_RATE       = 3.2
TIME_STEP           = 1.0 / SAMPLING_RATE
NOISE_EQUALIZE      = 100
MAX_SIGNAL          = 4095
WINDOW_SIZE         = 5.88*1e6
n_of_windows        = 2
SIMULATION_DURATION_NS = n_of_windows/(WINDOW_SIZE) * 1e9
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)
N_of_channels       = 8
N_REQ               = 1
COINC_NS            = SIMULATION_DURATION_NS
SCAN_RATE           = 1

CSW_THRESHOLD = 15.71

PULSE_AMPLITUDES = np.array([100])

N_CORES = 20

pulse_json_path = Path("../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)

impulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]


def load_delay_settings_json(json_path):
    with open(json_path, "r") as f:
        raw = json.load(f)

    settings = []
    for _, entry in raw.items():
        settings.append({
            "theta": float(entry["zenith_deg"]),
            "phi": float(entry["azimuth_deg"]),
            "delays_ns": list(entry["delays_ns"])
        })

    settings.sort(key=lambda d: (d["theta"], d["phi"]))
    return settings


def build_beam_inputs_from_settings(settings):
    beam_angles = []
    beam_delays = {}

    for s in settings:
        angle_key = (float(s["theta"]), float(s["phi"]))
        beam_angles.append(angle_key)
        beam_delays[angle_key] = np.array(s["delays_ns"], dtype=float)

    return beam_angles, beam_delays


def plot_recovered_power_map(results_by_angle, beam_angles, title="Normalized recovered beam power"):
    theta_vals = sorted(set(r["theta"] for r in results_by_angle))
    phi_vals   = sorted(set(r["phi"] for r in results_by_angle))

    theta_to_i = {th: i for i, th in enumerate(theta_vals)}
    phi_to_j   = {ph: j for j, ph in enumerate(phi_vals)}

    power_grid = np.full((len(theta_vals), len(phi_vals)), np.nan)

    for r in results_by_angle:
        i = theta_to_i[r["theta"]]
        j = phi_to_j[r["phi"]]
        power_grid[i, j] = r["normalized_power"]

    avg_normalized_power = float(np.nanmean(power_grid))

    PHI, THETA = np.meshgrid(phi_vals, theta_vals)

    plt.figure(figsize=(10, 7))
    pcm = plt.pcolormesh(PHI, THETA, power_grid, shading="auto")
    plt.colorbar(pcm, label="Normalized recovered max beam power")

    beam_phi = [ang[1] for ang in beam_angles]
    beam_theta = [ang[0] for ang in beam_angles]

    plt.scatter(
        beam_phi,
        beam_theta,
        marker="x",
        s=70,
        linewidths=1.8,
        label="Beam locations",
        color="red"
    )

    ax = plt.gca()

    ax.text(
        0.02, 0.98,
        "Average normalized power = {:.4f}".format(avg_normalized_power),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Zenith [deg]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"recovered_power_map_{title.replace(' ', '_')}.png", dpi=300)
    plt.show()


def save_results_json(results_by_angle, output_json_path):
    with open(output_json_path, "w") as f:
        json.dump(results_by_angle, f, indent=2)
    print(f"Saved results to {output_json_path}")


def run_one_angle_job(args):
    (
        amp_index,
        angle_index,
        run_pulse_amplitude,
        angle_setting,
        beam_angles,
        beam_delays,
        PLOT_FIRST_EVENT
    ) = args

    theta = angle_setting["theta"]
    phi = angle_setting["phi"]
    delay_list = np.array(angle_setting["delays_ns"], dtype=float)

    event_max_powers = []
    event_all_power_lists = []
    event_best_beam_angles = []

    for event_idx in range(SCAN_RATE):
        start_seed = random.uniform(0, TIME_STEP)

        channel_signals = [[] for _ in range(N_of_channels)]

        for ch in range(N_of_channels):
            t_axis, channel_signals[ch] = make_full_signal_with_delay_no_noise(
                SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                pulse_voltage=pulse_voltage,
                pulse_time=pulse_time,
                time_step=TIME_STEP,
                simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                amplitude_scale=run_pulse_amplitude,
                max_signal=MAX_SIGNAL,
                start_time=start_seed,
                pulse_delay_ns=delay_list[ch]
            )

        channel_signals = np.array(channel_signals, dtype=float)

        if PLOT_FIRST_EVENT and event_idx == 0 and amp_index == 1 and angle_index == 4:
            plot_channels_signals(
                t_axis,
                channel_signals,
                title=f"First event signals, theta={theta}, phi={phi}, amplitude={run_pulse_amplitude}"
            )

        max_power, power_list = ARA_beam_power_finder(
            channel_signals=channel_signals,
            time_axis=t_axis,
            sampling_rate=SAMPLING_RATE,
            beam_angles=beam_angles,
            beam_delays=beam_delays
        )

        best_beam_index = int(np.argmax(power_list))
        best_beam_angle = beam_angles[best_beam_index]

        event_max_powers.append(float(max_power))
        event_all_power_lists.append([float(x) for x in power_list])
        event_best_beam_angles.append({
            "theta": float(best_beam_angle[0]),
            "phi": float(best_beam_angle[1])
        })

    mean_max_power = float(np.mean(event_max_powers))
    std_max_power = float(np.std(event_max_powers))
    mean_power_list = np.mean(np.array(event_all_power_lists, dtype=float), axis=0)

    return {
        "amplitude": float(run_pulse_amplitude),
        "theta": float(theta),
        "phi": float(phi),
        "mean_max_power": mean_max_power,
        "std_max_power": std_max_power,
        "mean_power_list": [float(x) for x in mean_power_list],
        "beam_angles": [{"theta": float(a[0]), "phi": float(a[1])} for a in beam_angles],
        "best_beam_per_event": event_best_beam_angles
    }


def run_beam_recovery_study(
    event_delay_json_path,
    beam_delay_json_path,
    PULSE_AMPLITUDES,
    SCAN_RATE,
    N_OF_CHANNELS,
    TIME_STEP,
    SIMULATION_DURATION_NS,
    SAMPLING_RATE,
    pulse_voltage,
    pulse_time,
    SIMULATION_DURATION_SAMPLES,
    MAX_SIGNAL,
    PLOT_FIRST_EVENT=False,
    output_results_json_path=None
):
    selected_angle_settings = load_delay_settings_json(event_delay_json_path)

    beam_settings = load_delay_settings_json(beam_delay_json_path)
    beam_angles, beam_delays = build_beam_inputs_from_settings(beam_settings)

    n_total_jobs = len(selected_angle_settings) * len(PULSE_AMPLITUDES)
    job_counter = 0

    all_results = []

    n_workers = min(N_CORES, cpu_count())
    print(f"Using {n_workers} parallel cores")

    with Pool(processes=n_workers) as pool:

        for amp_index, run_pulse_amplitude in enumerate(PULSE_AMPLITUDES, start=1):
            print("\n############################################################")
            print(f"Amplitude {amp_index}/{len(PULSE_AMPLITUDES)} : {run_pulse_amplitude}")
            print("############################################################")

            amplitude_start = time.time()

            job_args = []
            for angle_index, angle_setting in enumerate(selected_angle_settings, start=1):
                job_counter += 1
                job_args.append((
                    amp_index,
                    angle_index,
                    float(run_pulse_amplitude),
                    angle_setting,
                    beam_angles,
                    beam_delays,
                    PLOT_FIRST_EVENT
                ))

            amplitude_results = []

            for result in pool.imap_unordered(run_one_angle_job, job_args):
                amplitude_results.append(result)

                print(f"Completed angle job {len(amplitude_results)}/{len(job_args)}")
                print(f"Mean recovered max power = {result['mean_max_power']:.6e}")

            amplitude_results.sort(key=lambda r: (r["theta"], r["phi"]))

            amplitude_peak = max(r["mean_max_power"] for r in amplitude_results)

            for r in amplitude_results:
                r["normalized_power"] = float(r["mean_max_power"] / amplitude_peak) if amplitude_peak > 0 else 0.0

            average_normalized_power = float(np.mean([r["normalized_power"] for r in amplitude_results]))

            all_results.append({
                "amplitude": float(run_pulse_amplitude),
                "average_normalized_power": average_normalized_power,
                "results": amplitude_results
            })

            amplitude_elapsed = time.time() - amplitude_start
            print(f"Amplitude finished in {amplitude_elapsed:.2f} s")
            print(f"Average normalized power = {average_normalized_power:.6f}")

            plot_recovered_power_map(
                results_by_angle=amplitude_results,
                beam_angles=beam_angles,
                title=f"Normalized recovered beam power {len(beam_angles)} beams all_angles_2"
            )

    if output_results_json_path is not None:
        with open(output_results_json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved full study results to {output_results_json_path}")

    return all_results


if __name__ == "__main__":

    EVENT_DELAY_JSON = "Event_sources_origins_45000.json"
    BEAM_DELAY_JSON = "Trigger_beams_1000_+-60.json"

    results = run_beam_recovery_study(
        event_delay_json_path=EVENT_DELAY_JSON,
        beam_delay_json_path=BEAM_DELAY_JSON,
        PULSE_AMPLITUDES=PULSE_AMPLITUDES,
        SCAN_RATE=SCAN_RATE,
        N_OF_CHANNELS=N_of_channels,
        TIME_STEP=TIME_STEP,
        SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
        SAMPLING_RATE=SAMPLING_RATE,
        pulse_voltage=pulse_voltage,
        pulse_time=pulse_time,
        SIMULATION_DURATION_SAMPLES=SIMULATION_DURATION_SAMPLES,
        MAX_SIGNAL=MAX_SIGNAL,
        PLOT_FIRST_EVENT=True,
        output_results_json_path="Events_most_extensive_parallel_1000beams_45000.json"
    )