import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

from sim_functions import *
from trig_functions_cop import *


# ============================================================
# Random seed
# ============================================================

RANDOM_SEED = 5561

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# Main simulation settings
# ============================================================

SAMPLING_RATE = 3.2                      # GHz
TIME_STEP = 1.0 / SAMPLING_RATE          # ns/sample

NOISE_RMS = 100                          # mV
NOISE_EQUALIZE = NOISE_RMS               # name used by make_full_signal_with_delay

SNR_RANGE = np.array([0.5, 1, 2, 2.5], dtype=float)
PULSE_AMPLITUDES = NOISE_RMS * SNR_RANGE

MAX_SIGNAL = 4095

WINDOW_SIZE = 5.88 * 1e6
n_of_windows = 2
SIMULATION_DURATION_NS = n_of_windows / WINDOW_SIZE * 1e9
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)

N_OF_CHANNELS = 8
SCAN_RATE = 40                           # tests per angle combination
N_SEGMENTS = 1                           # CSW power segmentation
N_CORES = 20                             # use up to 20 CPU cores

PLOT_FIRST_EVENT = False

PROGRESS_EVENT_INTERVAL = 1000           # print progress after this many events


# ============================================================
# Input JSON paths
# ============================================================

PULSE_JSON_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json"
).resolve()

IMPULSE_RESPONSE_PATH = Path(
    "../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json"
).resolve()


# ============================================================
# Load pulse waveform
# ============================================================

with open(PULSE_JSON_PATH) as f:
    pulse_data = json.load(f)

pulse_voltage = np.array(pulse_data["avg_wave"], dtype=float)
pulse_time = np.array(pulse_data["t_axis_ns"], dtype=float)

pulse_start_time, pulse_end_time = 450, 570

pulse_mask = (pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)

pulse_voltage = pulse_voltage[pulse_mask]
pulse_time = pulse_time[pulse_mask]

pulse_voltage = pulse_voltage / np.max(np.abs(pulse_voltage))
pulse_time = pulse_time - pulse_time[0]


# ============================================================
# Delay settings helpers
# ============================================================

def load_delay_settings_json(json_path):
    """
    Loads angle settings from a JSON file.

    Expected fields per entry:
        zenith_deg
        azimuth_deg
        delays_ns
    """
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


# ============================================================
# Plotting
# ============================================================

def plot_csw_power_map(results_by_angle, title, output_png_path=None):
    """
    Makes a theta-phi map of normalized average CSW max power.

    Each result in results_by_angle should contain:
        theta
        phi
        normalized_mean_power
    """
    theta_vals = sorted(set(r["theta"] for r in results_by_angle))
    phi_vals = sorted(set(r["phi"] for r in results_by_angle))

    theta_to_i = {theta: i for i, theta in enumerate(theta_vals)}
    phi_to_j = {phi: j for j, phi in enumerate(phi_vals)}

    power_grid = np.full((len(theta_vals), len(phi_vals)), np.nan)

    for r in results_by_angle:
        i = theta_to_i[r["theta"]]
        j = phi_to_j[r["phi"]]
        power_grid[i, j] = r["normalized_mean_power"]

    avg_normalized_power = float(np.nanmean(power_grid))

    PHI, THETA = np.meshgrid(phi_vals, theta_vals)

    plt.figure(figsize=(10, 7))

    pcm = plt.pcolormesh(PHI, THETA, power_grid, shading="auto")
    plt.colorbar(pcm, label="Normalized average CSW max power")

    ax = plt.gca()

    ax.text(
        0.02,
        0.98,
        f"Average normalized power = {avg_normalized_power:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.xlabel("Azimuth phi [deg]")
    plt.ylabel("Zenith theta [deg]")
    plt.title(title)
    plt.tight_layout()

    if output_png_path is not None:
        plt.savefig(output_png_path, dpi=300)

    plt.close()


# ============================================================
# One angle CSW job
# ============================================================

def run_one_angle_csw_job(args):
    """
    Runs SCAN_RATE noisy CSW tests for one angle and one SNR/amplitude.

    For each test:
        1. Generate delayed pulse + band-limited noise in all channels.
        2. Run ARA_CSW_trigger_FFT_return_max_power.
        3. Accumulate power statistics.

    Returns:
        mean CSW power, std, stderr, and number of completed events.

    To save memory and time, this does not store every event power.
    """
    (
        amp_index,
        snr_value,
        run_pulse_amplitude,
        angle_index,
        angle_setting,
        scan_rate,
        noise_rms,
        n_segments,
        plot_first_event
    ) = args

    theta = float(angle_setting["theta"])
    phi = float(angle_setting["phi"])
    delay_list = np.array(angle_setting["delays_ns"], dtype=float)

    # Different deterministic seed for each angle/SNR job.
    # This keeps the scan reproducible while preventing every worker
    # from generating identical noise streams.
    job_seed = (
        RANDOM_SEED
        + 1000003 * int(amp_index)
        + 9176 * int(angle_index)
        + int(1000 * abs(theta))
        + int(1000 * abs(phi))
    ) % (2**32 - 1)

    np.random.seed(job_seed)
    random.seed(job_seed)

    rng = np.random.default_rng(job_seed)

    # Store only enough to compute mean and std.
    # This avoids returning huge event-power lists.
    power_sum = 0.0
    power_sum_sq = 0.0

    for event_idx in range(scan_rate):

        # Random sub-sample start time within one sampling step.
        start_seed = rng.uniform(0, TIME_STEP)

        channel_signals = []

        for ch in range(N_OF_CHANNELS):

            t_axis, full_signal = make_full_signal_with_delay(
                impulse_json_path=IMPULSE_RESPONSE_PATH,
                SIMULATION_DURATION_NS=SIMULATION_DURATION_NS,
                SAMPLING_RATE=SAMPLING_RATE,
                NOISE_EQUALIZE=noise_rms,
                pulse_voltage=pulse_voltage,
                pulse_time=pulse_time,
                time_step=TIME_STEP,
                simulation_duration_samples=SIMULATION_DURATION_SAMPLES,
                amplitude_scale=run_pulse_amplitude,
                max_signal=MAX_SIGNAL,
                start_time=start_seed,
                pulse_delay_ns=delay_list[ch]
            )

            full_signal = np.asarray(full_signal, dtype=float)
            channel_signals.append(full_signal)

        channel_signals = np.array(channel_signals, dtype=float)

        if plot_first_event and event_idx == 0 and amp_index == 1 and angle_index == 1:
            plot_channels_signals(
                t_axis,
                channel_signals,
                title=(
                    f"First noisy CSW test event\n"
                    f"theta={theta}, phi={phi}, "
                    f"SNR={snr_value}, amplitude={run_pulse_amplitude}"
                )
            )

        peak_power = ARA_CSW_trigger_FFT_return_max_power(
            channel_signals=channel_signals,
            time_axis=t_axis,
            noise_rms=noise_rms,
            N_segments=n_segments
        )

        peak_power = float(peak_power)

        power_sum += peak_power
        power_sum_sq += peak_power**2

    mean_power = power_sum / scan_rate

    variance = (power_sum_sq / scan_rate) - mean_power**2
    variance = max(variance, 0.0)

    std_power = float(np.sqrt(variance))
    stderr_power = float(std_power / np.sqrt(scan_rate))

    return {
        "snr": float(snr_value),
        "amplitude": float(run_pulse_amplitude),
        "theta": theta,
        "phi": phi,
        "scan_rate": int(scan_rate),
        "noise_rms": float(noise_rms),
        "N_segments": int(n_segments),
        "mean_power": float(mean_power),
        "std_power": std_power,
        "stderr_power": stderr_power,
        "events_completed": int(scan_rate)
    }


# ============================================================
# Full CSW map scan
# ============================================================

def run_csw_power_map_scan(
    event_delay_json_path,
    snr_range,
    noise_rms,
    scan_rate,
    n_segments,
    output_plot_prefix="CSW_noisy_power_map"
):
    """
    Full scan.

    For each SNR:
        amplitude = noise_rms * SNR

        For each angle combination:
            Run SCAN_RATE noisy CSW tests.
            Average the returned CSW max powers.

        Normalize all angle powers by the largest mean power
        in that SNR map.

        Save only:
            1. Normalized PNG power map per SNR.

        Does not save JSON results.
    """
    selected_angle_settings = load_delay_settings_json(event_delay_json_path)

    snr_range = np.asarray(snr_range, dtype=float)
    pulse_amplitudes = noise_rms * snr_range

    n_workers = min(N_CORES, cpu_count())

    n_angles = len(selected_angle_settings)
    n_snr = len(snr_range)

    total_global_jobs = n_angles * n_snr
    total_events = total_global_jobs * scan_rate

    completed_global_jobs = 0
    completed_events = 0
    next_event_report = PROGRESS_EVENT_INTERVAL

    print("============================================================")
    print("Starting noisy CSW power map scan")
    print("============================================================")
    print(f"Using random seed: {RANDOM_SEED}")
    print(f"Using {n_workers} parallel CPU cores")
    print(f"Number of angle combinations: {n_angles}")
    print(f"Number of SNR values: {n_snr}")
    print(f"Global angle jobs: {total_global_jobs}")
    print(f"SCAN_RATE per angle job: {scan_rate}")
    print(f"Total events: {total_events}")
    print(f"Noise RMS: {noise_rms}")
    print(f"SNR range: {snr_range}")
    print(f"Pulse amplitudes: {pulse_amplitudes}")
    print(f"N_segments: {n_segments}")
    print("No JSON results will be saved.")
    print("============================================================")

    total_start = time.time()

    all_results_light = []

    with Pool(processes=n_workers) as pool:

        for amp_index, (snr_value, run_pulse_amplitude) in enumerate(
            zip(snr_range, pulse_amplitudes),
            start=1
        ):

            snr_start = time.time()

            job_args = []

            for angle_index, angle_setting in enumerate(selected_angle_settings, start=1):
                job_args.append((
                    amp_index,
                    float(snr_value),
                    float(run_pulse_amplitude),
                    angle_index,
                    angle_setting,
                    int(scan_rate),
                    float(noise_rms),
                    int(n_segments),
                    bool(PLOT_FIRST_EVENT)
                ))

            snr_results = []

            for result in pool.imap_unordered(run_one_angle_csw_job, job_args):

                snr_results.append(result)

                completed_global_jobs += 1
                completed_events += result["events_completed"]

                if completed_events >= next_event_report or completed_events == total_events:

                    percent_done = 100.0 * completed_events / total_events
                    elapsed = time.time() - total_start

                    print(
                        f"Progress: {percent_done:.2f}% | "
                        f"events {completed_events}/{total_events} | "
                        f"global jobs {completed_global_jobs}/{total_global_jobs} | "
                        f"current SNR={snr_value} | "
                        f"elapsed={elapsed:.1f} s"
                    )

                    while next_event_report <= completed_events:
                        next_event_report += PROGRESS_EVENT_INTERVAL

            snr_results.sort(key=lambda r: (r["theta"], r["phi"]))

            max_mean_power = max(r["mean_power"] for r in snr_results)

            for r in snr_results:
                if max_mean_power > 0:
                    r["normalized_mean_power"] = float(r["mean_power"] / max_mean_power)
                else:
                    r["normalized_mean_power"] = 0.0

            average_normalized_power = float(
                np.mean([r["normalized_mean_power"] for r in snr_results])
            )

            best_result = max(snr_results, key=lambda r: r["mean_power"])

            safe_snr = str(snr_value).replace(".", "p")

            plot_path = f"{output_plot_prefix}_SNR_{safe_snr}.png"

            plot_csw_power_map(
                results_by_angle=snr_results,
                title=(
                    f"CSW recovered power map, "
                    f"SNR={snr_value}, noise RMS={noise_rms}"
                ),
                output_png_path=plot_path
            )

            snr_elapsed = time.time() - snr_start

            print(
                f"SNR done: {snr_value} | "
                f"amplitude={run_pulse_amplitude} | "
                f"best theta={best_result['theta']:.2f}, "
                f"best phi={best_result['phi']:.2f} | "
                f"max mean power={max_mean_power:.6e} | "
                f"avg normalized power={average_normalized_power:.6f} | "
                f"time={snr_elapsed:.1f} s | "
                f"saved plot={plot_path}"
            )

            # Keep only lightweight SNR-level summaries in memory.
            # No JSON writing.
            all_results_light.append({
                "snr": float(snr_value),
                "amplitude": float(run_pulse_amplitude),
                "max_mean_power": float(max_mean_power),
                "average_normalized_power": average_normalized_power,
                "best_angle": {
                    "theta": float(best_result["theta"]),
                    "phi": float(best_result["phi"]),
                    "mean_power": float(best_result["mean_power"]),
                    "std_power": float(best_result["std_power"]),
                    "stderr_power": float(best_result["stderr_power"])
                },
                "plot_path": plot_path
            })

    total_elapsed = time.time() - total_start

    print("\n============================================================")
    print("Full noisy CSW power map scan complete")
    print(f"Total completed events: {completed_events}/{total_events}")
    print(f"Total completed global jobs: {completed_global_jobs}/{total_global_jobs}")
    print(f"Total time: {total_elapsed:.2f} s")
    print("No JSON results were saved.")
    print("============================================================")

    return all_results_light


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    EVENT_DELAY_JSON = "Event_sources_for_CSW_IDEAL_9600.json"

    results = run_csw_power_map_scan(
        event_delay_json_path=EVENT_DELAY_JSON,
        snr_range=SNR_RANGE,
        noise_rms=NOISE_RMS,
        scan_rate=SCAN_RATE,
        n_segments=N_SEGMENTS,
        output_plot_prefix="CSW_noisy_power_map_IDEAL_seed5561"
    )