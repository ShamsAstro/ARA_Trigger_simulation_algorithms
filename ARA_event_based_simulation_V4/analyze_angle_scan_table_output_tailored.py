import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from multiprocessing import Pool, cpu_count


# ============================================================
# User settings
# ============================================================
IN_JSON = Path("efficiency_scan_BEAM_100per_results_25000events_1000beams_FULL_all_angles.json").resolve()
IN_JSON_BEAMS = Path("Trigger_beams_1000_+-60.json").resolve()

OUT_JSON = Path("snr50_BEAM_25000_parallel.json").resolve()
OUT_PLOT = Path("snr50_BEAM_100per_25000_parallel.png").resolve()

THETA_BIN_WIDTH = None
PHI_BIN_WIDTH = None

# Parallel settings
N_CORES = 20

# Tick sparsity controls
XTICK_STRIDE = 20
YTICK_STRIDE = 5

# Beam marker controls
SHOW_BEAM_LOCATIONS = True
BEAM_MARKER_SIZE = 19
BEAM_MARKER_LINEWIDTH = 2


# ============================================================
# Sigmoid tools
# ============================================================
def sigmoid(x, a, b):
    """
    Sigmoid: 1 / (1 + exp(-a(x-b)))
    50% efficiency point is b
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def fit_sigmoid(snr, eff):
    """
    Fit sigmoid to (snr, pass_fraction).
    Returns dict with success, a, b, cov, reason.
    """
    snr = np.asarray(snr, dtype=float)
    eff = np.asarray(eff, dtype=float)

    mask = np.isfinite(snr) & np.isfinite(eff)
    snr = snr[mask]
    eff = eff[mask]

    if snr.size < 4:
        return {
            "success": False,
            "a": None,
            "b": None,
            "cov": None,
            "reason": "too_few_points"
        }

    eff_clip = np.clip(eff, 1e-6, 1 - 1e-6)

    b0 = float(snr[np.argmin(np.abs(eff_clip - 0.5))])
    a0 = 2.0

    snr_min = float(np.min(snr))
    snr_max = float(np.max(snr))

    lower = (1e-6, snr_min)
    upper = (1e6, snr_max)

    try:
        popt, pcov = curve_fit(
            sigmoid,
            snr,
            eff_clip,
            p0=[a0, b0],
            bounds=(lower, upper),
            maxfev=20000
        )

        a = float(popt[0])
        b = float(popt[1])

        return {
            "success": True,
            "a": a,
            "b": b,
            "cov": pcov.tolist(),
            "reason": None
        }

    except Exception as e:
        return {
            "success": False,
            "a": None,
            "b": None,
            "cov": None,
            "reason": str(e)
        }


# ============================================================
# Data tools
# ============================================================
def load_scan_data(in_json_path):
    with open(in_json_path, "r") as f:
        data = json.load(f)

    settings = data.get("settings", {})
    results = data.get("results", [])

    if not isinstance(results, list) or len(results) == 0:
        raise RuntimeError("Input JSON does not contain a non-empty list at data['results'].")

    return settings, results


def load_beam_locations(beam_json_path):
    """
    Loads beam locations from a beam-delay JSON.

    Expected format:
    {
      "zen_30.00_az_0.00": {
        "zenith_deg": 30.0,
        "azimuth_deg": 0.0,
        "delays_ns": [...]
      },
      ...
    }

    Returns
    -------
    beam_angles : list of tuple
        [(zenith, azimuth), ...]
    """
    with open(beam_json_path, "r") as f:
        raw = json.load(f)

    beam_angles = []

    for key, val in raw.items():
        if isinstance(val, dict) and "zenith_deg" in val and "azimuth_deg" in val:
            theta = float(val["zenith_deg"])
            phi = float(val["azimuth_deg"])
            beam_angles.append((theta, phi))
        else:
            raise ValueError(f"Unrecognized beam JSON format for key: {key}")

    beam_angles = sorted(beam_angles, key=lambda x: (x[0], x[1]))
    return beam_angles


def group_results_by_angle(results):
    """
    Group rows by angle combination key.
    """
    grouped = {}

    for row in results:
        key = row["key"]

        if key not in grouped:
            grouped[key] = {
                "key": key,
                "theta_deg": float(row["theta_deg"]),
                "phi_deg": float(row["phi_deg"]),
                "delays_ns": row.get("delays_ns", []),
                "rows": []
            }

        grouped[key]["rows"].append(row)

    return grouped


def sort_rows_by_snr(angle_entry):
    angle_entry["rows"] = sorted(angle_entry["rows"], key=lambda r: float(r["snr"]))
    return angle_entry


def fit_one_angle_job(args):
    """
    Worker job for one angle bin.
    This is parallelized across CPU cores.
    """
    i, n_total, key, angle_entry = args

    angle_entry = sort_rows_by_snr(angle_entry)

    theta = angle_entry["theta_deg"]
    phi = angle_entry["phi_deg"]

    snr_values = np.array([row["snr"] for row in angle_entry["rows"]], dtype=float)
    pass_fractions = np.array([row["pass_fraction"] for row in angle_entry["rows"]], dtype=float)
    amplitudes = np.array([row["amplitude_scale"] for row in angle_entry["rows"]], dtype=float)

    fit = fit_sigmoid(snr_values, pass_fractions)

    if fit["success"]:
        snr50_value = float(fit["b"])
    else:
        snr50_value = None

    fit_row = {
        "key": key,
        "theta_deg": float(theta),
        "phi_deg": float(phi),
        "delays_ns": angle_entry["delays_ns"],
        "snr_values": snr_values.tolist(),
        "amplitudes": amplitudes.tolist(),
        "pass_fractions": pass_fractions.tolist(),
        "fit_success": bool(fit["success"]),
        "fit_reason": fit["reason"],
        "fit_a": fit["a"],
        "snr_50": snr50_value
    }

    return {
        "index": i,
        "n_total": n_total,
        "key": key,
        "theta": theta,
        "phi": phi,
        "n_points": len(snr_values),
        "fit_success": bool(fit["success"]),
        "snr_50": snr50_value,
        "fit_reason": fit["reason"],
        "fit_row": fit_row
    }


# ============================================================
# Plotting
# ============================================================
def make_angle_map_plot(
    fit_rows,
    out_plot,
    theta_bin_width,
    phi_bin_width,
    beam_angles=None
):
    """
    Draw rectangle map:
      x-axis = azimuth
      y-axis = zenith
      color = SNR50

    Missing angles are simply absent.

    Optional:
      beam_angles = [(zenith, azimuth), ...]
    """
    valid_rows = [r for r in fit_rows if r["fit_success"] and r["snr_50"] is not None]

    if len(valid_rows) == 0:
        raise RuntimeError("No valid SNR_50 values available to plot.")

    theta_vals = np.array([r["theta_deg"] for r in valid_rows], dtype=float)
    phi_vals = np.array([r["phi_deg"] for r in valid_rows], dtype=float)
    snr50_vals = np.array([r["snr_50"] for r in valid_rows], dtype=float)

    average_snr50 = float(np.mean(snr50_vals))

    fig, ax = plt.subplots(figsize=(11, 7))

    norm = Normalize(vmin=np.min(snr50_vals), vmax=np.max(snr50_vals))
    cmap = plt.cm.coolwarm

    for row in valid_rows:
        theta = float(row["theta_deg"])   # zenith
        phi = float(row["phi_deg"])       # azimuth
        snr50 = float(row["snr_50"])

        rect = Rectangle(
            (phi - phi_bin_width / 2.0, theta - theta_bin_width / 2.0),
            phi_bin_width,
            theta_bin_width,
            facecolor=cmap(norm(snr50)),
            edgecolor="black",
            linewidth=0.05
        )
        ax.add_patch(rect)

    # ------------------------------------------------------------
    # Beam locations overlay
    # ------------------------------------------------------------
    if beam_angles is not None and len(beam_angles) > 0:
        beam_phi = [ang[1] for ang in beam_angles]
        beam_theta = [ang[0] for ang in beam_angles]

        ax.scatter(
            beam_phi,
            beam_theta,
            marker="x",
            s=BEAM_MARKER_SIZE,
            linewidths=BEAM_MARKER_LINEWIDTH,
            label="Beam locations",
            color="green",
            alpha=0.95,
            zorder=5,

        )

    ax.set_xlabel("Azimuth angle (deg)", fontsize=15)
    ax.set_ylabel("Zenith angle (deg)", fontsize=15)
    ax.set_title(r"SNR$_{50}$ angle map", fontsize=18)

    ax.text(
        0.02, 0.98,
        r"Average SNR$_{50}$ = " + "{:.3f}".format(average_snr50),
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Sensible limits from actual populated bins
    ax.set_xlim(
        np.min(phi_vals) - phi_bin_width,
        np.max(phi_vals) + phi_bin_width
    )
    ax.set_ylim(
        np.min(theta_vals) - theta_bin_width,
        np.max(theta_vals) + theta_bin_width
    )

    # Sparser ticks
    unique_phi = np.unique(phi_vals)
    unique_theta = np.unique(theta_vals)

    ax.set_xticks(unique_phi[::XTICK_STRIDE])
    ax.set_yticks(unique_theta[::YTICK_STRIDE])

    # Flip y-axis so smaller zenith, e.g. 30 deg, is at the top
    ax.invert_yaxis()

    ax.grid(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"SNR$_{50}$", fontsize=13)

    if beam_angles is not None and len(beam_angles) > 0:
        ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()


def estimate_bin_widths_from_angles(fit_rows):
    """
    Estimate rectangle bin widths from the actual selected angle grid.
    Uses the smallest nonzero spacing in theta and phi.
    """
    theta_vals = np.array(
        sorted(set(float(r["theta_deg"]) for r in fit_rows)),
        dtype=float
    )

    phi_vals = np.array(
        sorted(set(float(r["phi_deg"]) for r in fit_rows)),
        dtype=float
    )

    theta_diffs = np.diff(theta_vals)
    phi_diffs = np.diff(phi_vals)

    theta_diffs = theta_diffs[theta_diffs > 0]
    phi_diffs = phi_diffs[phi_diffs > 0]

    if len(theta_diffs) > 0:
        theta_bin_width = float(np.min(theta_diffs))
    else:
        theta_bin_width = 1.0

    if len(phi_diffs) > 0:
        phi_bin_width = float(np.min(phi_diffs))
    else:
        phi_bin_width = 1.0

    return theta_bin_width, phi_bin_width


# ============================================================
# Main
# ============================================================
def main():
    print("Opening input JSON:")
    print(IN_JSON)

    settings, results = load_scan_data(IN_JSON)
    grouped = group_results_by_angle(results)

    print("\nFound {} angle combinations.\n".format(len(grouped)))

    print("Opening beam-location JSON:")
    print(IN_JSON_BEAMS)

    if SHOW_BEAM_LOCATIONS:
        beam_angles = load_beam_locations(IN_JSON_BEAMS)
        print("Loaded {} beam locations.".format(len(beam_angles)))
    else:
        beam_angles = None

    grouped_items = list(grouped.items())
    n_total_angles = len(grouped_items)

    fit_jobs = [
        (i, n_total_angles, key, angle_entry)
        for i, (key, angle_entry) in enumerate(grouped_items, start=1)
    ]

    n_workers = min(N_CORES, cpu_count())
    print("\nUsing {} CPU cores for SNR_50 fits.\n".format(n_workers))

    fit_job_results = []

    with Pool(processes=n_workers) as pool:
        for out in pool.imap_unordered(fit_one_angle_job, fit_jobs, chunksize=10):
            fit_job_results.append(out)

            if (
                len(fit_job_results) % 50 == 0
                or len(fit_job_results) == 1
                or len(fit_job_results) == n_total_angles
            ):
                if out["fit_success"]:
                    print(
                        "Completed angle {}/{} | theta={:.3f}, phi={:.3f} | SNR_50={:.6f}".format(
                            len(fit_job_results),
                            n_total_angles,
                            out["theta"],
                            out["phi"],
                            out["snr_50"]
                        )
                    )
                else:
                    print(
                        "Completed angle {}/{} | theta={:.3f}, phi={:.3f} | fit failed: {}".format(
                            len(fit_job_results),
                            n_total_angles,
                            out["theta"],
                            out["phi"],
                            out["fit_reason"]
                        )
                    )

    # Sort back to the original grouped order for stable output
    fit_job_results = sorted(fit_job_results, key=lambda r: r["index"])
    fit_rows = [r["fit_row"] for r in fit_job_results]

    print("\nSaving extracted SNR_50 data to:")
    print(OUT_JSON)

    theta_bin_width, phi_bin_width = estimate_bin_widths_from_angles(fit_rows)

    print("\nEstimated bin widths from selected angles:")
    print("Theta bin width = {:.6f} deg".format(theta_bin_width))
    print("Phi bin width   = {:.6f} deg".format(phi_bin_width))

    out_data = {
        "source_json": str(IN_JSON),
        "beam_locations_json": str(IN_JSON_BEAMS),
        "settings_from_scan": settings,
        "theta_bin_width_deg": theta_bin_width,
        "phi_bin_width_deg": phi_bin_width,
        "n_parallel_cores": n_workers,
        "results_by_angle": fit_rows
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=4)

    print("\nMaking angle map plot...")
    make_angle_map_plot(
        fit_rows=fit_rows,
        out_plot=OUT_PLOT,
        theta_bin_width=theta_bin_width,
        phi_bin_width=phi_bin_width,
        beam_angles=beam_angles
    )

    print("Done.")
    print("Saved plot to:")
    print(OUT_PLOT)


if __name__ == "__main__":
    main()