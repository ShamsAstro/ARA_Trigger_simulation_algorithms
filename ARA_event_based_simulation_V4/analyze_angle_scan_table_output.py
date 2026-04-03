import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# User settings
# ============================================================
IN_JSON = Path("efficiency_scan_50per_results.json").resolve()
OUT_JSON = Path("snr50_all_angles_results.json").resolve()
OUT_PLOT = Path("snr50_angle_map.png").resolve()

THETA_BIN_WIDTH = 10.0   # deg
PHI_BIN_WIDTH = 10.0     # deg


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


# ============================================================
# Plotting
# ============================================================
def make_angle_map_plot(fit_rows, out_plot, theta_bin_width, phi_bin_width):
    """
    Draw rectangle map:
      theta on x-axis
      phi on y-axis
      color = SNR50
    Missing angles are simply absent.
    """
    valid_rows = [r for r in fit_rows if r["fit_success"] and r["snr_50"] is not None]

    if len(valid_rows) == 0:
        raise RuntimeError("No valid SNR_50 values available to plot.")

    theta_vals = np.array([r["theta_deg"] for r in valid_rows], dtype=float)
    phi_vals = np.array([r["phi_deg"] for r in valid_rows], dtype=float)
    snr50_vals = np.array([r["snr_50"] for r in valid_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 7))

    norm = Normalize(vmin=np.min(snr50_vals), vmax=np.max(snr50_vals))
    cmap = plt.cm.coolwarm

    for row in valid_rows:
        theta = float(row["theta_deg"])
        phi = float(row["phi_deg"])
        snr50 = float(row["snr_50"])

        rect = Rectangle(
            (theta - theta_bin_width / 2.0, phi - phi_bin_width / 2.0),
            theta_bin_width,
            phi_bin_width,
            facecolor=cmap(norm(snr50)),
            edgecolor="black",
            linewidth=0.6
        )
        ax.add_patch(rect)

    ax.set_xlabel("Theta (deg)", fontsize=14)
    ax.set_ylabel("Phi (deg)", fontsize=14)
    ax.set_title(r"SNR$_{50}$ angle map", fontsize=16)

    # sensible limits from actual populated bins
    ax.set_xlim(
        np.min(theta_vals) - theta_bin_width,
        np.max(theta_vals) + theta_bin_width
    )
    ax.set_ylim(
        np.min(phi_vals) - phi_bin_width,
        np.max(phi_vals) + phi_bin_width
    )

    # nice ticks based on actual values
    ax.set_xticks(np.unique(theta_vals))
    ax.set_yticks(np.unique(phi_vals))

    ax.grid(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"SNR$_{50}$", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    print("Opening input JSON:")
    print(IN_JSON)

    settings, results = load_scan_data(IN_JSON)
    grouped = group_results_by_angle(results)

    print("\nFound {} angle combinations.\n".format(len(grouped)))

    fit_rows = []

    for i, key in enumerate(grouped.keys(), start=1):
        angle_entry = sort_rows_by_snr(grouped[key])

        theta = angle_entry["theta_deg"]
        phi = angle_entry["phi_deg"]

        snr_values = np.array([row["snr"] for row in angle_entry["rows"]], dtype=float)
        pass_fractions = np.array([row["pass_fraction"] for row in angle_entry["rows"]], dtype=float)
        amplitudes = np.array([row["amplitude_scale"] for row in angle_entry["rows"]], dtype=float)

        print("------------------------------------------------------------")
        print("Angle {}/{}".format(i, len(grouped)))
        print("Key   : {}".format(key))
        print("Theta : {}".format(theta))
        print("Phi   : {}".format(phi))
        print("N pts : {}".format(len(snr_values)))

        fit = fit_sigmoid(snr_values, pass_fractions)

        if fit["success"]:
            print("Fit successful | SNR_50 = {:.6f}".format(fit["b"]))
            snr50_value = float(fit["b"])
        else:
            print("Fit failed | Reason: {}".format(fit["reason"]))
            snr50_value = None

        fit_rows.append({
            "key": key,
            "theta_deg": theta,
            "phi_deg": phi,
            "delays_ns": angle_entry["delays_ns"],
            "snr_values": snr_values.tolist(),
            "amplitudes": amplitudes.tolist(),
            "pass_fractions": pass_fractions.tolist(),
            "fit_success": bool(fit["success"]),
            "fit_reason": fit["reason"],
            "fit_a": fit["a"],
            "snr_50": snr50_value
        })

    print("\nSaving extracted SNR_50 data to:")
    print(OUT_JSON)

    out_data = {
        "source_json": str(IN_JSON),
        "settings_from_scan": settings,
        "theta_bin_width_deg": THETA_BIN_WIDTH,
        "phi_bin_width_deg": PHI_BIN_WIDTH,
        "results_by_angle": fit_rows
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=4)

    print("\nMaking angle map plot...")
    make_angle_map_plot(
        fit_rows=fit_rows,
        out_plot=OUT_PLOT,
        theta_bin_width=THETA_BIN_WIDTH,
        phi_bin_width=PHI_BIN_WIDTH
    )

    print("Done.")
    print("Saved plot to:")
    print(OUT_PLOT)


if __name__ == "__main__":
    main()