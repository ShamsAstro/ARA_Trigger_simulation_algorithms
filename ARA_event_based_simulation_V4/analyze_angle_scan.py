import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import curve_fit


# ============================================================
# User settings
# ============================================================
IN_JSON = Path("efficiency_scan_50per_results.json").resolve()
OUT_JSON = Path("snr50_fit_results_first_two_angles.json").resolve()
OUT_DIR = Path("sigmoid_test_plots_first_two_angles").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TEST_ANGLES = 5   


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


def make_sigmoid_curve(a, b, snr):
    snr = np.asarray(snr, dtype=float)
    xmin = float(np.min(snr))
    xmax = float(np.max(snr))
    span = max(1e-6, xmax - xmin)

    xg = np.linspace(xmin - 0.05 * span, xmax + 0.05 * span, 400)
    yg = sigmoid(xg, a, b)
    return xg, yg


# ============================================================
# Data loading and grouping
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
    Group scan results by angle combination key.

    Returns a dict like:
    {
        "theta_0_phi_70": {
            "theta_deg": ...,
            "phi_deg": ...,
            "delays_ns": ...,
            "rows": [...]
        },
        ...
    }
    """
    grouped = {}

    for row in results:
        key = row["key"]

        if key not in grouped:
            grouped[key] = {
                "key": key,
                "theta_deg": row["theta_deg"],
                "phi_deg": row["phi_deg"],
                "delays_ns": row["delays_ns"],
                "rows": []
            }

        grouped[key]["rows"].append(row)

    return grouped


def sort_angle_rows_by_snr(angle_entry):
    angle_entry["rows"] = sorted(angle_entry["rows"], key=lambda r: float(r["snr"]))
    return angle_entry


# ============================================================
# Plotting
# ============================================================
def save_angle_plot(angle_entry, fit_result, out_dir):
    snr = np.array([row["snr"] for row in angle_entry["rows"]], dtype=float)
    eff = np.array([row["pass_fraction"] for row in angle_entry["rows"]], dtype=float)
    amp = np.array([row["amplitude_scale"] for row in angle_entry["rows"]], dtype=float)

    theta = angle_entry["theta_deg"]
    phi = angle_entry["phi_deg"]
    key = angle_entry["key"]

    plt.figure(figsize=(10, 6))
    plt.plot(snr, eff, "o", label="Efficiency points", alpha=0.75)

    if fit_result["success"]:
        xg, yg = make_sigmoid_curve(fit_result["a"], fit_result["b"], snr)
        plt.plot(xg, yg, "-", label=f"Sigmoid fit (SNR$_{{50}}$ = {fit_result['b']:.3f})")
        plt.axvline(fit_result["b"], linestyle="--")
    else:
        plt.plot([], [], " ", label=f"Fit failed: {fit_result['reason']}")

    plt.axhline(0.5, linestyle="--", label="50% efficiency")

    plt.title(
        f"CSW trigger efficiency vs SNR\n"
        f"{key} | theta={theta}, phi={phi}"
    )
    plt.xlabel("SNR", fontsize=14)
    plt.ylabel("Pass fraction", fontsize=14)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_file = out_dir / f"{key}_sigmoid_fit.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    print("Opening input JSON:")
    print(IN_JSON)

    settings, results = load_scan_data(IN_JSON)
    grouped = group_results_by_angle(results)

    angle_keys = list(grouped.keys())
    angle_keys = angle_keys[:N_TEST_ANGLES]

    if len(angle_keys) == 0:
        raise RuntimeError("No angle combinations found in the JSON results.")

    print("\nFound {} total angle combinations.".format(len(grouped)))
    print("Testing first {} angle combinations.\n".format(len(angle_keys)))

    out_data = {
        "source_json": str(IN_JSON),
        "settings_from_scan": settings,
        "n_test_angles": len(angle_keys),
        "fit_results": []
    }

    for i, key in enumerate(angle_keys, start=1):
        angle_entry = grouped[key]
        angle_entry = sort_angle_rows_by_snr(angle_entry)

        theta = angle_entry["theta_deg"]
        phi = angle_entry["phi_deg"]

        snr_values = np.array([row["snr"] for row in angle_entry["rows"]], dtype=float)
        pass_fractions = np.array([row["pass_fraction"] for row in angle_entry["rows"]], dtype=float)
        amplitudes = np.array([row["amplitude_scale"] for row in angle_entry["rows"]], dtype=float)

        print("============================================================")
        print("Angle test {}/{}".format(i, len(angle_keys)))
        print("Key   : {}".format(key))
        print("Theta : {}".format(theta))
        print("Phi   : {}".format(phi))
        print("N pts : {}".format(len(snr_values)))
        print("SNRs  : {}".format(np.round(snr_values, 4)))
        print("Eff   : {}".format(np.round(pass_fractions, 4)))
        print("============================================================")

        fit_result = fit_sigmoid(snr_values, pass_fractions)

        if fit_result["success"]:
            print("Fit successful.")
            print("Slope a      = {:.6f}".format(fit_result["a"]))
            print("SNR_50 (b)   = {:.6f}".format(fit_result["b"]))
        else:
            print("Fit failed.")
            print("Reason: {}".format(fit_result["reason"]))

        save_angle_plot(angle_entry, fit_result, OUT_DIR)

        out_data["fit_results"].append({
            "key": key,
            "theta_deg": theta,
            "phi_deg": phi,
            "delays_ns": angle_entry["delays_ns"],
            "snr_values": snr_values.tolist(),
            "amplitudes": amplitudes.tolist(),
            "pass_fractions": pass_fractions.tolist(),
            "fit": fit_result,
            "snr_50": fit_result["b"] if fit_result["success"] else None
        })

        print("Saved test plot for {}".format(key))
        print()

    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=4)

    print("Done.")
    print("Saved fit JSON to:")
    print(OUT_JSON)
    print("Saved plots to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()