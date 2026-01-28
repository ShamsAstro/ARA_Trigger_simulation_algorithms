import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_JSON_ENV4  = Path("performance_TOT_trigger_results_4env.json")
IN_JSON_ENV10 = Path("performance_TOT_trigger_results_10env.json")

OUT_DIR = Path("plots_TOT_SNR50_compare_env4_env10")
OUT_DIR.mkdir(exist_ok=True)

OUT_COMBINED = OUT_DIR / "SNR50_vs_TOTmin_env4_vs_env10.png"

# Error bar definition for summary:
# - If True: error = half difference between (+) and (-) SNR50
# - If False: error = full difference
HALF_DIFF_ERROR = True
# ─────────────────────────────────────────────────────────────────────────────


def sigmoid(x, a, b):
    """Sigmoid: 1/(1+exp(-a(x-b))). 50% point is b."""
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def fit_sigmoid(snr, eff):
    snr = np.asarray(snr, dtype=float)
    eff = np.asarray(eff, dtype=float)

    m = np.isfinite(snr) & np.isfinite(eff)
    snr, eff = snr[m], eff[m]

    if snr.size < 4:
        return {"success": False, "a": None, "b": None, "cov": None, "reason": "too_few_points"}

    eff_clip = np.clip(eff, 1e-6, 1 - 1e-6)

    # initial guesses
    b0 = float(snr[np.argmin(np.abs(eff_clip - 0.5))])
    a0 = 2.0

    snr_min, snr_max = float(np.min(snr)), float(np.max(snr))
    lower = (1e-6, snr_min)
    upper = (1e6, snr_max)

    try:
        popt, pcov = curve_fit(
            sigmoid, snr, eff_clip,
            p0=[a0, b0],
            bounds=(lower, upper),
            maxfev=20000
        )
        return {"success": True, "a": float(popt[0]), "b": float(popt[1]), "cov": pcov, "reason": None}
    except Exception as e:
        return {"success": False, "a": None, "b": None, "cov": None, "reason": str(e)}


def load_results(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_minus_plus_scans(scans):
    minus_scan = None
    plus_scan = None
    for sc in scans:
        label = str(sc.get("label", "")).lower()
        if "minus" in label:
            minus_scan = sc
        if "plus" in label:
            plus_scan = sc
    return minus_scan, plus_scan


def summarize_snr50_vs_tot(json_path: Path, *, half_diff_error: bool = True):
    """
    Returns:
      tot (np.ndarray),
      snr50_mean (np.ndarray),
      snr50_err (np.ndarray),
      rows (list of tuples for debug)
    """
    data = load_results(json_path)
    results = data.get("results", {})
    if not isinstance(results, dict) or len(results) == 0:
        raise RuntimeError(f"{json_path}: missing results dict at data['results'].")

    rows = []
    tot_keys = sorted(results.keys(), key=lambda k: int(k))

    for tot_key in tot_keys:
        entry = results[tot_key]
        if entry.get("status") != "ok":
            continue

        scans = entry.get("efficiency_scans", [])
        if not isinstance(scans, list) or len(scans) == 0:
            continue

        minus_scan, plus_scan = find_minus_plus_scans(scans)
        if minus_scan is None or plus_scan is None:
            continue

        fit_m = fit_sigmoid(minus_scan.get("SNR_values", []), minus_scan.get("pass_fraction", []))
        fit_p = fit_sigmoid(plus_scan.get("SNR_values", []), plus_scan.get("pass_fraction", []))

        if not fit_m["success"] or not fit_p["success"]:
            continue

        b_minus = float(fit_m["b"])
        b_plus = float(fit_p["b"])
        b_mean = 0.5 * (b_minus + b_plus)

        diff = abs(b_plus - b_minus)
        b_err = 0.5 * diff if half_diff_error else diff

        rows.append((int(tot_key), b_minus, b_plus, b_mean, b_err))

    if len(rows) == 0:
        raise RuntimeError(f"{json_path}: no valid sigmoid fits found for summary plot.")

    rows = sorted(rows, key=lambda t: t[0])
    tot = np.array([r[0] for r in rows], dtype=float)
    snr50_mean = np.array([r[3] for r in rows], dtype=float)
    snr50_err = np.array([r[4] for r in rows], dtype=float)

    return tot, snr50_mean, snr50_err, rows


def main():
    tot4,  snr4,  err4,  rows4  = summarize_snr50_vs_tot(IN_JSON_ENV4,  half_diff_error=HALF_DIFF_ERROR)
    tot10, snr10, err10, rows10 = summarize_snr50_vs_tot(IN_JSON_ENV10, half_diff_error=HALF_DIFF_ERROR)

    # Plot both on one figure (no per-TOT sigmoid plots)
    plt.figure(figsize=(10, 6))

    plt.errorbar(tot4, snr4, yerr=err4, fmt="o", capsize=3, label="Envelope samples = 4", color="C0")
    plt.plot(tot4, snr4, "-", alpha=0.7, color="C0")

    plt.errorbar(tot10, snr10, yerr=err10, fmt="o", capsize=3, label="Envelope samples = 10", color="C1")
    plt.plot(tot10, snr10, "-", alpha=0.7, color="C1")
    plt.xlabel("Minimum allowed TOT (samples)", fontsize=14)
    plt.ylabel("SNR$_{50}$", fontsize=14)
    err_note = "half-difference" if HALF_DIFF_ERROR else "full difference"
    plt.title(f"TOT trigger: SNR_50 vs minimum TOT\n(mean of +/- threshold cases, error = {err_note})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_COMBINED, dpi=300)
    plt.close()

    print(f"Saved combined plot: {OUT_COMBINED}")
    print(f"env_4 points:  {len(rows4)} | env_10 points: {len(rows10)}")


if __name__ == "__main__":
    main()
