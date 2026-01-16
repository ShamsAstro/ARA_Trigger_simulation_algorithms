import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_JSON = Path("performance_TOT_trigger_results.json")

OUT_DIR = Path("plots_TOT_efficiency_sigmoids")
OUT_DIR.mkdir(exist_ok=True)

OUT_COMPREHENSIVE = OUT_DIR / "SNR50_vs_TOTmin.png"

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


def make_sigmoid_curve(a, b, snr):
    snr = np.asarray(snr, dtype=float)
    xmin, xmax = float(np.min(snr)), float(np.max(snr))
    span = max(1e-6, xmax - xmin)
    xg = np.linspace(xmin - 0.05 * span, xmax + 0.05 * span, 400)
    yg = sigmoid(xg, a, b)
    return xg, yg


def load_results(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_minus_plus_scans(scans):
    minus_scan = None
    plus_scan = None
    for sc in scans:
        label = str(sc.get("label", ""))
        if "minus" in label:
            minus_scan = sc
        if "plus" in label:
            plus_scan = sc
    return minus_scan, plus_scan


def plot_per_tot(min_tot, minus_scan, plus_scan, fit_m, fit_p):
    snr_m = np.asarray(minus_scan["SNR_values"], dtype=float)
    eff_m = np.asarray(minus_scan["pass_fraction"], dtype=float)
    snr_p = np.asarray(plus_scan["SNR_values"], dtype=float)
    eff_p = np.asarray(plus_scan["pass_fraction"], dtype=float)

    thr_m = float(minus_scan.get("threshold_used", np.nan))
    thr_p = float(plus_scan.get("threshold_used", np.nan))

    plt.figure(figsize=(10, 6))

    plt.plot(snr_m, eff_m, "o", label=f"pred - err data (thr={thr_m:.0f})")
    plt.plot(snr_p, eff_p, "o", label=f"pred + err data (thr={thr_p:.0f})")

    if fit_m["success"]:
        xg, yg = make_sigmoid_curve(fit_m["a"], fit_m["b"], snr_m)
        plt.plot(xg, yg, "-", label=f"pred - err fit (SNR50={fit_m['b']:.3f})")
        plt.axvline(fit_m["b"], linestyle="--")
    if fit_p["success"]:
        xg, yg = make_sigmoid_curve(fit_p["a"], fit_p["b"], snr_p)
        plt.plot(xg, yg, "-", label=f"pred + err fit (SNR50={fit_p['b']:.3f})")
        plt.axvline(fit_p["b"], linestyle="--")

    plt.axhline(0.5, linestyle="--", label="50% efficiency")
    plt.title(f"TOT trigger efficiency vs SNR (TOT≥{min_tot})\nTwo thresholds: predicted ± error")
    plt.xlabel("SNR")
    plt.ylabel("Pass fraction")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file = OUT_DIR / f"efficiency_sigmoid_TOT_{int(min_tot):02d}.png"
    plt.savefig(out_file)
    plt.close()


def main():
    data = load_results(IN_JSON)
    results = data.get("results", {})
    if not isinstance(results, dict) or len(results) == 0:
        raise RuntimeError("IN_JSON missing results dict: data['results'].")

    rows = []

    tot_keys = sorted(results.keys(), key=lambda k: int(k))

    for tot_key in tot_keys:
        entry = results[tot_key]
        if entry.get("status") != "ok":
            print(f"Skipping TOT≥{tot_key}: status={entry.get('status')}")
            continue

        scans = entry.get("efficiency_scans", [])
        if not isinstance(scans, list) or len(scans) == 0:
            print(f"Skipping TOT≥{tot_key}: no efficiency_scans")
            continue

        minus_scan, plus_scan = find_minus_plus_scans(scans)
        if minus_scan is None or plus_scan is None:
            print(f"Skipping TOT≥{tot_key}: missing minus or plus scan")
            continue

        fit_m = fit_sigmoid(minus_scan.get("SNR_values", []), minus_scan.get("pass_fraction", []))
        fit_p = fit_sigmoid(plus_scan.get("SNR_values", []), plus_scan.get("pass_fraction", []))

        if not fit_m["success"] or not fit_p["success"]:
            print(f"TOT≥{tot_key}: sigmoid fit failed "
                  f"(minus={fit_m['reason']}, plus={fit_p['reason']})")
            continue

        plot_per_tot(int(tot_key), minus_scan, plus_scan, fit_m, fit_p)

        b_minus = float(fit_m["b"])
        b_plus = float(fit_p["b"])
        b_mean = 0.5 * (b_minus + b_plus)

        diff = abs(b_plus - b_minus)
        b_err = 0.5 * diff if HALF_DIFF_ERROR else diff

        rows.append((int(tot_key), b_minus, b_plus, b_mean, b_err))

        print(f"TOT≥{tot_key}: SNR50(-)={b_minus:.4f}, SNR50(+)={b_plus:.4f}, mean={b_mean:.4f}, err={b_err:.4f}")

    if len(rows) == 0:
        raise RuntimeError("No valid sigmoid fits found to make summary plot.")

    rows = sorted(rows, key=lambda t: t[0])
    tot = np.array([r[0] for r in rows], dtype=float)
    snr50_mean = np.array([r[3] for r in rows], dtype=float)
    snr50_err = np.array([r[4] for r in rows], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.errorbar(tot, snr50_mean, yerr=snr50_err, fmt="o", capsize=3)
    plt.xlabel("Minimum allowed TOT (samples)")
    plt.ylabel("SNR at 50% efficiency (SNR_50)")
    err_note = "half-difference" if HALF_DIFF_ERROR else "full difference"
    plt.title(f"TOT trigger: SNR_50 vs minimum TOT\n(mean of +/- threshold cases, error = {err_note})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_COMPREHENSIVE)
    plt.close()

    print(f"\nSaved per-TOT sigmoid plots to: {OUT_DIR}")
    print(f"Saved comprehensive plot: {OUT_COMPREHENSIVE}")


if __name__ == "__main__":
    main()
