import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_JSON = Path("performance_CSW_trigger_results.json")

OUT_DIR = Path("plots_CSW_SNR50_vs_Nsegments")
OUT_DIR.mkdir(exist_ok=True)

OUT_COMPREHENSIVE = OUT_DIR / "SNR50_vs_Nsegments_CSW.png"

# If you want the error bar to be the FULL difference, set HALF_DIFF_ERROR=False
HALF_DIFF_ERROR = True
# ─────────────────────────────────────────────────────────────────────────────


def sigmoid(x, a, b):
    """Sigmoid: 1/(1+exp(-a(x-b))). 50% point is b."""
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def fit_sigmoid(snr, eff):
    """
    Fit sigmoid to (snr, pass_fraction).
    Returns dict with success, a, b, cov, reason.
    """
    snr = np.asarray(snr, dtype=float)
    eff = np.asarray(eff, dtype=float)

    m = np.isfinite(snr) & np.isfinite(eff)
    snr, eff = snr[m], eff[m]

    if snr.size < 4:
        return {"success": False, "a": None, "b": None, "cov": None, "reason": "too_few_points"}

    # clip slightly for numerical stability
    eff_clip = np.clip(eff, 1e-6, 1 - 1e-6)

    # initial guess: b is where eff close to 0.5, a moderate slope
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
        a, b = float(popt[0]), float(popt[1])
        return {"success": True, "a": a, "b": b, "cov": pcov, "reason": None}
    except Exception as e:
        return {"success": False, "a": None, "b": None, "cov": None, "reason": str(e)}


def make_sigmoid_curve(a, b, snr):
    snr = np.asarray(snr, dtype=float)
    xmin, xmax = float(np.min(snr)), float(np.max(snr))
    span = max(1e-6, xmax - xmin)
    xg = np.linspace(xmin - 0.05 * span, xmax + 0.05 * span, 400)
    yg = sigmoid(xg, a, b)
    return xg, yg


def save_per_nseg_plot(nseg, minus_scan, plus_scan, fit_m, fit_p):
    """
    Plot pass_fraction vs SNR for both minus and plus cases with sigmoid fits.
    """
    snr_m = np.asarray(minus_scan["SNR_values"], dtype=float)
    eff_m = np.asarray(minus_scan["pass_fraction"], dtype=float)
    snr_p = np.asarray(plus_scan["SNR_values"], dtype=float)
    eff_p = np.asarray(plus_scan["pass_fraction"], dtype=float)

    plt.figure(figsize=(10, 6))

    # data
    plt.plot(snr_m, eff_m, "o", label="pred - err (data)")
    plt.plot(snr_p, eff_p, "o", label="pred + err (data)")

    # fits
    if fit_m["success"]:
        xg, yg = make_sigmoid_curve(fit_m["a"], fit_m["b"], snr_m)
        plt.plot(xg, yg, "-", label=f"pred - err fit (SNR50={fit_m['b']:.3f})")
        plt.axvline(fit_m["b"], linestyle="--")
    if fit_p["success"]:
        xg, yg = make_sigmoid_curve(fit_p["a"], fit_p["b"], snr_p)
        plt.plot(xg, yg, "-", label=f"pred + err fit (SNR50={fit_p['b']:.3f})")
        plt.axvline(fit_p["b"], linestyle="--")

    plt.axhline(0.5, linestyle="--", label="50% efficiency")

    plt.title(f"CSW efficiency vs SNR (N_segments={nseg})")
    plt.xlabel("SNR")
    plt.ylabel("Pass fraction")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file = OUT_DIR / f"efficiency_sigmoid_Nsegments_{int(nseg):02d}.png"
    plt.savefig(out_file)
    plt.close()


def main():
    data = json.loads(IN_JSON.read_text())
    results = data.get("results", {})
    if not isinstance(results, dict) or len(results) == 0:
        raise RuntimeError("IN_JSON does not contain a dict at data['results'].")

    rows = []

    # Sort by N_segments numeric
    nseg_keys = sorted(results.keys(), key=lambda k: int(k))

    for k in nseg_keys:
        entry = results[k]
        if entry.get("status") != "ok":
            print(f"Skipping N_segments={k}: status={entry.get('status')}")
            continue

        scans = entry.get("efficiency_scans", [])
        if not isinstance(scans, list) or len(scans) == 0:
            print(f"Skipping N_segments={k}: no efficiency_scans")
            continue

        minus_scan = None
        plus_scan = None
        for sc in scans:
            label = str(sc.get("label", ""))
            if "minus" in label:
                minus_scan = sc
            if "plus" in label:
                plus_scan = sc

        if minus_scan is None or plus_scan is None:
            print(f"Skipping N_segments={k}: could not find both minus and plus scans")
            continue

        # Fit both
        fit_m = fit_sigmoid(minus_scan.get("SNR_values", []), minus_scan.get("pass_fraction", []))
        fit_p = fit_sigmoid(plus_scan.get("SNR_values", []), plus_scan.get("pass_fraction", []))

        if not fit_m["success"] or not fit_p["success"]:
            print(f"N_segments={k}: sigmoid fit failed "
                  f"(minus={fit_m['reason']}, plus={fit_p['reason']})")
            continue

        # Save per-nseg plot with both cases + both fits
        save_per_nseg_plot(int(k), minus_scan, plus_scan, fit_m, fit_p)

        # Compute mean + error from +/- difference
        b_minus = float(fit_m["b"])
        b_plus = float(fit_p["b"])
        b_mean = 0.5 * (b_minus + b_plus)
        diff = abs(b_plus - b_minus)
        b_err = 0.5 * diff if HALF_DIFF_ERROR else diff

        rows.append((int(k), b_minus, b_plus, b_mean, b_err))

        print(
            f"N_segments={k}: "
            f"SNR50(-)={b_minus:.4f}, SNR50(+)={b_plus:.4f}, "
            f"mean={b_mean:.4f}, err={b_err:.4f}"
        )

    if len(rows) == 0:
        raise RuntimeError("No valid (minus, plus) sigmoid fits found to plot.")

    # Comprehensive plot: SNR50 vs N_segments
    rows = sorted(rows, key=lambda t: t[0])
    nseg = np.array([r[0] for r in rows], dtype=float)
    snr50_mean = np.array([r[3] for r in rows], dtype=float)
    snr50_err = np.array([r[4] for r in rows], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.errorbar(nseg, snr50_mean, yerr=snr50_err, fmt="o", capsize=3)
    plt.xlabel("N_segments")
    plt.xticks(nseg.astype(int))
    plt.ylabel("SNR at 50% efficiency (SNR_50)")
    err_note = "half-difference" if HALF_DIFF_ERROR else "full difference"
    plt.title(f"CSW: SNR_50 vs N_segments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_COMPREHENSIVE)
    plt.close()

    print(f"\nSaved per-N_segments sigmoid plots to: {OUT_DIR}")
    print(f"Saved comprehensive plot: {OUT_COMPREHENSIVE}")


if __name__ == "__main__":
    main()
