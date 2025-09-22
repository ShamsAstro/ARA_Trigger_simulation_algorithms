import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
IN_JSON_A = Path("threshold_scan_rates_large.json")
IN_JSON_B = Path("threshold_scan_rates_Eliminate_20_tot_long.json")

labelA = "PURE noise"
labelB = "PURE noise, TOT>=20 samples"

X_axis_start = 10000  # for zooming in the plot
Y_axis_end = 10**10  # for zooming in the plot

EVENT_NS = 170.0            # event duration in ns (≈ 1 / 5.88 MHz)
TARGET_HZ = 5.0             # target trigger rate

# You can use the same fit-start threshold for both, or set distinct ones:
FIT_START_THRESHOLD_A = 60000
FIT_START_THRESHOLD_B = 60000

OUT_PNG = Path("trigger_rate_hz_vs_threshold_fit_compare_20tot_long.png")
# ------------------------------------------

def load_results(path: Path):
    with open(path) as f:
        results = json.load(f)
    return results

def get_pass_fraction(entry):
    """Return pass fraction = triggers/events. Use 'trigger_rate' if provided."""
    if "trigger_rate" in entry:
        return float(entry["trigger_rate"])
    return float(entry["triggers"]) / float(entry["events"])

def to_hz(pass_fraction, event_ns):
    """Convert pass fraction (per event) to Hz: (triggers/events) / event_duration."""
    return pass_fraction / (event_ns * 1e-9)

def fit_exponential(thresholds, rates_hz, fit_start_threshold):
    """
    Fit ln(y) = ln(A) + k * x on data with x >= fit_start_threshold and y > 0.
    Returns A, k, x_fit, y_fit.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    rates_hz   = np.asarray(rates_hz, dtype=float)

    mask = (thresholds >= fit_start_threshold) & (rates_hz > 0)
    x = thresholds[mask]
    y = rates_hz[mask]
    if x.size < 2:
        return None, None, x, y  # not enough points to fit

    lny = np.log(y)
    k, lnA = np.polyfit(x, lny, 1)  # slope, intercept
    A = np.exp(lnA)
    return A, k, x, y

def intersection_threshold_for_rate(A, k, target_hz):
    """Solve A * exp(k * x) = target_hz for x."""
    if A is None or k is None or k == 0 or target_hz <= 0:
        return None
    return (np.log(target_hz) - np.log(A)) / k

def prepare_dataset(results):
    """Return sorted arrays: thresholds, rates_hz, mask_pos (rates>0)."""
    thresholds = np.array([r["threshold"] for r in results], dtype=float)
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    pass_fracs = np.array([get_pass_fraction(results[i]) for i in order], dtype=float)
    rates_hz   = to_hz(pass_fracs, EVENT_NS)
    mask_pos   = rates_hz > 0
    return thresholds, rates_hz, mask_pos

def make_fit_curve(A, k, thresholds, target_thr=None):
    """Build (x_grid, y_grid) for plotting the exponential fit."""
    if A is None:
        return None, None
    xmin = float(np.min(thresholds))
    xmax = float(np.max(thresholds))
    if target_thr is not None and np.isfinite(target_thr):
        xmax = max(xmax, float(target_thr))
    # add 10% headroom
    span = max(1.0, xmax - xmin)
    xmax = xmax + 0.1 * span
    x_grid = np.linspace(xmin, xmax, 400)
    y_grid = A * np.exp(k * x_grid)
    return x_grid, y_grid

def main():
    # Load and prep both datasets
    res_A = load_results(IN_JSON_A)
    res_B = load_results(IN_JSON_B)

    thr_A, rate_A, pos_A = prepare_dataset(res_A)
    thr_B, rate_B, pos_B = prepare_dataset(res_B)

    # Fits
    A_A, k_A, xfit_A, yfit_A = fit_exponential(thr_A, rate_A, FIT_START_THRESHOLD_A)
    A_B, k_B, xfit_B, yfit_B = fit_exponential(thr_B, rate_B, FIT_START_THRESHOLD_B)

    thr_at_target_A = intersection_threshold_for_rate(A_A, k_A, TARGET_HZ)
    thr_at_target_B = intersection_threshold_for_rate(A_B, k_B, TARGET_HZ)

    xgrid_A, ygrid_A = make_fit_curve(A_A, k_A, thr_A, thr_at_target_A)
    xgrid_B, ygrid_B = make_fit_curve(A_B, k_B, thr_B, thr_at_target_B)

    # ---------- PLOT ----------
    plt.figure(figsize=(10, 6))

    # Measured points
    plt.scatter(thr_A[pos_A], rate_A[pos_A], s=40, label=labelA)
    plt.scatter(thr_B[pos_B], rate_B[pos_B], s=40, label=labelB)

    # Fit curves
    if xgrid_A is not None:
        plt.plot(xgrid_A, ygrid_A, lw=2, label=str(labelA+"Exp fit"))
    if xgrid_B is not None:
        plt.plot(xgrid_B, ygrid_B, lw=2, label=str(labelB+"Exp fit"))

    # Target line and intersections
    plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

    if thr_at_target_A is not None and np.isfinite(thr_at_target_A):
        plt.axvline(thr_at_target_A, linestyle="--", color="tab:green", label=f"A @ {TARGET_HZ:.1f} Hz")
        plt.scatter([thr_at_target_A], [TARGET_HZ], zorder=5, color="tab:green")
        plt.text(thr_at_target_A, TARGET_HZ, f"  {thr_at_target_A:.1f}", va="bottom", ha="left")

    if thr_at_target_B is not None and np.isfinite(thr_at_target_B):
        plt.axvline(thr_at_target_B, linestyle="--", color="tab:purple", label=f"B @ {TARGET_HZ:.1f} Hz")
        plt.scatter([thr_at_target_B], [TARGET_HZ], zorder=5, color="tab:purple")
        plt.text(thr_at_target_B, TARGET_HZ, f"  {thr_at_target_B:.1f}", va="bottom", ha="right")

    plt.yscale("log")
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title("Trigger Rate vs Threshold — Two Datasets (pure noise)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper right') #, bbox_to_anchor=(1, 1)
    plt.tight_layout()
    plt.xlim(left=X_axis_start)  # zoom in
    plt.ylim(top=Y_axis_end)  # zoom in
    plt.savefig(OUT_PNG)

    # ---------- Console summary ----------
    print(f"EVENT_NS = {EVENT_NS} ns, TARGET_HZ = {TARGET_HZ} Hz")
    print(f"Dataset A: points={len(thr_A)} (positive={int(pos_A.sum())})")
    if A_A is not None:
        print(f"  Fit region A: thr >= {FIT_START_THRESHOLD_A} (n={len(xfit_A)})")
        print(f"  A_A = {A_A:.3e}, k_A = {k_A:.3e} (1/ADC²)")
        if thr_at_target_A is not None and np.isfinite(thr_at_target_A):
            print(f"  Threshold at {TARGET_HZ:.1f} Hz (A): {thr_at_target_A:.3f} ADC²")
        else:
            print("  Could not compute intersection for A.")
    else:
        print("  Not enough points to fit A.")

    print(f"Dataset B: points={len(thr_B)} (positive={int(pos_B.sum())})")
    if A_B is not None:
        print(f"  Fit region B: thr >= {FIT_START_THRESHOLD_B} (n={len(xfit_B)})")
        print(f"  A_B = {A_B:.3e}, k_B = {k_B:.3e} (1/ADC²)")
        if thr_at_target_B is not None and np.isfinite(thr_at_target_B):
            print(f"  Threshold at {TARGET_HZ:.1f} Hz (B): {thr_at_target_B:.3f} ADC²")
        else:
            print("  Could not compute intersection for B.")
    else:
        print("  Not enough points to fit B.")

if __name__ == "__main__":
    main()
