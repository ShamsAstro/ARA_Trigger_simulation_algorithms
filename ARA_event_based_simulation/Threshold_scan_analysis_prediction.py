import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
IN_JSON = Path("threshold_scan_rates_large.json")

EVENT_NS = 170.0           # event duration in ns (≈ 1 / 5.88 MHz)
TARGET_HZ = 5.0            # target trigger rate
FIT_START_THRESHOLD = 48000 # choose where the exponential behavior starts

OUT_PNG = Path("trigger_rate_hz_vs_threshold_fit_large.png")

# ------------------------------------------

def load_results(path: Path):
    with open(path) as f:
        results = json.load(f)
    return results

def get_pass_fraction(entry):
    """Return pass fraction = triggers/events. Use 'trigger_rate' if provided."""
    if "trigger_rate" in entry:
        return float(entry["trigger_rate"])
    # Fallback if only raw counts are present
    return float(entry["triggers"]) / float(entry["events"])

def to_hz(pass_fraction, event_ns):
    """Convert pass fraction (per event) to Hz: (triggers/events) / event_duration."""
    return pass_fraction / (event_ns * 1e-9)

def fit_exponential(thresholds, rates_hz, fit_start_threshold):
    """
    Fit ln(y) = ln(A) + k * x on data with x >= fit_start_threshold and y>0.
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

def main():
    results = load_results(IN_JSON)

    # Extract arrays
    thresholds = np.array([r["threshold"] for r in results], dtype=float)
    # Ensure sorted by threshold
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    pass_fractions = np.array([get_pass_fraction(results[i]) for i in order], dtype=float)
    rates_hz = to_hz(pass_fractions, EVENT_NS)

    # Fit exponential on selected region
    A, k, x_fit, y_fit = fit_exponential(thresholds, rates_hz, FIT_START_THRESHOLD)
    thr_at_target = intersection_threshold_for_rate(A, k, TARGET_HZ)

    # Prepare fit curve (extend beyond target so the plot shows the crossing)
    if A is not None:
        # Choose x-range to include data and the target crossing with some margin
        xmin = thresholds.min()
        xmax_candidates = [thresholds.max()]
        if thr_at_target is not None and np.isfinite(thr_at_target):
            xmax_candidates.append(thr_at_target)
        xmax = max(xmax_candidates) + 0.1 * (max(xmax_candidates) - xmin if max(xmax_candidates) > xmin else 1.0)

        x_grid = np.linspace(xmin, xmax, 400)
        y_grid = A * np.exp(k * x_grid)
    else:
        x_grid, y_grid = None, None

    # ---------- PLOT ----------
    plt.figure(figsize=(9, 6))
    # scatter of measured rates (only plot positive values on log axis)
    mask_pos = rates_hz > 0
    plt.scatter(thresholds[mask_pos], rates_hz[mask_pos], s=40, label="Measured (Hz)")

    # fit curve
    if x_grid is not None:
        plt.plot(x_grid, y_grid, lw=2, label="Exponential fit")

    # 4 Hz target line and intersection marker
    plt.axhline(TARGET_HZ, color="tab:red", linestyle="--", label=f"Target = {TARGET_HZ:.1f} Hz")
    if thr_at_target is not None and np.isfinite(thr_at_target):
        plt.axvline(thr_at_target, color="tab:green", linestyle="--", label=f"Threshold @ {TARGET_HZ:.1f} Hz")
        plt.scatter([thr_at_target], [TARGET_HZ], color="tab:green", zorder=5)
        plt.text(thr_at_target, TARGET_HZ, f"  {thr_at_target:.1f}", va="bottom", ha="left")

    plt.yscale("log")  # log only on y
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title("Trigger Rate vs Threshold (pure noise)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG)

    # ---------- Console summary ----------
    print(f"Data points: {len(thresholds)} (positive rates: {mask_pos.sum()})")
    print(f"Event length: {EVENT_NS} ns")
    if A is not None:
        print(f"Fit region: thresholds >= {FIT_START_THRESHOLD} (points used: {len(x_fit)})")
        print(f"Fit model: rate ≈ A * exp(k * threshold)")
        print(f"A = {A:.3e},  k = {k:.3e}  (1/ADC²)")
        if thr_at_target is not None and np.isfinite(thr_at_target):
            print(f"Estimated threshold for {TARGET_HZ:.1f} Hz: {thr_at_target:.3f} ADC²")
        else:
            print("Could not compute intersection with target (check fit region and data).")
    else:
        print("Not enough points to fit. Increase data or lower FIT_START_THRESHOLD.")

if __name__ == "__main__":
    main()
