import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
IN_JSON_A = Path("threshold_scan_rates_large.json")
IN_JSON_B = Path("Full_threshold_scan_for_TOT_range.json")

labelA = "PURE noise"
EVENT_NS = 170.0            # ns
TARGET_HZ = 5.0             # Hz target rate

FIT_START_THRESHOLD_A = 60000
FIT_START_THRESHOLD_B = 50000

X_axis_start = 10000
Y_axis_end = 1e10

OUT_DIR = Path("plots_tot_comparisons")
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------
def load_results(path: Path):
    with open(path) as f:
        return json.load(f)

def get_pass_fraction(entry):
    if "trigger_rate" in entry:
        return float(entry["trigger_rate"])
    return float(entry["triggers"]) / float(entry["events"])

def to_hz(pass_fraction, event_ns):
    return pass_fraction / (event_ns * 1e-9)

def fit_exponential(thresholds, rates_hz, fit_start_threshold):
    thresholds = np.asarray(thresholds, dtype=float)
    rates_hz   = np.asarray(rates_hz, dtype=float)

    mask = (thresholds >= fit_start_threshold) & (rates_hz > 0)
    x = thresholds[mask]
    y = rates_hz[mask]
    if x.size < 2:
        return None, None, x, y
    lny = np.log(y)
    k, lnA = np.polyfit(x, lny, 1)
    A = np.exp(lnA)
    return A, k, x, y

def intersection_threshold_for_rate(A, k, target_hz):
    if A is None or k is None or k == 0 or target_hz <= 0:
        return None
    return (np.log(target_hz) - np.log(A)) / k

def prepare_dataset(results):
    thresholds = np.array([r["threshold"] for r in results], dtype=float)
    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    pass_fracs = np.array([get_pass_fraction(results[i]) for i in order], dtype=float)
    rates_hz   = to_hz(pass_fracs, EVENT_NS)
    mask_pos   = rates_hz > 0
    return thresholds, rates_hz, mask_pos

def make_fit_curve(A, k, thresholds, target_thr=None):
    if A is None:
        return None, None
    xmin = float(np.min(thresholds))
    xmax = float(np.max(thresholds))
    if target_thr is not None and np.isfinite(target_thr):
        xmax = max(xmax, float(target_thr))
    span = max(1.0, xmax - xmin)
    xmax = xmax + 0.1 * span
    x_grid = np.linspace(xmin, xmax, 400)
    y_grid = A * np.exp(k * x_grid)
    return x_grid, y_grid

# ---------- MAIN ----------
def main():
    res_A = load_results(IN_JSON_A)
    res_B_all = load_results(IN_JSON_B)  # dict of {TOT: [records]}

    thr_A, rate_A, pos_A = prepare_dataset(res_A)
    A_A, k_A, xfit_A, yfit_A = fit_exponential(thr_A, rate_A, FIT_START_THRESHOLD_A)
    xgrid_A, ygrid_A = make_fit_curve(A_A, k_A, thr_A)

    # Comparison plots for each TOT
    for tot_key, res_B in res_B_all.items():
        thr_B, rate_B, pos_B = prepare_dataset(res_B)
        A_B, k_B, xfit_B, yfit_B = fit_exponential(thr_B, rate_B, FIT_START_THRESHOLD_B)
        xgrid_B, ygrid_B = make_fit_curve(A_B, k_B, thr_B)

        thr_at_target_B = intersection_threshold_for_rate(A_B, k_B, TARGET_HZ)
        thr_at_target_A = intersection_threshold_for_rate(A_A, k_A, TARGET_HZ)

        plt.figure(figsize=(10, 6))
        plt.scatter(thr_A[pos_A], rate_A[pos_A], s=40, label=labelA)
        plt.scatter(thr_B[pos_B], rate_B[pos_B], s=40, label=f"TOT≥{tot_key}")

        if xgrid_A is not None:
            plt.plot(xgrid_A, ygrid_A, lw=2, label=labelA + " fit")
        if xgrid_B is not None:
            plt.plot(xgrid_B, ygrid_B, lw=2, label=f"TOT≥{tot_key} fit")

        plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

        if thr_at_target_A is not None:
            plt.axvline(thr_at_target_A, linestyle="--", color="tab:green")
            plt.scatter([thr_at_target_A], [TARGET_HZ], color="tab:green")
            plt.text(thr_at_target_A, TARGET_HZ*1.2, f"{thr_at_target_A:.0f}", color="tab:green", va="bottom", ha="left")
        if thr_at_target_B is not None:
            plt.axvline(thr_at_target_B, linestyle="--", color="tab:purple")
            plt.scatter([thr_at_target_B], [TARGET_HZ], color="tab:purple")
            #plot the intercept of the threshold with 5HZ
            plt.text(thr_at_target_B, TARGET_HZ*1.2, f"{thr_at_target_B:.0f}", color="tab:purple", va="bottom", ha="right")

        plt.yscale("log")
        plt.xlabel("Threshold (ADC²)")
        plt.ylabel("Trigger rate (Hz, log scale)")
        plt.title(f"Trigger Rate vs Threshold — PURE noise vs TOT≥{tot_key}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.xlim(left=X_axis_start)
        plt.ylim(top=Y_axis_end)
        out_file = OUT_DIR / f"compare_A_vs_TOT{tot_key}.png"
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved {out_file}")

    # Combined plot
    plt.figure(figsize=(12, 7))
    plt.scatter(thr_A[pos_A], rate_A[pos_A], s=40, label=labelA)
    if xgrid_A is not None:
        plt.plot(xgrid_A, ygrid_A, lw=2, label=labelA + " fit")

    for tot_key, res_B in res_B_all.items():
        thr_B, rate_B, pos_B = prepare_dataset(res_B)
        A_B, k_B, _, _ = fit_exponential(thr_B, rate_B, FIT_START_THRESHOLD_B)
        xgrid_B, ygrid_B = make_fit_curve(A_B, k_B, thr_B)
        plt.scatter(thr_B[pos_B], rate_B[pos_B], s=25, alpha=0.6, label=f"TOT≥{tot_key}")
        if xgrid_B is not None:
            plt.plot(xgrid_B, ygrid_B, lw=1.5, alpha=0.8)

    plt.yscale("log")
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title("Trigger Rate vs Threshold — PURE noise and all TOT eliminations")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.xlim(left=X_axis_start)
    plt.ylim(top=Y_axis_end)
    out_file = OUT_DIR / "compare_all_TOT.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
