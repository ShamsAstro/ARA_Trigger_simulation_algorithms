import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ----------------- CONFIG -----------------
IN_JSON = Path("threshold_CSW_segments_full_scan.json")   # <-- your earlier scan JSON

LABEL = "PURE noise (CSW FFT)"
EVENT_NS = 170.0               # ns per event record
TARGET_HZ = 5.0                # Hz target rate

# Fit-start scan (in threshold units)
FIT_START_CANDIDATES = list(range(5, 40, 2))  # 5..10 inclusive

# Plot controls
X_AXIS_START = None            # set to a float if you want, else None
Y_AXIS_TOP   = 1e10            # log axis upper bound
Y_AXIS_BOTTOM = None           # set if you want, else None

OUT_DIR = Path("plots_nsegments_CSW_full_scans")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_JSON = OUT_DIR / "summary_nsegments_analysis.json"
# ------------------------------------------


def load_scan_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["meta"], data["results"]
    # fallback if old format is just a list
    return {"note": "no meta found"}, data


def rate_and_error_hz(num_triggers: int, num_events: int, event_ns: float):
    """
    Convert triggers/events -> rate in Hz with binomial statistical error.
    p = T/N
    sigma_p = sqrt(p(1-p)/N)
    rate = p / (event_ns*1e-9)
    sigma_rate = sigma_p / (event_ns*1e-9)
    """
    if num_events <= 0:
        return np.nan, np.nan

    p = num_triggers / num_events
    # Binomial variance p(1-p)/N; handle edge cases p=0 or p=1 cleanly
    var_p = p * (1.0 - p) / num_events
    sigma_p = np.sqrt(max(var_p, 0.0))

    dt = event_ns * 1e-9
    rate = p / dt
    sigma_rate = sigma_p / dt
    return rate, sigma_rate


def prepare_dataset(records, event_ns: float):
    """
    Returns sorted arrays:
      thresholds, rates_hz, sigma_rates_hz
    Filters out non-finite entries.
    """
    thr = []
    rate = []
    sig = []

    for r in records:
        t = float(r["threshold"])
        T = int(r.get("num_triggers", r.get("triggers", 0)))
        N = int(r.get("num_events_scanned", r.get("events", 0)))

        rhz, srhz = rate_and_error_hz(T, N, event_ns)
        thr.append(t)
        rate.append(rhz)
        sig.append(srhz)

    thr = np.asarray(thr, dtype=float)
    rate = np.asarray(rate, dtype=float)
    sig = np.asarray(sig, dtype=float)

    order = np.argsort(thr)
    thr, rate, sig = thr[order], rate[order], sig[order]

    finite = np.isfinite(thr) & np.isfinite(rate) & np.isfinite(sig)
    thr, rate, sig = thr[finite], rate[finite], sig[finite]

    return thr, rate, sig


def weighted_log_fit(thresholds, rates_hz, sigma_rates_hz, fit_start_threshold):
    """
    Weighted fit in log-space:
      y = ln(rate)
      y = b + k x   where b = lnA
    weights from sigma_y = sigma_rate/rate  (propagation), W = 1/sigma_y^2

    Returns dict with params, cov, chi2, ndof, and fit mask arrays.
    """
    x = np.asarray(thresholds, dtype=float)
    r = np.asarray(rates_hz, dtype=float)
    sr = np.asarray(sigma_rates_hz, dtype=float)

    # Need positive rates to take log, and non-zero sigma
    mask = (x >= fit_start_threshold) & (r > 0) & np.isfinite(r) & np.isfinite(sr)

    xfit = x[mask]
    rfit = r[mask]
    srfit = sr[mask]

    if xfit.size < 2:
        return None

    y = np.log(rfit)

    # sigma_ln = sigma_r / r
    sigma_y = srfit / rfit
    # avoid zero weights
    sigma_y = np.where(sigma_y <= 0, np.nan, sigma_y)
    ok = np.isfinite(sigma_y) & (sigma_y > 0)
    xfit, y, sigma_y = xfit[ok], y[ok], sigma_y[ok]

    if xfit.size < 2:
        return None

    W = 1.0 / (sigma_y ** 2)

    # Design matrix for [b, k]
    X = np.column_stack([np.ones_like(xfit), xfit])

    # Weighted normal equations
    XT_W = X.T * W
    M = XT_W @ X
    v = XT_W @ y

    try:
        beta = np.linalg.solve(M, v)  # [b, k]
    except np.linalg.LinAlgError:
        return None

    b, k = beta[0], beta[1]

    # Covariance
    try:
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        cov = None

    y_model = b + k * xfit
    resid = (y - y_model) / sigma_y
    chi2 = float(np.sum(resid ** 2))
    ndof = int(len(y) - 2)

    A = float(np.exp(b))

    return {
        "A": A,
        "lnA": float(b),
        "k": float(k),
        "cov": cov,  # cov for [lnA, k]
        "chi2": chi2,
        "ndof": ndof,
        "xfit": xfit,
        "yfit": np.exp(y),       # back to rate units
        "mask_fit_start": float(fit_start_threshold),
        "sigma_ln_used": sigma_y,
    }


def choose_best_fit_start(thr, rate, sig, candidates):
    """
    Try multiple fit_start_threshold values, pick the one with smallest chi2 (with valid ndof).
    """
    best = None
    for fs in candidates:
        fit = weighted_log_fit(thr, rate, sig, fs)
        if fit is None:
            continue
        if fit["ndof"] <= 0:
            continue
        if (best is None) or (fit["chi2"] < best["chi2"]):
            best = fit
    return best


def threshold_at_target_with_error(fit, target_hz):
    """
    Solve target_hz = A * exp(k x) -> x = (ln(target)-lnA)/k
    Propagate uncertainty from cov(lnA, k).
    """
    if fit is None:
        return None, None

    lnA = fit["lnA"]
    k = fit["k"]
    cov = fit["cov"]

    if k == 0 or target_hz <= 0 or cov is None or not np.all(np.isfinite(cov)):
        # return x but no error if covariance is missing
        x = (np.log(target_hz) - lnA) / k if k != 0 else None
        return x, None

    x = (np.log(target_hz) - lnA) / k

    # x(lnA,k) gradients:
    # x = (lnT - lnA)/k
    # dx/d(lnA) = -1/k
    # dx/dk = -(lnT - lnA)/k^2 = -x/k
    d_lnA = -1.0 / k
    d_k = -x / k

    J = np.array([d_lnA, d_k], dtype=float)  # shape (2,)
    var_x = float(J.T @ cov @ J)
    sig_x = np.sqrt(max(var_x, 0.0))
    return float(x), float(sig_x)


def make_fit_curve(fit, xdata, extra_x=None):
    if fit is None:
        return None, None
    A = fit["A"]
    k = fit["k"]
    xmin = float(np.min(xdata))
    xmax = float(np.max(xdata))
    if extra_x is not None and np.isfinite(extra_x):
        xmax = max(xmax, float(extra_x))
    span = max(1.0, xmax - xmin)
    xmax = xmax + 0.1 * span
    xg = np.linspace(xmin, xmax, 400)
    yg = A * np.exp(k * xg)
    return xg, yg


def main():
    meta, results = load_scan_json(IN_JSON)

    # Group by N_segments
    by_seg = defaultdict(list)
    for r in results:
        if "N_segments" not in r:
            continue
        by_seg[int(r["N_segments"])].append(r)

    nseg_keys = sorted(by_seg.keys())
    if not nseg_keys:
        print("No N_segments records found in JSON.")
        return

    analysis_summary = {
        "input_json": str(IN_JSON),
        "event_ns": EVENT_NS,
        "target_hz": TARGET_HZ,
        "fit_start_candidates": FIT_START_CANDIDATES,
        "meta_from_scan_file": meta,
        "results": {}
    }

    # -------- per N_segments plots --------
    for nseg in nseg_keys:
        recs = by_seg[nseg]
        thr, rate, sig = prepare_dataset(recs, EVENT_NS)

        # Choose best fit start by chi2
        best_fit = choose_best_fit_start(thr, rate, sig, FIT_START_CANDIDATES)

        if best_fit is None:
            analysis_summary["results"][str(nseg)] = {
                "status": "fit_failed",
                "num_points_total": int(len(thr)),
                "note": "Not enough valid points for weighted log-fit."
            }
            continue

        thr_target, thr_target_err = threshold_at_target_with_error(best_fit, TARGET_HZ)
        xgrid, ygrid = make_fit_curve(best_fit, thr, extra_x=thr_target)

        # Store summary
        analysis_summary["results"][str(nseg)] = {
            "status": "ok",
            "num_points_total": int(len(thr)),
            "fit_start_threshold": float(best_fit["mask_fit_start"]),
            "A": float(best_fit["A"]),
            "k": float(best_fit["k"]),
            "chi2": float(best_fit["chi2"]),
            "ndof": int(best_fit["ndof"]),
            "chi2_reduced": float(best_fit["chi2"] / best_fit["ndof"]) if best_fit["ndof"] > 0 else None,
            "threshold_at_target_hz": thr_target,
            "threshold_at_target_hz_err": thr_target_err,
        }

        # Plot
        plt.figure(figsize=(10, 6))
        # error bars
        pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)
        plt.errorbar(thr[pos], rate[pos], yerr=sig[pos], fmt="o", ms=5, capsize=2,
                     label=f"N_segments={nseg} data")

        if xgrid is not None:
            plt.plot(xgrid, ygrid, lw=2, label=f"fit (start≥{best_fit['mask_fit_start']})")

        plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

        if thr_target is not None and np.isfinite(thr_target):
            plt.axvline(thr_target, linestyle="--", color="tab:green")
            plt.scatter([thr_target], [TARGET_HZ], color="tab:green", zorder=5)

            # annotate with uncertainty if available
            if thr_target_err is not None and np.isfinite(thr_target_err):
                txt = f"{thr_target:.3f} ± {thr_target_err:.3f}"
            else:
                txt = f"{thr_target:.3f}"
            plt.text(thr_target, TARGET_HZ * 1.2, txt, color="tab:green", va="bottom", ha="left")

        plt.yscale("log")
        plt.xlabel("Threshold (CSW units)")
        plt.ylabel("Trigger rate (Hz, log scale)")
        plt.title(f"Trigger Rate vs Threshold — N_segments={nseg}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        if X_AXIS_START is not None:
            plt.xlim(left=float(X_AXIS_START))
        if Y_AXIS_BOTTOM is not None:
            plt.ylim(bottom=float(Y_AXIS_BOTTOM))
        if Y_AXIS_TOP is not None:
            plt.ylim(top=float(Y_AXIS_TOP))

        out_file = OUT_DIR / f"nsegments_{nseg:02d}.png"
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()

    # -------- combined plot --------
    plt.figure(figsize=(12, 7))
    plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

    for nseg in nseg_keys:
        recs = by_seg[nseg]
        thr, rate, sig = prepare_dataset(recs, EVENT_NS)

        pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)
        plt.errorbar(thr[pos], rate[pos], yerr=sig[pos], fmt="o", ms=4, capsize=2, alpha=0.65,
                     label=f"Nseg={nseg}")

        best = analysis_summary["results"].get(str(nseg), {})
        if best.get("status") == "ok":
            # Rebuild curve from saved A,k if you want; simpler: re-fit from stored values is not necessary.
            # We'll just do a fit again quickly using the chosen start threshold to plot curve consistently.
            fs = float(best["fit_start_threshold"])
            fit = weighted_log_fit(thr, rate, sig, fs)
            thr_target = best.get("threshold_at_target_hz", None)
            xg, yg = make_fit_curve(fit, thr, extra_x=thr_target)
            if xg is not None:
                plt.plot(xg, yg, lw=1.5, alpha=0.8)

    plt.yscale("log")
    plt.xlabel("Threshold (CSW units)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title("Trigger Rate vs Threshold — all N_segments")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    if X_AXIS_START is not None:
        plt.xlim(left=float(X_AXIS_START))
    if Y_AXIS_BOTTOM is not None:
        plt.ylim(bottom=float(Y_AXIS_BOTTOM))
    if Y_AXIS_TOP is not None:
        plt.ylim(top=float(Y_AXIS_TOP))

    out_file = OUT_DIR / "compare_all_Nsegments.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

    # -------- write summary JSON --------
    with open(SUMMARY_JSON, "w") as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"Saved plots to: {OUT_DIR}")
    print(f"Saved summary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
