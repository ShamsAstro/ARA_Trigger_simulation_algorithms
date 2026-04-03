import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_JSON = Path("threshold_CSW_segments_N_segment_ODD_numbers_scan.json")   # <-- your earlier scan JSON

LABEL = "PURE noise (CSW)"
EVENT_NS = 170.0          # ns per event record
TARGET_HZ = 5.0           # Hz target rate

# Fit-start scan (in threshold units)
FIT_START_CANDIDATES = list(range(5, 40, 2))

# Plot controls
X_AXIS_START = None
Y_AXIS_TOP = 1e9
Y_AXIS_BOTTOM = 0.1 #None

OUT_DIR = Path("plots_nsegments_CSW_full_ODD_scans")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_JSON = OUT_DIR / "summary_ODD_nsegments_analysis.json"
# ---------------------------
# JSON FIELD MAPPING (edit if needed)
# ---------------------------
# Records list can be either:
#   - data["results"] (dict wrapper), or
#   - a raw list
#
# Per-record fields:
THRESHOLD_KEY = "threshold"

# Triggers/Events keys (script will try these in order)
TRIG_KEYS = ("num_triggers", "triggers")
EVT_KEYS = ("num_events_scanned", "events")

# Grouping key for separate curves (examples: "N_segments", "N_windows", "csw_windows", etc.)
# If your CSW scan does NOT vary a discrete parameter, set GROUP_KEY = None
GROUP_KEY = "N_segments"   # <--- change this to your CSW scan grouping key, or None
GROUP_LABEL = r"$N_{\mathrm{segments}}$" # <--- used only for plot labels/titles
# ─────────────────────────────────────────────────────────────────────────────


def load_scan_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data.get("meta", {"note": "no meta found"}), data["results"]
    return {"note": "no meta found"}, data


def rate_and_error_hz(num_triggers: int, num_events: int, event_ns: float):
    """
    triggers/events -> rate in Hz with binomial statistical error.
      p = T/N
      sigma_p = sqrt(p(1-p)/N)
      rate = p / dt
      sigma_rate = sigma_p / dt
    """
    if num_events <= 0:
        return np.nan, np.nan

    p = num_triggers / num_events
    var_p = p * (1.0 - p) / num_events
    sigma_p = np.sqrt(max(var_p, 0.0))

    dt = event_ns * 1e-9
    rate = p / dt
    sigma_rate = sigma_p / dt
    return rate, sigma_rate


def _first_present_int(d: dict, keys, default=0):
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except Exception:
                return default
    return default


def prepare_dataset(records, event_ns: float):
    """
    Returns sorted arrays:
      thresholds, rates_hz, sigma_rates_hz
    Filters out non-finite entries.
    """
    thr, rate, sig = [], [], []

    for r in records:
        if THRESHOLD_KEY not in r:
            continue

        t = float(r[THRESHOLD_KEY])
        T = _first_present_int(r, TRIG_KEYS, default=0)
        N = _first_present_int(r, EVT_KEYS, default=0)

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
      y = ln(rate) = b + k x, where b = lnA
    weights from sigma_y = sigma_rate/rate
    """
    x = np.asarray(thresholds, dtype=float)
    r = np.asarray(rates_hz, dtype=float)
    sr = np.asarray(sigma_rates_hz, dtype=float)

    mask = (x >= fit_start_threshold) & (r > 0) & np.isfinite(r) & np.isfinite(sr)
    xfit, rfit, srfit = x[mask], r[mask], sr[mask]

    if xfit.size < 2:
        return None

    y = np.log(rfit)
    sigma_y = srfit / rfit
    sigma_y = np.where(sigma_y <= 0, np.nan, sigma_y)

    ok = np.isfinite(sigma_y) & (sigma_y > 0)
    xfit, y, sigma_y = xfit[ok], y[ok], sigma_y[ok]

    if xfit.size < 2:
        return None

    W = 1.0 / (sigma_y ** 2)
    X = np.column_stack([np.ones_like(xfit), xfit])  # [b, k]

    XT_W = X.T * W
    M = XT_W @ X
    v = XT_W @ y

    try:
        beta = np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return None

    b, k = float(beta[0]), float(beta[1])

    try:
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        cov = None

    y_model = b + k * xfit
    resid = (y - y_model) / sigma_y
    chi2 = float(np.sum(resid ** 2))
    ndof = int(len(y) - 2)

    return {
        "A": float(np.exp(b)),
        "lnA": b,
        "k": k,
        "cov": cov,
        "chi2": chi2,
        "ndof": ndof,
        "mask_fit_start": float(fit_start_threshold),
    }


def choose_best_fit_start(thr, rate, sig, candidates):
    best = None
    for fs in candidates:
        fit = weighted_log_fit(thr, rate, sig, fs)
        if fit is None or fit["ndof"] <= 0:
            continue
        if (best is None) or (fit["chi2"] < best["chi2"]):
            best = fit
    return best


def threshold_at_target_with_error(fit, target_hz):
    """
    target_hz = A * exp(k x) -> x = (ln(target)-lnA)/k
    error propagation using cov(lnA, k)
    """
    if fit is None:
        return None, None

    lnA = fit["lnA"]
    k = fit["k"]
    cov = fit["cov"]

    if k == 0 or target_hz <= 0:
        return None, None

    x = (np.log(target_hz) - lnA) / k

    if cov is None or (not np.all(np.isfinite(cov))):
        return float(x), None

    d_lnA = -1.0 / k
    d_k = -x / k
    J = np.array([d_lnA, d_k], dtype=float)
    var_x = float(J.T @ cov @ J)
    sig_x = np.sqrt(max(var_x, 0.0))
    return float(x), float(sig_x)


def make_fit_curve(fit, xdata, extra_x=None):
    if fit is None:
        return None, None
    A, k = fit["A"], fit["k"]
    xmin = float(np.min(xdata))
    xmax = float(np.max(xdata))
    if extra_x is not None and np.isfinite(extra_x):
        xmax = max(xmax, float(extra_x))
    span = max(1.0, xmax - xmin)
    xmax = xmax + 0.1 * span
    xg = np.linspace(xmin, xmax, 400)
    yg = A * np.exp(k * xg)
    return xg, yg


def group_records(results):
    """
    Returns dict[group_value] -> list[records]
    If GROUP_KEY is None, returns {0: results}
    """
    if GROUP_KEY is None:
        return {0: list(results)}

    by_g = defaultdict(list)
    for r in results:
        if GROUP_KEY not in r:
            continue
        try:
            gv = int(r[GROUP_KEY])
        except Exception:
            # fallback if not int-like
            gv = str(r[GROUP_KEY])
        by_g[gv].append(r)

    return dict(by_g)


def main():
    meta, results = load_scan_json(IN_JSON)

    if not isinstance(results, list) or len(results) == 0:
        raise RuntimeError(f"{IN_JSON}: results is empty or not a list.")

    by_group = group_records(results)
    g_keys = sorted(by_group.keys(), key=lambda x: (isinstance(x, str), x))

    if not g_keys:
        print(f"No records found with GROUP_KEY={GROUP_KEY!r}.")
        print("Tip: set GROUP_KEY = None or update GROUP_KEY to match your JSON field.")
        return

    analysis_summary = {
        "input_json": str(IN_JSON),
        "label": LABEL,
        "event_ns": EVENT_NS,
        "target_hz": TARGET_HZ,
        "fit_start_candidates": FIT_START_CANDIDATES,
        "meta_from_scan_file": meta,
        "field_mapping": {
            "threshold_key": THRESHOLD_KEY,
            "trigger_keys": TRIG_KEYS,
            "event_keys": EVT_KEYS,
            "group_key": GROUP_KEY,
        },
        "results": {},
    }

    # -------- per-group plots --------
    for g in g_keys:
        recs = by_group[g]
        thr, rate, sig = prepare_dataset(recs, EVENT_NS)

        best_fit = choose_best_fit_start(thr, rate, sig, FIT_START_CANDIDATES)

        if best_fit is None:
            analysis_summary["results"][str(g)] = {
                "status": "fit_failed",
                "num_points_total": int(len(thr)),
                "note": "Not enough valid points for weighted log-fit."
            }
            continue

        thr_target, thr_target_err = threshold_at_target_with_error(best_fit, TARGET_HZ)
        xgrid, ygrid = make_fit_curve(best_fit, thr, extra_x=thr_target)

        analysis_summary["results"][str(g)] = {
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

        pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)
        plt.errorbar(thr[pos], rate[pos], yerr=sig[pos], fmt="o", ms=5, capsize=2,
                     label=f"{GROUP_LABEL} = {g} data" if GROUP_KEY is not None else "data")

        if xgrid is not None:
            plt.plot(xgrid, ygrid, lw=2, label=f"fit (start≥{best_fit['mask_fit_start']})")

        plt.axhline(TARGET_HZ, linestyle="--", color="tab:red",
                    label=f"Target = {TARGET_HZ:.1f} Hz")

        if thr_target is not None and np.isfinite(thr_target):
            plt.axvline(thr_target, linestyle="--", color="tab:green")
            plt.scatter([thr_target], [TARGET_HZ], color="tab:green", zorder=5)

            txt = f"{thr_target:.3f}"
            if thr_target_err is not None and np.isfinite(thr_target_err):
                txt = f"{thr_target:.3f} ± {thr_target_err:.3f}"
            plt.text(thr_target, TARGET_HZ * 1.2, txt, color="tab:green",
                     va="bottom", ha="left")

        plt.yscale("log")
        plt.xlabel("Threshold (CSW units)")
        plt.ylabel("Trigger rate (Hz, log scale)")
        title_group = f"{GROUP_LABEL}={g}" if GROUP_KEY is not None else "all"
        plt.title(f"{LABEL}\nTrigger Rate vs Threshold — {title_group}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        if X_AXIS_START is not None:
            plt.xlim(left=float(X_AXIS_START))
        if Y_AXIS_BOTTOM is not None:
            plt.ylim(bottom=float(Y_AXIS_BOTTOM))
        if Y_AXIS_TOP is not None:
            plt.ylim(top=float(Y_AXIS_TOP))

        out_name = f"N_segments_{g}.png" if GROUP_KEY is not None else "scan.png"
        out_file = OUT_DIR / out_name
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()

    #combined plot
    # -------- combined plot (TOT-style: dotted extrap regions + X markers at 5 Hz) --------
    plt.figure(figsize=(12, 7))
    plt.axhline(TARGET_HZ, linestyle="--", color="tab:red",
                label=f"Target = {TARGET_HZ:.1f} Hz")

    for i, g in enumerate(g_keys):
        recs = by_group[g]
        thr, rate, sig = prepare_dataset(recs, EVENT_NS)

        pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)

        # choose a consistent color per group (like TOT script did)
        color_choice = plt.get_cmap("tab10")(int(i) % 10)

        plt.errorbar(
            thr[pos], rate[pos], yerr=sig[pos],
            fmt="o", ms=4, capsize=2, alpha=0.55,
            label=(f"{GROUP_LABEL} = {g}" if GROUP_KEY is not None else "data")
        )

        best = analysis_summary["results"].get(str(g), {})
        if best.get("status") == "ok":
            fs = float(best["fit_start_threshold"])
            thr_pred = best.get("threshold_at_target_hz", None)

            # Rebuild the fit for this group (same as old combined plot)
            fit = weighted_log_fit(thr, rate, sig, fs)

            if fit is not None:
                # Build curve domain similar to TOT plotting logic
                xmin = float(np.min(thr[pos])) if np.any(pos) else float(np.min(thr))
                xmax = float(np.max(thr[pos])) if np.any(pos) else float(np.max(thr))

                if thr_pred is not None and np.isfinite(thr_pred):
                    xmax = max(xmax, float(thr_pred))

                span = max(1.0, xmax - xmin)
                xg = np.linspace(xmin, xmax + 0.1 * span, 400)
                yg = fit["A"] * np.exp(fit["k"] * xg)

                fit_start = float(fs)
                last_data = float(np.max(thr[pos])) if np.any(pos) else float(np.max(thr))

                # Dotted line for extrapolated regions (before fit start and after last data)
                mask_extrap = (xg < fit_start) | (xg > last_data)
                plt.plot(
                    xg[mask_extrap], yg[mask_extrap],
                    color=color_choice, lw=1.5, alpha=0.9, linestyle="--"
                )

                # Solid line for fitted region (fit_start -> last_data)
                mask_fitted = (xg >= fit_start) & (xg <= last_data)
                plt.plot(
                    xg[mask_fitted], yg[mask_fitted],
                    color=color_choice, lw=1.5, alpha=0.9
                )

            # X marker at the 5 Hz intersection
            if thr_pred is not None and np.isfinite(thr_pred):
                plt.scatter(
                    [float(thr_pred)], [TARGET_HZ],
                    s=40, alpha=0.9, color=color_choice, marker="X"
                )

    plt.yscale("log")
    plt.xlabel("Threshold (CSW units)", fontsize=14)
    plt.ylabel("Trigger rate (Hz, log scale)", fontsize=14)
    plt.title(f"{LABEL}\nTrigger Rate vs Threshold — combined (data + fits + {TARGET_HZ:.1f} Hz intersections)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=14)

    # axis controls (keep your template variables)
    if X_AXIS_START is not None:
        plt.xlim(left=float(X_AXIS_START))
    if Y_AXIS_BOTTOM is not None or Y_AXIS_TOP is not None:
        plt.ylim(bottom=Y_AXIS_BOTTOM, top=Y_AXIS_TOP*3.5)

    out_file = OUT_DIR / "compare_all_groups.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


    # -------- summary JSON --------
    with open(SUMMARY_JSON, "w") as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"Saved plots to: {OUT_DIR}")
    print(f"Saved summary JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
