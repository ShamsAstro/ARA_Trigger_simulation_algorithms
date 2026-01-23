import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_TOT_SCAN_JSON = Path("Full_threshold_scan_long_10env.json")

EVENT_NS = 170.0
TARGET_HZ = 5.0

# Candidate fit-start thresholds (same units as your thresholds, ADC^2)
FIT_START_CANDIDATES = list(range(20000, 50001, 2500))

OUT_DIR = Path("TOT_threshold_analysis_outputs_10env")
OUT_DIR.mkdir(exist_ok=True)

OUT_SUMMARY_JSON = OUT_DIR / "summary_TOT_threshold_analysis_10env.json"

# Plot limits (optional)
X_AXIS_LEFT = None
X_AXIS_RIGHT = None
Y_AXIS_TOP = 1e10
Y_AXIS_BOTTOM = None
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def extract_tot_results(scan_data):
    """
    Expected new format:
      {"meta":..., "results": {"0":[...], "1":[...], ...}}
    Also supports old format:
      {"0":[...], "1":[...], ...}
    """
    if isinstance(scan_data, dict) and "results" in scan_data and isinstance(scan_data["results"], dict):
        return scan_data.get("meta", {}), scan_data["results"]
    if isinstance(scan_data, dict):
        return {}, scan_data
    raise RuntimeError("Unsupported TOT scan JSON format.")


def get_triggers_per_threshold(meta: dict, default: int = 12) -> int:
    try:
        return int(meta.get("simulation_parameters", {}).get("TRIGGERS_PER_THRESHOLD", default))
    except Exception:
        return default


def drop_last_if_too_partial(records: list, triggers_per_threshold: int) -> list:
    """
    If the LAST threshold record has num_triggers < (2/3)*TRIGGERS_PER_THRESHOLD,
    drop it from the analysis.
    """
    if not records:
        return records

    last = records[-1]
    try:
        ntrig = int(last.get("num_triggers", last.get("triggers", 0)))
    except Exception:
        ntrig = 0

    min_required = (2.0 / 3.0) * float(triggers_per_threshold)
    if ntrig < min_required:
        return records[:-1]
    return records


def rate_and_error_hz(num_triggers: int, num_events: int, event_ns: float):
    """
    Binomial error on pass fraction:
      p = T/N, sigma_p = sqrt(p(1-p)/N)
    rate = p / dt, sigma_rate = sigma_p / dt, dt = event_ns*1e-9
    """
    if num_events <= 0:
        return np.nan, np.nan
    p = num_triggers / num_events
    var_p = p * (1.0 - p) / num_events
    sigma_p = np.sqrt(max(var_p, 0.0))
    dt = event_ns * 1e-9
    return p / dt, sigma_p / dt


def prepare_dataset(records, event_ns: float):
    """
    Build sorted arrays:
      thr, rate_hz, sigma_rate_hz
    """
    thr, rate, sig = [], [], []
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
    return thr[finite], rate[finite], sig[finite]


def weighted_log_fit(thresholds, rates_hz, sigma_rates_hz, fit_start_threshold):
    """
    Weighted fit in log space:
      ln(rate) = lnA + k*x
    sigma_ln = sigma_rate / rate
    """
    x = np.asarray(thresholds, dtype=float)
    r = np.asarray(rates_hz, dtype=float)
    sr = np.asarray(sigma_rates_hz, dtype=float)

    mask = (x >= fit_start_threshold) & (r > 0) & np.isfinite(r) & np.isfinite(sr)
    xfit = x[mask]
    rfit = r[mask]
    srfit = sr[mask]
    if xfit.size < 2:
        return None

    y = np.log(rfit)
    sigma_y = srfit / rfit
    ok = np.isfinite(sigma_y) & (sigma_y > 0)
    xfit, y, sigma_y = xfit[ok], y[ok], sigma_y[ok]
    if xfit.size < 2:
        return None

    W = 1.0 / (sigma_y ** 2)
    X = np.column_stack([np.ones_like(xfit), xfit])

    XT_W = X.T * W
    M = XT_W @ X
    v = XT_W @ y

    try:
        beta = np.linalg.solve(M, v)     # [lnA, k]
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None

    lnA = float(beta[0])
    k = float(beta[1])
    A = float(np.exp(lnA))

    y_model = lnA + k * xfit
    resid = (y - y_model) / sigma_y
    chi2 = float(np.sum(resid ** 2))
    ndof = int(len(y) - 2)

    return {
        "A": A,
        "lnA": lnA,
        "k": k,
        "cov": cov,  # cov for [lnA, k]
        "chi2": chi2,
        "ndof": ndof,
        "chi2_reduced": float(chi2 / ndof) if ndof > 0 else None,
        "fit_start_threshold": float(fit_start_threshold),
        "n_fit_points": int(len(xfit)),
    }


def choose_best_fit(thr, rate, sig, candidates):
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
    target = A exp(kx) => x = (ln(target)-lnA)/k
    propagate via cov(lnA,k)
    """
    if fit is None:
        return None, None
    lnA = fit["lnA"]
    k = fit["k"]
    cov = fit["cov"]
    if k == 0 or target_hz <= 0:
        return None, None

    x = (np.log(target_hz) - lnA) / k

    if cov is None or not np.all(np.isfinite(cov)):
        return float(x), None

    d_lnA = -1.0 / k
    d_k = -x / k
    J = np.array([d_lnA, d_k], dtype=float)

    var_x = float(J.T @ cov @ J)
    sig_x = np.sqrt(max(var_x, 0.0))
    return float(x), float(sig_x)


def make_fit_curve(A, k, x_min, x_max, n=400):
    xg = np.linspace(float(x_min), float(x_max), int(n))
    yg = A * np.exp(k * xg)
    return xg, yg


def plot_per_tot(tot_key, thr, rate, sig, best_fit, thr_pred, thr_err, out_dir: Path):
    """
    Scatter with error bars + fit curve + target line + predicted threshold annotation.
    NOTE: No colored error band (per your request).
    """
    plt.figure(figsize=(10, 6))

    pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)
    plt.errorbar(
        thr[pos], rate[pos], yerr=sig[pos],
        fmt="o", ms=5, capsize=2,
        label=f"TOT≥{tot_key} data"
    )

    if best_fit is not None:
        A = best_fit["A"]
        k = best_fit["k"]
        xmin = float(np.min(thr[pos])) if np.any(pos) else float(np.min(thr))
        xmax = float(np.max(thr[pos])) if np.any(pos) else float(np.max(thr))
        if thr_pred is not None and np.isfinite(thr_pred):
            xmax = max(xmax, float(thr_pred))
        span = max(1.0, xmax - xmin)
        xg, yg = make_fit_curve(A, k, xmin, xmax + 0.1 * span)
        plt.plot(xg, yg, lw=2, label=f"fit (start≥{best_fit['fit_start_threshold']:.0f})")

    plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

    if thr_pred is not None and np.isfinite(thr_pred):
        thr_pred = float(thr_pred)
        plt.axvline(thr_pred, linestyle="--", color="tab:green")
        plt.scatter([thr_pred], [TARGET_HZ], color="tab:green", zorder=5)

        if thr_err is not None and np.isfinite(thr_err) and float(thr_err) > 0:
            txt = f"{thr_pred:.0f} ± {float(thr_err):.0f}"
        else:
            txt = f"{thr_pred:.0f}"

        plt.text(thr_pred, TARGET_HZ * 1.2, txt, color="tab:green", va="bottom", ha="left")

    plt.yscale("log")
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title(f"Trigger Rate vs Threshold — TOT≥{tot_key}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    if X_AXIS_LEFT is not None or X_AXIS_RIGHT is not None:
        plt.xlim(left=X_AXIS_LEFT, right=X_AXIS_RIGHT)
    if Y_AXIS_BOTTOM is not None or Y_AXIS_TOP is not None:
        plt.ylim(bottom=Y_AXIS_BOTTOM, top=Y_AXIS_TOP)

    out_file = out_dir / f"TOT_{int(tot_key):02d}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_combined(all_series, out_dir: Path):
    """
    One plot with all TOT data+fits and all 5Hz intersections.
    NOTE: No colored error bands.
    """
    plt.figure(figsize=(12, 7))
    plt.axhline(TARGET_HZ, linestyle="--", color="tab:red", label=f"Target = {TARGET_HZ:.1f} Hz")

    for s in sorted(all_series, key=lambda d: d["tot_key"]):
        tot = s["tot_key"]
        thr = s["thr"]
        rate = s["rate"]
        sig = s["sig"]
        fit = s["best_fit"]
        thr_pred = s["thr_pred"]

        pos = (rate > 0) & np.isfinite(rate) & np.isfinite(sig)
        plt.errorbar(
            thr[pos], rate[pos], yerr=sig[pos],
            fmt="o", ms=4, capsize=2, alpha=0.55,
            label=f"TOT≥{tot}"
        )

        if fit is not None:
            A, k = fit["A"], fit["k"]
            xmin = float(np.min(thr[pos])) if np.any(pos) else float(np.min(thr))
            xmax = float(np.max(thr[pos])) if np.any(pos) else float(np.max(thr))
            if thr_pred is not None and np.isfinite(thr_pred):
                xmax = max(xmax, float(thr_pred))
            span = max(1.0, xmax - xmin)
            xg, yg = make_fit_curve(A, k, xmin, xmax + 0.1 * span)
            plt.plot(xg, yg, lw=1.5, alpha=0.8)

        if thr_pred is not None and np.isfinite(thr_pred):
            plt.scatter([float(thr_pred)], [TARGET_HZ], s=25, alpha=0.9)

    plt.yscale("log")
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (Hz, log scale)")
    plt.title("Trigger Rate vs Threshold — all TOT eliminations (data + fits + 5 Hz intersections)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    if X_AXIS_LEFT is not None or X_AXIS_RIGHT is not None:
        plt.xlim(left=X_AXIS_LEFT, right=X_AXIS_RIGHT)
    if Y_AXIS_BOTTOM is not None or Y_AXIS_TOP is not None:
        plt.ylim(bottom=Y_AXIS_BOTTOM, top=Y_AXIS_TOP)

    out_file = out_dir / "compare_all_TOT.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    scan_data = load_json(IN_TOT_SCAN_JSON)
    meta, results_by_tot = extract_tot_results(scan_data)

    triggers_per_threshold = get_triggers_per_threshold(meta, default=12)
    min_last_required = (2.0 / 3.0) * float(triggers_per_threshold)

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"tot_scan_json": str(IN_TOT_SCAN_JSON)},
        "analysis_parameters": {
            "event_ns": EVENT_NS,
            "target_hz": TARGET_HZ,
            "fit_start_candidates": FIT_START_CANDIDATES,
            "statistics_model": "binomial on pass fraction, weighted log-fit on ln(rate)",
            "drop_last_threshold_if_num_triggers_below": min_last_required,
            "triggers_per_threshold": int(triggers_per_threshold),
        },
        "scan_meta": meta,
        "results": {},
    }

    all_series = []

    for tot_key in sorted(results_by_tot.keys(), key=lambda k: int(k)):
        records_raw = results_by_tot[tot_key]
        if not isinstance(records_raw, list) or len(records_raw) == 0:
            summary["results"][str(tot_key)] = {"status": "fit_failed", "reason": "no records"}
            continue

        # Drop last threshold if it's too partial
        records = drop_last_if_too_partial(records_raw, triggers_per_threshold)

        if len(records) < 2:
            summary["results"][str(tot_key)] = {
                "status": "fit_failed",
                "reason": "too few records after dropping partial last threshold",
                "num_points_total_raw": int(len(records_raw)),
                "num_points_total_used": int(len(records)),
            }
            continue

        thr, rate, sig = prepare_dataset(records, EVENT_NS)

        best_fit = choose_best_fit(thr, rate, sig, FIT_START_CANDIDATES)
        if best_fit is None:
            summary["results"][str(tot_key)] = {
                "status": "fit_failed",
                "reason": "not enough valid points for weighted log-fit",
                "num_points_total_raw": int(len(records_raw)),
                "num_points_total_used": int(len(thr)),
            }
            continue

        thr_pred, thr_err = threshold_at_target_with_error(best_fit, TARGET_HZ)

        summary["results"][str(tot_key)] = {
            "status": "ok",
            "num_points_total_raw": int(len(records_raw)),
            "num_points_total_used": int(len(thr)),
            "fit_start_threshold": float(best_fit["fit_start_threshold"]),
            "A": float(best_fit["A"]),
            "k": float(best_fit["k"]),
            "chi2": float(best_fit["chi2"]),
            "ndof": int(best_fit["ndof"]),
            "chi2_reduced": best_fit["chi2_reduced"],
            "n_fit_points": int(best_fit["n_fit_points"]),
            "threshold_at_target_hz": thr_pred,
            "threshold_at_target_hz_err": thr_err,
        }

        plot_per_tot(
            tot_key=int(tot_key),
            thr=thr,
            rate=rate,
            sig=sig,
            best_fit=best_fit,
            thr_pred=thr_pred,
            thr_err=thr_err,
            out_dir=OUT_DIR,
        )

        all_series.append({
            "tot_key": int(tot_key),
            "thr": thr,
            "rate": rate,
            "sig": sig,
            "best_fit": best_fit,
            "thr_pred": thr_pred,
            "thr_err": thr_err,
        })

        print(f"TOT≥{tot_key}: thr@{TARGET_HZ}Hz = {thr_pred:.2f} ± {thr_err if thr_err is not None else float('nan'):.2f}")

    if all_series:
        plot_combined(all_series, OUT_DIR)

    save_json(summary, OUT_SUMMARY_JSON)

    print(f"\nSaved summary: {OUT_SUMMARY_JSON}")
    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
