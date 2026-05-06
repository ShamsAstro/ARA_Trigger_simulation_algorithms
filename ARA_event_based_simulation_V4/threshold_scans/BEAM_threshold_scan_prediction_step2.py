import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl
import numpy as np


# =======================
# GLOBAL PLOT STYLE PRESET
# =======================
mpl.rcParams.update({

    # ---- Figure ----
    "figure.figsize": (8, 5),
    "figure.dpi": 120,

    # ---- Fonts ----
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    # ---- Lines ----
    "lines.linewidth": 2,
    "lines.markersize": 6,

    # ---- Axes ----
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,

    # ---- Ticks ----
    "xtick.direction": "in",
    "ytick.direction": "in",

    # ---- Legend ----
    "legend.frameon": True,

    # ---- Savefig ----
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# ---- Custom color cycle (priority order) ----
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    "black",
    "red",
    "#003366",   # dark blue
    "orange",
    "green"
])


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IN_JSON = Path("Test_threshold_scan_BEAM_algorithm_1000beams.json")

LABEL = "PURE noise BEAMs = 1000"
EVENT_NS = 170.0 *2
TARGET_HZ = 5.0

FIT_START_CANDIDATES = list(range(int(10 * 1.0e5), int(2.0e8), int(2.0e6)))

X_AXIS_START = None
Y_AXIS_TOP = 1e9
Y_AXIS_BOTTOM = 0.1

OUT_DIR = Path("plots_BEAM_Nsegment_1_threshold_scan_BEAM")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_JSON = OUT_DIR / "summary_BEAM_analysis_parallel.json"

THRESHOLD_KEY = "threshold"
TRIG_KEYS = ("num_triggers", "triggers")
EVT_KEYS = ("num_events_scanned", "events")


def load_scan_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data.get("meta", {"note": "no meta found"}), data["results"]

    return {"note": "no meta found"}, data


def rate_and_error_hz(num_triggers, num_events, event_ns):
    if num_events <= 0:
        return np.nan, np.nan

    dt = event_ns * 1e-9

    p = num_triggers / num_events
    rate = p / dt

    # Use Poisson counting uncertainty instead of binomial.
    # This avoids zero error when num_triggers == num_events.
    sigma_triggers = np.sqrt(max(num_triggers, 1.0))
    sigma_rate = sigma_triggers / (num_events * dt)

    return rate, sigma_rate


def first_present_int(d, keys, default=0):
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except Exception:
                return default
    return default


def prepare_dataset(records):
    thresholds = []
    rates = []
    sigmas = []

    for r in records:
        if THRESHOLD_KEY not in r:
            continue

        threshold = float(r[THRESHOLD_KEY])
        num_triggers = first_present_int(r, TRIG_KEYS)
        num_events = first_present_int(r, EVT_KEYS)

        rate, sigma = rate_and_error_hz(num_triggers, num_events, EVENT_NS)

        thresholds.append(threshold)
        rates.append(rate)
        sigmas.append(sigma)

    thresholds = np.asarray(thresholds, dtype=float)
    rates = np.asarray(rates, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    rates = rates[order]
    sigmas = sigmas[order]

    finite = np.isfinite(thresholds) & np.isfinite(rates) & np.isfinite(sigmas)

    return thresholds[finite], rates[finite], sigmas[finite]


def weighted_log_fit(thresholds, rates_hz, sigma_rates_hz, fit_start_threshold):
    """
    Fit:
        rate = A exp(k threshold)

    Done in log space:
        ln(rate) = ln(A) + k threshold
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

    xfit = xfit[ok]
    y = y[ok]
    sigma_y = sigma_y[ok]

    if xfit.size < 2:
        return None

    W = 1.0 / sigma_y**2
    X = np.column_stack([np.ones_like(xfit), xfit])

    XT_W = X.T * W
    M = XT_W @ X
    v = XT_W @ y

    try:
        beta = np.linalg.solve(M, v)
        cov = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None

    lnA, k = beta

    y_model = lnA + k * xfit
    resid = (y - y_model) / sigma_y

    chi2 = float(np.sum(resid**2))
    ndof = int(len(y) - 2)

    return {
        "A": float(np.exp(lnA)),
        "lnA": float(lnA),
        "k": float(k),
        "cov": cov,
        "chi2": chi2,
        "ndof": ndof,
        "fit_start_threshold": float(fit_start_threshold),
    }


def choose_best_fit_start(thresholds, rates, sigmas, candidates):
    best = None

    for fs in candidates:
        fit = weighted_log_fit(thresholds, rates, sigmas, fs)

        if fit is None or fit["ndof"] <= 0:
            continue

        if best is None or fit["chi2"] < best["chi2"]:
            best = fit

    return best


def threshold_at_target_with_error(fit, target_hz):
    """
    target = A exp(k x)
    x = (ln(target) - lnA) / k
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

    J = np.array([d_lnA, d_k])
    var_x = float(J.T @ cov @ J)
    sigma_x = np.sqrt(max(var_x, 0.0))

    return float(x), float(sigma_x)


def main():
    meta, records = load_scan_json(IN_JSON)

    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError("JSON file is empty or not in expected list format.")

    thresholds, rates, sigmas = prepare_dataset(records)

    best_fit = choose_best_fit_start(
        thresholds,
        rates,
        sigmas,
        FIT_START_CANDIDATES
    )

    if best_fit is None:
        raise RuntimeError("Fit failed. Not enough valid nonzero-rate points.")

    threshold_5hz, threshold_5hz_err = threshold_at_target_with_error(
        best_fit,
        TARGET_HZ
    )

    # Make fit curve
    xmin = float(np.min(thresholds))
    xmax = float(np.max(thresholds))

    if threshold_5hz is not None and np.isfinite(threshold_5hz):
        xmax = max(xmax, threshold_5hz)

    span = max(1.0, xmax - xmin)
    xgrid = np.linspace(xmin, xmax + 0.1 * span, 500)
    ygrid = best_fit["A"] * np.exp(best_fit["k"] * xgrid)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 7))

    pos = (rates > 0) & np.isfinite(rates) & np.isfinite(sigmas)

    plt.errorbar(
        thresholds[pos],
        rates[pos],
        yerr=sigmas[pos],
        fmt="o",
        ms=5,
        capsize=2,
        alpha=0.7,
        label="data"
    )

    fit_start = best_fit["fit_start_threshold"]
    last_data = float(np.max(thresholds[pos]))

    fitted_region = (xgrid >= fit_start) & (xgrid <= last_data)
    extrap_region = (xgrid < fit_start) | (xgrid > last_data)

    plt.plot(
        xgrid[fitted_region],
        ygrid[fitted_region],
        lw=2,
        label=f"fit, start ≥ {fit_start:g}"
    )

    plt.plot(
        xgrid[extrap_region],
        ygrid[extrap_region],
        lw=2,
        linestyle="--",
        alpha=0.8,
        label="extrapolation"
    )

    plt.axhline(
        TARGET_HZ,
        linestyle="--",
        color="tab:red",
        label=f"Target = {TARGET_HZ:.1f} Hz"
    )

    if threshold_5hz is not None and np.isfinite(threshold_5hz):
        plt.axvline(threshold_5hz, linestyle="--", color="tab:green")
        plt.scatter([threshold_5hz], [TARGET_HZ], marker="X", s=70, color="tab:green")

        text = f"{threshold_5hz:.2e}"
        if threshold_5hz_err is not None and np.isfinite(threshold_5hz_err):
            text = f"{threshold_5hz:.2e} ±{threshold_5hz_err:.2e}"

        plt.text(
            threshold_5hz,
            TARGET_HZ * 1.2,
            text,
            color="tab:green",
            va="bottom",
            ha="left"
        )

    plt.yscale("log")
    plt.xlabel("Threshold (CSW units)", fontsize=14)
    plt.ylabel("Trigger rate (Hz, log scale)", fontsize=14)
    plt.title(f"{LABEL}\nTrigger Rate vs Threshold")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=12)

    if X_AXIS_START is not None:
        plt.xlim(left=float(X_AXIS_START))

    if Y_AXIS_BOTTOM is not None or Y_AXIS_TOP is not None:
        plt.ylim(bottom=Y_AXIS_BOTTOM, top=Y_AXIS_TOP)

    plt.tight_layout()

    out_plot = OUT_DIR / "threshold_scan_BEAM_trigger_rate.png"
    plt.savefig(out_plot, dpi=300)
    plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Save summary
    # ─────────────────────────────────────────────────────────────────────────
    summary = {
        "input_json": str(IN_JSON),
        "label": LABEL,
        "event_ns": EVENT_NS,
        "target_hz": TARGET_HZ,
        "fit_start_candidates": FIT_START_CANDIDATES,
        "meta_from_scan_file": meta,
        "num_points_total": int(len(thresholds)),
        "fit_start_threshold": best_fit["fit_start_threshold"],
        "A": best_fit["A"],
        "k": best_fit["k"],
        "chi2": best_fit["chi2"],
        "ndof": best_fit["ndof"],
        "chi2_reduced": (
            best_fit["chi2"] / best_fit["ndof"]
            if best_fit["ndof"] > 0 else None
        ),
        "threshold_at_5hz": threshold_5hz,
        "threshold_at_5hz_err": threshold_5hz_err,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best fit start threshold: {best_fit['fit_start_threshold']}")
    print(f"Threshold at {TARGET_HZ:.1f} Hz: {threshold_5hz:.4f}")

    if threshold_5hz_err is not None:
        print(f"Uncertainty: ± {threshold_5hz_err:.4f}")

    print(f"Saved plot to: {out_plot}")
    print(f"Saved summary JSON to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()