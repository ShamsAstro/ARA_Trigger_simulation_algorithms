import json
import numpy as np
from pathlib import Path    
import matplotlib.pyplot as plt
import random
import math
import os
import sys
from scipy.optimize import curve_fit
from scipy.signal import firwin, lfilter

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

def angle_delay_time(angle ):
    angle=np.deg2rad(angle)
    # === Physics ===
    n_ice= 1.75  #index of refraction in ice
    vertical_seperation= 1 #distance betwwen channel in meters
    c= 299792458 #speed of light in a vaccum in m/s
    
    time_delays= n_ice * vertical_seperation * np.sin(angle) / c
    return -time_delays*1e9 # ns  # negative because the pulse is delayed, not advanced and ch3 is closes to surface


def envelope_with_edge_rules(x: np.ndarray, window_points: int = 10) -> np.ndarray:
    """
    Envelope smoothing by rolling average with custom edge handling.

    Parameters
    ----------
    x : 1D array
        Input signal.
    window_points : int, default=10
        Window length used in the center region. Must be even for
        symmetric behavior (e.g. 10 → 5 before, 5 after).

    Returns
    -------
    y : 1D array
        Smoothed envelope.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    if window_points % 2 != 0:
        raise ValueError("window_points should be even (e.g. 10).")

    half = window_points // 2

    # cumulative sum for fast range means
    cs = np.empty(n + 1, dtype=float)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])

    y = np.empty_like(x)

    # 1) start edge: first `half` samples → mean of next `half` points
    for i in range(min(half, n)):
        r = min(i + half, n)
        y[i] = (cs[r] - cs[i]) / max(r - i, 1)

    # 2) center region: i = half .. n-half-1 → mean over [i-half, i+half)
    left = half
    right = n - half
    if right > left:
        i_center = np.arange(left, right, dtype=int)
        sums = cs[i_center + half] - cs[i_center - half]
        y[left:right] = sums / float(window_points)

    # 3) end edge: last `half` samples → mean of previous `half` points
    for i in range(max(n - half, 0), n):
        a = max(i - (half - 1), 0)
        b = i + 1
        y[i] = (cs[b] - cs[a]) / max(b - a, 1)

    return y


def find_ARA_env_triggers(
    channel_signals,
    time_axis,
    *,
    threshold,
    n_channels_required=3,
    envelope_window_points=10,   # fixed to 10 per your spec; kept for clarity
    min_separation_ns=0.0        # optional: ignore triggers closer than this
):
    """
    Envelope-based coincidence trigger:

    Steps per channel:
      1) square the signal (power)
      2) apply 'envelope' = rolling average with 10-point center window
         and special edge handling:
           - first 5 samples: mean of next 5 points
           - last 5 samples: mean of previous 5 points
      3) compare envelope to per-channel threshold

    A trigger occurs at the earliest sample index where the number of
    channels above threshold >= n_channels_required. We report one event
    per 'rising edge' of that condition and (optionally) enforce a
    minimum separation between reported triggers.

    Returns
    -------
    triggers : list[dict]
       Each: {"t_trigger": float, "channels": list[int]}
       'channels' lists the channels above threshold at that sample.
    """
    # --- inputs to arrays ---
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return []
    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")

    thr = np.asarray(threshold, dtype=float)
    if thr.size == 1:
        thr = np.repeat(thr, n_ch)
    if thr.size != n_ch:
        raise ValueError("threshold must be scalar or length == n_channels.")

    # --- 1) power and 2) envelope (vectorized across channels) ---
    # Build 2D array (n_ch, N)
    P = np.empty((n_ch, N), dtype=float)
    for ch in range(n_ch):
        # square
        p = X[ch] * X[ch]
        # envelope with custom 10-point rule
        P[ch] = envelope_with_edge_rules(p)

    # --- 3) thresholding per channel ---
    # Boolean mask: shape (n_ch, N)
    above = P >= thr[:, None]

    # Count how many channels are above at each time sample
    count = np.sum(above, axis=0)  # shape (N,)

    # Find “rising edges” where count crosses from <n_req to >=n_req
    cond = count >= int(n_channels_required)
    if not np.any(cond):
        return []

    rising = np.flatnonzero(cond & ~np.r_[False, cond[:-1]])

    # Optionally enforce a minimum time separation between triggers
    if rising.size and min_separation_ns > 0.0:
        kept = [rising[0]]
        last_t = t[rising[0]]
        for idx in rising[1:]:
            if (t[idx] - last_t) >= min_separation_ns:
                kept.append(idx)
                last_t = t[idx]
        rising = np.asarray(kept, dtype=int)

    # Build outputs; for each trigger index, list which channels are above
    triggers = []
    for idx in rising:
        chans = np.flatnonzero(above[:, idx]).tolist()
        triggers.append({"t_trigger": float(t[idx]), "channels": chans})
    return triggers


def TOT_finder(
    channel_signals,
    time_axis,
    *,
    threshold,
    n_channels_required=2
):
    """
    Time-Over-Threshold (TOT) for the FIRST coincidence event using the same
    envelope as your trigger path (_envelope_10_with_edge_rules).

    Inputs (same style as your trigger):
      channel_signals : list/array of shape (n_ch, N)
      time_axis       : 1D array of length N
      threshold       : scalar OR array-like length n_ch (POWER thresholds)
      n_channels_required : int

    Returns
    -------
    TOT_samples : int
        Number of consecutive samples, starting at the first trigger sample,
        for which the coincidence condition remains true.
    channels_triggered : int
        Number of channels above threshold at that trigger sample.
    """
    # ---- validate shapes ----
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return 0, 0
    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")

    thr = np.asarray(threshold, dtype=float)
    if thr.size == 1:
        thr = np.repeat(thr, n_ch)
    if thr.size != n_ch:
        raise ValueError("threshold must be scalar or length == n_channels.")

    # ---- square → envelope (using the SAME helper) ----
    # NOTE: thresholds must be in POWER units (since we square).
    P = np.empty((n_ch, N), dtype=float)
    for ch in range(n_ch):
        p = X[ch] * X[ch]  # power
        P[ch] = envelope_with_edge_rules(p)

    # ---- threshold per channel → coincidence condition ----
    above = P >= thr[:, None]       # (n_ch, N) bool
    count = np.sum(above, axis=0)   # (N,)
    cond  = count >= int(n_channels_required)

    if not np.any(cond):
        return 0, 0

    # first rising edge of coincidence
    rising = np.flatnonzero(cond & ~np.r_[False, cond[:-1]])
    if rising.size == 0:
        return 0, 0
    r0 = int(rising[0])

    # channels above at trigger sample
    channels_triggered = int(np.sum(above[:, r0]))

    # TOT: consecutive True samples from r0 onward
    seg = cond[r0:]
    end_false = np.flatnonzero(~seg)
    TOT_samples = int(end_false[0]) if end_false.size else int(seg.size)

    return TOT_samples, channels_triggered

def fit_sigmoid_get_b(snr, passfrac):
    params, _ = curve_fit(sigmoid, snr, passfrac, p0=[1, np.mean(snr)])
    a, b = params
    return a, b


#Phased trigger functions:

def shift_by_samples(sig, shift_samp):
    """
    Integer sample shift with symmetric zero padding.
    shift_samp > 0 delays in time. shift_samp < 0 advances.
    """
    L = sig.shape[0]
    pad = np.zeros(L, dtype=sig.dtype)
    ext = np.concatenate([pad, sig, pad])      # safe index range: [0 .. 3L-1]
    idx = np.arange(L) - int(shift_samp) + L   # map 0..L-1 into ext
    return ext[idx]

def per_channel_delay_ns(angle_deg, ch_idx):
    """
    Your firmware rule for geometric per-channel delays in ns.
    Uses global angle_delay_time(angle_deg).
    """
    step_ns = angle_delay_time(angle_deg)      # ns per channel step
    if step_ns < 0:
        return 3 * abs(step_ns) + step_ns * ch_idx
    return step_ns * ch_idx

def de_shifter(sig_up, angle_deg, ch_idx, dt_up):
    """
    De-shift one upsampled channel to ALIGN a plane wave from angle_deg.
    Uses integer-sample shift on the upsampled grid (no interpolation).
    """
    d_ns   = per_channel_delay_ns(angle_deg, ch_idx)   # arrival delay
    shift  = int(np.rint(-d_ns / dt_up))               # advance by delay to align
    return shift_by_samples(sig_up, shift)

def window_power(segment, division_factor):
    """One-liner: sum of squares with scaling."""
    return float(np.dot(segment, segment)) / float(division_factor)

def iter_overlapping_windows(x, window_size, step):
    """
    Yield (start_idx, center_idx, segment_view) for overlapping windows.
    window_size and step are integers in upsampled samples.
    """
    W = int(window_size)
    S = int(step)
    if W <= 0 or S <= 0 or len(x) < W:
        return
    last_start = len(x) - W
    for s in range(0, last_start + 1, S):
        c = s + W // 2
        yield s, c, x[s:s+W]

def scan_beam_for_triggers(beam, t_up, *, threshold, window_size, window_step, division_factor):
    """
    Slide overlapping windows across 'beam', compute power per window,
    and return list of (t_trigger, power_value) where power >= threshold.
    """
    hits = []
    for s, cidx, seg in iter_overlapping_windows(beam, window_size, window_step):
        p = window_power(seg, division_factor)
        if p >= threshold:
            hits.append((float(t_up[min(cidx, len(t_up)-1)]), float(p)))
    return hits

# ───────────────────── main phased trigger ─────────────────────

def find_phased_triggers(channel_signals, time_axis, phased_trigger_parameters):
    """
    Phased-array trigger (modular):
      1) Upsample each channel by UPSAMPLE_FACTOR (zero-stuff + FIR LPF with quantized taps)
      2) For each beam angle, de-shift each channel with generate_pulse-like integer shifting
      3) Coherent sum
      4) Overlapping window power (sum of squares / DIV)
      5) Trigger if any window power >= threshold

    Returns a list of dicts:
      {"t_trigger": float, "channels": [0,1,2,3], "beam_angle": float, "beam_index": int, "power_value": float}
    """
    (PHASED_THRESHOLD,
     UPSAMPLE_FACTOR,
     PHASED_BEAMS,
     POWER_WINDOW_SIZE,
     POWER_WINDOW_STEP,
     POWER_DIVISION_FACTOR) = phased_trigger_parameters

    n_ch = len(channel_signals)

    dt_ns  = float(time_axis[1] - time_axis[0])
    fs_orig = 1.0 / dt_ns
    fs_up   = fs_orig * UPSAMPLE_FACTOR
    dt_up   = dt_ns / UPSAMPLE_FACTOR


    taps = firwin(45, cutoff=fs_orig * 0.5, pass_zero='lowpass', fs=fs_up)
    taps = np.round(taps * 256) / 256.0
    up_ch = []
    for ch in range(n_ch):
        x = np.asarray(channel_signals[ch], dtype=float)
        up = np.zeros(len(x) * UPSAMPLE_FACTOR, dtype=float)
        up[::UPSAMPLE_FACTOR] = x
        up_filt = lfilter(taps, [1.0], up) * UPSAMPLE_FACTOR  # upsampled + FIR LPF
        up_ch.append(up_filt)
    up_ch = np.asarray(up_ch)              # (n_ch, n_up)
    n_up  = up_ch.shape[1]
    t0   = float(time_axis[0])
    t_up = t0 + np.arange(n_up) * dt_up

    triggers = []

    # 2..5) Loop over beams
    for b_idx, ang in enumerate(PHASED_BEAMS):
        # De-shift every channel to align arrivals for beam 'ang'
        aligned = np.empty((n_ch, n_up), dtype=float)
        for ch in range(n_ch):
            aligned[ch] = de_shifter(up_ch[ch], ang, ch, dt_up)

        # Coherent sum
        beam = aligned.mean(axis=0)

        # Overlapping window power and thresholding
        hits = scan_beam_for_triggers(
            beam, t_up,
            threshold=PHASED_THRESHOLD,
            window_size=POWER_WINDOW_SIZE,
            window_step=POWER_WINDOW_STEP,
            division_factor=POWER_DIVISION_FACTOR
        )

        for t_hit, pval in hits:
            triggers.append({
                "t_trigger": t_hit,
                "channels": list(range(n_ch)),    # PA uses all channels
                "beam_angle": float(ang),
                "beam_index": int(b_idx),
                "power_value": pval,
            })

    triggers.sort(key=lambda tr: tr["t_trigger"])
    return triggers

































