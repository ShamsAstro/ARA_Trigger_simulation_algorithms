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
    envelope_window_points=10,   # kept for interface; envelope helper is fixed to 10
    min_separation_ns=0.0        # ignored in event-wide mode (one trigger max)
):
    """
    EVENT-WIDE envelope trigger (no simultaneity requirement):
    - Square each channel, apply the same 10-point edge-handled envelope.
    - A channel 'fires' if its envelope exceeds its threshold at ANY time in the event.
    - If >= n_channels_required channels fire anywhere in the event, we report ONE trigger:
        * t_trigger = earliest threshold-crossing time among all fired channels
        * channels  = list of all channels that fired anywhere in the event

    Returns
    -------
    triggers : list[dict]  (length 0 or 1)
        Each: {"t_trigger": float, "channels": list[int]}
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

    # --- power + envelope (fixed 10-point, edge rules) ---
    P = np.empty((n_ch, N), dtype=float)
    for ch in range(n_ch):
        p = X[ch] * X[ch]
        P[ch] = envelope_with_edge_rules(p)  # same helper used in your path

    # --- channel 'fires' if it ever exceeds its own threshold in the event ---
    above = P >= thr[:, None]              # (n_ch, N) bool
    channel_hit = np.any(above, axis=1)    # (n_ch,) bool

    fired_channels = np.flatnonzero(channel_hit)
    if fired_channels.size < int(n_channels_required):
        return []

    # earliest crossing time among ALL fired channels
    first_idxs = []
    for ch in fired_channels:
        idxs = np.flatnonzero(above[ch])
        if idxs.size:
            first_idxs.append(idxs[0])
    if not first_idxs:
        return []

    r0 = int(min(first_idxs))
    return [{
        "t_trigger": float(t[r0]),
        "channels": fired_channels.tolist()
    }]



def TOT_finder(
    channel_signals,
    time_axis,
    *,
    threshold,
    n_channels_required=2  # kept for API compatibility; not used here
):
    """
    Event-wide per-channel TOT (samples), averaged over triggered channels.

    Steps:
      - Square each channel and apply the same 10-point envelope with edge rules.
      - For each channel, compute the longest consecutive run of samples where
        envelope >= that channel's threshold.
      - Consider only channels with a nonzero run (i.e., that ever crossed).
      - Return (average of per-channel maxima in samples, number of triggered channels).

    Returns
    -------
    TOT_avg_samples : float
        Average of the per-channel longest runs (in samples) over channels that fired.
        0.0 if no channel crossed threshold.
    channels_triggered : int
        Number of channels that had at least one sample above threshold.
    """

    # ---- validate shapes ----
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return 0.0, 0
    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")

    thr = np.asarray(threshold, dtype=float)
    if thr.size == 1:
        thr = np.repeat(thr, n_ch)
    if thr.size != n_ch:
        raise ValueError("threshold must be scalar or length == n_channels.")

    # ---- square → envelope (same helper as trigger path) ----
    P = np.empty((n_ch, N), dtype=float)
    for ch in range(n_ch):
        p = X[ch] * X[ch]                 # power
        P[ch] = envelope_with_edge_rules(p)

    # ---- per-channel longest run of True in (envelope >= threshold) ----
    def longest_true_run(mask: np.ndarray) -> int:
        """Length (in samples) of the longest consecutive True run in a 1D bool mask."""
        if not np.any(mask):
            return 0
        m = mask.astype(np.int8)
        d = np.diff(np.r_[0, m, 0])
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1)
        return int(np.max(ends - starts))  # samples

    runs = np.empty(n_ch, dtype=int)
    for ch in range(n_ch):
        above = P[ch] >= thr[ch]
        runs[ch] = longest_true_run(above)

    # ---- average over channels that actually crossed ----
    triggered_mask = runs > 0
    channels_triggered = int(np.sum(triggered_mask))
    if channels_triggered == 0:
        return 0.0, 0

    TOT_avg_samples = float(np.mean(runs[triggered_mask]))
    return TOT_avg_samples, channels_triggered


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

































