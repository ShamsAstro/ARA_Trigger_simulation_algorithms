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
from scipy.signal import fftconvolve

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

def _parabolic_interp(y, i):
    """Quadratic (parabolic) peak interpolation around index i. Returns sub-sample offset."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[i-1], y[i], y[i+1]
    denom = (y0 - 2*y1 + y2)
    return 0.5 * (y0 - y2) / denom if denom != 0 else 0.0

def normalized_xcorr_fft(x, y):
    """
    Normalized cross-correlation via FFT. 
    Returns (corr, lags), where corr is scaled in [-1, 1]-ish if lengths match.
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    # energy normalization (global, good when signals are same length)
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        lags = np.arange(-len(y) + 1, len(x))
        return np.zeros_like(lags, dtype=float), lags

    # correlate: x (*) y  ==  conv(x, flip(y))
    r = fftconvolve(x, y[::-1], mode='full') / denom
    lags = np.arange(-len(y) + 1, len(x))
    return r, lags

def estimate_peak_lag(x, y, max_lag_samples=None, sub_sample=True):
    """
    Estimate lag (in samples) that maximizes normalized cross-correlation of x vs y.
    Positive lag means y must be shifted RIGHT to align with x (i.e., y(t+lag) ~ x(t)).
    """
    r, lags = normalized_xcorr_fft(x, y)
    if max_lag_samples is not None:
        m = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
        r, lags = r[m], lags[m]

    k = int(np.argmax(r))
    lag = float(lags[k])

    if sub_sample and 0 < k < len(r) - 1:
        lag += _parabolic_interp(r, k)  # fractional correction

    return lag, r[k]

def fractional_delay_fft(x, delay_samples):
    """
    Apply a fractional delay using frequency-domain phase rotation.
    delay_samples can be fractional. Returns a time-shifted copy of x, same length.
    Notes:
      - This is circular for non-zero edges. If you worry about wrap-around, pad x with zeros,
        shift, then crop (see 'safe_fractional_delay_fft' below).
    """
    N = len(x)
    X = np.fft.rfft(x)
    k = np.arange(len(X))
    # e^{-j 2π k delay / N}
    phasor = np.exp(-2j * np.pi * k * delay_samples / N)
    Y = X * phasor
    y = np.fft.irfft(Y, n=N)
    return y

def safe_fractional_delay_fft(x, delay_samples, pad=0):
    """
    Safer variant: zero-pad to reduce circular wrap-around, then crop back.
    Useful if non-zero content touches edges.
    """
    if pad <= 0:
        return fractional_delay_fft(x, delay_samples)
    x_pad = np.pad(x, (pad, pad), mode='constant')
    Np = len(x_pad)
    X = np.fft.rfft(x_pad)
    k = np.arange(len(X))
    phasor = np.exp(-2j * np.pi * k * delay_samples / Np)
    Y = X * phasor
    y_pad = np.fft.irfft(Y, n=Np)
    return y_pad[pad:-pad]

# ---------- One-call alignment for a whole event ----------

def align_channels_fft_xcorr(
    X, 
    fs=None, 
    ref_idx=0, 
    max_lag_s=None, 
    sub_sample=True, 
    fractional=True, 
    edge_pad=0
):
    """
    Align channels to a reference using normalized FFT cross-correlation peak lag.

    Parameters
    ----------
    X : array, shape (nch, nsamp)
        Multi-channel waveforms (one event).
    fs : float or None
        Sampling rate [Hz]. Only used to convert max_lag_s to samples and to report lags in seconds.
    ref_idx : int
        Index of reference channel.
    max_lag_s : float or None
        Limit search to +/- max_lag_s seconds (converted to samples). If None, search all.
    sub_sample : bool
        Use quadratic interpolation for sub-sample lag estimate.
    fractional : bool
        If True, apply fractional delay (frequency-domain) when shifting. If False, round to nearest sample and use np.roll.
    edge_pad : int
        Zero padding on both sides before fractional delay to reduce circular artifacts (safe shift). 0 = no pad.

    Returns
    -------
    X_aligned : array, shape (nch, nsamp)
        Aligned waveforms.
    lags_samples : array, shape (nch,)
        Estimated lag (in samples) each channel should be shifted RIGHT by to align to ref (ref lag = 0).
    lags_seconds : array, shape (nch,)
        Same in seconds (NaN if fs is None).
    corr_peaks : array, shape (nch,)
        Peak normalized cross-correlation values (rough alignment quality metric).
    """
    X = np.asarray(X)
    nch, ns = X.shape
    lags_samples = np.zeros(nch, dtype=float)
    corr_peaks = np.zeros(nch, dtype=float)

    if fs is not None and max_lag_s is not None:
        max_lag_samples = int(np.floor(max_lag_s * fs))
    else:
        max_lag_samples = None

    x_ref = X[ref_idx]

    # 1) estimate lags wrt reference
    for i in range(nch):
        if i == ref_idx:
            corr_peaks[i] = 1.0
            continue
        lag_i, peak = estimate_peak_lag(
            x_ref, X[i], 
            max_lag_samples=max_lag_samples, 
            sub_sample=sub_sample
        )
        lags_samples[i] = lag_i
        corr_peaks[i] = float(peak)

    # 2) apply shifts
    X_aligned = np.zeros_like(X)
    for i in range(nch):
        if i == ref_idx:
            X_aligned[i] = X[i] - np.mean(X[i])  # also center it
            continue
        if fractional:
            X_aligned[i] = safe_fractional_delay_fft(X[i] - np.mean(X[i]), lags_samples[i], pad=edge_pad)
        else:
            # integer-sample fallback (fastest)
            X_aligned[i] = np.roll(X[i] - np.mean(X[i]), int(np.round(lags_samples[i])))

    # 3) seconds units if fs is known
    if fs is not None:
        lags_seconds = lags_samples / fs
    else:
        lags_seconds = np.full(nch, np.nan, dtype=float)

    return X_aligned, lags_samples, lags_seconds, corr_peaks

# ---------- Optional: coherent-sum (CSW) helper ----------

def coherent_sum(X_aligned):
    """
    Simple delay-and-sum beamformer (coherent sum).
    Returns coherent waveform and its instantaneous power.
    """
    csw = np.sum(X_aligned, axis=0)
    power = csw**2
    return csw, power

def ARA_CSW_trigger_no_shifting(
    channel_signals,
    time_axis,
    *,
    threshold,
    noise_rms
):
    """
    EVENT-WIDE CSW (Coherent Sum Window) trigger:
      1) For each channel, find the index of its maximum *magnitude* sample.
      2) Roll the waveform so that this max aligns at the center sample.
         (Samples shifted past one edge reappear on the other edge.)
      3) Coherently sum all aligned waveforms sample-by-sample.
      4) Compute the event total CSW power: sum_t [ (sum_ch x_ch(t))^2 ].
      5) Compare to an effective (single) threshold:
             power_threshold = threshold * len(time_axis) * noise_rms**2
         If CSW power exceeds this, report ONE trigger.
    Returns
    -------
    triggers : list[dict]  (length 0 or 1)
        Each: {"t_trigger": float, "channels": list[int]}
        - t_trigger is the time at the center sample after alignment.
        - channels are all channels (indices) used in CSW.
    """
    # --- inputs -> arrays ---
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return []
    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")
    # --- scalars ---
    try:
        thr = float(threshold)
    except Exception as e:
        raise ValueError("threshold must be a scalar float-like value.") from e
    try:
        sigma_n = float(noise_rms)
    except Exception as e:
        raise ValueError("noise_rms must be a scalar float-like value.") from e
    # --- align (roll) each channel so that |x| maximum is at the center ---

    aligned = X
    # --- coherent sum across channels, then power trace and total event power ---
    s = np.sum(aligned, axis=0)  # coherent sum
    csw_power_trace = s * s
    
    
    # --- effective threshold ---
    power_threshold = float(thr *(sigma_n**2))
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, csw_power_trace, label='CSW Power Trace')
    plt.xlabel('Time (ns)')
    plt.axhline(y=power_threshold, color='r', linestyle='--', label='CSW Power Threshold')
    plt.ylabel('CSW Power')
    plt.title('CSW Power Trace vs Time')
    plt.legend()
    plt.grid()
    plt.show()
    """
    # --- decision ---
    if np.max(csw_power_trace)<=power_threshold:
        return []

    # Report ONE trigger: center time after alignment
    t_center = float(t[mid])
    fired_channels = list(range(n_ch))
    return [{
        "t_trigger": t_center,
        "channels": fired_channels
    }]

def ARA_CSW_trigger_FFT_optimized(
    channel_signals,
    time_axis,
    *,
    threshold,
    noise_rms
):

       # --- inputs -> arrays ---
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return []

    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")

    # --- scalars ---
    try:
        thr = float(threshold)
    except Exception as e:
        raise ValueError("threshold must be a scalar float-like value.") from e

    try:
        sigma_n = float(noise_rms)
    except Exception as e:
        raise ValueError("noise_rms must be a scalar float-like value.") from e

    mid = N // 2
    #scan range to take 120ns
    scan_lim = int(120/(t[1]-t[0])) #120 ns is the longest possible time shift between two channels

    # --- pick reference channel: largest absolute peak amplitude ---
    peak_vals = [np.max(x)-np.min(x) for x in X]
    ref_idx = int(np.argmax(peak_vals))
    ref = X[ref_idx]

    # --- center the reference channel at mid using its |max| position ---
    ref_kmax = int(np.argmax(np.abs(ref)))
    ref_center_shift = mid - ref_kmax   
    ref_centered = np.roll(ref, ref_center_shift)
    shift_centers = []
    
    # --- for each channel, find the best roll (every 2 samples) vs centered reference ---
    X_aligned, lags_samp, lags_sec, corr_pk = align_channels_fft_xcorr(
        X, 
        fs=None, 
        ref_idx=ref_idx,          # pick your best SNR channel
        max_lag_s=15e-8,           # example: ±150 ns window
        sub_sample=True, 
        fractional=True, 
        edge_pad=64               # small pad helps avoid circular wrap
    )
    shift_centers = lags_samp.tolist()
    #remove the 0 values from shift_centers
    shift_centers = [int(shift) for shift in shift_centers if shift != 0]

    csw, csw_power = coherent_sum(X_aligned)

    # Now you can threshold on csw_power max, or integrate over a small gate, etc.
    #peak_power = np.max(csw_power)

    #divide the csw power into 20 sections, and taking the mean of the sections
    csw_power_sections = np.array_split(csw_power, 10)
    peak_power = np.max([np.mean(section) for section in csw_power_sections])
    
    power_threshold = float(thr * (sigma_n ** 2)) 

    # --- plot for debugging ---
    """
    
    
    #save name ref
    save_name_ref= str(round(t[0],1)) + '_'+ str(round(t[-1],1)) + '_refch'+ str(ref_idx)+'_CSW_FFT_trigger.png'
    plt.figure(figsize=(10, 6))
    plt.plot(t, csw_power, label='CSW Power Trace')
    plt.axhline(y=power_threshold, linestyle='--', label='CSW Power Threshold', color='red')
    plt.xlabel('Time (ns)')
    plt.ylabel('CSW Power')
    plt.title('CSW Power Trace vs Time')
    plt.legend()
    plt.grid()
    #plt.savefig(save_name_ref)
    plt.show()
    #plt.close()
    """
    
    # --- decision (single event-wide trigger or none) ---
    if peak_power <= power_threshold:
        return []

    t_center = float(t[mid])
    fired_channels = list(range(n_ch))
    return [{
        "Shifts": shift_centers,
        "channels": fired_channels
    }]

def ARA_CSW_trigger(
    channel_signals,
    time_axis,
    *,
    threshold,
    noise_rms,
    STEP=3
):
    """
    EVENT-WIDE CSW (Coherent Sum Window) trigger with correlation-based alignment:
      0) Choose reference channel = channel with highest absolute peak amplitude.
      1) Center the reference by rolling so its |max| sample is at the middle index.
      2) For each other channel, find the roll (tested every 2 samples) that maximizes
         its correlation (dot product) with the centered reference, then apply that roll.
         (Rolling is circular, i.e., samples shifted past one edge reappear on the other.)
      3) Coherently sum aligned waveforms sample-by-sample.
      4) Compute event CSW power: sum_t [ (sum_ch x_ch(t))^2 ] (we keep the trace too).
      5) Compare to a single effective threshold, same convention as your code:
             power_threshold = threshold * (noise_rms**2)
         If max power in the trace exceeds this, report ONE trigger.

    Returns
    -------
    triggers : list[dict]  (length 0 or 1)
        Each: {"t_trigger": float, "channels": list[int]}
        - t_trigger is the time at the center sample after alignment.
        - channels are all channels (indices) used in CSW.
    """
    # --- inputs -> arrays ---
    t = np.asarray(time_axis, dtype=float)
    X = [np.asarray(sig, dtype=float) for sig in channel_signals]
    n_ch = len(X)
    if n_ch == 0:
        return []

    N = X[0].size
    if any(x.size != N for x in X) or t.size != N:
        raise ValueError("All channels and time_axis must have the same length.")

    # --- scalars ---
    try:
        thr = float(threshold)
    except Exception as e:
        raise ValueError("threshold must be a scalar float-like value.") from e

    try:
        sigma_n = float(noise_rms)
    except Exception as e:
        raise ValueError("noise_rms must be a scalar float-like value.") from e

    mid = N // 2
    #scan range to take 169ns
    scan_lim = int(169/(t[1]-t[0])) #169 ns is the longest possible time shift between two channels

    # --- pick reference channel: largest absolute peak amplitude ---
    peak_vals = [np.max(x)-np.min(x) for x in X]
    ref_idx = int(np.argmax(peak_vals))
    ref = X[ref_idx]

    # --- center the reference channel at mid using its |max| position ---
    ref_kmax = int(np.argmax(np.abs(ref)))
    ref_center_shift = mid - ref_kmax   
    ref_centered = np.roll(ref, ref_center_shift)
    shift_centers = []
    # --- for each channel, find the best roll (every 2 samples) vs centered reference ---
    aligned = np.empty((n_ch, N), dtype=float)
    for ch in range(n_ch):
        x = X[ch]

        if ch == ref_idx:
            aligned[ch] = ref_centered
            continue

        best_shift = 0
        best_corr = -np.inf

        # Try all circular rolls s in [-N+1, ..., N-1] stepping by 2 samples
        # We include the reference centering so everything ends up aligned to mid.
        for s in range(-(scan_lim- 1), scan_lim, STEP):
            x_rolled = np.roll(x, ref_center_shift + s)
            # Dot product with reference; normalization is unnecessary for a fixed x across s
            corr = float(np.dot(x_rolled, ref_centered))
            if corr > best_corr:
                best_corr = corr
                best_shift = s
                
        shift_centers.append(best_shift)
        aligned[ch] = np.roll(x, ref_center_shift + best_shift)

    # --- coherent sum, power trace, threshold ---
    s = np.sum(aligned, axis=0)
    csw_power_trace = s * s

    power_threshold = float(thr * (sigma_n ** 2)) 

    # --- plot for debugging ---
    """
    
    #save name ref
    save_name_ref= str(round(t[0],1)) + '_'+ str(round(t[-1],1)) + '_refch'+ str(ref_idx)+'_CSW_trigger.png'
    plt.figure(figsize=(10, 6))
    plt.plot(t, csw_power_trace, label='CSW Power Trace')
    plt.axhline(y=power_threshold, linestyle='--', label='CSW Power Threshold', color='red')
    plt.xlabel('Time (ns)')
    plt.ylabel('CSW Power')
    plt.title('CSW Power Trace vs Time')
    plt.legend()
    plt.grid()
    plt.savefig(save_name_ref)
    plt.clf()
    """
    # --- decision (single event-wide trigger or none) ---
    if np.max(csw_power_trace) <= power_threshold:
        return []

    t_center = float(t[mid])
    fired_channels = list(range(n_ch))
    return [{
        "Shifts": shift_centers,
        "channels": fired_channels
    }]



def TOT_finder_mod(
    channel_signals,
    time_axis,
    *,
    threshold,
    n_channels_required=2,
    env_parameter=10  # kept for API compatibility; not used here
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
        P[ch] = envelope_with_edge_rules(p, window_points=env_parameter)

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

































