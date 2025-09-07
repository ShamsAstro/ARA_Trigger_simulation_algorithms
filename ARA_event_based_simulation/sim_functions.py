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

def make_band_limited_noise(json_path,
                            channel_key="ch0",
                            window_ns=1000.0,  # length of the output trace in ns
                            adc_rate_ghz=0.472,   # hardware sampling rate
                            oversample=1,         # use >1 for fractional-ns step
                            target_rms_mV=1.0,
                            rng=None):
    """
    Return (time_ns, noise_mV) for one random noise realisation whose
    spectrum follows the magnitude in `json_path[channel_key]`.

    Parameters
    ----------
    json_path : str | Path
        Path to impulse-response JSON containing keys 'freq_GHz' + channels.
    channel_key : str
        Which channel’s magnitude to use as the band-pass shape.
    window_ns : float
        Length of the output trace in nanoseconds.
    adc_rate_ghz : float
        Native ADC rate of the system (GHz).
    oversample : int
        Upsampling factor applied before the FFT (Δt = 1/(adc_rate*oversample)).
    target_rms_mV : float
        RMS of the returned waveform, in millivolts.
    rng : numpy.random.Generator | None
        Source of randomness (defaults to np.random.default_rng()).

    Returns
    -------
    time_ns : 1-D ndarray
        Time axis in nanoseconds.
    noise_mV : 1-D ndarray
        Band-limited noise in millivolts, rms ≈ `target_rms_mV`.
    """
    # ---------- load magnitude response ------------
    data = json.loads(Path(json_path).read_text())
    freq_ref   = np.asarray(data["freq_GHz"])        # GHz
    mag_ref    = np.asarray(data[channel_key])

    # ---------- define FFT grid --------------------
    dt_ns  = 1.0 / (adc_rate_ghz * oversample)       # ns
    N      = int(round(window_ns / dt_ns))
    dt_s   = dt_ns * 1e-9
    freq_Hz = np.fft.rfftfreq(N, d=dt_s)
    freq_GHz = freq_Hz / 1e9

    # ---------- interpolate → Rayleigh σ -----------
    sigma = np.interp(freq_GHz, freq_ref, mag_ref, left=0.0, right=0.0)

    # ---------- draw random spectrum ---------------
    rng = rng or np.random.default_rng()
    amp   = rng.rayleigh(scale=sigma)
    phase = rng.uniform(0, 2*np.pi, size=amp.size)
    spec  = amp * np.exp(1j * phase)

    # ---------- IFFT to time domain ----------------
    noise = np.fft.irfft(spec, n=N)

    # ---------- normalise RMS ----------------------
    rms = np.sqrt(np.mean(noise ** 2)) or 1.0
    noise_mV = noise / rms * target_rms_mV

    time_ns = np.arange(N) * dt_ns
    return time_ns, noise_mV


#def generate_pulse (pulse_v, pulse_t , STEP, simulation_index_duration, amplitude_scale, start_time=0):
#
#
#    #start_time= random.uniform(0, STEP)
#    #start_index= np.argmin(np.where(pulse_t >= start_time)[0])
#    start_index = np.argmin(np.abs(pulse_t - start_time))
#    pulse_indices= np.linspace(start_index, len(pulse_v)-1, simulation_index_duration, dtype=int)
#    
#
#    if pulse_indices[-1] > len(pulse_v):
#        raise ValueError("Pulse exceeds total duration when placed at the specified start time.")
#
#    signal = pulse_v[pulse_indices] * amplitude_scale  # Scale the pulse voltage
#    return signal

def generate_pulse (pulse_v, pulse_t , STEP, simulation_index_duration, amplitude_scale, start_time):

    start_index = np.argmin(np.abs(pulse_t - start_time))
    zeros_array=np.zeros(len(pulse_v), dtype=pulse_v.dtype)
    pulse_v_zeros= np.concatenate((pulse_v, zeros_array))
    pulse_indices = np.linspace(start_index, start_index + len(pulse_v), simulation_index_duration, dtype=int)
    #pulse_indices= np.linspace(start_index, len(pulse_v)-1, simulation_index_duration, dtype=int)
    
    signal = pulse_v_zeros[pulse_indices] * amplitude_scale  # Scale the pulse voltage
    return signal

def generate_pulse_at_angle(
    pulse_voltage,                 # 1D array of the pulse shape
    pulse_time,                    # 1D array of times (ns) for pulse_voltage (sorted)
    time_step,                     # simulation dt (ns)
    simulation_duration_samples,   # length of the output signal
    amplitude_scale,               # scale factor for the pulse
    angle,                         # beam angle
    start_seed,                    # base start time (ns), shared across channels per event
    channel_index                  # 0..Nch-1
):

        # Channel delay from your rule
    delay_dt = angle_delay_time(angle)  # ns
    if delay_dt < 0:
        delay_ns = 3 * abs(delay_dt) + delay_dt * channel_index
    else:
        delay_ns = delay_dt * channel_index
    
    start_time = start_seed + delay_ns  # ns shift applied to this channel

    start_index = np.argmin(np.abs(pulse_time - start_time))
    
    #fill the end of the signal with zeros if the pulse exceeds the total duration
    zeros=np.zeros(len(pulse_voltage), dtype=pulse_voltage.dtype)
    pulse_voltage_extended = np.concatenate((pulse_voltage, zeros))

    pulse_indices = np.linspace(start_index, start_index +len(pulse_voltage), simulation_duration_samples, dtype=int)
    signal= pulse_voltage_extended[pulse_indices] * amplitude_scale  # Scale the pulse voltage

    return signal

def digitize_signal(signal, max_signal):
    """
    Digitize the signal to the ADC range.
    """
    digitized_signal = np.clip(signal, -max_signal//2, max_signal//2)
    return digitized_signal

def make_full_signal(impulse_json_path, SIMULATION_DURATION_NS, SAMPLING_RATE, NOISE_EQUALIZE,
                     pulse_voltage, pulse_time, time_step, simulation_duration_samples, amplitude_scale, max_signal, start_time=0):

    t, noise=make_band_limited_noise(
        impulse_json_path,
        "ch2_2x_amp",
        window_ns=SIMULATION_DURATION_NS,
        adc_rate_ghz=SAMPLING_RATE,
        oversample=1,
        target_rms_mV=NOISE_EQUALIZE,
    ) 
    
    pulse = generate_pulse(pulse_voltage, pulse_time, time_step, simulation_duration_samples, amplitude_scale, start_time)
    full_signal = digitize_signal(noise+pulse, max_signal) #noise + pulse 
    full_signal = full_signal[:simulation_duration_samples]  # Ensure the signal length matches the
    return t, full_signal

def plot_channels_signals(time_axis, channel_signals, title="8 Channels Signals"):
    """
    Plot signals from 4 channels on the same graph.
    
    Parameters
    ----------
    time_axis : array-like
        Time stamps (ns) corresponding to the samples.
    channel_signals : list of list of float
        Each sublist contains signal values for one channel.
    title : str, optional
        Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    for ch, signal in enumerate(channel_signals):
        plt.plot(time_axis, signal, label=f'Channel {ch}')
    
    plt.title(title)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude (ADC counts)')
    plt.legend()
    plt.grid()
    full_title = f"{title} - {int(time_axis[0])} ns to {int(time_axis[-1])} ns"
    plt.savefig(f"{full_title}.png")

def find_triggers(channel_signals, time_axis, *,            # ← positional, keyword-only
                  threshold, coincidence_ns=160,
                  n_channels_required=2):
    """
    Detect coincidence triggers.

    Parameters
    ----------
    channel_signals : list[list[float]]
        One list/array per channel, same length as time_axis.
    time_axis : array-like
        Time stamps (ns) corresponding to the samples.
    threshold : float
        Voltage threshold for a channel to count as "hit".
    coincidence_ns : float, default 160
        Width of coincidence window (ns).  All hits whose times fall
        within ±(coincidence_ns/2) of the trigger centre are grouped.
    n_channels_required : int, default 2
        Minimum distinct channels required to declare a trigger.

    Returns
    -------
    triggers : list[dict]
        Each dict: {"t_trigger": float, "channels": list[int]}
    """
    half_window = coincidence_ns / 2.0

    # -------- 1. collect (time, channel) hits -------------------------------
    hits_t = []
    hits_ch = []
    for ch, sig in enumerate(channel_signals):
        sig = np.asarray(sig)
        idx = np.nonzero(sig > threshold[ch])[0]      # indices where above threshold
        hits_t.extend(time_axis[idx])
        hits_ch.extend([ch] * len(idx))

    if not hits_t:
        return []                                 # no hits → no triggers

    # -------- 2. sort hits by time ------------------------------------------
    hits = sorted(zip(hits_t, hits_ch))           # tuples (t, ch)

    # -------- 3. walk through hits, build coincidence groups ---------------
    triggers = []
    i = 0
    while i < len(hits):
        t0   = hits[i][0]                         # start of new group
        group_ch = {hits[i][1]}
        j = i + 1
        while j < len(hits) and hits[j][0] - t0 <= half_window:
            group_ch.add(hits[j][1])
            j += 1

        if len(group_ch) >= n_channels_required:
            triggers.append({"t_trigger": t0, "channels": sorted(group_ch)})

        # move i to first hit outside the current window
        while i < len(hits) and hits[i][0] - t0 <= half_window:
            i += 1

    return triggers


def make_full_signal_angle(impulse_json_path, SIMULATION_DURATION_NS, SAMPLING_RATE, NOISE_EQUALIZE,
                     pulse_voltage, pulse_time, time_step, simulation_duration_samples, amplitude_scale, max_signal, angle, delay_seed, channel_index):

    t, noise=make_band_limited_noise(
        impulse_json_path,
        "ch2_2x_amp",
        window_ns=SIMULATION_DURATION_NS,
        adc_rate_ghz=SAMPLING_RATE,
        oversample=1,
        target_rms_mV=NOISE_EQUALIZE,
    ) 
    
    pulse = generate_pulse_at_angle(pulse_voltage, pulse_time, time_step, simulation_duration_samples, amplitude_scale, angle, delay_seed, channel_index)
    full_signal = digitize_signal( noise+ pulse, max_signal) #
    full_signal = full_signal[:simulation_duration_samples]  # Ensure the signal length matches the
    return t, full_signal

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

































