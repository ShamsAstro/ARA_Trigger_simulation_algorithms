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

def angle_delay_time(angle ):
    angle=np.deg2rad(angle)
    # === Physics ===
    n_ice= 1.75  #index of refraction in ice
    vertical_seperation= 1 #distance betwwen channel in meters
    c= 299792458 #speed of light in a vaccum in m/s
    
    time_delays= n_ice * vertical_seperation * np.sin(angle) / c
    return -time_delays*1e9 # ns  # negative because the pulse is delayed, not advanced and ch3 is closes to surface


_impulse_cache = {}

def get_impulse_arrays(json_path, channel_key):
    key = (str(json_path), channel_key)
    arrs = _impulse_cache.get(key)
    if arrs is None:
        d = json.loads(Path(json_path).read_text())
        arrs = (np.asarray(d["freq_GHz"]), np.asarray(d[channel_key]))
        _impulse_cache[key] = arrs
    return arrs

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
    freq_ref, mag_ref = get_impulse_arrays(json_path, channel_key)

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






























