import numpy as np
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import math
import time
import random
import json
from pathlib import Path
from scipy.optimize import curve_fit
from sim_functions import *
from trig_functions_cop import *
from scipy.signal import fftconvolve


SAMPLING_RATE       = 3.2                      # GHz
TIME_STEP           = 1.0 / SAMPLING_RATE      # ns
NOISE_EQUALIZE      = 100                      # ADC (use as noise_rms)
MAX_SIGNAL          = 4095                     # ADC
WINDOW_SIZE         = 5.88*1e6                 # MHz (name kept from your script)
n_of_windows        = 2
SIMULATION_DURATION_NS = n_of_windows/(WINDOW_SIZE) * 1e9  # ns
SIMULATION_DURATION_SAMPLES = int(SIMULATION_DURATION_NS / TIME_STEP)
N_of_channels       = 8
N_REQ               = 1                        # not needed for CSW, but kept
COINC_NS            = SIMULATION_DURATION_NS
SCAN_RATE           = 50

# ---- define a single CSW trigger value (you can change this) ----
CSW_THRESHOLD =15.71   # <- “range of trigger of 5” interpreted as trigger value = 5

PULSE_AMPLITUDES = np.concatenate([
    np.arange(60, 200, 15),
    np.arange(200, 400, 10),
    np.arange(400, 550, 25)
])
#PULSE_AMPLITUDES= np.arange(100, 501,10)
#PULSE_AMPLITUDES = np.array([300]*8)

# ---------------- Load pulse and impulse response ----------------
pulse_json_path = Path("../ARA_event_based_simulation_V2/jsons/new_pulse_waveform_ARA_event_based_simulation_V2.json").resolve()
with open(pulse_json_path) as f:
    pulse_data = json.load(f)

impulse_response_path = Path("../ARA_event_based_simulation_V2/jsons/new_impulse_response_ARA_event_based_simulation_V2.json").resolve()

pulse_voltage = np.array(pulse_data['avg_wave'])
pulse_time = np.array(pulse_data['t_axis_ns'])
pulse_start_time, pulse_end_time = 450, 570  # ns
pulse_voltage = pulse_voltage[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)] / np.max(pulse_voltage)
pulse_time = pulse_time[(pulse_time >= pulse_start_time) & (pulse_time <= pulse_end_time)]
pulse_time = pulse_time - pulse_time[0]  # Start from 0 ns


ARA_channel_positions = {
    0: (10.5874, 2.3432, -170.247),
    1: (4.85167, -10.3981, -170.347),
    2: (-2.58128, 9.37815, -171.589),
    3: (-7.84111, -4.05791, -175.377),
    4: (10.5873, 2.3428, -189.502),
    5: (4.85157, -10.3985, -189.400),
    6: (-2.58138, 9.37775, -191.242),
    7: (-7.84131, -4.05821, -194.266),
}


#find center of array
x_coords = [pos[0] for pos in ARA_channel_positions.values()]
y_coords = [pos[1] for pos in ARA_channel_positions.values()]
z_coords = [pos[2] for pos in ARA_channel_positions.values()]
center_x = np.mean(x_coords)
center_y = np.mean(y_coords)
center_z = np.mean(z_coords)
center_of_array = (center_x, center_y, center_z)



def plane_wave_travel_times_from_R(
    channel_positions,
    zenith_deg,
    azimuth_deg,
    R,
    n=1.74,
    center=None,
    return_ns=True
):
    """
    Compute plane-wave travel times to all channels.

    Parameters
    ----------
    channel_positions : dict
        Dictionary {channel: (x, y, z)} in meters.
    zenith_deg : float
        Zenith angle in degrees, measured from +z.
    azimuth_deg : float
        Azimuth angle in degrees, measured from +x toward +y in the xy-plane.
    R : float
        Starting distance of the wavefront from the array center, in meters.
    n : float
        Refractive index of the medium.
    center : array-like or None
        Center of the array. If None, computed from channel positions.
    return_ns : bool
        If True, return times in ns. Otherwise in seconds.

    Returns
    -------
    times_list : list
        List of 8 travel times in channel order [0, 1, ..., 7].
    direction_hat : np.ndarray
        Propagation unit vector.
    start_point : np.ndarray
        Chosen wavefront start point.
    """
    c = 299792458.0
    v = c / n

    if center is None:
        coords = np.array(list(channel_positions.values()), dtype=float)
        center = np.mean(coords, axis=0)
    else:
        center = np.array(center, dtype=float)

    theta = np.deg2rad(zenith_deg)
    phi = np.deg2rad(azimuth_deg)

    direction_hat = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)

    # start point of the wavefront along the incoming direction
    start_point = center - R * direction_hat

    times_list = []

    for ch in range(len(channel_positions)):
        r_i = np.array(channel_positions[ch], dtype=float)

        # plane-wave propagation distance from the chosen start wavefront
        travel_distance = R + np.dot(r_i - center, direction_hat)

        t = travel_distance / v

        if return_ns:
            t *= 1e9

        times_list.append(t)

    return times_list, direction_hat, start_point

delta_times, direction_hat, start_point = plane_wave_travel_times_from_R(ARA_channel_positions, 85, 0, 100, center=center_of_array)

delta_times = -np.array(delta_times)
delta_times -= np.median(delta_times)  # Center times around zero
print("Delta times (ns):", delta_times)

