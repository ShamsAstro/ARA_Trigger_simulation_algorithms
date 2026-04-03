import numpy as np
from typing import List  # 3.6-style generics
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
import json

# ---------- Geometry ---------------------------------------------------
ice = medium.greenland_simple()

ch3_pos = np.array([0, 0, -97.55])
ch2_pos = np.array([0, 0, -98.55])
ch1_pos = np.array([0, 0, -99.55])
ch0_pos = np.array([0, 0, -100.55])
channel_positions = [ch0_pos, ch1_pos, ch2_pos, ch3_pos]

pulser_xy = np.array([30, 0])          # x-y of the pulser
N_REFLECTIONS = 0                       # direct ray only
# -----------------------------------------------------------------------


def delays_and_mean_angle(depth, channel_positions, pulser_xy, rays):
    """
    Return (delays, mean_angle_deg) for a single pulser depth.

    delays  – NumPy array of per-channel travel times, shifted so the
              earliest hit is 0 ns.
    """
    pulser_pos = np.array([pulser_xy[0], pulser_xy[1], depth])

    delays = []   # type: List[float]
    angles = []   # type: List[float]

    for ch in channel_positions:
        rays.set_start_and_end_point(ch, pulser_pos)
        rays.find_solutions()

        vec_x, _, vec_z = rays.get_receive_vector(0)
        angle_rad = np.arctan(vec_z / vec_x)

        delays.append(rays.get_travel_time(0))      # ns
        angles.append(np.rad2deg(angle_rad))        # deg

    delays = np.array(delays) - np.min(delays)       # normalize
    mean_angle = float(np.mean(angles))
    return delays, mean_angle


def build_lookup(depths, channel_positions, pulser_xy, ice, n_reflections=0):
    """
    Scan 'depths' and return two NumPy arrays:

        mean_angles  – shape (N,), degrees
        delay_table  – shape (N, 4), ns (each row is a 4-chan delay vector)

    Both arrays are sorted by mean angle for easy interpolation later.
    """
    prop = propagation.get_propagation_module('analytic')
    rays = prop(ice, n_reflections=n_reflections)

    mean_angles = []
    delay_rows = []

    for d in depths:
        delays, ang = delays_and_mean_angle(d, channel_positions, pulser_xy, rays)
        mean_angles.append(ang)
        delay_rows.append(delays)

    mean_angles = np.array(mean_angles)
    delay_table = np.vstack(delay_rows)

    order = np.argsort(mean_angles)
    return mean_angles[order], delay_table[order]


def interpolate_delays(target_angles, mean_angles, delay_table):
    """
    For each value in 'target_angles' (deg) return a 4-element delay vector (ns)
    by linear interpolation in angle space.
    """
    target_angles = np.asarray(target_angles)
    out = np.empty((target_angles.size, delay_table.shape[1]))

    for ch in range(delay_table.shape[1]):
        out[:, ch] = np.interp(target_angles, mean_angles, delay_table[:, ch])

    return out


if __name__ == "__main__":
    # --- 1. build fine depth→angle map ---------------------------------
    fine_depths = np.linspace(0, -150, 481)        
    mean_ang, delay_tab = build_lookup(fine_depths,
                                       channel_positions,
                                       pulser_xy,
                                       ice,
                                       N_REFLECTIONS)

    # --- 2. user-requested angles --------------------------------------
    req_angles = np.arange(60, -60.1, -0.5)           
    delay_lists = interpolate_delays(req_angles, mean_ang, delay_tab)
    delay_lists = np.round(delay_lists, 3)  # round to 3 decimal places
    # --- 3. show the result --------------------------------------------
    print("Angles and delays for pulser drop 30-150 m:")
    # --- 3. show the result --------------------------------------------
    print("Angle (deg) :  delays [ns] (ch0→ch3, earliest hit = 0)")
    for ang, dly in zip(req_angles, delay_lists):
        print("{:>6.1f}      {}".format(ang, np.round(dly, 4)))
    
    data = {
        "angles": req_angles.tolist(),      # 1-D list of floats
        "delays": delay_lists.tolist(),     # 2-D list (list of 4-elem lists)
    }
    """
    """
    with open("delay_presets_pulser_drop_30_150_full.json", "w") as f:
        json.dump(data, f, indent=4)
   