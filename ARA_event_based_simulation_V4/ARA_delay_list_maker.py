import numpy as np
from typing import List  # 3.6-style generics
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
import json

ice = medium.ARA_2022() #This is ARA ice model  

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


#Theta= 90  #[0 to 360]
#Phi = 150  #[70 to 150]
R = 500 

Theta_list = np.arange(0, 360, 10)
Phi_list = np.arange(85, 151, 10)   

print(Phi_list)

def vertex_coordinates(theta, phi, r):
    x = r * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
    y = r * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
    z = r * np.cos(np.deg2rad(phi))
    return x, y, z -180 # Shift z to be relative to the center of the array

def travel_time_between_points(start_point, end_point, ice, n_reflections=0):
    """
    Return the light travel time in ns from start_point to end_point
    using the same NuRadioMC propagation setup, ice model, and reflection settings.

    Parameters
    ----------
    start_point : array-like, shape (3,)
        Starting position [x, y, z] in meters.
    end_point : array-like, shape (3,)
        Ending position [x, y, z] in meters.
    ice : NuRadioMC ice model
        Example: medium.greenland_simple()
    n_reflections : int, optional
        Number of reflections to allow, same meaning as in your script.

    Returns
    -------
    float
        Travel time in ns for solution 0.
    """
    prop = propagation.get_propagation_module('analytic')
    rays = prop(ice, n_reflections=n_reflections)

    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)

    rays.set_start_and_end_point(start_point, end_point)
    rays.find_solutions()

    return rays.get_travel_time(0)


def calculate_delay_list(vertex, channel_positions, ice, n_reflections=0):
    vertex_loc = vertex_coordinates(vertex[0], vertex[1], vertex[2])
    delay_list = []
    for ch in range(8):
        ch_pos = channel_positions[ch]
        travel_time = travel_time_between_points(vertex_loc, ch_pos, ice, n_reflections)
        delay_list.append(travel_time)
    delay_list = np.array(delay_list) - np.median(delay_list)
    return delay_list 

#make a json file that has all the delay lists for the iteration over all angles at R=1km
delay_dict = {}
for theta in Theta_list:
    for phi in Phi_list:
        vertex = [theta, phi, R]
        delay_list = calculate_delay_list(vertex, ARA_channel_positions, ice)
        delay_dict[f"theta_{theta}_phi_{phi}"] = delay_list.tolist() 

with open("delay_list.json", "w") as f:
    json.dump(delay_dict, f, indent=4)
    


