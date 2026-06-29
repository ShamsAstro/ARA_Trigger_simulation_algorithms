import numpy as np
import json
from pathlib import Path

# ---- your geometry ----
Idealized_ARA_positions = {
    0: (10.9186, 2.4504, -171.890),
    1: (4.3884, -10.3483, -171.890),
    2: (-1.8802, 8.9806, -171.890),
    3: (-8.4104, -3.8181, -171.890),
    4: (10.9186, 2.4504, -191.102),
    5: (4.3884, -10.3483, -191.102),
    6: (-1.8802, 8.9806, -191.102),
    7: (-8.4104, -3.8181, -191.102), }

ARA_channel_positions = Idealized_ARA_positions

# center of array
coords = np.array(list(ARA_channel_positions.values()))
center_of_array = np.mean(coords, axis=0)


# ---- your function (unchanged) ----
def plane_wave_travel_times_from_R(
    channel_positions,
    zenith_deg,
    azimuth_deg,
    R,
    n=1.74,
    center=None,
    return_ns=True
):
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

    start_point = center - R * direction_hat

    times_list = []

    for ch in range(len(channel_positions)):
        r_i = np.array(channel_positions[ch], dtype=float)
        travel_distance = R + np.dot(r_i - center, direction_hat)
        t = travel_distance / (c / n)

        if return_ns:
            t *= 1e9 

        times_list.append(t)

    return times_list, direction_hat, start_point


# ---- generate JSON ----
def generate_beam_delay_json(
    filename="Event_sources_for_CSW_IDEAL_9600.json",
    R=100.0,
    azimuths=np.linspace(0, 360, 120),
    zeniths=np.linspace(0, 180, 80)
):
    data = {}

    for zen in zeniths:
        for az in azimuths:

            times_ns, direction_hat, start_point = plane_wave_travel_times_from_R(
                ARA_channel_positions,
                zenith_deg=float(zen),
                azimuth_deg=float(az),
                R=R,
                center=center_of_array
            )

            # times_ns to RELATIVE delays (important for beamforming)
            times_ns = np.array(times_ns)
            times_ns -= np.median(times_ns)

            key = f"zen_{zen:.2f}_az_{az:.2f}"

            data[key] = {
                "zenith_deg": float(zen),
                "azimuth_deg": float(az),
                "delays_ns": times_ns.tolist()
            }

    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved beam delays to {output_path}")


# ---- run ----
if __name__ == "__main__":
    generate_beam_delay_json()