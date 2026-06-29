import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Real ARA V-pol antenna positions
# -----------------------------
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


channels = np.array(sorted(ARA_channel_positions.keys()))
real = np.array([ARA_channel_positions[ch] for ch in channels])

# -----------------------------
# Build idealized cube-like geometry
# -----------------------------
# Channels 0-3 are upper V-pol antennas.
# Channels 4-7 are lower V-pol antennas.
top = real[:4]
bottom = real[4:]

# Average x-y position of each vertical string
string_xy = 0.5 * (top[:, :2] + bottom[:, :2])

# Center of the station in x-y
xy_center = string_xy.mean(axis=0)

# Average radial distance from center to the real strings
# For a square, radial distance = side / sqrt(2)
real_radii = np.linalg.norm(string_xy - xy_center, axis=1)
ideal_radius = real_radii.mean()
half_side = ideal_radius / np.sqrt(2)

# Use the direction from the center to channel/string 0 to set orientation.
# This keeps the ideal square oriented like the real station.
v0 = string_xy[0] - xy_center
angle_v0 = np.arctan2(v0[1], v0[0])

# In local square coordinates, corner [1, 1] has angle 45 degrees.
# Rotate the square so that corner [1, 1] points toward real channel 0.
rotation_angle = angle_v0 - np.pi / 4

R = np.array([
    [np.cos(rotation_angle), -np.sin(rotation_angle)],
    [np.sin(rotation_angle),  np.cos(rotation_angle)],
])

# Local ideal square corners.
# Ordered to match channels approximately:
# 0 and 4 -> corner near real string 0
# 1 and 5 -> corner near real string 1
# 2 and 6 -> corner near real string 2
# 3 and 7 -> corner near real string 3
square_local = np.array([
    [ half_side,  half_side],
    [ half_side, -half_side],
    [-half_side,  half_side],
    [-half_side, -half_side],
])

ideal_xy_unordered = xy_center + square_local @ R.T

# Match ideal corners to real string labels by nearest neighbor
ideal_xy = np.zeros_like(ideal_xy_unordered)
used = set()

for i in range(4):
    distances = np.linalg.norm(ideal_xy_unordered - string_xy[i], axis=1)
    for j in np.argsort(distances):
        if j not in used:
            ideal_xy[i] = ideal_xy_unordered[j]
            used.add(j)
            break

# Use flat ideal top and bottom depths
z_top = top[:, 2].mean()
z_bottom = bottom[:, 2].mean()

ideal_top = np.column_stack([ideal_xy, np.full(4, z_top)])
ideal_bottom = np.column_stack([ideal_xy, np.full(4, z_bottom)])
ideal = np.vstack([ideal_top, ideal_bottom])

idealized_ARA_channel_positions = {
    int(ch): tuple(pos) for ch, pos in zip(channels, ideal)
}

print("Idealized ARA V-pol positions:")
for ch, pos in idealized_ARA_channel_positions.items():
    print(f"{ch}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.3f})")

# -----------------------------
# Plotting
# -----------------------------
edges = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # top square
    (4, 5), (5, 7), (7, 6), (6, 4),  # bottom square
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical strings
]

views = [
    ("3D angled view", 25, -45),
    ("Top view: x-y geometry", 90, -90),
    ("Side view: depth structure", 0, -90),
]

def set_equal_3d(ax, pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    max_range = max(np.ptp(x), np.ptp(y), np.ptp(z))
    if max_range == 0:
        max_range = 1

    cx, cy, cz = x.mean(), y.mean(), z.mean()

    ax.set_xlim(cx - max_range / 2, cx + max_range / 2)
    ax.set_ylim(cy - max_range / 2, cy + max_range / 2)
    ax.set_zlim(cz - max_range / 2, cz + max_range / 2)

def plot_geometry(ax, pts, title, elev, azim, marker="o", linestyle="-", label=None):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=70,
        marker=marker,
        label=label,
    )

    for a, b in edges:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            linewidth=1.5,
            linestyle=linestyle,
            alpha=0.75,
        )

    for ch, (x, y, z) in zip(channels, pts):
        ax.text(x, y, z, f" {ch}", fontsize=10)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", alpha=0.45)

def make_single_geometry_figure(pts, fig_title):
    fig = plt.figure(figsize=(18, 6))

    for idx, (view_title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        plot_geometry(ax, pts, view_title, elev, azim)
        set_equal_3d(ax, pts)

    fig.suptitle(fig_title, fontsize=17)
    plt.tight_layout()
    plt.show()

def make_comparison_figure(real, ideal):
    fig = plt.figure(figsize=(18, 6))

    for idx, (view_title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")

        plot_geometry(
            ax,
            real,
            view_title,
            elev,
            azim,
            marker="o",
            linestyle="-",
            label="Real",
        )

        plot_geometry(
            ax,
            ideal,
            view_title,
            elev,
            azim,
            marker="x",
            linestyle="--",
            label="Idealized",
        )

        set_equal_3d(ax, np.vstack([real, ideal]))
        ax.legend()

    fig.suptitle("ARA V-pol geometry: real vs idealized same-size cube", fontsize=17)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Make plots
# -----------------------------
make_single_geometry_figure(
    real,
    "Real ARA V-pol antenna geometry"
)

make_single_geometry_figure(
    ideal,
    "Idealized same-size cube-like ARA V-pol geometry"
)

make_comparison_figure(real, ideal)