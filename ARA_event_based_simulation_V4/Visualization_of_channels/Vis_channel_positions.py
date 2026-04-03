import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Channel positions (XYZ)
positions = {
    0: (10.5874, 2.3432, -170.247),
    1: (4.85167, -10.3981, -170.347),
    2: (-2.58128, 9.37815, -171.589),
    3: (-7.84111, -4.05791, -175.377),
    4: (10.5873, 2.3428, -189.502),
    5: (4.85157, -10.3985, -189.400),
    6: (-2.58138, 9.37775, -191.242),
    7: (-7.84131, -4.05821, -194.266),
}

# Extract coordinates
xs = [pos[0] for pos in positions.values()]
ys = [pos[1] for pos in positions.values()]
zs = [pos[2] for pos in positions.values()]

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(xs, ys, zs, c="blue", s=60, depthshade=True)

# Annotate each channel
for ch, (x, y, z) in positions.items():
    ax.text(x, y, z, f"{ch}", color="red")

# Labels and title
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3D Positions of Channels 0–7")
plt.title("ARA02_Vpol Channel Positions")

plt.savefig("channel_positions_3d_ARA02_Vpol.png", dpi=300)