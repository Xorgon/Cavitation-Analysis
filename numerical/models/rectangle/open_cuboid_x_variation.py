import math
import numpy as np
import itertools
import matplotlib.pyplot as plt


def get_vel(x_b, y_b, z_b, w=1.0, h=1.0, n=25, Q=1.0):
    """
    Calculate the velocity for a bubble position with the given geometric parameters.
    :param x_b: Bubble x position, measured from the left (closed) end.
    :param y_b: Bubble y position, measured from the horizontal plane of symmetry.
    :param z_b: Bubble z position, measured from the vertical plane of symmetry.
    :param w: Width of the cuboid (along y axis).
    :param h: Height of the cuboid (along z axis).
    :param n: Number of mirror sinks to populate in each direction.
    :param Q: Sink strength.
    :return: Representative velocity, aligned with the jet direction.
    """
    sink_coords = []
    for i in [-1, 1]:
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                if j == 0 and k == 0 and i == 1:
                    continue  # Don't include the bubble itself
                s_x = i * x_b
                s_y = (-1) ** j * y_b + j * w
                s_z = (-1) ** k * z_b + k * h

                sink_coords.append([s_x, s_y, s_z])

    # Sum the contribution from all sinks
    vel = [0.0, 0.0, 0.0]
    for (s_x, s_y, s_z) in sink_coords:
        R = math.sqrt((x_b - s_x) ** 2 + (y_b - s_y) ** 2 + (z_b - s_z) ** 2)
        vel[0] += - Q * (x_b - s_x) / (4 * math.pi * R ** 3)
        vel[1] += - Q * (y_b - s_y) / (4 * math.pi * R ** 3)
        vel[2] += - Q * (z_b - s_z) / (4 * math.pi * R ** 3)
    return vel


# Geometry parameters
width = 400e-6
height = 100e-6

border = 200e-6  # Plotting border to avoid divide-by-zero when bubble is exactly on the boundary

N = 100  # Number of points to plot in y

# Generate points and matrices
xs = np.linspace(border, 0.1, 150)
y = 0

plt.figure(figsize=(5, 3.5))

# Not strictly necessary, but it looks nicer
plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 10, 'serif': ['cmr10']}
plt.rc('font', **font)

v_xs = []
# Get all the jet velocities
for x in xs:
    # Q here is arbitrarily chosen to avoid huge velocities.
    # This would need to be tuned to get relevant velocity magnitudes.
    vel = get_vel(x, y, 0, width, height, n=150, Q=1e-6)
    v_xs.append(vel[0])

plt.plot(xs * 1e6, -np.array(v_xs))
# plt.gca().set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
# plt.gca().set_yticklabels(["$-\\pi / 2$", "$-\\pi / 4$", "0", "$\\pi / 4$", "$\\pi / 2$"])
plt.xlabel("x ($\\mu$m)")
plt.ylabel("$-v_x$")
# plt.xscale("log")
# plt.yscale("log")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
