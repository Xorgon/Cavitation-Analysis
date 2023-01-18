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

border = 1e-6  # Plotting border to avoid divide-by-zero when bubble is exactly on the boundary

N = 30  # Number of points to plot in x and y

# Generate points and matrices
xs = np.linspace(border, 2 * width, N)
ys = np.linspace(-width / 2 + border, width / 2 - border, N)
U = np.empty((N, N))
V = np.empty((N, N))
thetas = np.empty((N, N))

# Get all the jet velocities
for i, j in itertools.product(range(N), range(N)):
    # Q here is arbitrarily chosen to avoid huge velocities.
    # This would need to be tuned to get relevant velocity magnitudes.
    vel = get_vel(xs[i], ys[j], 0, width, height, n=25, Q=1e-6)
    theta = math.atan2(vel[1], -vel[0])

    U[j, i] = vel[0]
    V[j, i] = vel[1]

    thetas[j, i] = theta

# Plot
plt.figure(figsize=(4, 3))

# Not strictly necessary, but it looks nicer
plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 10, 'serif': ['cmr10']}
plt.rc('font', **font)

c = plt.contourf(xs * 1e6, ys * 1e6, thetas)  # Jet angle contour

for i in c.collections:
    i.set_edgecolor("face")  # Reduce aliasing in output.

cbar = plt.colorbar(c, label="$\\theta$", ticks=[-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])  # Add the colour bar
cbar.ax.set_yticklabels(["$-\\pi / 2$", "$-\\pi / 4$", "0", "$\\pi / 4$", "$\\pi / 2$"])

plt.quiver(xs * 1e6, ys * 1e6, U / np.sqrt(U ** 2 + V ** 2), V / np.sqrt(U ** 2 + V ** 2), pivot='mid')  # Plot direction arrows

# Plot geometry boundary
plt.plot([1.1 * width * 1e6, 0, 0, 1.1 * width * 1e6],
         [0.5 * width * 1e6, 0.5 * width * 1e6, -0.5 * width * 1e6, -0.5 * width * 1e6], "k", linewidth=2)

# Make everything pretty
plt.xlim(-0.1 * width * 1e6, 1.1 * width * 1e6)
plt.xlabel("x ($\mu$m)")
plt.ylim(-0.6 * width * 1e6, 0.6 * width * 1e6)
plt.ylabel("y ($\mu$m)")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
