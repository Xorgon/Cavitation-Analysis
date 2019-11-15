"""
Testing the 'bem/run_analysis.py' code against solutions developed by Tagawa and Peters (2018).
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.linalg import inv
from scipy.interpolate import griddata
import itertools
import math
import numpy as np

import numerical.bem as bem
import numerical.util.gen_utils as gen
import numerical.potential_flow.element_utils as eu
import common.util.plotting_utils as pu
from numerical.potential_flow.elements import Source3D
import common.util.vector_utils as vect


def get_corner_elements(sink_pos, peters_n, incl_bubble=True):
    r = vect.mag(sink_pos)
    theta_b = math.atan2(sink_pos[1], sink_pos[0])

    elements = [Source3D(r * math.cos(2 * math.pi - theta_b),  # One sink at (2pi - theta_b)
                         r * math.sin(2 * math.pi - theta_b), 0, -1)]

    if incl_bubble:
        elements.append(Source3D(r * math.cos(theta_b),  # Bubble
                                 r * math.sin(theta_b), 0, -1))

    for k in range(1, peters_n):
        # n - 1 sinks at (2pik/n  - theta_b), 1 <= k <= n - 1
        elements.append(Source3D(r * math.cos(2 * math.pi * k / peters_n - theta_b),
                                 r * math.sin(2 * math.pi * k / peters_n - theta_b), 0, -1))

        # n - 1 sinks at (2pik/n + theta_b), 1 <= k <= n - 1
        elements.append(Source3D(r * math.cos(2 * math.pi * k / peters_n + theta_b),
                                 r * math.sin(2 * math.pi * k / peters_n + theta_b), 0, -1))

    return elements


def get_peters_corner_jet_dir(sink_pos, peters_n):
    elements = get_corner_elements(sink_pos, peters_n, incl_bubble=False)
    return eu.get_all_vel(elements, sink_pos[0], sink_pos[1])


n = 10000
bubble_dist = 5
m_0 = 1
peters_n = 2
include_bubble = False
bubble_radius = 1  # If None, don't plot the bubble boundary

corner_length = 50

corner_angle = math.pi / peters_n
theta_b = corner_angle / 3
centroids, normals, areas = gen.gen_varied_corner(n, length=corner_length, angle=corner_angle, depth=25,
                                                  density_ratio=0.25, thresh=bubble_dist)
# centroids, normals, areas = gen.gen_corner(n, length=corner_length, angle=corner_angle, depth=50)
print(f"{np.sum(areas)}, {np.mean(areas)}, {np.std(areas)}")
print("Creating R matrix")
R_matrix = bem.get_R_matrix(centroids, normals, areas)
R_inv = inv(R_matrix)

print("    theta_b =", theta_b)
bubble_x = bubble_dist * math.cos(theta_b)
bubble_y = bubble_dist * math.sin(theta_b)

R_b = bem.get_R_vector([bubble_x, bubble_y, 0], centroids, normals)
sigma = bem.calculate_sigma([bubble_x, bubble_y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)

corner_elements = get_corner_elements([bubble_x, bubble_y], peters_n, incl_bubble=include_bubble)

offset = 0.01
max_dist = bubble_dist * 2
thetas = np.linspace(offset, corner_angle - offset, 80)
dists = np.linspace(offset, bubble_dist * 2, 80)

xs = []
ys = []
bem_us = []
bem_vs = []
bem_speeds = []

mir_us = []
mir_vs = []
mir_speeds = []

dif_us = []
dif_vs = []
dif_speeds = []

for theta, dist in itertools.product(thetas, dists):
    x = dist * math.cos(theta)
    y = dist * math.sin(theta)

    if np.linalg.norm(np.array([x, y]) - np.array([bubble_x, bubble_y])) == 0:
        continue

    xs.append(x)
    ys.append(y)

    #
    # BEM
    #
    bubble_pos = [bubble_x, bubble_y, 0] if include_bubble else None
    bem_vel = bem.get_vel([x, y, 0], centroids, areas, sigma, bubble_pos=bubble_pos)
    bem_speed = np.linalg.norm(bem_vel)

    bem_us.append(bem_vel[0] / bem_speed)
    bem_vs.append(bem_vel[1] / bem_speed)
    bem_speeds.append(bem_speed)

    #
    # Mirrored sinks
    #
    mir_vel = eu.get_all_vel_3d(corner_elements, x, y, 0)
    mir_speed = np.linalg.norm(mir_vel)

    mir_us.append(mir_vel[0] / mir_speed)
    mir_vs.append(mir_vel[1] / mir_speed)
    mir_speeds.append(mir_speed)

    #
    # Difference
    #
    dif_vel = np.array(bem_vel) - np.array(mir_vel)
    dif_speed = np.linalg.norm(dif_vel)

    # dif_us.append(dif_vel[0] / dif_speed)
    # dif_vs.append(dif_vel[1] / dif_speed)
    dif_us.append(dif_vel[0])
    dif_vs.append(dif_vel[1])
    dif_speeds.append(dif_speed)

mir_speeds = np.array(mir_speeds)
bem_speeds = np.array(bem_speeds)
dif_speeds = np.array(dif_speeds)

fig, axes = plt.subplots(1, 3, figsize=(19, 6), num=f"n = {peters_n}")
fig.patch.set_facecolor('white')
scale = 4

if bubble_radius is not None:
    for ax in axes:
        circle = plt.Circle((bubble_x, bubble_y), bubble_radius, color="k", fill=False)
        ax.add_artist(circle)

centroids_to_plot = np.array([centroid for centroid in centroids if
                              centroid[0] < max_dist and centroid[1] < max_dist * np.sin(corner_angle)])

mir_ax = axes[0]
mir_ax.set_title("Mirror Sinks")
lines = LineCollection([[(0, 0), (max_dist, 0)],
                        [(0, 0), (max_dist * np.cos(corner_angle), max_dist * np.sin(corner_angle))]])
mir_ax.add_collection(lines)
mir_ax.quiver(xs, ys, mir_us, mir_vs, mir_speeds, scale=scale, pivot='mid', scale_units="xy")
mir_ax.set_aspect('equal')
mir_ax.axis('equal')

bem_ax = axes[1]
bem_ax.set_title("BEM")
lines = LineCollection([[(0, 0), (max_dist, 0)],
                        [(0, 0), (max_dist * np.cos(corner_angle), max_dist * np.sin(corner_angle))]])
bem_ax.add_collection(lines)
quiver = bem_ax.quiver(xs, ys, bem_us, bem_vs, bem_speeds, scale=scale, pivot='mid', scale_units="xy")
bem_ax.scatter(centroids_to_plot[:, 0], centroids_to_plot[:, 1])
bem_ax.set_aspect('equal')
bem_ax.axis('equal')

dif_ax = axes[2]
dif_ax.set_title("Difference")
lines = LineCollection([[(0, 0), (max_dist, 0)],
                        [(0, 0), (max_dist * np.cos(corner_angle), max_dist * np.sin(corner_angle))]])
dif_ax.add_collection(lines)
dif_ax.scatter(xs, ys, c=np.array(dif_speeds), cmap=cm.get_cmap("coolwarm"))
dif_ax.quiver(xs, ys, dif_us, dif_vs, scale=0.5, pivot='mid', scale_units="xy")
dif_ax.scatter(centroids_to_plot[:, 0], centroids_to_plot[:, 1])
dif_ax.set_aspect('equal')
dif_ax.axis('equal')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.01, wspace=0.01)
plt.tight_layout()

# fig = plt.figure()
# ax = fig.gca()
# ax.scatter(xs, ys, c=np.array(dif_speeds), cmap=cm.get_cmap("coolwarm"))
# ax.quiver(xs, ys, mir_us, mir_vs, scale=scale, color="b", pivot='mid', scale_units="xy", label="Mirror")
# ax.quiver(xs, ys, bem_us, bem_vs, scale=scale, color="r", pivot='mid', scale_units="xy", label="BEM")
# ax.legend()
plt.show()
