"""
Testing the 'bem/run_analysis.py' code against solutions developed by Tagawa and Peters (2018).
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import inv
import numpy as np
import math

import util.bem as bem
import util.gen_utils as gen
import util.element_utils as eu
import util.plotting_utils as pu
from util.elements import Source3D
import util.vector_utils as vect


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
    return eu.get_all_vel_3d(elements, sink_pos[0], sink_pos[1], 0)


def vel_to_angle(vel):
    angle = -math.atan2(vel[1], vel[0]) - math.pi / 2
    if angle < 0 and theta_b > corner_angle / 2:
        angle += 2 * math.pi
    if angle > math.pi and theta_b < corner_angle / 2:
        angle -= 2 * math.pi
    if normalize:
        angle /= math.pi - corner_angle
    return angle


if __name__ == "__main__":
    theta_j_sweeps = {}
    n = 10000
    dist = 5
    m_0 = 1
    n_theta_bs = 15
    normalize = True

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0
    for peters_n in np.arange(2, 6, 1):
        label = "Angle = $\\pi / " + str(peters_n) + "$"
        print("Testing", label)
        corner_angle = math.pi / peters_n
        theta_bs = np.linspace(0.1, corner_angle - 0.1, n_theta_bs)
        centroids, normals, areas = gen.gen_varied_corner(n, length=50, angle=corner_angle, depth=50,
                                                          density_ratio=0.25, thresh=2 * dist)
        # centroids, normals, areas = gen.gen_corner(n, length=50, angle=corner_angle, depth=50)
        # pu.plot_3d_point_sets([centroids])
        # print(normals)
        # print(areas)
        print("Creating R matrix")
        R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
        R_inv = inv(R_matrix)

        condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
        condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
        print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

        color = colors[color_idx]
        color_idx += 1
        theta_j_sweeps[label] = [theta_bs, [], [], color, corner_angle, condition_number_inf]  # theta_bs, theta_j numerical, theta_j Peters
        for theta_b in theta_bs:
            print("    theta_b =", theta_b)
            x = dist * math.cos(theta_b)
            y = dist * math.sin(theta_b)

            # Numerical theta_j
            R_b = bem.get_R_vector([x, y, 0], centroids, normals)
            res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, m_0, R_inv=R_inv, R_b=R_b)

            # pu.plot_3d_point_sets([centroids], [sigma], colorbar=True)

            print("        max_res =", np.max(np.abs(R_b + np.dot(R_matrix, sigma))))
            sigma_and_bubble = np.append(sigma, 1)
            theta_j_sweeps[label][1].append(vel_to_angle(res_vel))

            # Analytic theta_j
            res_vel = get_peters_corner_jet_dir([x, y], peters_n)
            theta_j_sweeps[label][2].append(vel_to_angle(res_vel))

        theta_j_sweeps[label][0] /= corner_angle

    pu.initialize_plt()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    legend_elements = [Line2D([0], [0], color='k', label="Analytic"),
                       Line2D([0], [0], color='k', marker='o', linestyle="--", label="Numeric")]
    for label in sorted(theta_j_sweeps.keys()):
        inf_norm_symbol = "||\\mathsfbi{R}||_\\infty"
        legend_elements.append(Line2D([0], [0], color=theta_j_sweeps[label][3], label=f"{label}, ${inf_norm_symbol} = {theta_j_sweeps[label][5]:.1f}$"))
        ax.plot(theta_j_sweeps[label][0], theta_j_sweeps[label][1], theta_j_sweeps[label][3] + "o--", markersize=3)
        ax.plot(theta_j_sweeps[label][0], theta_j_sweeps[label][2], theta_j_sweeps[label][3])
        if not normalize:
            ax.axhline(math.pi - theta_j_sweeps[label][4], color="gray", linestyle="--")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("$\\theta_b$", fontsize=18)
    ax.set_ylabel("$\\theta_j$", fontsize=18)
    ax.legend(handles=legend_elements, loc=4, frameon=False)
    plt.tight_layout()
    plt.show()
