"""
Testing the 'bem/run_analysis.py' code against solutions developed by Tagawa and Peters (2018).
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import inv
from scipy.integrate import quad
import numpy as np
import math

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
    return eu.get_all_vel_3d(elements, sink_pos[0], sink_pos[1], 0)


def vel_to_angle(vel, theta_b):
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
    peters_n = 2
    dist = 5
    m_0 = 1
    n_theta_bs = 15
    normalize = True

    for length in [10, 30, 50, 100, 150, 250]:
        ns = []
        areas = []
        rmsds = []

        f = open(f"uniform_corner_rms_bp_limited_{length}.csv", 'a')

        increment = round(length / dist)
        while 130 / increment > 16:
            increment *= 2
        for panels_per_side in range(20, 130 + increment, increment):
            if panels_per_side > 130:
                break
            n = panels_per_side ** 2 * 2
            panel_length = length / panels_per_side
            print(f"n = {n}, panels per side = {panels_per_side}, panel length = {panel_length}, "
                  f"position = {np.mod(dist, panel_length) / panel_length}")
            corner_angle = math.pi / peters_n
            centroids, normals, areas = gen.gen_corner(n, length=length, angle=corner_angle, depth=length)
            print(f"Creating R matrix, actual n = {len(centroids)}")
            R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
            R_inv = inv(R_matrix)

            condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
            condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
            print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")


            def square_difference(theta_b):
                # print("    theta_b =", theta_b)
                x = dist * math.cos(theta_b)
                y = dist * math.sin(theta_b)

                # Numerical theta_j
                R_b = bem.get_R_vector([x, y, 0], centroids, normals)
                res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, m_0, R_inv=R_inv,
                                                           R_b=R_b)
                # print("        max_res =", np.max(np.abs(R_b + np.dot(R_matrix, sigma))))
                num_theta_j = vel_to_angle(res_vel, theta_b)

                # Analytic theta_j
                res_vel = get_peters_corner_jet_dir([x, y], peters_n)
                analytic_theta_j = vel_to_angle(res_vel, theta_b)

                return (num_theta_j - analytic_theta_j) ** 2


            total_sqr_dif = quad(square_difference, corner_angle / 4, 3 * corner_angle / 4)[
                0]  # TODO: Plot these out and see what they look like
            rms_dif = np.sqrt(total_sqr_dif / (corner_angle - 0.2))  # TODO: OOPS, THAT'S NOT corner_angle / 2
            ns.append(len(centroids))
            rmsds.append(rms_dif)
            f.write(f"{len(centroids)}, {np.mean(areas)}, {rms_dif}\n")
            f.flush()

        f.close()

    # pu.initialize_plt()
    # plt.scatter(ns, rmsds)
    # plt.xlabel("$n$")
    # plt.ylabel("RMSD")
    # plt.show()
