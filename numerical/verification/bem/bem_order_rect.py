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


def get_rectangle_jet_vel(x, y, length, height, k=25, m_0=1):
    elements = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if i == 0 and j == 0:
                continue  # Don't include the bubble.
            s_x = i * length + x + (2 * x - length) * ((-1) ** i - 1) / 2
            s_y = j * height + y + (2 * y - height) * ((-1) ** j - 1) / 2
            elements.append(Source3D(s_x, s_y, 0, -m_0))
    return eu.get_all_vel_3d(elements, x, y, 0)


def vel_to_angle(vel, theta_b):
    angle = -math.atan2(vel[1], vel[0]) - math.pi / 2
    # if angle > 3 * math.pi / 2 and theta_b < math.pi:
    #     angle -= 2 * math.pi
    # if angle < math.pi / 2 and theta_b > math.pi:
    #     angle += 2 * math.pi
    return angle


if __name__ == "__main__":
    theta_j_sweeps = {}
    w = 5
    h = 5
    r = 2
    n_theta_bs = 15

    ns = []
    rmsds = []

    f = open("rect_rms.csv", 'a')

    for n in range(1000, 30000, 1000):
        print(f"n = {n}")
        centroids, normals, areas = gen.gen_rectangle(n=n, w=w, h=h, depth=50)
        print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
        R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
        R_inv = np.linalg.inv(R_matrix)

        condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
        condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
        print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")


        def square_difference(theta_b):
            # print("    theta_b =", theta_b)
            x = r * math.cos(theta_b - math.pi / 2) + w / 2
            y = r * math.sin(theta_b - math.pi / 2) + h / 2

            # Numerical theta_j
            R_b = bem.get_R_vector([x, y, 0], centroids, normals)
            res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, R_inv=R_inv, R_b=R_b)
            # print("        max_res =", np.max(np.abs(R_b + np.dot(R_matrix, sigma))))
            num_theta_j = vel_to_angle(res_vel, theta_b)

            # Analytic theta_j
            res_vel = get_rectangle_jet_vel(x, y, w, h)
            analytic_theta_j = vel_to_angle(res_vel, theta_b)

            dif = num_theta_j - analytic_theta_j
            if dif > np.pi:
                num_theta_j -= np.pi * 2
            elif dif < -np.pi:
                num_theta_j += np.pi * 2

            return (num_theta_j - analytic_theta_j) ** 2


        total_sqr_dif = quad(square_difference, 0, 2 * math.pi)[0]
        rms_dif = np.sqrt(total_sqr_dif / (2 * math.pi))
        ns.append(len(centroids))
        rmsds.append(rms_dif)
        f.write(f"{len(centroids)}, {rms_dif}\n")
        f.flush()

    f.close()

    pu.initialize_plt()
    plt.scatter(ns, rmsds)
    plt.xlabel("$n$")
    plt.ylabel("RMSD")
    plt.show()
