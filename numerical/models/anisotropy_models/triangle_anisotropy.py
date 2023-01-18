import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
from util.file_utils import lists_to_csv
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
from common.util.plotting_utils import plot_3d_point_sets
import random


def rp(t, lhs, phi_prime, density=997, kin_visc=0.10533e-5, surf_tension=0, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            (delta_P(t, R) / density
             - R_dot ** 2 * (3 / 2 - 8 * np.pi * phi_prime * R)
             - 4 * kin_visc * R_dot / R
             - 2 * surf_tension / (density * R)) / (R - 4 * np.pi * phi_prime * R ** 2)]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


dirs = ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
        "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

total_readings = 0

for i, dir_path in enumerate(dirs):
    zetas = []
    disps = []
    radius_ratios = []

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    readings = au.load_readings(dir_path + "readings_dump.csv")

    center = (np.array(params.corner_3) + np.array(params.corner_2) + np.array(params.corner_1)) / 3

    n = 10000
    centroids, normals, areas = gen.gen_clipped_triangular_prism(n,
                                                         (np.array(params.corner_1) - center) / 1000,
                                                         (np.array(params.corner_2) - center) / 1000,
                                                         (np.array(params.corner_3) - center) / 1000,
                                                         0.05, 0.1)

    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float64)
    R_inv = scipy.linalg.inv(R_matrix)

    condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
    condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
    print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

    # random.shuffle(readings)
    for j, reading in enumerate(readings):
        # if j % 75 != 0: continue
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(readings)} | Total readings: {total_readings}")
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        x = (pos[0] - center[0]) / 1000
        y = (pos[1] - center[1]) / 1000

        # if y < 0:
        #     continue

        R_init = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px / 1000
        displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px / 1000

        bubble_pos = [x, y, 0]

        # plt.scatter(centroids[:, 0], centroids[:, 1])
        # plt.scatter((centroids + 0.001 * normals)[:, 0], (centroids + 0.0005 * normals)[:, 1])
        # plt.scatter([bubble_pos[0]], [bubble_pos[1]])
        # plt.gca().set_aspect('equal')
        # plt.show()

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        phi_prime = bem.calculate_phi_prime(bubble_pos, centroids, areas, sigmas=sigmas)
        force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)

        # plot_3d_point_sets([centroids], [sigmas], colorbar=True)

        out = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(phi_prime, density, kin_visc, surf_tension, delta_P))

        peaks = find_peaks(-out.y[0])[0]
        if len(peaks) >= 2:
            kelvin_impulse = trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1,
                                                                                                   peaks[0]:peaks[
                                                                                                       1]] ** 2,
                x=out.t[peaks[0]:peaks[1]])
        else:
            # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
            # helpfully still cover exactly the right period for half an expansion-collapse cycle.
            kelvin_impulse = 2 * trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                x=out.t[:])
        anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

        vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)
        reading.model_anisotropy = vect_anisotropy

        print(vect_anisotropy)

        total_readings += 1
        zetas.append(anisotropy)
        disps.append(displacement / R_init)
        radius_ratios.append(np.sqrt(reading.sec_max_area / reading.max_bubble_area))

    lists_to_csv(dir_path, "zetas_vs_disps.csv", [zetas, disps, radius_ratios],
                 headers=["zeta", "disp", "radius_ratio"], overwrite=True)
    au.save_readings(dir_path, readings)  # Add anisotropy
    plt.scatter(zetas, disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
