import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
from util.file_utils import lists_to_csv, csv_to_lists
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
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


dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/60 degree corner/",
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/90 degree corner/"]

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

total_readings = 0

for i, dir_path in enumerate(dirs):
    # TODO: Convert this to use my version of the data
    zetas = []
    norm_disps = []

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    corner_angle = np.deg2rad(params.corner_angle_deg)

    column_lists = csv_to_lists(dir_path, "data.csv", True)
    init_xs = np.array(column_lists[21])
    init_ys = np.array(column_lists[22])

    second_xs = np.array(column_lists[23])
    second_ys = np.array(column_lists[24])

    radii = np.array(column_lists[25]) / 2

    disps = np.sqrt((second_xs - init_xs) ** 2 + (second_ys - init_ys) ** 2)

    n = 12000
    density_ratio = 0.5

    centroids, normals, areas = gen.gen_vertical_varied_corner(n, 50 / 1000, 50 / 1000, angle=corner_angle,
                                                               thresh=30 / 1000, density_ratio=density_ratio)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float64)
    R_inv = scipy.linalg.inv(R_matrix)

    condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
    condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
    print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

    # random.shuffle(readings)
    for j, pos in enumerate(zip(init_xs, init_ys)):
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(init_xs)} | Total readings: {total_readings}")

        R_init = radii[j]
        displacement = disps[j]

        bubble_pos = [pos[0], pos[1], 0]

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        phi_prime = bem.calculate_phi_prime(bubble_pos, centroids, areas, sigmas=sigmas)
        force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)

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

        total_readings += 1
        zetas.append(anisotropy)
        norm_disps.append(disps[j] / radii[j])

    lists_to_csv(dir_path, "zetas_vs_disps.csv", [zetas, norm_disps], headers=["zeta", "disp"], overwrite=True)
    plt.scatter(zetas, norm_disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
