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
    return - P_inf
    # return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


density = 997
kin_visc = 0  # 1.003e-6
surf_tension = 0  # 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

R_init = 3 / 1000
bubble_pos = [0, R_init * 2.5, 0]

for plane_length in [0.01, 0.02, 0.03, 0.04]:
    print(f"L = {plane_length:.2f}")
    hs = []
    anisotropies = []
    for n in np.linspace(1000, 23000, 12):
        # centroids, normals, areas = gen.gen_varied_simple_plane(0.1 * plane_length, 0.925 * n, plane_length, 0.075 * n)
        centroids, normals, areas = gen.gen_plane([- plane_length / 2, 0, - plane_length / 2],
                                                  [- plane_length / 2, 0, plane_length / 2],
                                                  [plane_length / 2, 0, - plane_length / 2],
                                                  n)
        R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
        R_inv = scipy.linalg.inv(R_matrix)

        print(f"    {n:5.0f} -- {len(centroids):5d}")

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        phi_prime = bem.calculate_phi_prime(bubble_pos, centroids, areas, sigmas=sigmas)
        force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density,
                                                dtype=np.float32)

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

        hs.append(np.sqrt(np.min(areas)))
        anisotropies.append(anisotropy)

        vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)

    lists_to_csv("", f"uniform_hs_vs_zetas_L{100 * plane_length:.0f}", [hs, anisotropies], ["min panel size", "zeta"])
    plt.scatter(hs, anisotropies, label=f"$L = {plane_length:.2f}$")
plt.legend(frameon=False)
plt.axhline(0.195 * (bubble_pos[1] / R_init) ** -2, color="grey", linestyle="--")
plt.show()
