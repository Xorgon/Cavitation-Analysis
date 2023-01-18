import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import os

import numerical.bem as bem
import numerical.util.gen_utils as gen
from numerical.util.moi_utils import sink_positions_equilateral_triangle
from common.util.file_utils import lists_to_csv


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

L = 0.015  # 15 mm

R_init = 1 / 1000  # 1 mm
offset = R_init

xs = np.linspace(2 * offset, L * np.sqrt(3) / 2 - offset, 20)
ys = np.sort(np.concatenate([-(xs[1:] - 2 * offset) * np.sqrt(3) / 3, (xs - 2 * offset) * np.sqrt(3) / 3]))

X, Y = np.meshgrid(xs, ys)

Z = np.full(X.shape, np.nan)

img_ranges = 10

bubble_pos = [3 * R_init, 0, 0]
ns = range(1000, 48000, 2000)
anis = []
for n in ns:
    centroids, normals, areas = gen.gen_triangular_prism(n,
                                                         [0, 0],
                                                         [L * np.sqrt(3) / 2, L / 2],
                                                         [L * np.sqrt(3) / 2, -L / 2],
                                                         0.2)

    sink_positions = sink_positions_equilateral_triangle(L, bubble_pos,
                                                         x_range=img_ranges, y_range=img_ranges, rotate=True)

    ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
    sim_length = 4 * ray_col_time

    force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)

    out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                    args=(density, delta_P))

    peaks = find_peaks(-out.y[0])[0]
    if len(peaks) >= 2:
        kelvin_impulse = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0,
                                                            peaks[0]:peaks[1]] ** 4 * out.y[1, peaks[0]:peaks[1]] ** 2,
            x=out.t[peaks[0]:peaks[1]])
    else:
        # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
        # helpfully still cover exactly the right period for half an expansion-collapse cycle.
        kelvin_impulse = 2 * trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
            x=out.t[:])
    anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
    anis.append(anisotropy)

plt.plot(ns, anis)
plt.xlabel("Pressure grid points")
plt.ylabel("$\\zeta$")
plt.tight_layout()
plt.show()
