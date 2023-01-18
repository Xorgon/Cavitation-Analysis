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

# img_ranges = 10

# Pressure grid settings
n = 10000
centroids, normals, areas = gen.gen_triangular_prism(n,
                                                     [0, 0],
                                                     [L * np.sqrt(3) / 2, L / 2],
                                                     [L * np.sqrt(3) / 2, -L / 2],
                                                     0.2)

rng = np.random.default_rng(12345)  # fixed seed
num_points = 16

bubble_positions = []
for i in range(num_points):
    inner_L = (L - 2 * np.sqrt(3) * R_init)
    c1 = np.array([inner_L * np.sqrt(3) / 2, inner_L / 2])
    c2 = np.array([inner_L * np.sqrt(3) / 2, - inner_L / 2])

    fc1 = rng.random()
    fc2 = rng.random()

    pos = np.array([2 * R_init, 0]) + (rng.random() * c1 + rng.random() * c2) / 2

    bubble_positions.append([pos[0], pos[1], 0])

print(bubble_positions)
irs = range(1, 48)
anis = []
ms = []

for img_ranges in irs:
    anis.append([])
    for i, bubble_pos in enumerate(bubble_positions):
        sink_positions = sink_positions_equilateral_triangle(L, bubble_pos,
                                                             x_range=img_ranges, y_range=img_ranges, rotate=True)
        if i == 0:
            ms.append(len(sink_positions))

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)


        def peak_event(t, lhs, *args):
            return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


        peak_event.terminal = True  # End the integration at the peak
        peak_event.direction = -1  # Only detect the maximum

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P), events=peak_event)

        kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                                   x=out.t[:])

        anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        anis[-1].append(anisotropy)

for i in range(num_points):
    plt.plot(irs, np.array(anis)[:, i] / np.max(np.array(anis)[:, i]))
plt.xlabel("Image ranges")
plt.ylabel("$\\zeta'$")
plt.tight_layout()

plt.figure()
for i in range(num_points):
    plt.plot(ms, np.array(anis)[:, i] / np.max(np.array(anis)[:, i]))
plt.xlabel("$M$")
plt.ylabel("$\\zeta'$")
plt.tight_layout()

plt.figure()
plt.plot(irs, ms)
plt.xlabel("Image ranges")
plt.ylabel("$M$")
plt.tight_layout()
plt.show()
