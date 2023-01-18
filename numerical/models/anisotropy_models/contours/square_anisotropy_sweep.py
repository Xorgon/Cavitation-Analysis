import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import os

import numerical.bem as bem
import numerical.util.gen_utils as gen
from numerical.util.moi_utils import sink_positions_square
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

N = 32

xs = np.linspace(-L / 2 + offset, L / 2 - offset, N)
ys = np.linspace(-L / 2 + offset, L / 2 - offset, N)

X, Y = np.meshgrid(xs, ys)

Z = np.full(X.shape, np.nan)

img_ranges = 32

# Pressure grid settings
n = 20000
centroids, normals, areas = gen.gen_rectangle(n, L, L, depth=0.2)
# Shift centroids so that the center is at the origin
centroids = centroids - np.array([L / 2, L / 2, 0])

for i, j in itertools.product(range(len(xs)), range(len(ys))):
    print(i, j)
    bubble_pos = [xs[i], ys[j], 0]

    sink_positions = sink_positions_square(L, bubble_pos, x_range=img_ranges, y_range=img_ranges)

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

    Z[j, i] = anisotropy

if not os.path.exists("../../model_outputs/square_anisotropy_data"):
    os.makedirs("../../model_outputs/square_anisotropy_data")
lists_to_csv(f'../../model_outputs/square_anisotropy_data/', f'R{1000 * R_init:.1f}_L{1000 * L:.0f}_n{n}_i{img_ranges}',
             [X.flatten(), Y.flatten(), Z.flatten()])

Z_min = np.min(Z, where=np.invert(np.isnan(Z)), initial=np.inf)
Z_max = np.max(Z, where=np.invert(np.isnan(Z)), initial=0)
cnt = plt.contourf(X, Y, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64))
plt.colorbar(cnt)

plt.gca().set_aspect('equal')
plt.show()
