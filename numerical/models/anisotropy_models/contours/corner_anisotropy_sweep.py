import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import os

import numerical.bem as bem
import numerical.util.gen_utils as gen
from numerical.util.moi_utils import sink_positions_corner
from common.util.file_utils import lists_to_csv


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


def peak_event(t, lhs, *args):
    return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


peak_event.terminal = True  # End the integration at the peak
peak_event.direction = -1  # Only detect the maximum

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

corner_n = 3
H = 0.016875  # match the plot height of the other two semi-analytic geometries.
L = 0.02  # sets scale for pressure mesh

N = 32

R_init = 1 / 1000  # 1 mm
offset = R_init

ys = np.linspace(offset / np.sin(np.pi / (2 * corner_n)), H, N)
half_xs = (ys - ys[0]) / np.tan(np.pi / 2 - np.pi / (2 * corner_n))
xs = np.sort(np.concatenate([-half_xs[1:], half_xs]))

X, Y = np.meshgrid(xs, ys)

Z = np.full(X.shape, np.nan)

# Pressure grid settings
n = 50000
centroids, normals, areas = gen.gen_vertical_varied_corner(n, 10 * L, 50 * L, np.pi / corner_n, thresh=5 * L)

for i, j in itertools.product(range(len(xs)), range(len(ys))):
    print(i, j)
    if ys[j] - ys[0] + offset * 0.1 < abs(xs[i]) * np.tan(np.pi / 2 - np.pi / (2 * corner_n)):
        continue  # Do it this way around so that the if clause is short

    bubble_pos = [xs[i], ys[j], 0]

    sink_positions = sink_positions_corner(bubble_pos, corner_n)

    ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
    sim_length = 4 * ray_col_time

    force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)

    out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                    args=(density, delta_P), events=peak_event)

    kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                               x=out.t)
    anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

    Z[j, i] = anisotropy

if not os.path.exists("../../model_outputs/corner_anisotropy_data"):
    os.makedirs("../../model_outputs/corner_anisotropy_data")
lists_to_csv(f'../../model_outputs/corner_anisotropy_data/', f'R{1000 * R_init:.1f}_L{1000 * L:.0f}_n{n}_cn{corner_n}',
             [X.flatten(), Y.flatten(), Z.flatten()])

Z_min = np.min(Z, where=np.invert(np.isnan(Z)), initial=np.inf)
Z_max = np.max(Z, where=np.invert(np.isnan(Z)), initial=0)
cnt = plt.contourf(X, Y, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64))
plt.colorbar(cnt)

plt.gca().add_patch(Polygon(np.array([[0, L * np.sin(np.pi / (2 * corner_n)), -L * np.sin(np.pi / (2 * corner_n))],
                                      [0, L * np.cos(np.pi / (2 * corner_n)), L * np.cos(np.pi / (2 * corner_n))]]).T,
                            facecolor="white", edgecolor="black", linewidth=1))

plt.gca().set_aspect('equal')
plt.show()
