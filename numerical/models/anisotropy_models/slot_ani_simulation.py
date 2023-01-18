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
R_init = 1.5e-3

total_readings = 0

min_n = np.inf
max_n = 0

zetas = []
disps = []
radius_ratios = []


# Convert to meters
w = 4 / 1000
h = 4 / 1000

n = 2000
density_ratio = 0.25
w_thresh = 5

centroids, normals, areas = gen.gen_varied_slot(n=n, H=h, W=w, length=0.1, depth=0.05, w_thresh=w_thresh,
                                                density_ratio=density_ratio)

if len(centroids) < min_n:
    min_n = len(centroids)
if len(centroids) > max_n:
    max_n = len(centroids)

R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)

p = 1.5 * w / 2
q = 3 * R_init

bubble_pos = [p, q, 0]

ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
sim_length = 4 * ray_col_time

sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)


def peak_event(t, lhs, *args):
    return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


peak_event.terminal = True  # End the integration at the peak
peak_event.direction = -1  # Only detect the maximum

out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                args=(density, delta_P), events=peak_event)

kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                           x=out.t[:])
anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)

print(anisotropy)