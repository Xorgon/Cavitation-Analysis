import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import scipy
import sys

import util.gen_utils as gen
from util.plotting_utils import initialize_plt, format_axis_ticks_decimal
import util.bem as bem

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

# R_init = 1.5 / 1000
dist = 5 / 1000

standoffs = np.logspace(np.log10(1), np.log10(10), 5)

ns = np.linspace(360, 18000, 16)
min_lengths = []

standoff_1_anis_sa = []
standoff_1_anis_bem = []

for n in ns:
    sa_anisotropies = []
    bem_anisotropies = []

    # centroids, normals, areas = gen.gen_varied_simple_plane(0.25, 16000, 3, 6000)
    centroids, normals, areas = gen.gen_varied_simple_plane(0.045, 2 * n / 3, 1, n / 3)
    print(f"Minimum element = {np.sqrt(np.min(areas)) * 1000} mm, "
          f"maximum element = {np.sqrt(np.max(areas)) * 1000} mm")
    print(set(areas))

    min_lengths.append(np.sqrt(np.min(areas)))

    print(f"Using {len(centroids)} elements.")

    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)

    initialize_plt()

    bubble_position = np.array([0, dist, 0])

    sink_positions = np.array([bubble_position,
                               -bubble_position])

    # F' values can be calculated here because depend only on the bubble position which we are keeping constant.
    force_prime_sa = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)

    sigmas = bem.calculate_sigma(bubble_position, centroids, normals, areas, 1, R_inv)
    force_prime_bem = bem.calculate_force_prime(bubble_position, centroids, normals, areas, sigmas, density)

    force_prime_a = -density / (16 * np.pi * dist ** 2)
    print(f"F' = \n"
          f"[A] {force_prime_a} (100 %) \n"
          f"[SA] {force_prime_sa[1]} ({100 * force_prime_sa[1] / force_prime_a:.1f} %) \n"
          f"[BEM] {force_prime_bem[1]} ({100 * force_prime_bem[1] / force_prime_a:.1f} %)")

    for standoff in standoffs:
        print(f"Standoff = {standoff:.2f}")

        R_init = dist / standoff

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time


        def peak_event(t, lhs, *args):
            return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


        peak_event.terminal = True  # End the integration at the peak
        peak_event.direction = -1  # Only detect the maximum

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P), events=peak_event)

        kelvin_impulse_sa = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime_sa) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
            x=out.t[:])
        kelvin_impulse_bem = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime_bem) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
            x=out.t[:])

        anisotropy_sa = kelvin_impulse_sa / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        sa_anisotropies.append(anisotropy_sa)

        anisotropy_bem = kelvin_impulse_bem / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        bem_anisotropies.append(anisotropy_bem)

        if standoff == 1:
            standoff_1_anis_sa.append(anisotropy_sa)
            standoff_1_anis_bem.append(anisotropy_bem)

fig, (comp_ax, conv_ax) = plt.subplots(2, 1, figsize=(4, 3))

comp_ax.plot(standoffs, 0.195 * standoffs ** -2, label="Analytic")
comp_ax.plot(standoffs, sa_anisotropies, label="Semi-analytic", linestyle="dashed")
comp_ax.plot(standoffs, bem_anisotropies, label="BEM", linestyle="dotted")

comp_ax.set_xlabel("$\\gamma = Y / R_0$")
comp_ax.set_ylabel("$\\zeta$")
comp_ax.set_xscale('log')
comp_ax.set_yscale('log')

comp_ax.legend(frameon=False, fontsize="x-small")

min_lengths = np.array(min_lengths)

# Using dist as R_0 since standoff = 1
eps_sa = (0.195 - np.array(standoff_1_anis_sa)) / 0.195
a, b = np.polyfit(np.log(min_lengths), np.log(eps_sa), 1)
conv_ax.plot(min_lengths / dist, np.exp(b) * min_lengths ** a, label="Semi-analytic", color="C1", linestyle="dashed")
conv_ax.scatter(min_lengths / dist, eps_sa, label="Semi-analytic", color="C1")

eps_bem = (0.195 - np.array(standoff_1_anis_bem)) / 0.195
a, b = np.polyfit(np.log(min_lengths), np.log(eps_bem), 1)
conv_ax.plot(min_lengths / dist, np.exp(b) * min_lengths ** a, label="BEM", color="C2", linestyle="dotted")
conv_ax.scatter(min_lengths / dist, eps_bem, label="BEM", color="C2")
conv_ax.set_xscale('log')
conv_ax.set_yscale('log')
conv_ax.set_xlabel("$L_{\\mathrm{fine}} / R_0$")
conv_ax.set_ylabel("$\\epsilon$")
format_axis_ticks_decimal(conv_ax.xaxis, decimal_places=1)
format_axis_ticks_decimal(conv_ax.yaxis, decimal_places=2)
conv_ax.set_yticks([0.025, 0.05, 0.1, 0.2])

label_subplot(comp_ax, "($a$)")
label_subplot(conv_ax, "($b$)")

plt.tight_layout()
plt.show()