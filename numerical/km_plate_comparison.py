import numpy as np
import numerical.util.gen_utils as gen
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import numerical.bem as bem
import matplotlib.pyplot as plt
from common.util.plotting_utils import initialize_plt, fig_width
import scipy


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return p_init * (R_init / R) ** (3 * polytropic_const) + p_vapour - p_inf


def km_inertial(t, lhs, p_v, p_inf, c, density, R_init, p_g0, gamma):
    """ Based on equation 6 in Tinguely et al. (2012). """
    R = lhs[0]
    R_dot = lhs[1]

    v = R_dot / c
    p_g = p_g0 * (R_init / R) ** (3 * gamma)
    p_g_dot = -3 * gamma * p_g0 * R_init ** (3 * gamma) * R ** (-3 * gamma - 1) * R_dot  # Chain rule

    R_dotdot = ((p_g - (p_inf - p_v)) * (1 + v)
                + R * p_g_dot / c - (3 - v) * R_dot ** 2 * density / 2) / ((1 - v) * R * density)

    return [R_dot, R_dotdot]


density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
p_vapour = 2.3388e3
p_inf = 100e3
p_init = p_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

R_init = 1.5 / 1000
dist = 5 / 1000

standoffs = np.logspace(np.log10(1), np.log10(10), 5)

rp_anisotropies = []

initialize_plt()

bubble_position = np.array([0, dist, 0])

sink_positions = np.array([bubble_position,
                           -bubble_position])

force_prime_a = -density / (16 * np.pi * dist ** 2)

km_p_inits = np.logspace(np.log10(0.01 * p_vapour), np.log10(5 * p_vapour), 5)
km_anisotropies = []
radius_ratios = []
for _ in km_p_inits:
    km_anisotropies.append([])
    radius_ratios.append([])

for standoff in standoffs:
    print(f"Standoff = {standoff:.2f}")

    R_init = dist / standoff

    ray_col_time = 0.915 * R_init * (density / (p_inf - p_vapour)) ** 0.5
    sim_length = 4 * ray_col_time

    # RAYLEIGH-PLESSET
    out_rp = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                       args=(density, delta_P))
    peaks_rp = find_peaks(out_rp.y[0])[0]
    kelvin_impulse_rp = trapezoid(
        16 * np.pi ** 2 * np.linalg.norm(force_prime_a) * out_rp.y[0, 0:peaks_rp[0]] ** 4 * out_rp.y[1,
                                                                                            0:peaks_rp[0]] ** 2,
        x=out_rp.t[0:peaks_rp[0]])
    rp_anisotropy = kelvin_impulse_rp / (4.789 * R_init ** 3 * np.sqrt(density * (p_inf - p_vapour)))
    rp_anisotropies.append(rp_anisotropy)

    # KELLER-MIKSIS
    for i, km_p_init in enumerate(km_p_inits):
        out_km = solve_ivp(km_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                           args=(p_vapour, p_inf, 1480, density, R_init, km_p_init, polytropic_const))
        peaks_km = find_peaks(out_km.y[0])[0]
        kelvin_impulse_km = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime_a) * out_km.y[0, 0:peaks_km[0]] ** 4 * out_km.y[1,
                                                                                                0:peaks_km[0]] ** 2,
            x=out_km.t[0:peaks_km[0]])
        km_anisotropy = kelvin_impulse_km / (4.789 * R_init ** 3 * np.sqrt(density * (p_inf - p_vapour)))
        # km_anisotropy = kelvin_impulse_km / (5.234 * R_init ** 2 * (p_inf - p_vapour) * (out_km.t[peaks_km[0]] - out_km.t[0]))
        km_anisotropies[i].append(km_anisotropy)
        radius_ratios[i].append(out_km.y[0, peaks_km[0]] / R_init)

plt.figure(figsize=(fig_width(), fig_width() * 0.75))

plt.plot(standoffs, rp_anisotropies, label="R-P", linestyle="dashed")

for zetas, km_p_init, rrs in zip(km_anisotropies, km_p_inits, radius_ratios):
    plt.plot(standoffs, zetas,
             label=f"K-M, " + "$p_{g0} / p_v =$" + f" {km_p_init / p_vapour:.2f}, $R_1 / R_0 = $ {np.mean(rrs):.2f}",
             linestyle="dashdot")

plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$\\zeta$")

plt.loglog()

plt.legend(frameon=False, fontsize="small")
plt.tight_layout()

plt.show()
