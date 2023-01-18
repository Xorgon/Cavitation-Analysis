import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from common.util.plotting_utils import initialize_plt


def rp(t, lhs, density=997, kin_visc=0.10533e-5, surf_tension=0, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            -3 * R_dot ** 2 / (2 * R)
            - 4 * kin_visc * R_dot / (R ** 2)
            - 2 * surf_tension / (density * R ** 2)
            + delta_P(t, R) / (R * density)]


def non_dim_rp(t, lhs, Re, We, pressure_ratio, polytropic_const=1.33):
    """ Non-dimensional Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            0.915 ** 2 * (pressure_ratio * R ** (-3 * polytropic_const) - 1) / R
            - 3 * R_dot ** 2 / (2 * R)
            - 4 * R_dot / (R ** 2 * Re)
            - 2 / (R ** 2 * We)]


def inertia_only_rp(t, lhs):
    """ Non-dimensional Rayleigh-Plesset formulation as two first order ODEs with only inertia terms. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            -95322.4 / R
            - 3 * R_dot ** 2 / (2 * R)]


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
c = 1480
# kin_visc = 0
surf_tension = 0.0728
R_init = 0.002
P_vapour = 2.3388e3
P_init = P_vapour
P_inf = 100e3
polytropic_const = 1.33  # ratio of specific heats of water vapour

ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
u = R_init / ray_col_time
print("Rayleigh Collapse Time = ", ray_col_time)
We = density * R_init * u ** 2 / surf_tension
print("Weber number = ", We)
Re = u * R_init / kin_visc
print("Reynolds number = ", Re)
# Eu = (P_init + P_vapour - P_inf) / inertial stuff  VARIES IN TIME
# print("Euler number = ", Eu)
pressure_ratio = P_init / (P_inf - P_vapour)
print("Pressure ratio = ", pressure_ratio)

sim_length = 5 * ray_col_time


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


initialize_plt()
plt.figure(figsize=(5, 3))
out_rp = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                   args=(density, kin_visc, surf_tension, delta_P))

out_km = solve_ivp(km_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                   args=(P_vapour, P_inf, c, density, R_init, P_init, polytropic_const))

# inertia_only_out = solve_ivp(inertia_only_rp, (0, 5), (1, 0), max_step=sim_length / 100)
plt.plot(out_rp.t / ray_col_time, out_rp.y[0, :] / R_init, label="Rayleigh-Plesset")
plt.plot(out_km.t / ray_col_time, out_km.y[0, :] / R_init, label="Keller-Miksis")
# plt.plot(inertia_only_out.t, inertia_only_out.y[0, :])
plt.xlabel("$t / t_{TC}$")
plt.ylabel("$R / R_0$")
plt.legend(frameon=True, loc="lower right", fancybox=False)
plt.tight_layout()

plt.show()
