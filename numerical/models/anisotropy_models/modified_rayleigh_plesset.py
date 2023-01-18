import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
import numpy as np
from common.util.plotting_utils import initialize_plt


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


density = 997
kin_visc = 1.003e-6
# kin_visc = 0
surf_tension = 0.0728
R_init = 0.002
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
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

phi_prime = 0

# initialize_plt()
out = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                args=(phi_prime, density, kin_visc, surf_tension, delta_P))
non_dim_out = solve_ivp(non_dim_rp, (0, 5), (1, 0), max_step=sim_length / 100,
                        args=(Re, We, pressure_ratio, polytropic_const))
# inertia_only_out = solve_ivp(inertia_only_rp, (0, 5), (1, 0), max_step=sim_length / 100)
plt.plot(out.t / ray_col_time, out.y[0, :] / R_init)
plt.plot(non_dim_out.t, non_dim_out.y[0, :])
# plt.plot(inertia_only_out.t, inertia_only_out.y[0, :])
plt.xlabel("$t / t_{TC}$")
plt.ylabel("$R / R_0$")

plt.figure()
integ = []
for i in range(len(out.t)):
    integ.append(
        trapezoid(out.y[0, :i + 1] ** 3  # Buoyancy (ish)
                  - 16 * np.pi ** 2 * out.y[0, :i + 1] ** 4 * out.y[1, :i + 1] ** 2))
plt.plot(out.t / ray_col_time, integ)

plt.show()
