import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import numpy as np
from common.util.plotting_utils import initialize_plt
import numerical.bem as bem
import numerical.util.gen_utils as gen


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
surf_tension = 0.0728
R_init = 0.002
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
u = R_init / ray_col_time
print("Rayleigh Collapse Time = ", ray_col_time)
# We = density * R_init * u ** 2 / surf_tension
# print("Weber number = ", We)
# Re = u * R_init / kin_visc
# print("Reynolds number = ", Re)
# Eu = (P_init + P_vapour - P_inf) / inertial stuff  VARIES IN TIME
# print("Euler number = ", Eu)
pressure_ratio = P_init / (P_inf - P_vapour)
print("Pressure ratio = ", pressure_ratio)

sim_length = 5 * ray_col_time

centroids, normals, areas = gen.gen_varied_simple_plane(0.12, 10000, 3, 6000)
R_matrix = bem.get_R_matrix(centroids, normals, areas)
R_inv = np.linalg.inv(R_matrix)

bubbles_pos = np.array([0, 2 * R_init, 0])
sigmas = bem.calculate_sigma(bubbles_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)

phi_prime = bem.calculate_phi_prime(bubbles_pos, centroids, areas, sigmas=sigmas)
# phi_prime = 0
print(f"phi_prime = {phi_prime}")

force_prime = bem.calculate_force_prime(bubbles_pos, centroids, normals, areas, sigmas, density)
print(f"force_prime = {force_prime}")
print(f"force prime from Blake et al 2015. = {density / (16 * np.pi * bubbles_pos[1] ** 2)}")
print(f"...which is a factor of {density / (16 * np.pi * bubbles_pos[1] ** 2) / force_prime[1]} bigger.")

standoff = bubbles_pos[1] / R_init
supp_impulse = 0.934 * R_init ** 3 * (density * (P_inf - P_vapour)) ** 0.5 / (standoff ** 2)
print(f"Impulse estimate from Supponen = {supp_impulse}")

# initialize_plt()
out = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                args=(phi_prime, density, kin_visc, surf_tension, delta_P))
# non_dim_out = solve_ivp(non_dim_rp, (0, 5), (1, 0), max_step=sim_length / 100,
#                         args=(Re, We, pressure_ratio, polytropic_const))
# inertia_only_out = solve_ivp(inertia_only_rp, (0, 5), (1, 0), max_step=sim_length / 100)
plt.plot(out.t / ray_col_time, out.y[0, :] / R_init, label="Modified")
# plt.plot(non_dim_out.t, non_dim_out.y[0, :], label="Non-dimensional Original")
plt.legend()
# plt.plot(inertia_only_out.t, inertia_only_out.y[0, :])
plt.xlabel("$t / t_{TC}$")
plt.ylabel("$R / R_0$")

peaks = find_peaks(-out.y[0])[0]
# kelvin_impulse = trapezoid(
#     16 * np.pi ** 2 * force_prime[1] * out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1, peaks[0]:peaks[1]] ** 2,
#     x=out.t[peaks[0]:peaks[1]])
# print(f"Kelvin Impulse = {kelvin_impulse}")
# print(f"Anisotropy = {kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))}")
print(f"Supponen anisotropy = {0.195 * (bubbles_pos[1] / R_init) ** -2}")

plt.figure()
integ = []
integ_blake = []
for i in range(len(out.t)):
    integ.append(
        trapezoid(  # out.y[0, :i + 1] ** 3  # Buoyancy (ish)
            16 * np.pi ** 2 * force_prime[1] * out.y[0, :i + 1] ** 4 * out.y[1, :i + 1] ** 2,
            x=out.t[:i + 1]))
    integ_blake.append(
        trapezoid(  # out.y[0, :i + 1] ** 3  # Buoyancy (ish)
            (-np.pi * density / bubbles_pos[1] ** 2) * out.y[0, :i + 1] ** 4 * out.y[1, :i + 1] ** 2,
            x=out.t[:i + 1]))
plt.plot(out.t / ray_col_time, integ, label="BEM")
plt.plot(out.t / ray_col_time, integ_blake, label="Blake et al. 2015")
i = 0
while -supp_impulse * i >= np.min([integ_blake, integ]) or i < 2:
    plt.axhline(-supp_impulse * i, linestyle="dashed", color="grey")
    i += 1
plt.xlabel("$t / t_{TC}$")
plt.ylabel("Kelvin Impulse")
plt.legend()

plt.show()
