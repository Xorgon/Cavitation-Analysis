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
pressure_ratio = P_init / (P_inf - P_vapour)
print("Pressure ratio = ", pressure_ratio)

sim_length = 4 * ray_col_time

centroids, normals, areas = gen.gen_varied_simple_plane(0.05, 16000, 2, 2000)
R_matrix = bem.get_R_matrix(centroids, normals, areas)
R_inv = np.linalg.inv(R_matrix)

standoffs = np.linspace(1, 8, 8)
anisotropies = []
for standoff in standoffs:
    print(f"Standoff = {standoff:.2f}")
    bubbles_pos = np.array([0, standoff * R_init, 0])
    sigmas = bem.calculate_sigma(bubbles_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)

    phi_prime = bem.calculate_phi_prime(bubbles_pos, centroids, areas, sigmas=sigmas)
    phi_prime = 0

    force_prime = bem.calculate_force_prime(bubbles_pos, centroids, normals, areas, sigmas, density)

    # initialize_plt()
    out = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                    args=(phi_prime, density, kin_visc, surf_tension, delta_P))

    peaks = find_peaks(-out.y[0])[0]
    if len(peaks) >= 2:
        kelvin_impulse = trapezoid(
            16 * np.pi ** 2 * force_prime[1] * out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1, peaks[0]:peaks[1]] ** 2,
            x=out.t[peaks[0]:peaks[1]])
    else:
        # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
        # helpfully still cover exactly the right period for half an expansion-collapse cycle.
        kelvin_impulse = 2 * trapezoid(16 * np.pi ** 2 * force_prime[1] * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                                       x=out.t[:])
    anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

    anisotropies.append(anisotropy)

plt.scatter(standoffs, anisotropies, label="BEM")
plt.scatter(standoffs, -0.195 / standoffs ** 2, label="Supponen et al. (2016)")
plt.xlabel("$\\gamma$")
plt.ylabel("$\\zeta$")
plt.legend()
plt.show()
