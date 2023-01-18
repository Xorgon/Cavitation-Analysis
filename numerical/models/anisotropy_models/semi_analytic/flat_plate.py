import numpy as np
import numerical.util.gen_utils as gen
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import numerical.bem as bem
import matplotlib.pyplot as plt


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
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

R_init = 1.5 / 1000

standoffs = np.linspace(1, 10, 25)

sa_anisotropies = []

for standoff in standoffs:
    bubble_position = np.array([0, R_init * standoff, 0])

    sink_positions = np.array([bubble_position,
                               -bubble_position])

    # Mesh on which to integrate pressure
    centroids, normals, areas = gen.gen_varied_simple_plane(0.25, 20000, 3, 10000)

    force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)

    ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
    sim_length = 4 * ray_col_time

    out = solve_ivp(rp, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                    args=(0, density, kin_visc, surf_tension, delta_P))

    peaks = find_peaks(-out.y[0])[0]
    if len(peaks) >= 2:
        kelvin_impulse = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1,
                                                                                               peaks[0]:peaks[
                                                                                                   1]] ** 2,
            x=out.t[peaks[0]:peaks[1]])
    else:
        # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
        # helpfully still cover exactly the right period for half an expansion-collapse cycle.
        kelvin_impulse = 2 * trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
            x=out.t[:])
    anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

    vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)

    sa_anisotropies.append(anisotropy)


plt.plot(standoffs, 0.195 * standoffs ** -2, label="Analytic")
plt.plot(standoffs, sa_anisotropies, label="Semi-analytic")

plt.xlabel("$\\gamma = y / R_0$")
plt.ylabel("$\\zeta$")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
