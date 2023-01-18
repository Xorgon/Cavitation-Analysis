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
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


initialize_plt()
plt.figure(figsize=(fig_width(), fig_width() * 0.75))

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

# R_init = 1.5 / 1000
dist = 5 / 1000

standoffs = np.logspace(np.log10(1.5), np.log10(3.2), 5)

bubble_position = np.array([0, dist, 0])

sink_positions = np.array([bubble_position,
                           -bubble_position])

sizes = [0.05, 0.1]

for size in sizes:

    sa_anisotropies = []
    bem_anisotropies = []

    # centroids, normals, areas = gen.gen_varied_simple_plane(0.25, 16000, 3, 6000)
    centroids, normals, areas = gen.gen_varied_simple_plane(0.045, 11000, size, 3000)
    areas = areas * (1 - 0.24)

    print(f"Using {len(centroids)} elements.")

    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)

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

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P))

        peaks = find_peaks(-out.y[0])[0]
        if len(peaks) >= 2:
            kelvin_impulse_sa = trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime_sa) *
                out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1, peaks[0]:peaks[1]] ** 2,
                x=out.t[peaks[0]:peaks[1]])

            kelvin_impulse_bem = trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime_bem) *
                out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1, peaks[0]:peaks[1]] ** 2,
                x=out.t[peaks[0]:peaks[1]])
        else:
            # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
            # helpfully still cover exactly the right period for half an expansion-collapse cycle.
            kelvin_impulse_sa = 2 * trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime_sa) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                x=out.t[:])

            kelvin_impulse_bem = 2 * trapezoid(
                16 * np.pi ** 2 * np.linalg.norm(force_prime_bem) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                x=out.t[:])

        anisotropy_sa = kelvin_impulse_sa / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        sa_anisotropies.append(anisotropy_sa)

        anisotropy_bem = kelvin_impulse_bem / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        bem_anisotropies.append(anisotropy_bem)

    # plt.plot(standoffs, 0.195 * standoffs ** -2, label="Analytic")
    # plt.plot(standoffs, sa_anisotropies, label="Semi-analytic")
    # plt.plot(standoffs, bem_anisotropies, label=f"BEM - $L = {1000 * size:.2f} mm$")
    print(bem_anisotropies)
    print(standoffs)
    print(5.672 * np.array(bem_anisotropies) ** 0.556)
    plt.plot(standoffs, 5.672 * np.array(bem_anisotropies) ** 0.556, label=f"BEM - $L = {1000 * size:.2f} mm$")

plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$\\Delta / R_0$")

plt.loglog()

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
