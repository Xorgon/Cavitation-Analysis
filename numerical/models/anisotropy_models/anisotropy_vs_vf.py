import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import curve_fit

import numerical.bem as bem
import numerical.util.gen_utils as gen
from common.util.plotting_utils import initialize_plt


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

R_init = 1.5 / 1000
dist = 5 / 1000

standoffs = np.logspace(np.log10(1), np.log10(10), 5)

color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=1), cmap=cm.get_cmap('viridis'))

vfs = np.linspace(0, 1, 15)
prefactors = []

W = 1.2e-3
delt = 1e-3

initialize_plt()

plt.figure(figsize=(4, 3))

for vf in vfs:
    print(f"Void fraction = {vf:.3f}")
    bem_anisotropies = []
    centroids, normals, areas = gen.gen_varied_simple_plane(0.045, 17000, 0.05, 3000)
    areas = (1 - vf) * areas
    print(f"N = {len(centroids)}")
    # areas = areas if vf == 0 else areas * (1 - vf + 4 * vf * delt / W)

    # print(1 - vf + 4 * vf * delt / W)

    # R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    # R_inv = scipy.linalg.inv(R_matrix)

    # All elements are independent in the normal direction so we can take this shortcut
    R_inv = np.diag(np.array([2] * len(centroids), dtype=np.float32))

    # condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
    # condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
    # print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

    bubble_position = np.array([0, dist, 0])

    sigmas = bem.calculate_sigma(bubble_position, centroids, normals, areas, 1, R_inv)
    force_prime_bem = bem.calculate_force_prime(bubble_position, centroids, normals, areas, sigmas, density)

    for standoff in standoffs:
        print(f"    Standoff = {standoff:.2f}")

        R_init = dist / standoff

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time


        def peak_event(t, lhs, *args):
            return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


        peak_event.terminal = True  # End the integration at the peak
        peak_event.direction = -1  # Only detect the maximum

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P), events=peak_event)

        kelvin_impulse_bem = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime_bem) * out.y[0, :] ** 4 * out.y[1, :] ** 2, x=out.t)

        anisotropy_bem = kelvin_impulse_bem / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))
        bem_anisotropies.append(anisotropy_bem)

    plt.plot(standoffs, bem_anisotropies, color=color_map.to_rgba(vf))
    (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, standoffs, bem_anisotropies)
    print(c, d)
    prefactors.append(c)

plt.loglog()
plt.colorbar(color_map, label="$\\phi$")
plt.xlabel("$\\gamma$")
plt.ylabel("$\\zeta$")
plt.tight_layout()

plt.figure(figsize=(4, 2))
print(vfs, prefactors)
plt.plot(vfs, prefactors, label="Numerical")

# plt.plot(1 - np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.]),
#          [0.1831211283232972, 0.1356203207068641, 0.09653215161309309, 0.06507716721389999, 0.040475913681138224,
#           0.021948937186654156, 0.008716783902318965, -4.019655235379999e-18], color="C0", label="Numerical")

exp_vfs = [0, 0.115, 0.115, 0.073, 0.073, 0.146, 0.146, 0.134, 0.134, 0.216, 0.216, 0.293, 0.293, 0.259, 0.259, 0.356,
           0.356, 0.407, 0.407, 0.441, 0.441, 0.229, 0.229, 0.447, 0.447, 0.527, 0.527, 0.594, 0.594, 0.233, 0.233,
           0.231, 0.231]
exp_pres = [0.195, 0.10959469277949292, 0.11817767980710502, 0.15295006372618478, 0.15551974344483963,
            0.11332525600177362, 0.12210333567240782, 0.13335264543270944, 0.13425712159976447, 0.10023286376971761,
            0.101945517280134, 0.08738366956403104, 0.08650247226768933, 0.08433097890034748, 0.08995658463058363,
            0.07129914365757016, 0.07791661138010292, 0.06739189638185361, 0.06507065634083135, 0.0658439041151578,
            0.06496650909741153, 0.11819299192773591, 0.10254874116515131, 0.07075885422079974, 0.07001411077768599,
            0.06611595435131866, 0.06330772419912876, 0.052458401056923565, 0.05017870797787934, 0.12676427463352702,
            0.11434680991733788, 0.11661715958706882, 0.1103944697802057]

plt.scatter(np.array(exp_vfs), exp_pres, color="C1", label="Experimental")

plt.xlabel("$\\phi$")
plt.ylabel("$c$ where $\\zeta = c \\gamma^{-2}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
