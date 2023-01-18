import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import util.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from util.moi_utils import sink_positions_square

import util.analysis_utils as au
import util.gen_utils as gen


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


dirs = [
    "../Data/Square/Square/",
    "../Data/Square/Square 2/",
    "../Data/Square/Square 3/"
]

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

total_readings = 0

for i, dir_path in enumerate(dirs):
    zetas = []
    disps = []
    radius_ratios = []

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    readings = au.load_readings(dir_path + "readings_dump.csv")

    # Center of square, mm ([ref_x, ref_y] is bottom right)
    origin = [params.ref_x - params.length / 2, params.ref_y + params.length / 2]

    n = 50000
    centroids, normals, areas = gen.gen_rectangle(n, params.length / 1000, params.length / 1000, depth=0.2)
    # Shift centroids so that the center is at the origin
    centroids = centroids - np.array([params.length / 2000, params.length / 2000, 0])

    for j, reading in enumerate(readings):
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(readings)} | Total readings: {total_readings}")
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        x = (pos[0] - origin[0]) / 1000
        y = (pos[1] - origin[1]) / 1000

        R_init = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px / 1000
        displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px / 1000

        bubble_pos = [x, y, 0]

        sink_positions = sink_positions_square(params.length / 1000, bubble_pos, x_range=32, y_range=32)
        print("    M = ", len(sink_positions))
        sink_positions = np.array(sink_positions)

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)
        phi_prime = 0

        print(f"F' = {force_prime}")


        def peak_event(t, lhs, *args):
            return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


        peak_event.terminal = True  # End the integration at the peak
        peak_event.direction = -1  # Only detect the maximum

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P))

        kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                                   x=out.t[:])
        anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

        vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)
        reading.model_anisotropy = vect_anisotropy

        print(vect_anisotropy)

        total_readings += 1
        zetas.append(anisotropy)
        disps.append(displacement / R_init)
        radius_ratios.append(np.sqrt(reading.sec_max_area / reading.max_bubble_area))

    au.save_readings(dir_path, readings)  # Add anisotropy
    plt.scatter(zetas, disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
