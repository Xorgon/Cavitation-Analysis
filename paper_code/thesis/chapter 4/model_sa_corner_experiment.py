import importlib
import sys

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

import util.analysis_utils as au
import util.bem as bem
import util.gen_utils as gen


def get_corner_sinks(bubble_pos, corner_n):
    sinks = []
    dist = np.linalg.norm(bubble_pos)
    theta_b = np.arctan2(bubble_pos[1], bubble_pos[0])  # Original bubble angle
    theta_b2 = np.pi * (1 - 1 / corner_n) - theta_b  # Mirrored bubble angle

    for k in range(corner_n):
        angle_b = theta_b + k * 2 * np.pi / corner_n
        angle_b2 = theta_b2 + k * 2 * np.pi / corner_n

        sinks.append([dist * np.cos(angle_b), dist * np.sin(angle_b), 0])
        sinks.append([dist * np.cos(angle_b2), dist * np.sin(angle_b2), 0])

    return np.array(sinks)


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


dirs = ["../Data/60 degree corner/",
        "../Data/90 degree corner/90 degree corner/",
        "../Data/90 degree corner/90 degree corner 2/",
        "../Data/90 degree corner/90 degree corner 3/",
        "../Data/90 degree corner/90 degree corner 4/"]

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
    norm_disps = []

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    corner_angle = np.deg2rad(params.corner_angle_deg)

    readings = au.load_readings(dir_path + "readings_dump.csv")

    corner = params.corner

    n = 50000
    density_ratio = 0.5

    centroids, normals, areas = gen.gen_vertical_varied_corner(n, 50 / 1000, 50 / 1000, angle=corner_angle,
                                                               thresh=30 / 1000, density_ratio=density_ratio)

    for j, reading in enumerate(readings):
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(readings)} | Total readings: {total_readings}")

        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        x = (pos[0] - corner[0]) / 1000
        y = (pos[1] - corner[1]) / 1000

        R_init = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px / 1000
        displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px / 1000

        bubble_pos = [x, y, 0]

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        if params.corner_angle_deg == 60:
            sink_positions = get_corner_sinks(bubble_pos, 3)
            N = 3
        elif params.corner_angle_deg == 90:
            sink_positions = get_corner_sinks(bubble_pos, 2)
            N = 2
        else:
            sink_positions = []
            print("Invalid corner angle.")

        force_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density)


        def peak_event(t, lhs, *args):
            return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


        peak_event.terminal = True  # End the integration at the peak
        peak_event.direction = -1  # Only detect the maximum

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P), events=peak_event)

        kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                                   x=out.t[:])
        anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

        vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)  # Use F' to get direction
        reading.model_anisotropy = vect_anisotropy

        total_readings += 1

    au.save_readings(dir_path, readings)  # Add anisotropy

print(f"Total readings = {total_readings}")
