import numpy as np
import numerical.util.gen_utils as gen
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import numerical.bem as bem
import matplotlib.pyplot as plt


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

standoff = 3

sa_anisotropies = []

bubble_position = np.array([0, R_init * standoff, 0])

sink_positions = np.array([bubble_position,
                           -bubble_position])

n = 6561

# Mesh on which to integrate pressure
# centroids, normals, areas = gen.gen_varied_simple_plane(0.25, 20000, 3, 10000)
pv_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Miscellaneous/Paraview testing/"
length = 0.05
centroids, normals, areas = gen.gen_plane([- length / 2, 0, - length / 2],
                                          [- length / 2, 0, length / 2],
                                          [length / 2, 0, - length / 2],
                                          n, filename=pv_dir + "plate.csv")

force_prime, press_prime = bem.calculate_force_prime_semi_analytic(sink_positions, centroids, normals, areas, density,
                                                                   output_pressures=True)

ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
sim_length = 4 * ray_col_time


def peak_event(t, lhs, *args):
    return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


peak_event.terminal = True  # End the integration at the peak
peak_event.direction = -1  # Only detect the maximum

out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                args=(density, delta_P), events=peak_event)

target_length = 3  # seconds
fps = 30
total_frames = target_length * fps
available_frames = len(out.t)
step = np.floor(available_frames / total_frames)

for frame, (t, r, r_dot) in enumerate(zip(out.t[:], out.y[0, :], out.y[1, :])):
    if frame % step != 0:
        continue

    file = open(pv_dir + f"pressures/n{n}t{t * 1e9:.0f}.csv", 'w')
    file.write("x, y, z, pressure\n")
    for i, p in enumerate(press_prime):
        file.write(f"{centroids[i][0]}, {centroids[i][1]}, {centroids[i][2]}, "
                   f" {p * 16 * np.pi ** 2 * r ** 4 * r_dot ** 2}\n")
    file.close()

    bfile = open(pv_dir + f"bubbles/t{t * 1e9:.0f}.csv", 'w')
    bfile.write("x, y, z, radius\n")
    bfile.write(f"{bubble_position[0]}, {bubble_position[1]}, {bubble_position[2]}, {r}\n")
    bfile.close()

kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                           x=out.t[:])

anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)

sa_anisotropies.append(anisotropy)
