import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
from util.file_utils import lists_to_csv
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import random


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


groups = [
    ["1 mm acrylic 50 mm plate, $\\phi = 0.525$",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/Between 3 holes/",
         "1mm acrylic 50mm plate, between 3 holes, 52.5% VF", "C0", "^"],
         [
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/On hole/",
             "1mm acrylic 50mm plate, above a hole, 52.5% VF", "C0", "o"]], 0.525],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.38$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 38% VF", "C9", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/On hole/",
       "1mm acrylic 50mm plate, above a hole, 38% VF", "C9", "o"]], 0.38],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.24$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 24% VF", "C5", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
       "1mm acrylic 50mm plate, above a hole, 24% VF", "C5", "o"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C5", "^"]], 0.24],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.16$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 16% VF", "C7", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/On hole/",
       "1mm acrylic 50mm plate, above a hole, 16% VF", "C7", "o"]], 0.16],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.12$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 12% VF", "C8", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/On hole/",
       "1mm acrylic 50mm plate, above a hole, 12% VF", "C8", "o"]], 0.12],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.08$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 8% VF", "C10", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/On hole/",
       "1mm acrylic 50mm plate, above a hole, 8% VF", "C10", "o"]], 0.08],

    ["1 mm acrylic 50 mm solid plate",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
       "1mm acrylic 50mm solid plate", "C6", "s"]], 0],
]

dirs = []
vfs = []
for g in groups:
    for d in g[1]:
        dirs.append(d[0])
        vfs.append(g[2])
print(f"Found {len(dirs)} data sets")
print(dirs)

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

total_readings = 0

reanalyse = False

for i, dir_path in enumerate(dirs):
    zetas = []
    disps = []

    vf = vfs[i]

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv")

    if any([r.model_anisotropy is not None for r in readings]) and not reanalyse:
        print(f"Already analysed {dir_path}")
        continue

    centroids, normals, areas = gen.gen_varied_simple_plane(0.045, 10000, 0.05, 2000)
    areas *= (1 - vf)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float64)
    R_inv = scipy.linalg.inv(R_matrix)

    # random.shuffle(readings)
    for j, reading in enumerate(readings):
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(readings)} | Total readings: {total_readings}")
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        y = (pos[1] - y_offset) / 1000

        if y < 0:
            continue

        R_init = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px / 1000
        displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px / 1000

        bubble_pos = [0, y, 0]

        ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        phi_prime = bem.calculate_phi_prime(bubble_pos, centroids, areas, sigmas=sigmas)
        force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)

        out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(density, delta_P))

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
        reading.model_anisotropy = vect_anisotropy

        total_readings += 1
        zetas.append(anisotropy)
        disps.append(displacement / R_init)

    lists_to_csv(dir_path, "zetas_vs_disps.csv", [zetas, disps], headers=["zeta", "disp"], overwrite=True)
    au.save_readings(dir_path, readings)  # Add anisotropy
    plt.scatter(zetas, disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
