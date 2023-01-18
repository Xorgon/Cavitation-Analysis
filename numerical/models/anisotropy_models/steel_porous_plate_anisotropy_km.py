import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks

import experimental.util.analysis_utils as au
import numerical.bem as bem
import numerical.util.gen_utils as gen
from util.file_utils import lists_to_csv


def km_inertial(t, lhs, p_v, p_inf, c, density, R_init, p_g0, gamma):
    """ Based on equation 6 in Tinguely et al. (2012). """
    R = lhs[0]
    R_dot = lhs[1]

    v = R_dot / c
    p_g = p_g0 * (R_init / R) ** (3 * gamma)
    p_g_dot = -3 * gamma * p_g0 * R_init ** (3 * gamma) * R ** (-3 * gamma - 1) * R_dot  # Chain rule

    R_dotdot = ((p_g - (p_inf - p_v)) * (1 + v)
                + R * p_g_dot / c - (3 - v) * R_dot ** 2 * density / 2) / ((1 - v) * R * density)

    return [R_dot, R_dotdot]


def peak_event(t, lhs, *args):
    return np.nan if lhs[0] == R_init else lhs[1]  # R_dot


peak_event.terminal = True  # End the integration at the peak
peak_event.direction = -1  # Only detect the maximum


def km_pg0_opt(p_v, p_inf, c, density, R_init, radius_ratio, gamma):
    ray_col_time = 0.915 * R_init * (density / (p_inf - p_v)) ** 0.5
    max_length = 5 * ray_col_time

    def get_rr_diff(p_g0):
        ivp_out = solve_ivp(km_inertial, (0, max_length), (R_init, 0), events=peak_event,
                            args=(p_v, p_inf, c, density, R_init, p_g0, gamma))
        rr = ivp_out.y[0, -1] / ivp_out.y[0, 0]
        return np.abs(rr - radius_ratio)

    res = minimize(get_rr_diff, p_v, method="Nelder-Mead", bounds=[(1e-5, 5 * p_v)])
    return res.x


def km_pg0_fit(p_v, p_inf, c, density, gamma, data_Rs, data_ts):
    """ This would need to be integrated into an area with access to R and t data, maybe during analysis? TODO """
    ray_col_time = 0.915 * R_init * (density / (p_inf - p_v)) ** 0.5
    max_length = 5 * ray_col_time

    def get_km_rs_ts(p_g0):
        ivp_out = solve_ivp(km_inertial, (0, max_length), (R_init, 0), events=peak_event,
                            args=(p_v, p_inf, c, density, R_init, p_g0, gamma))
        return ivp_out.y[0, :], ivp_out.t

    p_g0, _ = curve_fit(get_km_rs_ts, data_Rs, data_ts, p_v, bounds=[(1e-5, 5 * p_v)])
    return p_g0[0]


groups = [
    ["1 mm steel 50 mm plate, $\\phi = 0.24$ - circles",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 1/"],
      [
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 2/"]],
     0.24],
    ["1 mm steel 50 mm plate, $\\phi = 0.26$ - triangles",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
       "w12vf32triangles/On a hole/"],
      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
       "w12vf32triangles/Between 6 holes/"]],
     0.258],
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
p_vapour = 2.3388e3
p_inf = 100e3
polytropic_const = 1.33  # ratio of specific heats of water vapour
c = 1480  # speed of sound in water

reanalyse = True

total_readings = 0

for i, dir_path in enumerate(dirs):
    zetas = []
    disps = []
    radius_ratios = []

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

        ray_col_time = 0.915 * R_init * (density / (p_inf - p_vapour)) ** 0.5
        sim_length = 4 * ray_col_time

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        phi_prime = bem.calculate_phi_prime(bubble_pos, centroids, areas, sigmas=sigmas)
        force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)

        p_g0 = km_pg0_opt(p_vapour, p_inf, c, density, R_init, reading.get_radius_ratio(), polytropic_const)[0]
        print(f"p_g0 = {p_g0:.2f}")

        out = solve_ivp(km_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                        args=(p_vapour, p_inf, c, density, R_init, p_g0, polytropic_const),
                        events=peak_event)

        kelvin_impulse = trapezoid(16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
                                   x=out.t)
        anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (p_inf - p_vapour)))

        vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)
        reading.model_anisotropy = vect_anisotropy

        total_readings += 1
        zetas.append(anisotropy)
        disps.append(displacement / R_init)
        radius_ratios.append(np.sqrt(reading.sec_max_area / reading.max_bubble_area))

    lists_to_csv(dir_path, "zetas_vs_disps.csv", [zetas, disps, radius_ratios],
                 headers=["zeta", "disp", "radius ratios"], overwrite=True)
    au.save_readings(dir_path, readings)  # Add anisotropy
    plt.scatter(zetas, disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
