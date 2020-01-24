import math
import sys
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

from experimental.plotting.analyse_slot import SweepData, select_data_series
import experimental.util.analysis_utils as au
from numerical.models.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem
from common.util.plotting_utils import initialize_plt

dirs = select_data_series(use_all_dirs=False, num_series=1)

sweeps = []
for dir_path in dirs:
    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    x_offset = params.left_slot_wall_x + params.slot_width / 2
    y_offset = params.upper_surface_y

    if os.path.exists(dir_path + "invalid_readings.txt") and os.path.exists(dir_path + "readings_dump.csv"):
        au.flag_invalid_readings(dir_path)
    readings = au.load_readings(dir_path + "readings_dump.csv")
    readings = sorted(readings, key=lambda r: r.m_x)

    reading_ys = set([reading.m_y for reading in readings])

    for reading_y in reading_ys:
        if hasattr(params, 'title'):
            label = f"{params.title}"
        else:
            label = f"{dir_path}"

        sweep_data = SweepData(label, reading_y, params.slot_width, params.slot_height)
        sweep_readings = [reading for reading in readings if reading.m_y == reading_y]
        # Post-process data to get jet angles.
        for reading in sweep_readings:
            theta_j = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
            pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
            p_bar = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
            q = pos_mm[1] - y_offset
            sweep_data.add_point(reading.m_x, p_bar, theta_j, q)
        if np.mean(sweep_data.qs) < 0:
            continue  # Ignore bubbles generated inside the slot.
        sweeps.append(sweep_data)

num_rejected_sets = 0
print(f"Found {len(sweeps)} sweeps")
markers = [".", "v", "s", "x", "^", "+", "D", "1", "*", "P", "X", "4", "2", "<", "3", ">", "H", "o", "p", "|"]
if len(markers) < len(sweeps):
    raise ValueError("Too few markers are available for the data sets.")

ws = []
hs = []
qs = []
exp_p_bar_stars = []
exp_theta_j_stars = []
for sweep in sweeps:
    (max_fitted_peak_p, max_fitted_peak, _), (min_fitted_peak_p, min_fitted_peak, _) = sweep.get_curve_fits()

    qs.append(np.mean(sweep.qs))
    hs.append(sweep.h)
    ws.append(sweep.w)

    p_bar_star = (max_fitted_peak_p - min_fitted_peak_p) / 2
    theta_j_star = (max_fitted_peak - min_fitted_peak) / 2

    exp_p_bar_stars.append(p_bar_star)
    exp_theta_j_stars.append(theta_j_star)

# Should just be equal
h = np.mean(hs)
w = np.mean(ws)

m_0 = 1
n = 20000
w_thresh = 15
density_ratio = 0.15

centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=100, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

num_p_bar_stars = []
num_theta_js_stars = []
num_qs = np.linspace(0.5 * w, 5 * w, 10)
for q in num_qs:
    _, p, theta_j, _ = find_slot_peak(w, q, h, n, 100, 50, density_ratio, w_thresh, centroids, normals, areas, R_inv,
                                      m_0)
    num_p_bar_stars.append(2 * p / w)
    num_theta_js_stars.append(theta_j)

initialize_plt()
fig, axes = plt.subplots(2, 1, sharex='all', figsize=(5.31445, 4))
axes[0].scatter(np.divide(qs, w), exp_theta_j_stars, color='k', label="Experimental")
axes[0].plot(np.divide(num_qs, w), num_theta_js_stars, 'r', label="Numerical")
axes[0].legend()
axes[0].set_ylabel("$\\theta_j^\\star$")
axes[1].scatter(np.divide(qs, w), exp_p_bar_stars, color='k', label="Experimental")
axes[1].plot(np.divide(num_qs, w), num_p_bar_stars, 'r', label="Numerical")
axes[1].legend()
axes[1].set_ylabel("$\\bar{p}^\\star$")
axes[-1].set_xlabel("$q / w$")
plt.tight_layout()
plt.show()
