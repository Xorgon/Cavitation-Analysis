import math
import sys
import importlib
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scipy
from scipy import stats

from experimental.plotting.analyse_slot import SweepData, select_data_series
import experimental.util.analysis_utils as au
from numerical.models.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem
from common.util.plotting_utils import initialize_plt

dirs = ['C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W1H3/',
        'C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3a/']

initialize_plt()
fig, axes = plt.subplots(2, 1, sharex='all', figsize=(5.31445, 4))
legend_handles = []

colors = ['r', 'g', 'm', 'orange', 'b']
for i in range(2):
    dir_path = dirs[i]

    # TODO: Figure out why this causes Matplotlib to crash.
    # dir = select_data_series(use_all_dirs=False, num_series=1)
    # print(dir)

    sweeps = []
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
            x = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
            Y = pos_mm[1] - y_offset
            sweep_data.add_point(reading.m_x, x, theta_j, Y)
        if np.mean(sweep_data.Ys) < 0:
            continue  # Ignore bubbles generated inside the slot.
        sweeps.append(sweep_data)

    num_rejected_sets = 0
    print(f"Found {len(sweeps)} sweeps")
    markers = [".", "v", "s", "x", "^", "+", "D", "1", "*", "P", "X", "4", "2", "<", "3", ">", "H", "o", "p", "|"]
    if len(markers) < len(sweeps):
        raise ValueError("Too few markers are available for the data sets.")

    Ws = []
    Hs = []
    Ys = []
    exp_x_stars = []
    exp_theta_j_stars = []
    theta_j_errs = []
    for sweep in sweeps:
        (max_fitted_peak_X, max_fitted_peak, _, max_std_theta_j), \
        (min_fitted_peak_X, min_fitted_peak, _, min_std_theta_j) \
            = sweep.get_curve_fits()
        Ys.append(np.mean(sweep.Ys))
        Hs.append(sweep.H)
        Ws.append(sweep.W)

        x_star = (max_fitted_peak_X - min_fitted_peak_X) / 2
        theta_j_star = (max_fitted_peak - min_fitted_peak) / 2

        exp_x_stars.append(x_star)
        exp_theta_j_stars.append(theta_j_star)
        mean_std = (max_std_theta_j + min_std_theta_j) / 2
        # https://stackoverflow.com/a/28243282/5270376

        interval = stats.norm.interval(0.99, loc=np.mean(theta_j_star), scale=mean_std)
        theta_j_errs.append(interval[1] - theta_j_star)

    # Should just be equal
    H = np.mean(Hs)
    W = np.mean(Ws)

    m_0 = 1
    n = 2000
    w_thresh = 15
    density_ratio = 0.15

    print(theta_j_errs)

    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=100, depth=50, w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)

    num_x_stars = []
    num_theta_js_stars = []
    num_Ys = np.linspace(0.5 * W, 5 * W, 16)
    for Y in num_Ys:
        _, X, theta_j, _ = find_slot_peak(W, Y, H, n, 100, 50, density_ratio, w_thresh,
                                          centroids, normals, areas, R_inv, m_0)
        num_x_stars.append(2 * X / W)
        num_theta_js_stars.append(theta_j)

    color = f'C{i}'
    legend_handles.append(mpatches.Patch(color=color, label=params.title, linewidth=0.1, capstyle='butt'))
    axes[0].plot(np.divide(num_Ys, W), num_theta_js_stars, color, label="Numerical")
    axes[0].errorbar(np.divide(Ys, W), exp_theta_j_stars, yerr=theta_j_errs, color=color, capsize=3, fmt='.',
                     label="Experimental")
    axes[0].set_ylabel("$\\theta_j^\\star$")
    axes[1].plot(np.divide(num_Ys, W), num_x_stars, color, label="Numerical")
    axes[1].errorbar(np.divide(Ys, W), exp_x_stars, yerr=1, color=color, capsize=3, fmt='.', linestyle=" ",
                     label="Experimental")
    axes[1].set_ylabel("$x^\\star$")
    axes[-1].set_xlabel("$y$")

# TODO: Error bars!

legend_handles.append(mlines.Line2D([], [], color='k', marker='.', label='Experimental', linestyle=' ',
                                    markersize=5))
legend_handles.append(mlines.Line2D([], [], color='k', label='Numerical'))
axes[0].legend(handles=legend_handles, frameon=False)
axes[0].annotate("($a$)", xy=(0.5, 0), xytext=(0.025, 0.95), textcoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')
axes[1].annotate("($b$)", xy=(0.5, 0), xytext=(0.025, 0.95), textcoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.show()
