import math
import sys
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scipy
from scipy import stats

from experimental.plotting.analyse_slot import SweepData
import experimental.util.analysis_utils as au
from numerical.models.slot.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem
from common.util.plotting_utils import initialize_plt
from common.util.file_utils import lists_to_csv

tags = ['W1H3', 'W2H3a']

initialize_plt()
fig, axes = plt.subplots(2, 1, sharex='all', figsize=(5.31445, 4))
legend_handles = []

colors = ['r', 'g', 'm', 'orange', 'b']
for i in range(len(tags)):
    dir_path = f'C:/Users/eda1g15/OneDrive - University of Southampton/' \
               f'Research/Slot Geometries/Data/SlotSweeps/{tags[i]}/'
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
            theta = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
            pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
            x = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
            Y = pos_mm[1] - y_offset
            sweep_data.add_point(reading.m_x, x, theta, Y)
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
    exp_theta_stars = []
    theta_errs = []
    x_star_errs = []
    for sweep in sweeps:
        (max_fitted_peak_X, max_fitted_peak, _), \
        (min_fitted_peak_X, min_fitted_peak, _), combined_tj_std, _ \
            = sweep.get_curve_fits()
        Ys.append(np.mean(sweep.Ys))
        Hs.append(sweep.H)
        Ws.append(sweep.W)

        x_star = (max_fitted_peak_X - min_fitted_peak_X) / 2
        theta_star = (max_fitted_peak - min_fitted_peak) / 2

        exp_x_stars.append(x_star)
        exp_theta_stars.append(theta_star)
        # https://stackoverflow.com/a/28243282/5270376

        interval = stats.norm.interval(0.99, loc=theta_star, scale=combined_tj_std)
        theta_errs.append(interval[1] - theta_star)

        x_star_errs.append(0.075)  # Based on synthesised data testing.

    # Should just be equal
    H = np.mean(Hs)
    W = np.mean(Ws)

    m_0 = 1
    n = 20000
    w_thresh = 15
    density_ratio = 0.15

    print(theta_errs)

    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=100, depth=50, w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)

    num_x_stars = []
    num_theta_stars = []
    num_Ys = np.linspace(0.5 * W, 5 * W, 16)
    for Y in num_Ys:
        _, X, theta, _ = find_slot_peak(W, Y, H, n, 100, 50, density_ratio, w_thresh,
                                        centroids, normals, areas, R_inv, m_0)
        num_x_stars.append(2 * X / W)
        num_theta_stars.append(theta)

    num_ys = np.divide(num_Ys, W)
    exp_ys = np.divide(Ys, W)

    lists_to_csv("model_outputs/peak_position_comparison/", f"{tags[i]}_numerical",
                 [num_ys, num_theta_stars, num_x_stars], headers=["y", "theta_star", "x_star"])
    lists_to_csv("model_outputs/peak_position_comparison/", f"{tags[i]}_experimental",
                 [exp_ys, exp_theta_stars, theta_errs, exp_x_stars, x_star_errs],
                 headers=["y", "theta_star", "theta_star_err", "x_star", "x_star_err"])

    color = f'C{i}'
    legend_handles.append(mpatches.Patch(color=color, label=params.title, linewidth=0.1, capstyle='butt'))
    axes[0].plot(num_ys, num_theta_stars, color, label="Numerical")
    axes[0].errorbar(exp_ys, exp_theta_stars, yerr=theta_errs, color=color, capsize=3, fmt='.',
                     label="Experimental")
    axes[0].set_ylabel("$\\theta^\\star$ (rad)")
    axes[1].plot(num_ys, num_x_stars, color, label="Numerical")
    axes[1].errorbar(exp_ys, exp_x_stars, yerr=x_star_errs, color=color, capsize=3, fmt='.', linestyle=" ",
                     label="Experimental")
    axes[1].set_ylabel("$x^\\star$")
    axes[-1].set_xlabel("$y$")

legend_handles.append(mlines.Line2D([], [], color='k', marker='.', label='Experimental', linestyle=' ',
                                    markersize=5))
legend_handles.append(mlines.Line2D([], [], color='k', label='Numerical'))
axes[0].legend(handles=legend_handles, frameon=False)
axes[0].annotate("($a$)", xy=(np.mean(axes[0].get_xlim()), np.mean(axes[0].get_ylim())), xytext=(0.025, 0.95),
                 textcoords='axes fraction', horizontalalignment='left', verticalalignment='top')
axes[1].annotate("($b$)", xy=(np.mean(axes[1].get_xlim()), np.mean(axes[1].get_ylim())), xytext=(0.025, 0.95),
                 textcoords='axes fraction', horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.show()
