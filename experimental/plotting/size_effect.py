import math
from typing import List
import sys

import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt

dir_path = "../../../../Data/SlotSweeps/W2H3b/"

sys.path.append(dir_path)
import params

x_offset = params.left_slot_wall_x + params.slot_width / 2
y_offset = params.upper_surface_y

readings = au.load_readings(dir_path + "readings_dump.csv")
readings = sorted(readings, key=lambda r: r.m_x)  # type: List[au.Reading]

reading_ys = [1.0, 2.0]
x_ranges = []
std_ranges = []
initialize_plt()
plt.tight_layout()
for reading_y in reading_ys:

    m_xs = []
    p_bars = []
    theta_js = []
    theta_j_deviations = []
    mean_theta_js = []
    theta_js_normalized = []
    areas = []
    stds = []
    # Post-process data to get jet angles.
    for reading in readings:
        if reading.m_y != reading_y:
            continue
        m_xs.append(reading.m_x)
        pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
        p_bar = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
        p_bars.append(p_bar)
        theta_j = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
        theta_js.append(theta_j)
        areas.append(reading.max_bubble_area)

    for k, (x, theta_j) in enumerate(zip(m_xs, theta_js)):
        theta_j_set = [j for i, j in zip(m_xs, theta_js) if i == x]
        mean_theta_j = np.mean(theta_j_set)
        mean_theta_js.append(mean_theta_j)
        theta_j_deviations.append(theta_j - mean_theta_j)
        stds.append(np.std(theta_j_set))
        if p_bars[k] < 0:
            theta_js_normalized.append(theta_j_deviations[-1] * -1)
        else:
            theta_js_normalized.append(theta_j_deviations[-1])

    radii_mm = 2 * np.sqrt(np.divide(areas, np.pi)) * params.mm_per_px
    print(f"Spearman Correlation: {stats.spearmanr(theta_js_normalized, areas)}")
    lin_fit_coeffs = np.polyfit(radii_mm, theta_js_normalized, 1)
    print(f"Gradient = {lin_fit_coeffs[0]}")
    fit_areas = np.linspace(min(areas), max(areas), 25)
    fit_theta_js = lin_fit_coeffs[0] * fit_areas + lin_fit_coeffs[1]
    plt.xlabel("Bubble diameter (mm)")
    plt.ylabel("Normalized $\\theta_j$ deviation (rads)")
    plt.scatter(radii_mm, theta_js_normalized)
    plt.axhline(0, color="gray")
    plt.axvline(np.mean(radii_mm), color="gray")
    print(len([r for r in radii_mm if r < np.mean(radii_mm)]) / len(
        radii_mm))
    print(len([r for r, theta in zip(radii_mm, theta_js_normalized) if r < np.mean(radii_mm) and theta < 0]) / len(
        radii_mm))
    # plt.plot(radii_mm, fit_theta_js, 'r')
    # plt.colorbar(label="$\\bar{p}$")
    plt.show()

    # plt.xlabel("Bubble area (px)")
    # plt.ylabel("$\\theta_j$ deviation (rads)")
    # plt.scatter(areas, theta_j_deviations, c=p_bars)
    # plt.colorbar(label="$\\bar{p}$")
    # plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("$\\theta_j$ deviation (rads)")
    plt.scatter(p_bars, theta_j_deviations, c=radii_mm)
    plt.colorbar(label="Bubble radius (mm)")
    plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("Standard deviation of $\\theta_j$")
    plt.scatter(p_bars, stds)
    plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("Mean $\\theta_j$")
    plt.scatter(p_bars, mean_theta_js)
    plt.show()

    plt.hist(radii_mm, bins=100)
    plt.xlabel("Bubble radius (mm)")
    plt.show()
