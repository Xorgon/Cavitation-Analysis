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

min_vert = None
max_vert = None
min_hor = None
max_hor = None
min_rad = None
max_rad = None
for reading_y in reading_ys:

    m_xs = []
    p_bars = []
    thetas = []
    theta_deviations = []
    mean_thetas = []
    thetas_normalised = []
    areas = []
    stds = []
    displacements = []
    # Post-process data to get jet angles.
    for reading in readings:
        if reading.m_y != reading_y:
            continue
        m_xs.append(reading.m_x)
        pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
        p_bar = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
        p_bars.append(p_bar)
        theta = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
        thetas.append(theta)
        areas.append(reading.max_bubble_area)
        radius = params.mm_per_px * np.sqrt(reading.max_bubble_area / np.pi)
        displacements.append((np.linalg.norm(reading.disp_vect) * params.mm_per_px)
                             / radius)

        hor = abs((pos_mm[0] - x_offset) / radius)
        vert = abs((pos_mm[1] - y_offset) / radius)

        if min_hor is None or hor < min_hor:
            min_hor = hor
        if max_hor is None or hor > max_hor:
            max_hor = hor
        if min_vert is None or vert < min_vert:
            min_vert = vert
        if max_vert is None or vert > max_vert:
            max_vert = vert
        if min_rad is None or radius < min_rad:
            min_rad = radius
        if max_rad is None or radius > max_rad:
            max_rad = radius

    for k, (x, theta) in enumerate(zip(m_xs, thetas)):
        theta_set = [j for i, j in zip(m_xs, thetas) if i == x]
        mean_theta = np.mean(theta_set)
        mean_thetas.append(mean_theta)
        theta_deviations.append(theta - mean_theta)
        stds.append(np.std(theta_set))
        if p_bars[k] < 0:
            thetas_normalised.append(theta_deviations[-1] * -1)
        else:
            thetas_normalised.append(theta_deviations[-1])

    radii_mm = 2 * np.sqrt(np.divide(areas, np.pi)) * params.mm_per_px
    print(f"Spearman Correlation: {stats.spearmanr(thetas_normalised, areas)}")
    lin_fit_coeffs = np.polyfit(radii_mm, thetas_normalised, 1)
    print(f"Gradient = {lin_fit_coeffs[0]}")
    fit_areas = np.linspace(min(areas), max(areas), 25)
    fit_thetas = lin_fit_coeffs[0] * fit_areas + lin_fit_coeffs[1]
    plt.xlabel("Bubble diameter (mm)")
    plt.ylabel("Normalised $\\theta$ deviation (rads)")
    plt.scatter(radii_mm, thetas_normalised)
    plt.axhline(0, color="gray")
    plt.axvline(np.mean(radii_mm), color="gray")
    print(len([r for r in radii_mm if r < np.mean(radii_mm)]) / len(
        radii_mm))
    print(len([r for r, theta in zip(radii_mm, thetas_normalised) if r < np.mean(radii_mm) and theta < 0]) / len(
        radii_mm))
    # plt.plot(radii_mm, fit_thetas, 'r')
    # plt.colorbar(label="$\\bar{p}$")
    plt.show()

    # plt.xlabel("Bubble area (px)")
    # plt.ylabel("$\\theta$ deviation (rads)")
    # plt.scatter(areas, theta_deviations, c=p_bars)
    # plt.colorbar(label="$\\bar{p}$")
    # plt.show()

    plt.figure(figsize=(7, 4))
    plt.xlabel("$\\Delta / R$")
    plt.ylabel("Normalised $\\theta$ deviation (rads)")
    plt.axhline(0, color="black", linestyle="--")
    plt.scatter(displacements, thetas_normalised, c='k')
    if reading_y == 1:
        plt.xlim((0.45, 1))
        plt.ylim((-0.075, 0.075))
    if reading_y == 2:
        plt.xlim((0.38, 0.83))
        plt.ylim((-0.075, 0.075))
    plt.tight_layout()
    plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("$\\theta$ deviation (rads)")
    plt.scatter(p_bars, theta_deviations, c=radii_mm)
    plt.colorbar(label="Bubble radius (mm)")
    plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("Standard deviation of $\\theta$")
    plt.scatter(p_bars, stds)
    plt.show()

    plt.xlabel("$\\bar{p}$")
    plt.ylabel("Mean $\\theta$")
    plt.scatter(p_bars, mean_thetas)
    plt.show()

    plt.hist(radii_mm, bins=100)
    plt.xlabel("Bubble radius (mm)")
    plt.show()

print(f"Horizontal: min = {min_hor}, max = {max_hor}")
print(f"Vertical: min = {min_vert}, max = {max_vert}")
print(f"Diameter: min = {min_rad * 2}, max = {max_rad * 2}")
