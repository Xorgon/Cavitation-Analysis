import os
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from experimental.util.analysis_utils import load_readings, Reading
import numpy as np
from common.util.plotting_utils import initialize_plt
from scipy.optimize import curve_fit
from scipy.stats import anderson

# label, max_ecc, alpha, marker, m_size, colour, dirs
sets = [
    ["Slots", 0.20, 1, 'o', 1.5, 'C0',
     ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W1H3/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H12/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3a/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H6/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H9/",
      "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W4H12/"]],

    ["Square", 0.205, 1, 's', 1.5, 'C5',
     ["E:/Data/Lebo/Restructured Data/Square/",
      "E:/Data/Lebo/Restructured Data/Square 2/"]],

    ["Triangle", 0.22, 1, '<', 1.5, 'C4',
     ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
      "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]],

    ["$90^\\circ$ corner", 0.22, 1, 'x', 1.5, 'C2',
     ["E:/Data/Ivo/Restructured Data/90 degree corner/",
      "E:/Data/Ivo/Restructured Data/90 degree corner 2/",
      "E:/Data/Ivo/Restructured Data/90 degree corner 3/",
      "E:/Data/Ivo/Restructured Data/90 degree corner 4/"]],

    ["$60^\\circ$ corner", 0.28, 1, '+', 1.5, 'C1',
     ["E:/Data/Ivo/Restructured Data/60 degree corner/"]],

    ["Flat plate", 0.28, 1, '_', 1.7, 'C3',
     ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]],
]

# initialize_plt(font_size=14, line_scale=1.5, dpi=300)

initialize_plt()
x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

plt.figure(figsize=(5, 3))

max_eccs = np.logspace(np.log10(0.15), np.log10(0.65), 50)
stds = []
n_points = []
last_filter = 0
for k, max_ecc in enumerate(max_eccs):
    min_anisotropy = np.inf
    max_anisotropy = 0

    all_anisotropies = []
    all_disps = []

    total_readings = 0


    def r_filter(r: Reading):
        # mm_per_px scaling doesn't matter here, just using this function to get the right direction conventions
        theta = np.arccos(np.dot(r.get_disp_mm(1), r.model_anisotropy) /
                          (np.linalg.norm(r.get_disp_mm(1)) * np.linalg.norm(r.model_anisotropy)))
        return r.ecc_at_max < max_ecc and r.model_anisotropy is not None and theta < np.deg2rad(1)


    for j, (label, _, alpha, marker, m_size, colour, dirs) in enumerate(sets):
        set_readings = 0
        used_readings = 0
        radii = []
        eccs = []
        for i, dir_path in enumerate(dirs):
            if i == 0:
                this_label = label
            else:
                this_label = None
            try:
                # continue
                readings = load_readings(dir_path + "readings_dump.csv")

                f_vals = [(np.linalg.norm(r.model_anisotropy),
                           np.linalg.norm(r.get_normalised_displacement()),
                           r.get_radius_ratio(),
                           np.sqrt(r.max_bubble_area / np.pi),
                           r.ecc_at_max)
                          for r in readings if r_filter(r)]

                if len(f_vals) == 0:
                    continue

                (filtered_anisotropies,
                 filtered_displacements,
                 filtered_radius_ratios,
                 filtered_max_radii,
                 filtered_eccs) = zip(*f_vals)

                radii.extend(filtered_max_radii)
                eccs.extend(filtered_eccs)

                set_readings += len(readings)
                used_readings += len(filtered_anisotropies)
                total_readings += len(filtered_anisotropies)

                min_anisotropy = np.min([np.min(filtered_anisotropies), min_anisotropy])
                max_anisotropy = np.max([np.max(filtered_anisotropies), max_anisotropy])

                all_anisotropies.extend(filtered_anisotropies)
                all_disps.extend([np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                                  r.ecc_at_max < max_ecc and r.model_anisotropy is not None])
            except FileNotFoundError:
                print(f"Not yet processed {dir_path}")

        if set_readings == 0:
            continue
        print(f"{label:20s}: {used_readings:4d} of {set_readings:4d} "
              f"({100 * used_readings / set_readings:2.2f} %) | max_ecc = {max_ecc:0.3f}"
              f" | Mean bubble radius {np.mean(radii):2.2f} px"
              f" | Mean eccentricity {np.mean(eccs):.2f}")

        if used_readings / set_readings < 1:
            last_filter = k

    print(f"Total readings = {total_readings}")
    if total_readings < 3:
        stds.append(np.nan)
        n_points.append(np.nan)
        continue

    filt_anisotropies, filt_disps = zip(*[(z, dis) for z, dis in sorted(zip(all_anisotropies, all_disps))
                                          if min_anisotropy <= z <= max_anisotropy])
    (a, b) = np.polyfit(np.log10(filt_anisotropies), np.log10(filt_disps), 1)

    (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, filt_anisotropies, filt_disps)

    # Error analysis code
    lin_diffs = filt_disps - c * np.array(filt_anisotropies) ** d
    # plt.hist(lin_diffs, bins=96, alpha=0.5, label="Linear space")
    print(f"Linear: {anderson(lin_diffs)}")
    # plt.hist(np.log10(filt_disps) - (a * np.log10(filt_anisotropies) + b), bins=96, alpha=0.5, label="Log space")
    print(f"Log: {anderson(np.log10(filt_disps) - (a * np.log10(filt_anisotropies) + b))}")
    print(len(lin_diffs))
    print(f"Mean = {np.mean(lin_diffs):.2f}, standard deviation = {np.std(lin_diffs):.4f}")
    stds.append(np.std(lin_diffs))
    n_points.append(len(lin_diffs))
    # plt.xlabel("$\\Delta / R_0$ Difference from curve fit")
    # plt.ylabel("Frequency")
    # plt.legend()

# plt.axvline(max_eccs[last_filter + 1], color="gray", linestyle="dashed")
std_scat = plt.scatter(max_eccs, stds, color="C0", label="Standard deviation $\\sigma$", marker="o", s=2 ** 2)
plt.xlabel("Maximum eccentricity")
plt.ylabel("Standard deviation $\\sigma$")
plt.gca().twinx()
n_scat = plt.scatter(max_eccs, n_points, color="C1", label="Number of data points", marker="s", s=2 ** 2)
plt.xlabel("Maximum eccentricity")
plt.ylabel("Number of data points")
plt.legend(handles=[std_scat, n_scat], frameon=False)
plt.tight_layout()

plt.figure(figsize=(5, 3))
n_points = np.array(n_points)
plt.scatter((max_eccs[1:] + max_eccs[0:-1]) / 2, (n_points[1:] - n_points[0:-1]) / (max_eccs[1:] - max_eccs[0:-1]))
plt.xlabel("Eccentricity")
plt.ylabel("Gradient of number of data points")
plt.tight_layout()
plt.show()
