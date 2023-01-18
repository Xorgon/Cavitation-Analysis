import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit, minimize
from scipy.stats import anderson

import experimental.util.file_utils as file
from common.util.file_utils import lists_to_csv
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal, label_subplot
from experimental.util.analysis_utils import load_readings, Reading, analyse_frame
from experimental.util.mraw import mraw

# label, max_ecc, alpha, marker, m_size, colour, dirs
sets = [["Slots", 0.20, 1, 'o', 1.5, 'C0',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W1H3/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H12/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3a/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H6/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H9/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W4H12/"]],

        ["Square", 0.205, 1, 's', 1.5, 'C5',  # 0.205 | 0.234
         ["E:/Data/Lebo/Restructured Data/Square/",
          "E:/Data/Lebo/Restructured Data/Square 2/",
          "E:/Data/Lebo/Restructured Data/Square 3/"]],

        ["Triangle", 0.24, 1, '<', 1.5, 'C3',  # 0.22 | 0.257
         ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
          "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]],

        ["$90^\\circ$ corner", 0.22, 1, 'x', 1.5, 'C1',
         ["E:/Data/Ivo/Restructured Data/90 degree corner/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 2/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 3/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 4/"]],

        ["$60^\\circ$ corner", 0.28, 1, (6, 1, 30), 1.5, 'C4',  # (6, 1, 30) makes a 6-pointed star rotated 30 degrees
         ["E:/Data/Ivo/Restructured Data/60 degree corner/"]],

        ["Flat plate", 0.28, 1, '_', 3, 'C2',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]],
        ]

show_data_labels = False
plot_additionals = True

if not show_data_labels:
    initialize_plt()  # Do not want to plot thousands of text labels with this on

x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

min_anisotropy = np.inf
max_anisotropy = 0

min_rr = np.inf
max_rr = 0

all_anisotropies = []
all_disps = []
all_ratios = []
all_readings = []

fig_width = 7

fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.5))

total_readings = 0

# all_eccs = []
# all_pct_diffs = []

for j, (label, max_ecc, alpha, marker, m_size, colour, dirs) in enumerate(sets):
    # max_ecc = 0.18
    # max_ecc = 1
    set_readings = 0
    used_readings = 0
    radii = []
    eccs = []
    radii_mm = []


    def r_filter(r: Reading):
        return r.ecc_at_max < max_ecc and r.model_anisotropy is not None and r.get_angle_dif() < np.deg2rad(100)


    for i, dir_path in enumerate(dirs):  # type: int, str
        if i == 0:
            this_label = label
        else:
            this_label = None
        try:

            sys.path.append(dir_path)
            import params

            importlib.reload(params)
            sys.path.remove(dir_path)

            readings = load_readings(dir_path + "readings_dump.csv", include_invalid=True)
            filtered_anisotropies = [np.linalg.norm(r.model_anisotropy) for r in readings if r_filter(r)]
            filtered_displacements = [np.linalg.norm(r.sup_disp_vect) / np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)]
            print(filtered_displacements)
            filtered_radius_ratios = [r.get_radius_ratio() for r in readings if r_filter(r)]
            filtered_max_radii = [np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)]
            filtered_eccs = [r.ecc_at_max for r in readings if r_filter(r)]
            filtered_labels = [f"{r.idx}:{r.repeat_number}" for r in readings if r_filter(r)]
            radii.extend(filtered_max_radii)
            radii_mm.extend(np.array(filtered_max_radii) * params.mm_per_px)
            eccs.extend(filtered_eccs)

            # for r in readings:
            #     if any(np.isclose(r.ecc_at_max, look_for_eccs, atol=0.001, rtol=0)):
            #         print(r.ecc_at_max, dir_path, r.idx, r.repeat_number)

            all_readings.extend(readings)  # ALL readings

            set_readings += len(readings)
            used_readings += len(filtered_anisotropies)
            total_readings += len(filtered_anisotropies)

            if len(filtered_anisotropies) == 0:
                continue

            # RUDIMENTARY ERROR OBSERVATION
            # syst_px_err = 3  # Systematic pixel error
            # rand_px_err = 1  # Random pixel error
            # radius_0 = np.array([np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)])
            # radius_1 = np.array([np.sqrt(r.sec_max_area / np.pi) for r in readings if r_filter(r)])
            #
            # ratio_dif = (radius_1 + syst_px_err + rand_px_err) / \
            #             (radius_0 + syst_px_err - rand_px_err) - radius_1 / radius_0
            # print(f"Mean ratio_dif = {np.mean(ratio_dif):.8f}")
            #
            # r_ax.errorbar(filtered_anisotropies,
            #               filtered_radius_ratios,
            #               yerr=ratio_dif,
            #               marker=marker, color=colour, alpha=alpha, markersize=m_size, linestyle='',
            #               linewidth=0.5, capsize=0.5)

            if show_data_labels:
                for l, x, y in zip(filtered_labels, filtered_anisotropies, filtered_displacements):
                    disp_ax.annotate(f'{l}', xy=[x, y], textcoords='data')

            disp_ax.scatter(filtered_anisotropies, filtered_displacements, marker=marker, color=colour,
                            label=this_label, alpha=alpha, s=m_size ** 2)

            r_ax.scatter(filtered_anisotropies, filtered_radius_ratios,
                         marker=marker, color=colour, alpha=alpha, s=m_size ** 2)

            min_anisotropy = np.min([np.min(filtered_anisotropies), min_anisotropy])
            max_anisotropy = np.max([np.max(filtered_anisotropies), max_anisotropy])

            min_rr = np.min([np.min(filtered_radius_ratios), min_rr])
            max_rr = np.max([np.max(filtered_radius_ratios), max_rr])

            all_anisotropies.extend(filtered_anisotropies)
            all_disps.extend(filtered_displacements)
            all_ratios.extend(filtered_radius_ratios)

        except FileNotFoundError:
            print(f"Not yet processed {dir_path}")

    print(f"{label:20s}: {used_readings:4d} of {set_readings:4d} "
          f"({100 * used_readings / set_readings:2.2f} %) | max_ecc = {max_ecc:0.3f}"
          f" | Mean bubble radius {np.mean(radii):2.2f} px ({np.mean(radii_mm):.2f} mm)"
          f" | Mean eccentricity {np.mean(eccs):.2f}")

print(f"Total readings = {total_readings} ({100 * total_readings / len(all_readings):.2f} %)")

filt_anisotropies, filt_disps = zip(*[(z, dis) for z, dis in sorted(zip(all_anisotropies, all_disps))
                                      if min_anisotropy <= z <= max_anisotropy])
(a, b) = np.polyfit(np.log10(filt_anisotropies), np.log10(filt_disps), 1)

(c, d), _ = curve_fit(lambda x, c, d: c * x ** d, filt_anisotropies, filt_disps)
c = 2.5
d = 3/5

def get_contained_data_dif(dc, target=0.95):
    total_in_range = 0
    for zeta, delta in zip(all_anisotropies, all_disps):
        if (c - dc) * zeta ** d < delta < (c + dc) * zeta ** d:
            total_in_range += 1
    return abs(total_in_range / len(all_anisotropies) - target)


opt_dc = minimize(get_contained_data_dif, 0.1, method="Nelder-Mead").x
print(f"Optimum dc = {opt_dc}")

# Plotting the linear space curve fit because...
# a) It puts less weight on the outlier regions (particularly very low anisotropy)
# b) The curve fit qualitatively matches better
# c) The 'errors' in linear space seem to be closer to a normal distribution than in log space
disp_ax.plot(filt_anisotropies, c * np.array(filt_anisotropies) ** d, "k", linestyle="-.",
             label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (c, d))

# disp_ax.plot(filt_anisotropies, (c + opt_dc) * np.array(filt_anisotropies) ** d, "k", linestyle="-.", linewidth=0.5)
# disp_ax.plot(filt_anisotropies, (c - opt_dc) * np.array(filt_anisotropies) ** d, "k", linestyle="-.", linewidth=0.5)

# disp_ax.plot(filt_anisotropies, (10 ** b) * np.array(filt_anisotropies) ** a, "k", linestyle="dotted,
#              label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (10 ** b, a))

legend = disp_ax.legend(loc='lower right', frameon=False, fontsize='x-small', markerfirst=False)
for handle in legend.legendHandles:
    if type(handle) == PathCollection:
        handle.set_sizes(5 * np.array(handle.get_sizes()))

disp_ax.set_xlabel("$\\zeta$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.loglog()
format_axis_ticks_decimal(disp_ax.yaxis, 1)

r_ax_zetas = np.logspace(np.log10(min_anisotropy), np.log10(max_anisotropy), 100)
supp_rrs = (0.1 * np.log(r_ax_zetas) + 0.7) ** (1 / 3)
r_ax.plot(r_ax_zetas, [rr if min_rr < rr < max_rr else np.nan for rr in supp_rrs],
          "k--", label="Supponen \\textit{et al.} (2018)")

r_ax.set_xlabel("$\\zeta$")
r_ax.set_ylabel("$R_1 / R_0$")
r_ax.set_xscale("log", base=10)
r_ax.legend(loc='lower right', frameon=False, fontsize='x-small')

label_subplot(fig.axes[0], "(a)")
label_subplot(fig.axes[1], "(b)")

plt.tight_layout()

lists_to_csv("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/"
             "Code/Cavitation-Analysis/numerical/models/anisotropy_models/", "complex_geometry_data.csv",
             [all_anisotropies, all_disps, all_ratios],
             ["anisotropy", "normalised displacement", "rebound radius ratio"],
             overwrite=True)

plt.show()
