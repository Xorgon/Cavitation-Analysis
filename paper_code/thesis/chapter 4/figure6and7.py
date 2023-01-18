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

import util.file_utils as file
from util.file_utils import lists_to_csv
from util.plotting_utils import initialize_plt, format_axis_ticks_decimal, label_subplot
from util.analysis_utils import load_readings, Reading, analyse_frame
from util.mp4 import MP4

# label, max_ecc, alpha, marker, m_size, colour, dirs
sets = [["Slots", 0.20, 1, 'o', 1.5, 'C0',
         ["../Data/Slots/W1H3/",
          "../Data/Slots/W2H12/",
          "../Data/Slots/W2H3a/",
          "../Data/Slots/W2H3b/",
          "../Data/Slots/W2H6/",
          "../Data/Slots/W2H9/",
          "../Data/Slots/W4H12/"]],

        ["Square", 0.205, 1, 's', 1.5, 'C5',  # 0.205 | 0.234
         ["../Data/Square/Square/",
          "../Data/Square/Square 2/",
          "../Data/Square/Square 3/"]],

        ["Triangle", 0.24, 1, '<', 1.5, 'C3',  # 0.22 | 0.257
         ["../Data/Triangle/Equilateral triangle/",
          "../Data/Triangle/Equilateral triangle 2/"]],

        ["$90^\\circ$ corner", 0.22, 1, 'x', 1.5, 'C1',
         ["../Data/90 degree corner/90 degree corner/",
          "../Data/90 degree corner/90 degree corner 2/",
          "../Data/90 degree corner/90 degree corner 3/",
          "../Data/90 degree corner/90 degree corner 4/"]],

        ["$60^\\circ$ corner", 0.28, 1, (6, 1, 30), 1.5, 'C4',  # (6, 1, 30) makes a 6-pointed star rotated 30 degrees
         ["../Data/60 degree corner/"]],

        ["Flat plate", 0.28, 1, '_', 3, 'C2',
         ["../Data/Flat plate/"]],
        ]

show_data_labels = False
plot_additionals = True

if not show_data_labels:
    initialize_plt()  # Do not want to plot thousands of text labels with this on
else:
    plt.rc('lines', linewidth=3, markersize=9)

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

disp_inset_ax = disp_ax.inset_axes([0.1, 0.6, 0.375, 0.375])
disp_inset_ax.loglog()
disp_inset_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                          labelbottom=False, right=False, left=False, labelleft=False)

r_inset_ax = r_ax.inset_axes([0.1, 0.6, 0.375, 0.375])
r_inset_ax.set_xscale('log')
r_inset_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       labelbottom=False, right=False, left=False, labelleft=False)

total_readings = 0

for j, (label, max_ecc, alpha, marker, m_size, colour, dirs) in enumerate(sets):
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
            filtered_displacements = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r_filter(r)]
            filtered_radius_ratios = [r.get_radius_ratio() for r in readings if r_filter(r)]
            filtered_max_radii = [np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)]
            filtered_eccs = [r.ecc_at_max for r in readings if r_filter(r)]
            filtered_labels = [f"{r.idx}:{r.repeat_number}" for r in readings if r_filter(r)]
            radii.extend(filtered_max_radii)
            radii_mm.extend(np.array(filtered_max_radii) * params.mm_per_px)
            eccs.extend(filtered_eccs)

            all_readings.extend(readings)  # ALL readings

            set_readings += len(readings)
            used_readings += len(filtered_anisotropies)
            total_readings += len(filtered_anisotropies)

            if len(filtered_anisotropies) == 0:
                continue

            if show_data_labels:
                for l, x, y in zip(filtered_labels, filtered_anisotropies, filtered_displacements):
                    disp_ax.annotate(f'{l}', xy=[x, y], textcoords='data')

            disp_ax.scatter(filtered_anisotropies, filtered_displacements, marker=marker, color=colour,
                            label=this_label, alpha=alpha, s=m_size ** 2)

            disp_inset_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings],
                                  [np.linalg.norm(r.get_normalised_displacement()) for r in readings],
                                  marker=marker, color=colour, label=this_label, alpha=alpha, s=(0.2 * m_size) ** 2)

            r_ax.scatter(filtered_anisotropies, filtered_radius_ratios,
                         marker=marker, color=colour, alpha=alpha, s=m_size ** 2)

            r_inset_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings],
                               [r.get_radius_ratio() for r in readings],
                               marker=marker, color=colour, alpha=alpha, s=(0.2 * m_size) ** 2)

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

####################################################
# FIGURE 7                                         #
####################################################

all_eccs = [float(r.ecc_at_max) for r in all_readings]
all_fit_points = c * np.array([r.get_scalar_anisotropy() for r in all_readings]) ** d
all_scalar_disps = np.array([np.linalg.norm(r.get_normalised_displacement()) for r in all_readings])
all_diffs_normalised = np.abs(all_scalar_disps - all_fit_points) / all_fit_points

fig = plt.figure(figsize=(5, 3), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
fig.add_subplot(gs[1:, :])
bin_edges = np.linspace(np.min(all_eccs), np.max(all_eccs), 25)
full_bin_centers = []
mean_diffs = []
std_diffs = []
ns = []
min_edge_idx = np.inf
max_edge_idx = 0
for i in range(len(bin_edges) - 1):
    bin_diffs = [d for (d, e) in zip(all_diffs_normalised, all_eccs) if bin_edges[i] <= e < bin_edges[i + 1]]
    if len(bin_diffs) > 4:
        full_bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        mean_diffs.append(np.mean(bin_diffs))
        std_diffs.append(np.std(bin_diffs))
        ns.append(len(bin_diffs))
        if i < min_edge_idx:
            min_edge_idx = i
        if i + 1 > max_edge_idx:
            max_edge_idx = i + 1

for edge in bin_edges[min_edge_idx:max_edge_idx + 1]:
    plt.axvline(edge, color='#eeeeee', linewidth=0.5)

err_bars = 1.96 * np.array(std_diffs) / np.sqrt(ns)

mean_diffs = np.array(mean_diffs)
mean_line = plt.errorbar(full_bin_centers, 100 * mean_diffs, yerr=100 * err_bars, color="C0", linestyle="--",
                         marker="o")
plt.ylim((0, 27))
plt.xlabel("Eccentricity $\\epsilon$")
plt.ylabel("\\% difference from curve fit")

frame_dirs = [
    "../Data/Slots/W1H3/movie_S0008/",
    "../Data/Square/Square 2/movie0627/",
    "../Data/90 degree corner/90 degree corner 4/movie1204/"
]
repeats = [2, 91, 0]
frame_idxs = [26, 23, 32]

for i in range(len(frame_dirs)):
    fig.add_subplot(gs[0, i])
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)

    mov = MP4(frame_dirs[i] + f"video_{repeats[i]}.mp4")

    x, y, area, ecc, _, _ = analyse_frame(np.int32(mov[frame_idxs[i]]),
                                          np.int32(mov[0]), debug=True)

    r = np.sqrt(area / np.pi)
    plt.gca().add_patch(Circle((x, y), 1.1 * r, fill=False, color=f"C0"))

    plt.imshow(mov[frame_idxs[i]], cmap=plt.cm.gray)

    zoom = 1.25
    plt.xlim((x - zoom * r, x + zoom * r))
    plt.ylim((y - zoom * r, y + zoom * r))
    plt.xlabel(f"$\\epsilon = {ecc:0.2f}$", fontsize="small")

plt.show()
