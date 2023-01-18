import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt, fig_width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

groups = [
    ["1 mm acrylic 50 mm plate, $\\phi = 0.525$",
     [[
         "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/Between 3 holes/",
         "1mm acrylic 50mm plate, between 3 holes, 52.5% VF", "C0", "^"],
         [
             "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/On hole/",
             "1mm acrylic 50mm plate, above a hole, 52.5% VF", "C0", "o"]], 0.525],

    # ["1 mm acrylic 100 mm plate, $\\phi = 0.39$",
    #  [[
    #      "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/Between 2 holes/",
    #      "1mm acrylic 100mm plate, between 2 holes, 39% VF", "C1", "_"],
    #      [
    #          "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/Between 3 holes/",
    #          "1mm acrylic 100mm plate, between 3 holes, 39% VF", "C1", "^"],
    #      [
    #          "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/On hole/",
    #          "1mm acrylic 100mm plate, above a hole, 39% VF", "C1", "o"]], 0.39],

    # ["0.7 mm steel 100 mm plate, $\\phi = 0.47$",
    #  [[
    #      "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2mm Hole Steel Plate/Between 3 holes/",
    #      "0.7 mm steel 100mm plate, between 3 holes, 47% VF", "C2", "^"],
    #      [
    #          "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/2mm Hole Steel Plate/On hole/",
    #          "0.7 mm steel 100mm plate, above a hole, 47% VF", "C2", "o"]], 0.47],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.38$",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 38% VF", "C9", "^"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/On hole/",
       "1mm acrylic 50mm plate, above a hole, 38% VF", "C9", "o"]], 0.38],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.24$",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 24% VF", "C5", "^"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
       "1mm acrylic 50mm plate, above a hole, 24% VF", "C5", "o"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C5", "^"]], 0.24],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.16$",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 16% VF", "C7", "^"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/On hole/",
       "1mm acrylic 50mm plate, above a hole, 16% VF", "C7", "o"]], 0.16],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.12$",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 12% VF", "C8", "^"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/On hole/",
       "1mm acrylic 50mm plate, above a hole, 12% VF", "C8", "o"]], 0.12],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.08$",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 8% VF", "C10", "^"],

      ["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/On hole/",
       "1mm acrylic 50mm plate, above a hole, 8% VF", "C10", "o"]], 0.08],

    # ["3 mm acrylic 100 mm quad plate, $\\phi = 0.12$",
    #  [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/Porous plate/",
    #    "3mm acrylic 100mm quad plate, between 4 holes, 12% VF", "C3", "x"]], 0.12],
    #
    # ["3 mm acrylic 100 mm solid plate",
    #  [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/",
    #    "3mm acrylic 100mm solid plate", "C4", "s"]], 0],

    ["1 mm acrylic 50 mm solid plate",
     [["C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
       "1mm acrylic 50mm solid plate", "C6", "s"]], 0],
]

cross_section_gammas = np.array([2, 2.75, 3.5])
cross_section_phis = [[], [], []]
cross_section_disps = [[], [], []]

min_gamma = 1.8
max_gamma = 3.8

# initialize_plt(font_size=8, line_scale=1)
disp_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

size_ratio_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

cross_section_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

all_standoffs = []
all_radius_ratios = []

additional_legend_handles = []
additional_legend_labels = []
for label, dirs, phi in groups:
    g_norm_disps = []
    g_standoffs = []
    g_ys = []
    g_theta_js = []
    g_radius_ratios = []
    for i, (dir_path, _, colour, marker) in enumerate(dirs):
        sys.path.append(dir_path)
        print(dir_path)

        params = importlib.import_module("params")
        importlib.reload(params)

        sys.path.remove(dir_path)

        y_offset = params.upper_surface_y

        try:
            au.flag_invalid_readings(dir_path)
        except ValueError:
            print(f"Skipping flagging invalid readings for {dir_path}")
        readings = au.load_readings(dir_path + "readings_dump.csv", include_invalid=False)

        norm_disps = []
        standoffs = []
        ys = []
        theta_js = []
        radius_ratios = []
        for j, reading in enumerate(readings):
            pos = reading.get_bubble_pos_mm(params.mm_per_px)
            y = pos[1] - y_offset

            radius_ratio = np.sqrt(reading.sec_max_area / reading.max_bubble_area)

            if y >= 0:
                ys.append(y)
                radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
                displacement = np.linalg.norm(reading.sup_disp_vect) * params.mm_per_px

                inter_max_time = reading.inter_max_frames / 1e5  # 100,000 FPS
                density = 997
                R_init = radius / 1000  # meters
                P_vapour = 2.3388e3
                P_inf = 100e3
                polytropic_const = 1.33  # ratio of specific heats of water vapour
                ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5

                if inter_max_time > 2 * ray_col_time:
                    print(inter_max_time / ray_col_time)
                    continue

                theta_js.append(reading.get_jet_angle())
                standoff = y / radius
                standoffs.append(standoff)
                norm_disps.append(displacement / radius)
                radius_ratios.append(radius_ratio)

                # disp_fig.gca().text(standoff, displacement / radius, f"{reading.idx}:{reading.repeat_number}")
                # size_ratio_fig.gca().text(standoff, radius_ratio, f"{reading.idx}:{reading.repeat_number}")

        disp_fig.gca().scatter(standoffs, norm_disps, color=colour, marker=marker, alpha=0.25)
        size_ratio_fig.gca().scatter(standoffs, radius_ratios, color=colour, marker=marker, alpha=0.75)

        g_norm_disps.extend(norm_disps)
        g_standoffs.extend(standoffs)
        g_ys.extend(ys)
        g_theta_js.extend(theta_js)
        g_radius_ratios.extend(radius_ratios)

    filt_standoffs, filt_norm_disps = zip(
        *[(s, n) for s, n in zip(g_standoffs, g_norm_disps) if max_gamma > s > min_gamma])

    # (a, b) = np.polyfit(np.log10(filt_standoffs), np.log10(filt_norm_disps), 1)
    # print(label, a, b)

    # def disp_from_gamma(gamma):
    #     return 10 ** (a * np.log10(gamma) + b)

    # disp_fig.gca().plot(np.linspace(min_gamma, max_gamma, 2),
    #                     disp_from_gamma(np.linspace(min_gamma, max_gamma, 2)), color=colour,
    #                     markersize=5, label=label)

    additional_legend_handles.append(
        mpatches.Patch(color=colour, label=label, linewidth=0.1, alpha=0.75, capstyle='butt'))
    additional_legend_labels.append(label)

disp_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')

xs = np.linspace(disp_fig.gca().get_xlim()[0], disp_fig.gca().get_xlim()[1], 50)
ys = 2.5 * 0.195 ** (3 / 5) * xs ** (-6 / 5)
disp_fig.gca().plot(xs, ys)

for gamma in cross_section_gammas:
    disp_fig.gca().axvline(gamma, color="grey", linestyle="--", zorder=-1, alpha=0.5)

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
# disp_fig.gca().set_xlim((1, 6))
# disp_fig.gca().set_ylim((0.15, 1.25))
disp_fig.gca().legend(frameon=False)
disp_fig.tight_layout()

size_ratio_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
size_ratio_fig.gca().set_ylabel("$R_{max2} / R_{max}$")
size_ratio_fig.gca().set_xlim((0.5, 10))
size_ratio_fig.gca().set_ylim((0.2, 1))

x_range = size_ratio_fig.gca().get_xlim()
gammas = np.linspace(x_range[0], x_range[1], 50)
size_ratio_fig.gca().plot(gammas, np.power(0.1 * np.log(0.195 * np.power(gammas, -2)) + 0.7, 1 / 3),
                          label="Supponen et al. (2018)")
size_ratio_fig.gca().set_xlim(x_range)

handles, labels = size_ratio_fig.gca().get_legend_handles_labels()
handles.extend(additional_legend_handles)
labels.extend(additional_legend_labels)
size_ratio_fig.gca().legend(handles, labels, frameon=False)

size_ratio_fig.tight_layout()

plt.show()
