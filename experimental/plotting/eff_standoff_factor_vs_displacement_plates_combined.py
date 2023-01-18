import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt, fig_width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

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

min_gamma = 0
max_gamma = 6

initialize_plt(font_size=8, line_scale=1)
disp_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

size_ratio_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

all_standoffs = []
all_radius_ratios = []

additional_legend_handles = []
additional_legend_labels = []

phis = []
colours = []
standoff_lists = []
displacement_lists = []
ratio_lists = []

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
                displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px

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

        # disp_fig.gca().scatter(np.array(standoffs) * (1 / (1 - phi) ** 0.5), norm_disps, color=colour, marker=marker,
        #                        alpha=0.25)
        # size_ratio_fig.gca().scatter(standoffs, radius_ratios, color=colour, marker=marker, alpha=0.75)

        if i == 0:
            colours.append(colour)

        g_norm_disps.extend(norm_disps)
        g_standoffs.extend(standoffs)
        g_ys.extend(ys)
        g_theta_js.extend(theta_js)
        g_radius_ratios.extend(radius_ratios)

    phis.append(phi)
    standoff_lists.append(g_standoffs)
    displacement_lists.append(g_norm_disps)
    ratio_lists.append(g_radius_ratios)

phis, colours, standoff_lists, displacement_lists, ratio_lists = zip(
    *sorted(zip(phis, colours, standoff_lists, displacement_lists, ratio_lists)))


def opt_polyfit(facts):
    adjusted_standoffs = standoff_lists[0].copy()
    all_disps = displacement_lists[0].copy()
    for n in range(1, len(facts)):
        for stdoff in standoff_lists[n]:
            adjusted_standoffs.append(stdoff * facts[n])
        for disp in displacement_lists[n]:
            all_disps.append(disp)
    _, res, _, _, _ = np.polyfit(np.log(adjusted_standoffs), np.log(all_disps), 1, full=True)
    return res[0]  # Should be (1,) array


opt_res = minimize(opt_polyfit, np.ones((len(phis), 1)))
optimal_facts = opt_res.x

stdoff_ratio_phis = []
mean_stdoff_ratios = []

combined_adjusted_standoffs = []
combined_disps = []

for n in range(0, len(optimal_facts)):
    adjusted_standoffs = []
    all_disps = []
    all_radius_ratios = []
    for stdoff, disp, ratio in zip(standoff_lists[n], displacement_lists[n], ratio_lists[n]):
        adjusted_standoffs.append(stdoff * optimal_facts[n])
        combined_adjusted_standoffs.append(stdoff * optimal_facts[n])
        all_disps.append(disp)
        combined_disps.append(disp)
        all_radius_ratios.append(ratio)
    stdoff_ratio_phis.append(phis[n])
    mean_stdoff_ratios.append(np.mean(np.divide(adjusted_standoffs, standoff_lists[n])))
    disp_fig.gca().scatter(adjusted_standoffs, all_disps, label=f"$\\phi = {phis[n]}$")
    size_ratio_fig.gca().scatter(adjusted_standoffs, all_radius_ratios, label=f"$\\phi = {phis[n]}$")

plt.figure()
plt.scatter(stdoff_ratio_phis, mean_stdoff_ratios)
plt.plot(np.linspace(plt.xlim()[0], plt.xlim()[1]), np.power(1 / (1 - np.linspace(plt.xlim()[0], plt.xlim()[1])), 0.5),
         color="red", label="$\\frac{\\gamma_e}{\\gamma} = \\sqrt{\\frac{1}{1 - \\phi}}$")
plt.xlabel("$\\phi$")
plt.ylabel("$\\gamma_e / \\gamma$")
plt.legend(frameon=False)
plt.tight_layout()

disp_fig.gca().set_xlabel("$\\gamma_e = \\frac{Y}{R} + \\delta$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')

p = np.polyfit(np.log(combined_adjusted_standoffs), np.log(combined_disps), 1)
fit_stdoffs = np.linspace(1.7, 5.5)
fit_disps = np.exp(p[0] * np.log(fit_stdoffs) + p[1])
disp_fig.gca().plot(fit_stdoffs, fit_disps,
                    label="$\\Delta / R = %.3f\\gamma_e^{%.3f}$"
                          % (np.exp(p[1]), p[0]))

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
# disp_fig.gca().set_xlim((1, 6))
disp_fig.gca().set_ylim((0.15, 1.25))
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
