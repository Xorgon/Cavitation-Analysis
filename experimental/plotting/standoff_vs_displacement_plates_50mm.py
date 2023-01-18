import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt, fig_width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

groups = [
    ["$\\phi = 0.525$",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/Between 3 holes/",
         "1mm acrylic 50mm plate, between 3 holes, 52.5% VF", "C0", "^"],
         [
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/On hole/",
             "1mm acrylic 50mm plate, above a hole, 52.5% VF", "C0", "o"]], 0.525],

    ["$\\phi = 0.38$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 38% VF", "C1", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/On hole/",
       "1mm acrylic 50mm plate, above a hole, 38% VF", "C1", "o"]], 0.38],

    ["$\\phi = 0.24$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 24% VF", "C2", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
       "1mm acrylic 50mm plate, above a hole, 24% VF", "C2", "o"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C2", "^"],

      [
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3/",
          "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C2", "^"],

      [
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 2.0/",
          "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C2", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C2", "^"]], 0.24],

    ["$\\phi = 0.16$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 16% VF", "C3", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/On hole/",
       "1mm acrylic 50mm plate, above a hole, 16% VF", "C3", "o"]], 0.16],

    ["$\\phi = 0.12$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 12% VF", "C4", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/On hole/",
       "1mm acrylic 50mm plate, above a hole, 12% VF", "C4", "o"]], 0.12],

    ["$\\phi = 0.08$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 8% VF", "C5", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/On hole/",
       "1mm acrylic 50mm plate, above a hole, 8% VF", "C5", "o"]], 0.08],

    ["Solid plate",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
       "1mm acrylic 50mm solid plate", "C6", "s"]], 0],
]

groups = sorted(groups, key=lambda g: g[2])
n = 6
show_cross_sections = True

cross_section_gammas = np.array([2, 3, 4])
cross_section_phis = [[], [], []]
cross_section_disps = [[], [], []]
cross_section_errs = [[], [], []]

min_gamma = 1.5
max_gamma = 7

initialize_plt(font_size=10, line_scale=1)
disp_fig = plt.figure(figsize=(20 / 4, 12.5 / 4))

size_ratio_fig = plt.figure(figsize=(20 / 4, 12.5 / 4))

cross_section_fig = plt.figure(figsize=(20 / 4, 12.5 / 4))

all_standoffs = []
all_radius_ratios = []

additional_legend_handles = []
additional_legend_labels = []
for label, dirs, phi in groups[:n + 1]:
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

            if y >= 0 and reading.ecc_at_max is not None and reading.ecc_at_max < 0.3:
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

    (a, b), cov = np.polyfit(np.log10(filt_standoffs), np.log10(filt_norm_disps), 1, cov=True)
    print(label, a, b)
    print(cov)


    def disp_from_gamma(gamma):
        return 10 ** (a * np.log10(gamma) + b)


    disp_fig.gca().plot(np.linspace(min_gamma, max_gamma, 100),
                        disp_from_gamma(np.linspace(min_gamma, max_gamma, 100)), color=colour,
                        markersize=5, label=label)

    cross_section_fig.gca().scatter(np.array([phi] * 3), disp_from_gamma(cross_section_gammas), color=dirs[0][2])
    for i, gamma in enumerate(cross_section_gammas):
        a_uncertainty = 10 ** b * np.log10(gamma) * np.log(gamma) * np.sqrt(np.diag(cov)[0])
        b_uncertainty = 10 ** b * np.log(10) * gamma ** a * np.sqrt(np.diag(cov)[1])
        err = np.sqrt(a_uncertainty ** 2 + b_uncertainty ** 2)
        print(err)

        cross_section_phis[i].append(phi)
        cross_section_disps[i].append(disp_from_gamma(gamma))
        cross_section_errs[i].append(err)

    additional_legend_handles.append(
        mpatches.Patch(color=colour, label=label, linewidth=0.1, alpha=0.75, capstyle='butt'))
    additional_legend_labels.append(label)

disp_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')

if show_cross_sections:
    for gamma in cross_section_gammas:
        disp_fig.gca().axvline(gamma, color="grey", linestyle="--", zorder=-1, alpha=0.5)

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
disp_fig.gca().set_xlim((1.4, 9))
disp_fig.gca().set_ylim((0.15, 1.25))

yticks = [0.2, 0.3, 0.4, 0.6, 0.8]
disp_fig.gca().set_yticks(yticks)
disp_fig.gca().set_yticklabels([f"{i:.1f}" for i in yticks])

xticks = [2, 3, 4, 5, 6]
disp_fig.gca().set_xticks(xticks)
disp_fig.gca().set_xticklabels([f"{i:.0f}" for i in xticks])

disp_fig.gca().legend(frameon=False)
disp_fig.tight_layout()

size_ratio_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
size_ratio_fig.gca().set_ylabel("$R_{max2} / R_{max}$")
size_ratio_fig.gca().set_xlim((0.5, 10))
size_ratio_fig.gca().set_ylim((0.2, 1))

x_range = size_ratio_fig.gca().get_xlim()
gammas = np.linspace(x_range[0], x_range[1], 50)
# size_ratio_fig.gca().plot(gammas, np.power(0.1 * np.log(0.195 * np.power(gammas, -2)) + 0.7, 1 / 3),
#                           label="Supponen et al. (2018)")
size_ratio_fig.gca().set_xlim(x_range)

# size_ratio_fig.gca().plot([1.6, 2.0, 3.0], np.power([0.8077, 0.6400, 0.4345], 1 / 3), marker="o", label="Wang (2016)",
#                           c="red")

handles, labels = size_ratio_fig.gca().get_legend_handles_labels()
handles.extend(additional_legend_handles)
labels.extend(additional_legend_labels)
size_ratio_fig.gca().legend(handles, labels, frameon=False)

size_ratio_fig.tight_layout()

cross_section_disps = np.array(cross_section_disps)
for i, gamma in enumerate(cross_section_gammas):
    cross_section_phis[i], cross_section_disps[i] = zip(*sorted(zip(cross_section_phis[i], cross_section_disps[i])))
    # cross_section_fig.gca().plot(cross_section_phis[i], cross_section_disps[i], linestyle="--", color="grey", zorder=-1)
    cross_section_fig.gca().errorbar(cross_section_phis[i], cross_section_disps[i], yerr=cross_section_errs[i],
                                     linestyle="--", color="grey", zorder=-1)
    cross_section_fig.gca().text(np.mean(cross_section_phis[i]) + 0.05, np.mean(cross_section_disps[i]),
                                 f"$\\gamma = {gamma:.2f}$", ha='left', va='bottom')

# for phi in [0.08, 0.12, 0.16, 0.38]:
#     cross_section_fig.gca().axvline(phi, color="green", zorder=-1, alpha=0.5, linestyle="dotted")
cross_section_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
cross_section_fig.gca().set_xlabel("$\\phi$")
cross_section_fig.tight_layout()

# disp_fig.savefig(
#     f"C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Conferences/APS DFD 2021/"
#     f"figures/displacements/disps_{n}{'_cs' if show_cross_sections else ''}.png",
#     dpi=600)
#
# cross_section_fig.savefig(
#     f"C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Conferences/APS DFD 2021/figures/cross_sections.png",
#     dpi=600)

plt.show()
