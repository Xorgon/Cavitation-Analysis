import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt, fig_width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

groups = [

    ["MO (1 mm acrylic)",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 24% VF", "C0", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
       "1mm acrylic 50mm plate, above a hole, 24% VF", "C0", "o"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "Microscope objective", "C0", "^"]], 0.24],

    ["MO (3 mm acrylic)",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF 3mm acrylic/Between 3 holes/",
         "3 mm acrylic, between 3 holes", "C2", "^"],
         [
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF 3mm acrylic/On a hole/",
             "3 mm acrylic, on a hole", "C2", "o"], ], 0.24],

    ["OAPM - Acrylic arm",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/OAPM - between 3/",
       "OAP Mirror", "C1", "^"],
      ], 0.24],

    ["OAPM - Reinforced",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/OAPM - between 3 - sturdier/",
         "OAP Mirror", "C3", "^"],
     ], 0.24],

    ["OAPM - Reinforced",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/OAPM - between 3 - sturdier and smaller/",
         "OAP Mirror", "C4", "^"],
     ], 0.24],

    ["OAPM - Steel arm",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/OAPM - steel arm/",
         "OAP Mirror", "C5", "^"],
     ], 0.24],

    # ["Solid plate",
    #  [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
    #    "1mm acrylic 50mm solid plate", "C6", "s"]], 0]
]

min_gamma = 1.5
max_gamma = 3.2

initialize_plt(font_size=8, line_scale=1)
disp_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

size_ratio_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

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

    radii = []
    for i, (dir_path, _, colour, marker) in enumerate(dirs):
        sys.path.append(dir_path)
        # print(dir_path)

        params = importlib.import_module("params")
        importlib.reload(params)

        sys.path.remove(dir_path)

        y_offset = params.upper_surface_y

        # try:
        #     au.flag_invalid_readings(dir_path)
        # except ValueError:
        #     print(f"Skipping flagging invalid readings for {dir_path}")
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

            if y >= 0 and reading.ecc_at_max < 0.8:
                ys.append(y)
                radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
                radii.append(radius)
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

    (a, b) = np.polyfit(np.log10(filt_standoffs), np.log10(filt_norm_disps), 1)
    # print(label, a, b)

    print(f"{label} average radius = {np.mean(radii):.2f} mm")


    def disp_from_gamma(gamma):
        return 10 ** (a * np.log10(gamma) + b)


    disp_fig.gca().plot(np.linspace(min_gamma, max_gamma, 50),
                        disp_from_gamma(np.linspace(min_gamma, max_gamma, 50)), color=colour, markersize=5,
                        label=label + " - $\\bar{R_0} = %.2f$ mm" % np.mean(radii))

    additional_legend_handles.append(
        mpatches.Patch(color=colour, label=label, linewidth=0.1, alpha=0.75, capstyle='butt'))
    additional_legend_labels.append(label)

bem_standoffs = [1.5, 1.81282524, 2.19089023, 2.64780074, 3.2]
bem_disps_50 = [1.01824217, 0.82484359, 0.66817211, 0.54124997, 0.43844186]
bem_disps_100 = [1.04457772, 0.84617713, 0.68545354, 0.55524872, 0.44978161]

disp_fig.gca().plot(bem_standoffs, bem_disps_50, "k--", label="BEM 50 mm plate")
disp_fig.gca().plot(bem_standoffs, bem_disps_100, "k-.", label="BEM 100 mm plate")

disp_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')
disp_fig.gca().set_xlim((1.35, 4.5))
disp_fig.gca().set_ylim((0.08, 1.2))

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
# disp_fig.gca().set_xlim((1, 6))
# disp_fig.gca().set_ylim((0.15, 1.25))
disp_fig.gca().legend(frameon=False)
disp_fig.tight_layout()

size_ratio_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
size_ratio_fig.gca().set_ylabel("$R_{max2} / R_{max}$")
# size_ratio_fig.gca().set_xlim((0.5, 10))
# size_ratio_fig.gca().set_ylim((0.2, 1))

handles, labels = size_ratio_fig.gca().get_legend_handles_labels()
handles.extend(additional_legend_handles)
labels.extend(additional_legend_labels)
size_ratio_fig.gca().legend(handles, labels, frameon=False)

size_ratio_fig.tight_layout()

plt.show()
