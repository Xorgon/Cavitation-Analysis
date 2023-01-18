import os
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from experimental.util.analysis_utils import load_readings
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width
from scipy.optimize import curve_fit

flat_plate_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]

porous_groups = [
    ["1 mm acrylic 50 mm plate, $\\phi = 0.525$",
     [[
         "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/Between 3 holes/",
         "1mm acrylic 50mm plate, between 3 holes, 52.5% VF", "C0", "^"],
         [
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/On hole/",
             "1mm acrylic 50mm plate, above a hole, 52.5% VF", "C0", "o"]], 0.525],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.38$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 38% VF", "C9", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF38/On hole/",
       "1mm acrylic 50mm plate, above a hole, 38% VF", "C9", "o"]], 0.38],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.24$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 24% VF", "C5", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
       "1mm acrylic 50mm plate, above a hole, 24% VF", "C5", "o"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
       "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C5", "^"]], 0.24],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.16$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 16% VF", "C7", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF16/On hole/",
       "1mm acrylic 50mm plate, above a hole, 16% VF", "C7", "o"]], 0.16],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.12$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 12% VF", "C8", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF12/On hole/",
       "1mm acrylic 50mm plate, above a hole, 12% VF", "C8", "o"]], 0.12],

    ["1 mm acrylic 50 mm plate, $\\phi = 0.08$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/Between 3 holes/",
       "1mm acrylic 50mm plate, between 3 holes, 8% VF", "C10", "^"],

      ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/s2.8VF8/On hole/",
       "1mm acrylic 50mm plate, above a hole, 8% VF", "C10", "o"]], 0.08],

    ["1 mm acrylic 50 mm solid plate",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
       "1mm acrylic 50mm solid plate", "C6", "s"]], 0],
]

# porous_groups = [
#     ["$\\phi = 0.24$ - circles",
#      [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 1/"],
#       [
#           "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 2/"]],
#      0.24],
#     ["$\\phi = 0.26$ - triangles",
#      [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
#        "w12vf32triangles/On a hole/"],
#       ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
#        "w12vf32triangles/Between 6 holes/"]],
#      0.258],
# ]

porous_dirs = []
vfs = []
labels = []
for g in porous_groups:
    for d in g[1]:
        porous_dirs.append(d[0])
        vfs.append(g[2])
        labels.append(g[0])

# initialize_plt(font_size=14, line_scale=1.5, dpi=300)

initialize_plt()
x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

min_anisotropy = np.inf
max_anisotropy = 0

all_anisotropies = []
all_disps = []

fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(fig_width(), fig_width() * 0.5))

total_readings = 0

for i, dir_path in enumerate(porous_dirs):
    max_ecc = 0.30
    label = labels[i]
    # if i == 0:
    #     label = "Porous plate"
    # else:
    #     label = None
    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        filt_anisotropies = np.array([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc])
        filt_disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc]
        disp_ax.scatter(filt_anisotropies, filt_disps, marker=".", color=f'C{1 + i}', label=label, alpha=0.5)
        r_ax.scatter(filt_anisotropies,
                     [np.linalg.norm(r.get_radius_ratio()) ** 3 for r in readings if r.ecc_at_max < max_ecc],
                     marker=".", color=f'C{1 + i}', alpha=0.5)

        # (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, filt_anisotropies, filt_disps)
        #
        # print(c, d)

        min_anisotropy = np.min([np.min([np.linalg.norm(r.model_anisotropy) for r in readings if
                                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None]), min_anisotropy])
        max_anisotropy = np.max([np.max([np.linalg.norm(r.model_anisotropy) for r in readings if
                                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None]), max_anisotropy])

        all_anisotropies.extend([np.linalg.norm(r.model_anisotropy) for r in readings if
                                 r.ecc_at_max < max_ecc and r.model_anisotropy is not None])
        all_disps.extend([np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                          r.ecc_at_max < max_ecc and r.model_anisotropy is not None])

        print(f"{label} - {np.mean([r.ecc_at_max for r in readings if r.ecc_at_max < max_ecc])}")
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(flat_plate_dirs):
    max_ecc = 0.5
    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        disp_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc],
                        marker=".", color=f'C0', label="Solid plate", alpha=0.5)
        r_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                     [np.linalg.norm(r.get_radius_ratio()) ** 3 for r in readings if r.ecc_at_max < max_ecc],
                     marker=".", color=f'C0', alpha=0.5)

        min_anisotropy = np.min([np.min([np.linalg.norm(r.model_anisotropy) for r in readings if
                                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None]), min_anisotropy])
        max_anisotropy = np.max([np.max([np.linalg.norm(r.model_anisotropy) for r in readings if
                                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None]), max_anisotropy])

        all_anisotropies.extend([np.linalg.norm(r.model_anisotropy) for r in readings if
                                 r.ecc_at_max < max_ecc and r.model_anisotropy is not None])
        all_disps.extend([np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                          r.ecc_at_max < max_ecc and r.model_anisotropy is not None])

        print(f"Solid plate - {np.mean([r.ecc_at_max for r in readings])}")
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

print(f"Total readings = {total_readings}")

fit_anisotropies = np.linspace(min_anisotropy, max_anisotropy, 10)
disp_ax.plot(fit_anisotropies, 4.54 * np.array(fit_anisotropies) ** 0.50, "k-.",
             label="$\\frac{\\Delta}{R_0} = 4.54 \\zeta^{0.50}$")

disp_ax.legend(loc='lower right', frameon=False, fontsize='xx-small')

disp_ax.set_xlabel("$\\zeta$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.loglog()
# disp_ax.xaxis.set_major_formatter(x_formatter)
# disp_ax.xaxis.set_minor_formatter(x_formatter)
# disp_ax.yaxis.set_major_formatter(y_formatter)
# disp_ax.yaxis.set_minor_formatter(y_formatter)

min_anisotropy = 1e-4
max_anisotropy = 1e-1
r_ax.set_ylim((0, 0.5))
r_ax.set_xlim((min_anisotropy, max_anisotropy))
r_ax_zetas = np.linspace(min_anisotropy, max_anisotropy, 50)
r_ax.plot(r_ax_zetas, (0.1 * np.log(r_ax_zetas) + 0.7) ** (3 / 3), "k--", label="Supponen \\textit{et al.} (2018)")

r_ax.set_xlabel("$\\zeta$")
r_ax.set_ylabel("$E_1 / E_0$")
# r_ax.loglog()
r_ax.set_xscale("log", base=10)
# r_ax.xaxis.set_major_formatter(x_formatter)
# r_ax.xaxis.set_minor_formatter(x_formatter)
# r_ax.yaxis.set_major_formatter(y_formatter)
# r_ax.yaxis.set_minor_formatter(y_formatter)
r_ax.legend(loc='upper right', frameon=False, fontsize='xx-small')

disp_ax.annotate(f"($a$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
                 textcoords='axes fraction', color="k",
                 horizontalalignment='left', verticalalignment='top')
r_ax.annotate(f"($b$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
              textcoords='axes fraction', color="k",
              horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
