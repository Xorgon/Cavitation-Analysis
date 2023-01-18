import os
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from experimental.util.analysis_utils import load_readings
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width
from scipy.optimize import curve_fit

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
dirs = []

for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

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

base_zetas = []
base_disps = []

for i, dir_path in enumerate(dirs):
    if "layers" in dir_path or "w48" in dir_path:
        continue

    max_ecc = 0.20
    label = dir_path.replace(root_dir, "").replace("\\", "/")
    # if i == 0:
    #     label = "Porous plate"
    # else:
    #     label = None

    if "circles" in dir_path:
        marker = "o"
    elif "triangles" in dir_path:
        marker = "^"
    elif "squares" in dir_path:
        marker = "s"
    else:
        marker = "D"

    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        if readings[0].model_anisotropy is None:
            print(f"Not yet processed {dir_path}")
            continue
        total_readings += len(readings)
        filt_anisotropies = np.array([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc])
        filt_disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc]
        disp_ax.scatter(filt_anisotropies, filt_disps, marker=marker, color=f'C{1 + i}', label=label, alpha=0.5, s=0.5)
        r_ax.scatter(filt_anisotropies,
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if r.ecc_at_max < max_ecc],
                     marker=marker, color=f'C{1 + i}', alpha=0.5, s=0.5)

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

print(f"Total readings = {total_readings}")

old_fit_c, old_fit_d, old_fit_c_off = 4.57, 0.51, 0.6  # From plot_anisotropy.py
fit_anisotropies = np.linspace(min_anisotropy, max_anisotropy, 10)
disp_ax.plot(fit_anisotropies, old_fit_c * np.array(fit_anisotropies) ** old_fit_d, "k-.",
             label="$\\frac{\\Delta}{R_0} = %0.2f \\zeta^{%0.2f}$" % (old_fit_c, old_fit_d))
disp_ax.plot(fit_anisotropies, (old_fit_c + old_fit_c_off) * np.array(fit_anisotropies) ** old_fit_d, "k",
             linestyle="-.", linewidth=0.5)
disp_ax.plot(fit_anisotropies, (old_fit_c - old_fit_c_off) * np.array(fit_anisotropies) ** old_fit_d, "k",
             linestyle="-.", linewidth=0.5)

disp_ax.legend(loc='lower right', frameon=False, fontsize=2)

disp_ax.set_xlabel("$\\zeta$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.loglog()
# disp_ax.xaxis.set_major_formatter(x_formatter)
# disp_ax.xaxis.set_minor_formatter(x_formatter)
# disp_ax.yaxis.set_major_formatter(y_formatter)
# disp_ax.yaxis.set_minor_formatter(y_formatter)

min_anisotropy = 1e-3
max_anisotropy = 1e-1
# r_ax.set_ylim((0, 0.5))
# r_ax.set_xlim((min_anisotropy, max_anisotropy))
r_ax_zetas = np.logspace(np.log10(min_anisotropy), np.log10(max_anisotropy), 50)
r_ax.plot(r_ax_zetas, (0.1 * np.log(r_ax_zetas) + 0.7) ** (1 / 3), "k--", label="Supponen \\textit{et al.} (2018)")

r_ax.set_xlabel("$\\zeta$")
r_ax.set_ylabel("$R_1 / R_0$")
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
