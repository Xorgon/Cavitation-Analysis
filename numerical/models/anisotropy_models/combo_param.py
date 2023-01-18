import os
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from experimental.util.analysis_utils import load_readings, Reading
import numpy as np
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal
from scipy.optimize import curve_fit, minimize
from scipy.stats import anderson

# label, max_ecc, alpha, marker, m_size, colour, dirs
sets = [["$\\phi = 0.24$ - circles", 0.20, 1, 'o', 1.5, 'C0',
         [
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 1/",

             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 2/"]],

        ["$\\phi = 0.26$ - triangles", 0.20, 1, 'o', 1.5, 'C1',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
          "w12vf32triangles/On a hole/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
          "w12vf32triangles/Between 6 holes/"]],

        ["Flat plate", 0.28, 1, '_', 1.7, 'C3',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]],

        ]

initialize_plt()  # Do not want to plot thousands of text labels with this on

x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

min_anisotropy = np.inf
max_anisotropy = 0

all_anisotropies = []
all_disps = []
all_ratios = []

fig_width = 7

fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.5))

total_readings = 0

for j, (label, max_ecc, alpha, marker, m_size, colour, dirs) in enumerate(sets):
    # max_ecc = 0.18
    # max_ecc = 1
    set_readings = 0
    used_readings = 0
    radii = []
    eccs = []


    def r_filter(r: Reading):
        # mm_per_px scaling doesn't matter here, just using this function to get the right direction conventions
        theta = np.arccos(np.dot(r.get_disp_mm(1), r.model_anisotropy) /
                          (np.linalg.norm(r.get_disp_mm(1)) * np.linalg.norm(r.model_anisotropy)))
        return r.ecc_at_max < max_ecc and r.model_anisotropy is not None and theta < np.deg2rad(5)


    for i, dir_path in enumerate(dirs):
        if i == 0:
            this_label = label
        else:
            this_label = None
        try:
            # continue
            readings = load_readings(dir_path + "readings_dump.csv")
            filtered_anisotropies = [np.linalg.norm(r.model_anisotropy) for r in readings if r_filter(r)]
            filtered_displacements = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r_filter(r)]
            filtered_radius_ratios = [r.get_radius_ratio() for r in readings if r_filter(r)]
            filtered_max_radii = [np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)]
            filtered_eccs = [r.ecc_at_max for r in readings if r_filter(r)]
            filtered_labels = [f"{r.idx}:{r.repeat_number}" for r in readings if r_filter(r)]
            radii.extend(filtered_max_radii)
            eccs.extend(filtered_eccs)

            set_readings += len(readings)
            used_readings += len(filtered_anisotropies)
            total_readings += len(filtered_anisotropies)

            if len(filtered_anisotropies) == 0:
                continue

            filtered_displacements = np.array(filtered_displacements)
            filtered_radius_ratios = np.array(filtered_radius_ratios)

            disp_ax.scatter(filtered_anisotropies, filtered_displacements ** -0.1061 * np.exp(filtered_radius_ratios ** 3), marker=marker, color=colour,
                            label=this_label, alpha=alpha, s=m_size ** 2)
            r_ax.scatter(filtered_anisotropies, filtered_radius_ratios,
                         marker=marker, color=colour, alpha=alpha, s=m_size ** 2)

            min_anisotropy = np.min([np.min(filtered_anisotropies), min_anisotropy])
            max_anisotropy = np.max([np.max(filtered_anisotropies), max_anisotropy])

            all_anisotropies.extend(filtered_anisotropies)
            all_disps.extend(filtered_displacements)
            all_ratios.extend(filtered_radius_ratios)

        except FileNotFoundError:
            print(f"Not yet processed {dir_path}")

    print(f"{label:20s}: {used_readings:4d} of {set_readings:4d} "
          f"({100 * used_readings / set_readings:2.2f} %) | max_ecc = {max_ecc:0.3f}"
          f" | Mean bubble radius {np.mean(radii):2.2f} px"
          f" | Mean eccentricity {np.mean(eccs):.2f}")

print(f"Total readings = {total_readings}")

all_anisotropies = np.array(all_anisotropies)
all_ratios = np.array(all_ratios)
all_disps = np.array(all_disps)

filt_anisotropies, filt_disps = zip(*[(z, dis) for z, dis in sorted(zip(all_anisotropies, all_disps))
                                      if min_anisotropy <= z <= max_anisotropy])
(a, b) = np.polyfit(np.log10(filt_anisotropies), np.log10(filt_disps), 1)

(c, d), _ = curve_fit(lambda x, c, d: c * x ** d, filt_anisotropies, filt_disps)

disp_ax.plot(filt_anisotropies, c * np.array(filt_anisotropies) ** d, "k", linestyle="dotted",
             label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (c, d))

disp_ax.legend(loc='lower right', frameon=False, fontsize='x-small', markerfirst=False)

disp_ax.set_xlabel("$\\zeta$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.loglog()
format_axis_ticks_decimal(disp_ax.yaxis, 1)

r_ax_zetas = np.logspace(np.log10(min_anisotropy), np.log10(max_anisotropy), 100)
r_ax.plot(r_ax_zetas, (0.1 * np.log(r_ax_zetas) + 0.7) ** (1 / 3), "k--", label="Supponen \\textit{et al.} (2018)")

r_ax.set_xlabel("$\\zeta$")
r_ax.set_ylabel("$R_1 / R_0$")
r_ax.set_xscale("log", base=10)
r_ax.legend(loc='lower right', frameon=False, fontsize='x-small')

disp_ax.annotate(f"($a$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
                 textcoords='axes fraction', color="k",
                 horizontalalignment='left', verticalalignment='top')
r_ax.annotate(f"($b$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
              textcoords='axes fraction', color="k",
              horizontalalignment='left', verticalalignment='top')

plt.tight_layout()


def combination(disp, ratio, beta):
    return disp ** beta * np.exp(ratio ** 3)


def get_fit_quality(beta):
    combo = combination(all_disps, all_ratios, beta)
    (p, q), _ = curve_fit(lambda x, p, q: p * x ** q, all_anisotropies, combo)
    lin_diffs = combo - p * np.array(all_anisotropies) ** q

    (p, q), _ = curve_fit(lambda x, p, q: np.log10(p) + q * x, np.log10(all_anisotropies), np.log10(combo))
    log_diffs = np.log10(combo) - (np.log10(p) + q * np.log10(all_anisotropies))
    rmsd = np.sqrt(np.mean(lin_diffs ** 2))
    return rmsd


plt.figure()
res = minimize(get_fit_quality, 1)
beta = res.x
# beta = -2
print(beta)
combo = combination(all_disps, all_ratios, beta)
(p, q), _ = curve_fit(lambda x, p, q: p * x ** q, all_anisotropies, combo)
plt.scatter(all_anisotropies, combo)
fit_anis = np.linspace(np.min(all_anisotropies), np.max(all_anisotropies), 20)
plt.plot(fit_anis, p * fit_anis ** q)
plt.loglog()
plt.tight_layout()

plt.figure()
plt.scatter(all_anisotropies, all_disps / all_ratios)
plt.show()
