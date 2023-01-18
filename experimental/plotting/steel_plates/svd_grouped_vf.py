import os
import sys
import importlib
import matplotlib.pyplot as plt
from matplotlib import cm
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal, fig_width, label_subplot
from scipy.optimize import curve_fit
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()

vf_standoffs = {}
vf_disps = {}
vf_ratios = {}


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=0.6), cmap=cm.get_cmap('viridis'))

standoff_cuts = [2, 3, 4]

bin_width = 0.15

fit_for_disp_cuts = True
fit_for_ratio_cuts = True

cut_vfs = [[], [], []]
cut_disps = [[], [], []]
cut_ratios = [[], [], []]

all_vfs = []
all_stdoffs = []
all_disps = []
all_ratios = []

total_readings = 0

fig, ((disp_ax, ratio_ax), (disp_cuts_ax, ratio_cuts_ax)) = plt.subplots(2, 2, figsize=(fig_width(), fig_width() * 0.8))

for i, geom_dir in enumerate(geoms):
    # if "w12" not in geom_dir and "solid" not in geom_dir:
    #     continue
    # if "w48" in geom_dir or "w24" in geom_dir:
    #     print(geom_dir)
    #     continue

    vf = None
    labelled = False

    if "circles" in geom_dir:
        marker = "o"
    elif "triangles" in geom_dir:
        marker = "^"
    elif "squares" in geom_dir:
        marker = "s"
    else:
        marker = "D"

    for root, _, files in os.walk(root_dir + geom_dir):
        if "layers" in root:
            continue
        if "params.py" in files:
            sys.path.append(root)
            import params

            importlib.reload(params)
            sys.path.remove(root)

            if vf is None and hasattr(params, "vf"):
                vf = params.vf
            elif vf is None:
                vf = 0

            if hasattr(params, "W"):
                W = params.W
            else:
                W = 0

            if hasattr(params, "A"):
                A = params.A
            else:
                A = 0

            y_offset = params.upper_surface_y

            if not os.path.exists(root + "/readings_dump.csv"):
                continue

            readings = load_readings(root + "/readings_dump.csv")  # Type: List[Reading]

            mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
            A_prime = A / (vf * np.pi * mean_radius ** 2)

            if A_prime > 1.5 and W / mean_radius > 1:
                print(geom_dir, A_prime)
                break

            total_readings += len(readings)
            standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                         for r in readings if filter_by(r)]
            disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if filter_by(r)]
            radius_ratios = [r.get_radius_ratio() for r in readings if filter_by(r)]

            all_stdoffs.extend(standoffs)
            all_vfs.extend([vf] * len(standoffs))
            all_disps.extend(disps)
            all_ratios.extend(radius_ratios)

            if vf in vf_standoffs:
                vf_standoffs[vf].extend(standoffs)
                vf_disps[vf].extend(disps)
                vf_ratios[vf].extend(radius_ratios)

            else:
                vf_standoffs[vf] = standoffs
                vf_disps[vf] = disps
                vf_ratios[vf] = radius_ratios

for i, vf in enumerate(sorted(vf_standoffs.keys())):
    standoffs = vf_standoffs[vf]
    disps = vf_disps[vf]
    ratios = vf_ratios[vf]
    disp_ax.scatter(standoffs, disps, s=2, color=color_map.to_rgba(vf), label=f"$\\phi = {vf * 100:.1f} \\%$")
    ratio_ax.scatter(standoffs, ratios, s=2, color=color_map.to_rgba(vf), label=f"$\\phi = {vf * 100:.1f} \\%$")

    if fit_for_disp_cuts:
        (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, standoffs, disps)
        fit_standoffs = np.linspace(np.min(standoffs), np.max(standoffs))
        fit_disps = c * fit_standoffs ** d

        if i == 0 or i == round(len(vf_standoffs.keys()) / 2) or i == len(vf_standoffs.keys()) - 1:
            disp_ax.plot(fit_standoffs, fit_disps, color=color_map.to_rgba(vf))

        for j, stdoff in enumerate(standoff_cuts):
            cut_vfs[j].append(vf)
            cut_disps[j].append(c * stdoff ** d)

    else:
        for j, stdoff in enumerate(standoff_cuts):
            filt_stdoffs, filt_disps = zip(*[(s, d) for s, d in zip(standoffs, disps)
                                             if stdoff - bin_width / 2 <= s <= stdoff + bin_width / 2])
            disp_ax.scatter(filt_stdoffs, filt_disps, color="magenta", s=0.25)
            cut_vfs[j].append(vf)
            cut_disps[j].append(np.mean(filt_disps))

    if fit_for_ratio_cuts:
        (a, b), _ = curve_fit(lambda x, a, b: a * np.log(x) + b, standoffs, ratios)
        fit_standoffs = np.linspace(np.min(standoffs), np.max(standoffs))
        fit_ratios = a * np.log(fit_standoffs) + b

        if i == 0 or i == round(len(vf_standoffs.keys()) / 2) or i == len(vf_standoffs.keys()) - 1:
            ratio_ax.plot(fit_standoffs, fit_ratios, color=color_map.to_rgba(vf))

        for j, stdoff in enumerate(standoff_cuts):
            cut_ratios[j].append(a * np.log(stdoff) + b)
    else:
        for j, stdoff in enumerate(standoff_cuts):
            filt_stdoffs, filt_disps, filt_ratios = zip(*[(s, d, r) for s, d, r in zip(standoffs, disps, ratios)
                                                          if stdoff - bin_width / 2 <= s <= stdoff + bin_width / 2])
            # ratio_ax.scatter(filt_stdoffs, filt_ratios, color="magenta", s=0.25)
            cut_ratios[j].append(np.mean(filt_ratios))

for stdoff in standoff_cuts:
    disp_ax.axvline(stdoff, color="gray", linestyle="dashed")
    ratio_ax.axvline(stdoff, color="gray", linestyle="dashed")

cax = disp_ax.inset_axes([0.05, 0.05, 0.4, 0.05])
cbar = plt.colorbar(color_map, cax=cax, orientation='horizontal', ticks=np.linspace(0, 0.6, 4))
cbar.ax.tick_params(labelsize='xx-small', pad=1)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("$\\phi$")

disp_ax.loglog()
ratio_ax.set_xscale('log')
disp_ax.set_xticks(range(1, 10))
ratio_ax.set_xticks(range(1, 10))
format_axis_ticks_decimal(disp_ax.xaxis, decimal_places=0)
format_axis_ticks_decimal(ratio_ax.xaxis, decimal_places=0)

disp_ax.set_xlabel("$\\gamma = Y / R_0$")
ratio_ax.set_xlabel("$\\gamma = Y / R_0$")
disp_ax.set_ylabel("$\\Delta / R_0$")
ratio_ax.set_ylabel("$R_1 / R_0$")
print(total_readings)

for j, stdoff in enumerate(standoff_cuts):
    cut_vfs[j], cut_disps[j], cut_ratios[j] = zip(*sorted(zip(cut_vfs[j], cut_disps[j], cut_ratios[j])))
    disp_cuts_ax.plot(np.array(cut_vfs[j]), cut_disps[j], ".", label=f"$\\gamma = {stdoff:.0f}$")
    ratio_cuts_ax.plot(np.array(cut_vfs[j]), cut_ratios[j], ".", label=f"$\\gamma = {stdoff:.0f}$")
disp_cuts_ax.legend(frameon=False, fontsize="small", loc="upper right")
disp_cuts_ax.set_xlabel("$\\phi$")
disp_cuts_ax.set_ylabel("$\\Delta / R_0$")

disp_cuts_ax.set_ylim((0, disp_cuts_ax.get_ylim()[1]))
disp_cuts_ax.set_xlim((-0.15, 1))
ratio_cuts_ax.set_xlim((-0.15, 1))

ratio_cuts_ax.set_xlabel("$\\phi$")
ratio_cuts_ax.set_ylabel("$R_1 / R_0$")

label_subplot(disp_ax, "($a$)")
label_subplot(ratio_ax, "($b$)")
label_subplot(disp_cuts_ax, "($c$)")
label_subplot(ratio_cuts_ax, "($d$)")

plt.tight_layout()

plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
            "paper figures/svd_vf.eps")

# plt.figure()
# plt.tricontourf(all_stdoffs, all_vfs, all_disps, levels=16)
# plt.xlabel("$\\gamma$")
# plt.ylabel("$\\phi$")

plt.show()
