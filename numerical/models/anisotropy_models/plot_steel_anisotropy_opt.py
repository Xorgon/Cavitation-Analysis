import os
import sys
import importlib
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
from experimental.util.analysis_utils import load_readings
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width
from scipy.stats import pearsonr
from scipy.optimize import curve_fit, minimize

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

color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=0.6), cmap=cm.get_cmap('viridis'))

min_anisotropy = np.inf
max_anisotropy = 0

all_anisotropies = []
all_disps = []

fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(fig_width(), fig_width() * 0.5))

prev_ani, prev_disp, prev_rad = csv_to_lists(
    "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/"
    "Code/Cavitation-Analysis/numerical/models/anisotropy_models/", "complex_geometry_data.csv", True)

disp_ax.scatter(prev_ani, prev_disp, color="silver", s=0.3)
r_ax.scatter(prev_ani, prev_rad, color="silver", s=0.3)

(prev_c, prev_d), _ = curve_fit(lambda x, c, d: c * x ** d, prev_ani, prev_disp)
fit_prev_ani = np.linspace(np.min(prev_ani), np.max(prev_ani), 50)
disp_ax.plot(fit_prev_ani, prev_c * np.array(fit_prev_ani) ** prev_d, "k", linestyle="-.",
             label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (prev_c, prev_d), linewidth=0.5)

total_readings = 0

base_stdoffs = None
base_anis = None
base_disps = None

min_vfs = []
vfs = []
ws = []
max_vfs = []
pres = []
labels = []

for i, dir_path in enumerate(dirs):
    if "layers" in dir_path or "w48" in dir_path:
        continue

    max_ecc = 0.20
    label = dir_path.replace(root_dir, "").replace("\\", "/")

    if "circles" in dir_path:
        marker = "o"
    elif "triangles" in dir_path:
        marker = "^"
    elif "squares" in dir_path:
        marker = "s"
    else:
        marker = "D"

    try:
        sys.path.append(dir_path)
        import params

        importlib.reload(params)
        sys.path.remove(dir_path)

        min_vf = None
        vf = None
        max_vf = None
        W = None
        if vf is None and hasattr(params, "min_vf") and hasattr(params, "vf") and hasattr(params, "max_vf"):
            min_vf = params.min_vf
            vf = params.vf
            max_vf = params.max_vf
        elif vf is None:
            min_vf = 0
            vf = 0
            max_vf = 0
        if hasattr(params, "W"):
            W = params.W
        else:
            W = 0

        y_offset = params.upper_surface_y

        readings = load_readings(dir_path + "readings_dump.csv")
        if readings[0].model_anisotropy is None:
            print(f"Not yet processed {dir_path}")
            continue
        total_readings += len(readings)
        filt_standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                          for r in readings if r.ecc_at_max < max_ecc]
        # filt_anisotropies = np.array([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc])
        filt_disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc]

        mean_rad = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if r.ecc_at_max < max_ecc])

        min_vfs.append(min_vf)
        vfs.append(vf)
        ws.append(W / mean_rad)
        max_vfs.append(max_vf)
        if "solid" in dir_path:
            base_stdoffs = np.array(filt_standoffs)
            base_anis = 0.195 * np.array(base_stdoffs) ** -2
            base_disps = np.array(filt_disps)
            pres.append(0.195)

            filt_anisotropies = 0.195 * np.array(filt_standoffs) ** -2
        else:
            def ani_curve_opt_pearson(pre):
                anis = pre * np.array(filt_standoffs) ** -2
                comb_anis = np.concatenate([base_anis, anis])
                comb_disps = np.concatenate([base_disps, filt_disps])
                return -abs(pearsonr(comb_anis, comb_disps)[0])


            def ani_curve_opt_fitting(pre):
                anis = pre * np.array(filt_standoffs) ** -2
                comb_anis = np.concatenate([base_anis, anis])
                comb_disps = np.concatenate([base_disps, filt_disps])

                (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, comb_anis, comb_disps)
                fitted_disps = c * comb_anis ** d
                rmsd = np.sqrt(np.mean((np.log(fitted_disps) - np.log(comb_disps)) ** 2))
                return rmsd


            opt_pre = minimize(ani_curve_opt_fitting, 0.195, bounds=[(1e-9, 0.195)]).x[0]
            filt_anisotropies = opt_pre * np.array(filt_standoffs) ** -2
            pres.append(opt_pre)
            labels.append(label)

        # (c, d), _ = curve_fit(lambda x, c, d: c * x ** d, filt_anisotropies, filt_disps)
        #
        # print(c, d)

        disp_ax.scatter(filt_anisotropies, filt_disps, marker=marker, color=color_map.to_rgba(vf),
                        alpha=0.5, s=0.5)
        r_ax.scatter(filt_anisotropies,
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if r.ecc_at_max < max_ecc],
                     marker=marker, color=color_map.to_rgba(vf), alpha=0.5, s=0.5)

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

disp_ax.legend(loc='upper center', frameon=False, fontsize='xx-small')

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
r_ax.legend(loc='upper center', frameon=False, fontsize='xx-small')

disp_ax.annotate(f"($a$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
                 textcoords='axes fraction', color="k",
                 horizontalalignment='left', verticalalignment='top')
r_ax.annotate(f"($b$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
              textcoords='axes fraction', color="k",
              horizontalalignment='left', verticalalignment='top')

cax = disp_ax.inset_axes([0.5, 0.05, 0.45, 0.05])
cbar = plt.colorbar(color_map, cax=cax, orientation='horizontal', ticks=np.linspace(0, 0.6, 4))
cbar.ax.tick_params(labelsize='xx-small', pad=1)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("$\\phi$")

plt.tight_layout()


plt.figure()
# Negative because using 1 - vf to plot
err_min = -np.subtract(vfs, min_vfs)
err_max = -np.subtract(max_vfs, vfs)
plt.errorbar(1 - np.array(vfs), pres, xerr=[err_min, err_max], fmt="o", label="Experimental", capsize=0)
plt.scatter(1 - np.array(vfs), pres, c=ws, zorder=10)
plt.plot(1 - np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.]),
         [0.1831211283232972, 0.1356203207068641, 0.09653215161309309, 0.06507716721389999, 0.040475913681138224,
          0.021948937186654156, 0.008716783902318965, -4.019655235379999e-18], color="C1", label="Numerical")
print(f"exp_vfs = {vfs}")
print(f"exp_pres = {pres}")
plt.xlabel("$1 - \\phi$")
plt.ylabel("$c$ where $\\zeta = c\\gamma^{-2}$")
plt.legend(frameon=False)
# for vf, pre, label in zip(vfs, pres, labels):
#     plt.text(1 - vf, pre, label, fontsize="xx-small")
plt.show()
