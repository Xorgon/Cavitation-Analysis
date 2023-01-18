import os
import sys
import importlib
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
from matplotlib.collections import PathCollection
from experimental.util.analysis_utils import load_readings
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width, label_subplot
from scipy.stats import pearsonr
from scipy.optimize import curve_fit, minimize, leastsq

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
dirs = []

for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

# initialize_plt(font_size=14, line_scale=1.5, dpi=300)

initialize_plt(dpi=300)
x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=0.6), cmap=cm.get_cmap('viridis'))

min_anisotropy = np.inf
max_anisotropy = 0

all_anisotropies = []
all_disps = []

fig, (opt_disp_ax, opt_r_ax) = plt.subplots(1, 2, figsize=(6, 2.75), sharex='all')

prev_ani, prev_disp, prev_rad = csv_to_lists(
    "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/"
    "Code/Cavitation-Analysis/numerical/models/anisotropy_models/", "complex_geometry_data.csv", True)

(prev_c, prev_d), _ = curve_fit(lambda x, c, d: c * x ** d, prev_ani, prev_disp)
fit_prev_ani = np.linspace(np.min(prev_ani), np.max(prev_ani), 50)
# opt_disp_ax.plot(fit_prev_ani, prev_c * np.array(fit_prev_ani) ** prev_d, "k", linestyle="-.",
#                  label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (prev_c, prev_d), linewidth=1)

# opt_disp_ax.scatter(prev_ani, prev_disp, color="silver", s=0.3)
# opt_r_ax.scatter(prev_ani, prev_rad, color="silver", s=0.3)

total_readings = 0

base_stdoffs = None
base_anis = None
base_disps = None

solid_c, solid_d = (None, None)

min_vfs = []
vfs = []
ws = []
max_vfs = []
pres = []
pre_stds = []
labels = []

for i, dir_path in enumerate(dirs):
    if "layers" in dir_path:  # or "w48" in dir_path:
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

        if hasattr(params, "A"):
            A = params.A
        else:
            A = 0

        y_offset = params.upper_surface_y

        readings = load_readings(dir_path + "readings_dump.csv")
        if readings[0].model_anisotropy is None:
            print(f"Not yet processed {dir_path}")
            continue

        mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if r.ecc_at_max < max_ecc])
        A_prime = A / (vf * np.pi * mean_radius ** 2)

        if A_prime > 1.5 and W / mean_radius > 1:
            continue

        total_readings += len(readings)
        filt_standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                          for r in readings if r.ecc_at_max < max_ecc]
        filt_anisotropies = np.array([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc])
        filt_disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc]
        filt_radius_ratios = [np.linalg.norm(r.get_radius_ratio()) for r in readings if r.ecc_at_max < max_ecc]

        mean_rad = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if r.ecc_at_max < max_ecc])

        # bem_disp_ax.scatter(filt_anisotropies, filt_disps, marker=marker, color=color_map.to_rgba(vf), label=label,
        #                     s=0.5)
        # bem_r_ax.scatter(filt_anisotropies, filt_radius_ratios, marker=marker, color=color_map.to_rgba(vf), s=0.5)

        min_vfs.append(min_vf)
        vfs.append(vf)
        print(label, W, mean_rad)
        ws.append(W / mean_rad)
        max_vfs.append(max_vf)
        if "solid" in dir_path:
            base_stdoffs = np.array(filt_standoffs)
            base_anis = 0.195 * np.array(base_stdoffs) ** -2
            base_disps = np.array(filt_disps)
            pres.append(0.195)
            pre_stds.append(0)

            opt_anisotropies = 0.195 * np.array(filt_standoffs) ** -2

            (solid_c, solid_d), _ = curve_fit(lambda x, c, d: c * x ** d, base_anis, base_disps)
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


            def lstsq_opt_fun(pre):
                anis = pre * np.array(filt_standoffs) ** -2
                return np.log(np.array(filt_disps)) - np.log(solid_c * np.array(anis) ** solid_d)


            opt_pre, cov_x, _, _, _ = leastsq(lstsq_opt_fun, 0.195, full_output=True)
            # opt_res = minimize(ani_curve_opt_fitting, 0.195, bounds=[(1e-9, 0.195)], method="L-BFGS-B")
            # std = np.sqrt(opt_res.hess_inv * np.array([1]))
            std = np.sqrt(cov_x[0][0])
            print("sqrt(inv hess) = ", std)
            # opt_pre = opt_res.x[0]
            opt_anisotropies = opt_pre[0] * np.array(filt_standoffs) ** -2
            pres.append(opt_pre[0])
            pre_stds.append(std)
            labels.append(label)

        opt_disp_ax.scatter(opt_anisotropies, filt_disps, marker=marker, color=color_map.to_rgba(vf), s=0.5)
        opt_r_ax.scatter(opt_anisotropies, filt_radius_ratios,
                         marker=marker, color=color_map.to_rgba(vf), s=0.5)

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

(fit_c, fit_d), _ = curve_fit(lambda x, c, d: d * np.log(x) + np.log(c), all_anisotropies, np.log(all_disps))
fit_ani = np.linspace(np.min(all_anisotropies), np.max(all_anisotropies), 50)
opt_disp_ax.plot(fit_ani, fit_c * np.array(fit_ani) ** fit_d, "k", linestyle="--",
                 label="$\\frac{\\Delta}{R_0} = %.2f \\zeta^{%.2f}$" % (fit_c, fit_d), linewidth=1)

handles, labels = opt_disp_ax.get_legend_handles_labels()
order = [0]
legend = opt_disp_ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                            loc='upper left', frameon=False, fontsize='xx-small', markerfirst=True)
for handle in legend.legendHandles:
    if type(handle) == PathCollection:
        handle.set_sizes(25 * np.array(handle.get_sizes()))

opt_disp_ax.set_xlabel("$\\zeta$")
opt_disp_ax.set_ylabel("$\\Delta / R_0$")
# bem_disp_ax.set_ylabel("$\\Delta / R_0$")
opt_disp_ax.loglog()
# bem_disp_ax.loglog()

min_anisotropy = 1e-3
max_anisotropy = 1e-1
# r_ax.set_ylim((0, 0.5))
# r_ax.set_xlim((min_anisotropy, max_anisotropy))
r_ax_zetas = np.logspace(np.log10(min_anisotropy), np.log10(max_anisotropy), 50)
# opt_r_ax.plot(r_ax_zetas, (0.1 * np.log(r_ax_zetas) + 0.7) ** (1 / 3), "k", linestyle=(0, (5, 2, 1, 2, 1, 2)),
#               label="Supponen \\textit{et al.} (2018)", linewidth=1)

opt_r_ax.set_xlabel("$\\zeta$")
opt_r_ax.set_ylabel("$R_1 / R_0$")
# bem_r_ax.set_ylabel("$R_1 / R_0$")
# bem_r_ax.set_xscale("log", base=10)
opt_r_ax.set_xscale("log", base=10)
# r_ax.xaxis.set_major_formatter(x_formatter)
# r_ax.xaxis.set_minor_formatter(x_formatter)
# r_ax.yaxis.set_major_formatter(y_formatter)
# r_ax.yaxis.set_minor_formatter(y_formatter)
opt_r_ax.legend(loc='upper center', frameon=False, fontsize='xx-small')

# label_subplot(bem_disp_ax, "($a$)")
# label_subplot(bem_r_ax, "($b$)")
# label_subplot(opt_disp_ax, "($c$)")
# label_subplot(opt_r_ax, "($d$)")

cax = opt_disp_ax.inset_axes([0.5, 0.05, 0.45, 0.05])
cbar = plt.colorbar(color_map, cax=cax, orientation='horizontal', ticks=np.linspace(0, 0.6, 4))
cbar.ax.tick_params(labelsize='xx-small', pad=1, length=2, width=0.5)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("$\\phi$")

plt.tight_layout()
# plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
#             "paper figures/all_anisotropy_collapse.eps")

plt.figure(figsize=(fig_width() * 0.75, fig_width() * 0.4))
err_min = np.subtract(vfs, min_vfs)
err_max = np.subtract(max_vfs, vfs)
plt.errorbar(np.array(vfs), pres, xerr=[err_min, err_max], yerr=pre_stds, fmt="o", label="Experiment", capsize=0,
             color="k", markersize=1.5, elinewidth=0.75)

colours = cm.get_cmap('viridis').copy()
over_colour = '#ff2222'
colours.set_over(over_colour)
color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=np.min(ws), vmax=1.5), cmap=colours)

for vf, pre, w in zip(vfs, pres, ws):
    if w > 1.5:
        marker = "D"
    else:
        marker = "o"
    scat = plt.scatter(np.array(vf), pre, s=2.5 ** 2, c=color_map.to_rgba([w]), zorder=10, marker=marker)
cbar = plt.colorbar(color_map, label="$W / R_0$")
cbar.ax.set_ylim((np.min(ws), np.max(ws)))
cbar.ax.patch.set_facecolor(over_colour)


def fit_func(phi, b):
    a = 1 / 0.195 - 1
    return a / (phi ** b + a) - a / (1 + a)


filt_vfs, filt_pres = zip(*[(vf, pre) for vf, pre, w in zip(vfs, pres, ws) if w <= 1.5])

(fit_b), cov = curve_fit(fit_func, filt_vfs, filt_pres)
print(fit_b, np.sqrt(cov))

# fit_b = 0.5
print(fit_b)
plt.plot(np.linspace(0, 1, 100), fit_func(np.linspace(0, 1, 100), fit_b), label="Curve fit")

# From anisotropy_vs_vf.py with gen_varied_simple_plane(0.03, 16000, 0.05, 4000)
# plt.plot(np.array(
#     [0., 0.07142857, 0.14285714, 0.21428571, 0.28571429, 0.35714286, 0.42857143, 0.5, 0.57142857, 0.64285714,
#      0.71428571, 0.78571429, 0.85714286, 0.92857143, 1.]),
#     [0.17571664492313893, 0.15178543883228815, 0.12998472476493259, 0.11021957268312195, 0.0923950525490503,
#      0.07641623432478636, 0.06218818797250579, 0.04961598345429517, 0.03860469073230888, 0.029059379768668223,
#      0.020885120525486932, 0.013986982964907631, 0.008270037049047215, 0.0036393527400351866,
#      -4.019655235379999e-18], color="C1", label="BEM")

# From anisotropy_vs_vf.py with gen_varied_simple_plane(0.045, 12000, 1, 6000)
# Matching anisotropy paper validation case
# plt.plot(np.array(
#     [0., 0.07142857, 0.14285714, 0.21428571, 0.28571429, 0.35714286, 0.42857143, 0.5, 0.57142857, 0.64285714,
#      0.71428571, 0.78571429, 0.85714286, 0.92857143, 1.]),
#     [0.18362296143382972, 0.1587034810513032, 0.13599076541187188, 0.11538710619773507, 0.09679479509101364,
#      0.08011612377381234, 0.06525338392827, 0.052108867236548354, 0.04058486538074465, 0.030583670043019685,
#      0.02200757290548406, 0.014758865650284428, 0.00873983995955036, 0.0038527875154099078, -4.019655235379999e-18],
#     color="C1", label="BEM")

# From anisotropy_vs_vf.py with gen_varied_simple_plane(0.045, 17000, 0.05, 3000)
# Matching steel_porous_plate_anisotropy
# plt.plot(np.array(
#     [0., 0.07142857, 0.14285714, 0.21428571, 0.28571429, 0.35714286, 0.42857143, 0.5, 0.57142857, 0.64285714,
#      0.71428571, 0.78571429, 0.85714286, 0.92857143, 1.]),
#     [0.17450036446437237, 0.15077388441185346, 0.1291545051462996, 0.10954861179040958, 0.09186258946683544,
#      0.07600282329826794, 0.061875698407415955, 0.04938759991695053, 0.038444912949561906, 0.028954022627921034,
#      0.020821314074716605, 0.013953172412640692, 0.008255982764374259, 0.0036361302525979374, -4.019655235379999e-18],
#     color="C1", label="BEM")
print(f"exp_vfs = {vfs}")
print(f"exp_pres = {pres}")
plt.xlabel("$\\phi$")
plt.ylabel("$g(\\phi)$")
plt.legend(frameon=False)
plt.tight_layout()
# for vf, pre, label in zip(vfs, pres, labels):
#     plt.text(1 - vf, pre, label, fontsize="xx-small")
plt.show()
