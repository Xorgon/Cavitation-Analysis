import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
import numpy as np
import matplotlib.pyplot as plt

filter_by = ""

dirs = []
root_dir = "../../../../../Porous Materials/Data/Steel plates/"  # w20vf48squares/"
for root, _, files in os.walk(root_dir):
    if "params.py" in files and (filter_by is None or filter_by in root):
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

radii = []
displacements = []
avg_vels = []
ps = []
qs = []
sup_disps = []
times = []
idxs = []

highlight = None

# initialize_plt(font_size=14, line_scale=2)
disp_fig = plt.figure()
angle_fig = plt.figure()

total_readings = 0

for i, dir_path in enumerate(dirs):
    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv", include_invalid=True)
    label = dir_path.split("/")[-2]

    zetas = []
    norm_disps = []
    inv_sqr_standoffs = []
    standoffs = []
    ys = []
    theta_js = []
    for j, reading in enumerate(readings):
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        y = pos[1] - y_offset

        if abs(y) >= 0 and reading.ecc_at_max < 0.922:
            ys.append(y)
            radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
            radii.append(radius)
            displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px
            displacements.append(displacement)
            # avg_vels.append((displacement / 1000) * 1e5 / reading.inter_max_frames)  # m/s
            theta_js.append(reading.get_jet_angle())
            standoff = y / radius
            # if standoff ** (-2) < 0.012:
            #     print(reading.idx, reading.repeat_number)
            inv_sqr_standoffs.append(standoff ** (-2))
            standoffs.append(standoff)
            norm_disps.append(displacement / radius)
            # if norm_disps[-1] > 1.1:
                # print(label, reading.idx, reading.repeat_number)
            # sup_disps.append(np.linalg.norm(reading.sup_disp_vect) * params.mm_per_px)
            # times.append(reading.inter_max_frames)
            # idxs.append(j)
            total_readings += 1

    if highlight is None or highlight == "" or highlight in label:
        alpha = 1
    else:
        alpha = 0.25
    disp_fig.gca().scatter(standoffs, norm_disps, label=label, alpha=alpha)
    angle_fig.gca().scatter(standoffs, theta_js, label=label, alpha=alpha)
    print(label, "average theta_j", np.mean([theta_j for theta_j in theta_js if np.abs(theta_j) < 0.5]))

# filt_zetas, filt_disps, filt_radii = \
#     zip(*[(z, d, r) for z, d, r, p in zip(zetas, displacements, radii, ps) if np.abs(p) > 4])
# fit_params = np.polyfit(np.log(filt_zetas), np.log(np.divide(filt_disps, filt_radii)), 1)
# fit_zetas = np.linspace(min(filt_zetas), max(filt_zetas), 100)
# fit_disps = np.exp(fit_params[1]) * np.power(fit_zetas, fit_params[0])
# fit_label = "$%.3f \\zeta^{%.3f}$ (fitted)" % (np.exp(fit_params[1]), fit_params[0])
# print(len(filt_zetas))
# plt.plot(fit_zetas, fit_disps, label=fit_label, color="C1")

# plt.xlabel("$\\gamma^{-2} = \\left(\\frac{y}{r}\\right)^{-2}$")
# plt.xlabel("$\\gamma = \\frac{y}{r}$")
disp_fig.gca().set_xlabel("$\\gamma = \\frac{y}{r}$")
disp_fig.gca().set_ylabel("$\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')
disp_fig.gca().legend(frameon=False, fontsize="xx-small")
disp_fig.tight_layout()

angle_fig.gca().set_xlabel("$\\gamma = \\frac{y}{r}$")
angle_fig.gca().set_ylabel("$\\theta_j$")
angle_fig.gca().legend(frameon=False, fontsize="xx-small")
angle_fig.tight_layout()

print(f"Total readings: {total_readings}")

plt.show()
