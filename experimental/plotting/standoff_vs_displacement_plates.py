import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
import numpy as np
import matplotlib.pyplot as plt

dirs = []
root_dir = "../../../../../Porous Materials/Data/"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

radii = []
displacements = []
avg_vels = []
ps = []
qs = []
theta_js = []
sup_disps = []
times = []
idxs = []

initialize_plt(font_size=14, line_scale=2)
plt.figure()

for i, dir_path in enumerate(dirs):
    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv")

    zetas = []
    norm_disps = []
    standoffs = []
    for j, reading in enumerate(readings):
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        # p = pos[0]
        q = pos[1] - y_offset

        if q >= 0:
            # ps.append(p)
            # qs.append(q)
            radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
            radii.append(radius)
            displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px
            displacements.append(displacement)
            # avg_vels.append((displacement / 1000) * 1e5 / reading.inter_max_frames)  # m/s
            # theta_js.append(reading.get_jet_angle())
            standoff = q / radius
            if standoff ** (-2) < 0.012:
                print(reading.idx, reading.repeat_number)
            standoffs.append(standoff ** (-2))
            norm_disps.append(displacement / radius)
            # sup_disps.append(np.linalg.norm(reading.sup_disp_vect) * params.mm_per_px)
            # times.append(reading.inter_max_frames)
            # idxs.append(j)

    plt.scatter(standoffs, norm_disps, label=dir_path.split("/")[-2])


# filt_zetas, filt_disps, filt_radii = \
#     zip(*[(z, d, r) for z, d, r, p in zip(zetas, displacements, radii, ps) if np.abs(p) > 4])
# fit_params = np.polyfit(np.log(filt_zetas), np.log(np.divide(filt_disps, filt_radii)), 1)
# fit_zetas = np.linspace(min(filt_zetas), max(filt_zetas), 100)
# fit_disps = np.exp(fit_params[1]) * np.power(fit_zetas, fit_params[0])
# fit_label = "$%.3f \\zeta^{%.3f}$ (fitted)" % (np.exp(fit_params[1]), fit_params[0])
# print(len(filt_zetas))
# plt.plot(fit_zetas, fit_disps, label=fit_label, color="C1")

plt.xlabel("$\\gamma^{-2}$")
plt.ylabel("$\Delta / R$ = displacement / radius")
plt.xscale('log')
plt.yscale('log')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
