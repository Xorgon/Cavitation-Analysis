import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
import numpy as np
import matplotlib.pyplot as plt

dirs = ["../../../../Data/SlotErrorMeasurement/"]
root_dir = "../../../../Data/SlotSweeps"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

radii = []
displacements = []
avg_vels = []
ps = []
qs = []
ws = []
hs = []
theta_js = []
zetas = []
sup_disps = []
times = []
idxs = []
for i, dir_path in enumerate(dirs):
    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    x_offset = params.left_slot_wall_x + params.slot_width / 2
    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv")
    # readings = sorted(readings, key=lambda r: r.m_x)

    for j, reading in enumerate(readings):
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        p = pos[0] - x_offset
        q = pos[1] - y_offset

        if q >= 0:
            ps.append(p)
            qs.append(q)
            ws.append(params.slot_width)
            hs.append(params.slot_height)
            radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
            radii.append(radius)
            displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px
            displacements.append(displacement)
            avg_vels.append((displacement / 1000) * 1e5 / reading.inter_max_frames)  # m/s
            theta_js.append(reading.get_jet_angle())
            standoff = q / radius
            zetas.append(0.195 * standoff ** (-2))  # Supponen et al. 2016
            sup_disps.append(np.linalg.norm(reading.sup_disp_vect) * params.mm_per_px)
            times.append(reading.inter_max_frames)
            idxs.append(j)

initialize_plt(font_size=14, line_scale=2)
plt.figure()
plt.scatter(np.divide(qs, radii), np.divide(displacements, radii),
            c=np.log(np.abs(2 * np.divide(ps, ws))) > 1, marker=".", cmap=plt.cm.get_cmap('brg'))
plt.colorbar(label="$log(|\\bar{p}|)$")
plt.xlabel("$q / R$")
plt.ylabel("$\Delta / R$ = displacement / radius")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()

plt.figure()
plt.scatter(np.divide(qs, radii), avg_vels,
            c=np.log(np.abs(2 * np.divide(ps, ws))), marker=".", cmap=plt.cm.get_cmap('brg'))
plt.xlabel("$q / R$")
plt.ylabel("Average translation velocity")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()

plt.figure()
fit_params = np.polyfit(np.log(zetas), np.log(np.divide(displacements, radii)), 1)
fit_zetas = np.linspace(min(zetas), max(zetas), 100)
fit_disps = np.exp(fit_params[1]) * np.power(fit_zetas, fit_params[0])
fit_label = "$%.3f \\zeta^{%.3f}$ (fitted)" % (np.exp(fit_params[1]), fit_params[0])

sup_pow_disps = 2.5 * np.power(fit_zetas, 0.6)
eye_disps = 2.8 * np.power(fit_zetas, 0.42)

plt.plot(fit_zetas, fit_disps, label=fit_label)
plt.plot(fit_zetas, sup_pow_disps, label="$2.5 \\zeta^{0.6}$ (Supponen et al. 2016)")
plt.plot(fit_zetas, eye_disps, color="purple", label="$2.8 \\zeta^{0.42}$ (by eye)")
plt.scatter(zetas, np.divide(displacements, radii),
            c=np.log(np.abs(2 * np.divide(ps, ws))), marker=".", cmap=plt.cm.get_cmap('brg'))
plt.colorbar(label="$log(|\\bar{p}|)$")
plt.xlabel("$\\zeta$ (Supponen et al. 2016)")
plt.ylabel("$\Delta / R$ = displacement / radius")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()

plt.figure()
fit_params = np.polyfit(np.log(zetas), np.log(np.divide(sup_disps, radii)), 1)
fit_zetas = np.linspace(min(zetas), max(zetas), 100)
fit_disps = np.exp(fit_params[1]) * np.power(fit_zetas, fit_params[0])
fit_label = "$%.3f \\zeta^{%.3f}$ (fitted)" % (np.exp(fit_params[1]), fit_params[0])

sup_pow_disps = 2.5 * np.power(fit_zetas, 0.6)

plt.plot(fit_zetas, fit_disps, label=fit_label)
plt.plot(fit_zetas, sup_pow_disps, label="$2.5 \\zeta^{0.6}$ (Supponen et al. 2016)")
plt.scatter(zetas, np.divide(displacements, radii), marker="1", c="k", label="Original displacement")
plt.scatter(zetas, np.divide(sup_disps, radii),
            # c=times)
            c=np.log(np.abs(2 * np.divide(ps, ws))), marker=".", cmap=plt.cm.get_cmap('brg'),
            label="Supponen displacement")

# WARNING: Do not have LaTeX plotting on if you want to use this
# for k in zip(zetas, np.divide(sup_disps, radii), idxs):
#     plt.annotate(f'{k[2] + 1}', xy=[k[0], k[1]], textcoords='data')

plt.colorbar(label="$log(|\\bar{p}|)$")
plt.xlabel("$\\zeta$ (Supponen et al. 2016)")
plt.ylabel("Normalised bubble centroid displacement = $\Delta z / R_0$")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()

plt.figure()
plt.scatter(radii, displacements, c=qs, marker="s")
plt.colorbar(label="$q (mm)$")
plt.xlabel("$R$ = Radius (mm)")
plt.ylabel("$\Delta$ = displacement (mm)")
plt.tight_layout()
plt.show()
