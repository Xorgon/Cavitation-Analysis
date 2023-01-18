import os
from util.file_utils import csv_to_lists
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from experimental.util.analysis_utils import load_readings
import numpy as np
from common.util.plotting_utils import initialize_plt

corner_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/60 degree corner/",
               "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/90 degree corner/"]
corner_angles = [60, 90]

flat_plate_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]

triangle_dirs = ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
                 "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]

slot_dirs = []
root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        slot_dirs.append(root + "/")
print(f"Found {len(slot_dirs)} data sets")
print(slot_dirs)

porous_groups = [
    ["1 mm steel 50 mm plate, $\\phi = 0.24$",
     [["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 1/"],
      [
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 2/"]],
     0.24],
]

porous_dirs = []
vfs = []
for g in porous_groups:
    for d in g[1]:
        porous_dirs.append(d[0])
        vfs.append(g[2])

# initialize_plt(font_size=14, line_scale=1.5, dpi=300)

disp_fig = plt.figure()
plt.xlabel("$\\zeta$")
plt.ylabel("$\\Delta / R_0$")
plt.loglog()
disp_ax = plt.gca()
# disp_ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
# disp_ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter('{x:.2f}'))

r_fig = plt.figure()
plt.xlabel("$\\zeta$")
plt.ylabel("$R_{max2} / R_{max}$")
plt.loglog()
r_ax = plt.gca()

total_readings = 0

for i, dir_path in enumerate(corner_dirs):
    try:
        zetas, disps = csv_to_lists(dir_path, "zetas_vs_disps_sa.csv", True)
        total_readings += len(zetas)
        disp_ax.scatter(zetas, disps, marker=".", color=f'C{3 + 3 * i}', alpha=1,
                        label=f"${corner_angles[i]}^\\circ$ corner MOI")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(corner_dirs):
    try:
        zetas, disps = csv_to_lists(dir_path, "zetas_vs_disps.csv", True)
        total_readings += len(zetas)
        disp_ax.scatter(zetas, disps, marker=".", color=f'C{5 + 3 * i}', alpha=1,
                        label=f"${corner_angles[i]}^\\circ$ corner BEM")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(triangle_dirs):
    max_ecc = 0.22
    try:
        # continue
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        disp_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if
                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                        marker=".", label="Triangle", alpha=0.1)
        r_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if
                      r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if
                      r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                     marker=".", label="Triangle", alpha=0.1)
        zetas, disps, _ = csv_to_lists(dir_path, "zetas_vs_disps.csv", True)
        total_readings += len(zetas)
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(porous_dirs):
    max_ecc = 0.22
    if i == 0:
        label = "Porous plate"
    else:
        label = None
    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        disp_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc],
                        marker=".", color=f'C1', label=label, alpha=0.1)
        r_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if r.ecc_at_max < max_ecc],
                     marker=".", color=f'C1', alpha=0.1)
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(flat_plate_dirs):
    max_ecc = 0.22
    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        disp_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r.ecc_at_max < max_ecc],
                        marker=".", color=f'C2', label="Solid plate", alpha=0.1)
        r_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if r.ecc_at_max < max_ecc],
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if r.ecc_at_max < max_ecc],
                     marker=".", color=f'C2', alpha=0.1)
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

for i, dir_path in enumerate(slot_dirs):
    max_ecc = 0.22
    if i == 0:
        label = "Slots\\textsuperscript{[3]}"
    else:
        label = None
    try:
        readings = load_readings(dir_path + "readings_dump.csv")
        total_readings += len(readings)
        disp_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if
                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                         r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                        marker=".", color=f'C0', alpha=0.1, label=label, zorder=-1)

        r_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings if
                      r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if
                      r.ecc_at_max < max_ecc and r.model_anisotropy is not None],
                     marker=".", color=f'C0', alpha=0.1)
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

print(f"Total readings = {total_readings}")

disp_ax.legend(loc='lower right', frameon=False)
disp_fig.tight_layout()
r_fig.tight_layout()
plt.show()
