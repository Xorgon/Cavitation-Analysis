import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

from common.util.plotting_utils import initialize_plt, fig_width
from experimental.util.analysis_utils import load_readings

slot_dirs = []
root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        slot_dirs.append(root + "/")
print(f"Found {len(slot_dirs)} data sets")
print(slot_dirs)

max_ecc = 0.22

initialize_plt()
x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(fig_width(), fig_width() * 0.5))

for i, dir_path in enumerate(slot_dirs):
    try:
        sys.path.append(dir_path)
        import params

        importlib.reload(params)
        sys.path.remove(dir_path)

        x_offset = params.left_slot_wall_x + params.slot_width / 2
        y_offset = params.upper_surface_y

        readings = load_readings(dir_path + "readings_dump.csv")

        standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                     for r in readings if r.ecc_at_max < max_ecc]

        disp_ax.scatter(standoffs,
                        [np.linalg.norm(r.get_normalised_displacement()) for r in readings if
                         r.ecc_at_max < max_ecc],
                        marker=".", color=f'C{i}', alpha=0.25, zorder=-1)

        r_ax.scatter(standoffs,
                     [np.linalg.norm(r.get_radius_ratio()) for r in readings if
                      r.ecc_at_max < max_ecc],
                     marker=".", color=f'C{i}', alpha=0.25)
    except FileNotFoundError:
        print(f"Not yet processed {dir_path}")

disp_ax.set_xlabel("$\\gamma = Y / R_0$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.loglog()
disp_ax.xaxis.set_major_formatter(x_formatter)
disp_ax.xaxis.set_minor_formatter(x_formatter)
disp_ax.yaxis.set_major_formatter(y_formatter)
disp_ax.yaxis.set_minor_formatter(y_formatter)

r_ax.set_xlabel("$\\gamma = Y / R_0$")
r_ax.set_ylabel("$R_1 / R_0$")
r_ax.loglog()
r_ax.xaxis.set_major_formatter(x_formatter)
r_ax.xaxis.set_minor_formatter(x_formatter)
r_ax.yaxis.set_major_formatter(y_formatter)
r_ax.yaxis.set_minor_formatter(y_formatter)

disp_ax.annotate(f"($a$)", xy=(3, 0.5), xytext=(0.025, 0.975),
            textcoords='axes fraction', color="k",
            horizontalalignment='left', verticalalignment='top')
r_ax.annotate(f"($b$)", xy=(3, 0.5), xytext=(0.025, 0.975),
            textcoords='axes fraction', color="k",
            horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
