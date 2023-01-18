import os
import sys
import importlib
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()
fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(5, 2.5))


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


total_readings = 0
c_counter = 0

for i, geom_dir in enumerate(geoms):
    if "w20vf48squares" not in geom_dir and "solid" not in geom_dir:
        continue

    vf = None
    labelled = [False, False, False, False, False]

    if "circles" in geom_dir:
        marker = "o"
    elif "triangles" in geom_dir:
        marker = "^"
    elif "squares" in geom_dir:
        marker = "s"
    else:
        marker = "D"

    for root, _, files in os.walk(root_dir + geom_dir):
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

            y_offset = params.upper_surface_y

            readings = load_readings(root + "/readings_dump.csv")  # Type: List[Reading]
            total_readings += len(readings)
            standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                         for r in readings if filter_by(r)]
            disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if filter_by(r)]
            rrs = [r.get_radius_ratio() for r in readings if filter_by(r)]
            if "solid" not in geom_dir:
                layers = int(root.split('layer')[0][-2])
                disp_ax.scatter(standoffs, disps, color=f"C{layers}", marker=marker, s=2,
                                label=f"$H / W = {layers / W:.2f}$" if not labelled[layers - 1] else None)
                r_ax.scatter(standoffs, rrs, color=f"C{layers}", marker=marker, s=2,
                             label=f"$H / W = {layers / W:.2f}$" if not labelled[layers - 1] else None)
                labelled[layers - 1] = True
            else:
                disp_ax.scatter(standoffs, disps, color=f"C{0}", marker=marker, s=2, label=f"Solid")
                r_ax.scatter(standoffs, rrs, color=f"C{0}", marker=marker, s=2, label=f"Solid")

legend = disp_ax.legend(frameon=False, fontsize="x-small", ncol=1, loc="lower left")
for handle in legend.legendHandles:
    if type(handle) == PathCollection:
        handle.set_sizes(5 * np.array(handle.get_sizes()))
disp_ax.loglog()
disp_ax.set_xlabel("$\\gamma = Y / R_0$")
disp_ax.set_ylabel("$\\Delta / R_0$")
format_axis_ticks_decimal(disp_ax.xaxis, 0)

r_ax.set_xscale('log')
r_ax.set_xlabel("$\\gamma = Y / R_0$")
r_ax.set_ylabel("$R_1 / R_0$")
format_axis_ticks_decimal(r_ax.xaxis, 0)

print(total_readings)

plt.tight_layout()

# plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
#             "paper figures/svd_thickness.eps", dpi=300)
plt.show()
