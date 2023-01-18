import os
import sys
import importlib
import matplotlib.pyplot as plt
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


total_readings = 0

for i, geom_dir in enumerate(geoms):
    if "w24" not in geom_dir:
        continue

    vf = None

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
            continue  # Don't plot the thicker ones here
        if "params.py" in files:
            sys.path.append(root)
            import params

            importlib.reload(params)
            sys.path.remove(root)

            if vf is None and hasattr(params, "vf"):
                vf = params.vf
            elif vf is None:
                vf = 0

            y_offset = params.upper_surface_y

            readings = load_readings(root + "/readings_dump.csv")  # Type: List[Reading]
            total_readings += len(readings)
            standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                         for r in readings if filter_by(r)]
            disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if filter_by(r)]
            plt.scatter(standoffs, disps, marker=marker, s=2,
                        label=root.split('\\')[-1])
            labelled = True

plt.legend(frameon=False, fontsize="xx-small", ncol=2, loc="lower left")
plt.loglog()
plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$\\Delta / R_0$")

print(total_readings)

plt.tight_layout()
plt.show()
