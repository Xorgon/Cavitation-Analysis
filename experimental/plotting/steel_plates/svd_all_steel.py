import os
import sys
import importlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, Rbf
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()
plt.figure(figsize=(4.5, 3))


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


def geom_filter(g: str):
    include_filter = [
        # "vf24",
    ]
    exclude_filter = [
        # "w06",
    ]
    return (any([f in g for f in include_filter]) or len(include_filter) == 0) \
           and all([f not in g for f in exclude_filter])


tags = []
vfs = []

min_radius = np.inf
max_radius = 0

min_W = np.inf
max_W = 0

total_readings = 0

for i, geom_dir in enumerate([geom_dir for geom_dir in geoms if geom_filter(geom_dir)]):
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

            if W != 0:
                min_W = min(min_W, W)
                max_W = max(max_W, W)

            y_offset = params.upper_surface_y

            readings = load_readings(root + "/readings_dump.csv")  # Type: List[Reading]
            total_readings += len(readings)
            standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                         for r in readings if filter_by(r)]
            disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if filter_by(r)]

            mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])

            min_radius = min(min_radius, min([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)]))
            max_radius = max(max_radius, max([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)]))

            plt.scatter(standoffs, disps, color=f"C{i}", marker=marker, s=2,
                        label="$W / \\bar{R_0}" + f" = {W / mean_radius:.2f}$,  $\\phi = {vf * 100:.1f}$ %"
                        if not labelled else None)
            labelled = True

            standoffs, disps = zip(*sorted(zip(standoffs, disps)))
            cs = Rbf(standoffs, disps, smooth=1)
            line_stdoffs = np.linspace(np.min(standoffs), np.max(standoffs), 50)
            line_disps = cs(line_stdoffs)
            plt.plot(line_stdoffs, line_disps, color=f"C{i}")

            tags.append(geom_dir)
            vfs.append(vf)

if len(tags) / 2 > 8:
    plt.legend(frameon=False, fontsize="xx-small", ncol=2)
else:
    plt.legend(frameon=False, fontsize="xx-small", ncol=1)
plt.loglog()
plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$\\Delta / R_0$")
format_axis_ticks_decimal(plt.gca().xaxis, 0)

print(total_readings)

plt.tight_layout()

for vf, tag in sorted(zip(vfs, tags)):
    print(f"{tag:10s} - {100 * vf:4.1f} %")

print(f"Minimum radius = {min_radius:.2f} mm, maximum radius = {max_radius:.2f} mm.")
print(f"Minimum hole = {min_W:.2f} mm, maximum hole = {max_W:.2f} mm.")

plt.show()
