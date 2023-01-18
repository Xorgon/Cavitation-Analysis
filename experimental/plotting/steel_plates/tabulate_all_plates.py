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


rows = []
vfs = []

for i, geom_dir in enumerate([geom_dir for geom_dir in geoms if geom_filter(geom_dir)]):
    shape = None

    if "circles" in geom_dir:
        shape = "circles"
    elif "triangles" in geom_dir:
        shape = "triangles"
    elif "squares" in geom_dir:
        shape = "squares"
    else:
        shape = "solid"

    vf, W, A = None, None, None

    readings = []

    for root, _, files in os.walk(root_dir + geom_dir):
        if "layers" in root:
            continue
        if "params.py" in files:
            sys.path.append(root)
            import params

            importlib.reload(params)
            sys.path.remove(root)

            if hasattr(params, "vf"):
                vf = params.vf
            else:
                vf = 0

            if hasattr(params, "W"):
                W = params.W
            else:
                W = 0

            if hasattr(params, "A"):
                A = params.A
            else:
                A = 0

            readings.extend(load_readings(root + "/readings_dump.csv"))  # Type: List[Reading]

    min_radius = min([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
    mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
    max_radius = max([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
    rows.append(f"{shape} & {vf * 100:4.1f} & {W:.2f} & {A:.2f} & "
                f"{mean_radius:.2f} \\\\")
    vfs.append(vf)

vfs, rows = zip(*sorted(zip(vfs, rows)))

for row in rows:
    print(row)
