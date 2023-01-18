import os
import sys
import importlib
import matplotlib.pyplot as plt
from matplotlib import cm
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal
from scipy.optimize import curve_fit
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()

vf_standoffs = {}
vf_disps = {}


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


color_map = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=0.6), cmap=cm.get_cmap('viridis'))

standoff_cuts = [2, 3, 4]
cut_vfs = [[], [], []]
cut_rrs = [[], [], []]

total_readings = 0

for i, geom_dir in enumerate(geoms):
    # if "w12" not in geom_dir and "solid" not in geom_dir:
    #     continue
    if "w48" in geom_dir or "w24" in geom_dir:
        continue

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
        if "layers" in root:
            continue
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
            radius_ratios = [r.get_radius_ratio() for r in readings if filter_by(r)]

            if vf in vf_standoffs:
                vf_standoffs[vf].extend(standoffs)
                vf_disps[vf].extend(radius_ratios)
            else:
                vf_standoffs[vf] = standoffs
                vf_disps[vf] = radius_ratios

for i, vf in enumerate(vf_standoffs.keys()):
    standoffs = vf_standoffs[vf]
    radius_ratios = vf_disps[vf]
    plt.scatter(standoffs, radius_ratios, s=2, color=color_map.to_rgba(vf), label=f"$\\phi = {vf * 100:.1f} \\%$")

    (a, b), _ = curve_fit(lambda x, a, b: a * np.log(x) + b, standoffs, radius_ratios)
    fit_standoffs = np.linspace(np.min(standoffs), np.max(standoffs))
    fit_rrs = a * np.log(fit_standoffs) + b
    plt.plot(fit_standoffs, fit_rrs, color=color_map.to_rgba(vf))

    for j, stdoff in enumerate(standoff_cuts):
        cut_vfs[j].append(vf)
        cut_rrs[j].append(a * np.log(stdoff) + b)

plt.colorbar(color_map, label="$\\phi$")
plt.xscale('log')
plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$R_1 / R_0$")
print(total_readings)
plt.tight_layout()

plt.figure()
for j, stdoff in enumerate(standoff_cuts):
    cut_vfs[j], cut_rrs[j] = zip(*sorted(zip(cut_vfs[j], cut_rrs[j])))
    plt.plot(1 - np.array(cut_vfs[j]), cut_rrs[j], ".", label=f"$\\gamma = {stdoff:.0f}$")
plt.legend(frameon=False)
plt.xlabel("$1 - \\phi$")
plt.ylabel("$R_1 / R_0$")
plt.tight_layout()

plt.show()
