import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from scipy.interpolate import Rbf

from util.plotting_utils import initialize_plt, format_axis_ticks_decimal, label_subplot
from util.analysis_utils import load_readings, Reading

root_dir = "../Data/"

geoms = os.listdir(root_dir)

initialize_plt()
fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(5, 2.5))


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


def geom_filter(g: str):
    include_filter = [
        "vf24",
    ]
    exclude_filter = [
        "w06",
        "w12",
    ]
    return (any([f in g for f in include_filter]) or len(include_filter) == 0) \
           and all([f not in g for f in exclude_filter])


tags = []
vfs = []

total_readings = 0

for i, geom_dir in enumerate([geom_dir for geom_dir in geoms if geom_filter(geom_dir)]):
    vf = None
    labelled = False

    L_fact = None
    if "circles" in geom_dir:
        marker = "o"
        L_fact = 1 / np.sqrt(3)
    elif "triangles" in geom_dir:
        marker = "^"
        L_fact = 1
    elif "squares" in geom_dir:
        marker = "s"
        L_fact = 1 / np.sqrt(2)
    else:
        marker = "D"
        L_fact = 0

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

            if hasattr(params, "W"):
                W = params.W
            else:
                W = 0

            if hasattr(params, "A"):
                A = params.A
            else:
                A = 0

            if hasattr(params, "S"):
                S = params.S
            else:
                S = 0

            y_offset = params.upper_surface_y

            readings = load_readings(root + "/readings_dump.csv")  # Type: List[Reading]
            total_readings += len(readings)
            standoffs = [(r.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset) / r.get_max_radius(params.mm_per_px)
                         for r in readings if filter_by(r)]
            disps = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if filter_by(r)]
            rrs = [r.get_radius_ratio() for r in readings if filter_by(r)]

            mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
            disp_ax.scatter(standoffs, disps, color=f"C{i}", marker=marker, s=2,
                            label=f"{W / mean_radius:.2f}, " + "\\hspace{8pt}"
                                  + f"{A / (vf * np.pi * mean_radius ** 2):5.2f}," + "\\hspace{8pt}"
                                  + f"{vf * 100:.1f} \\%"
                            # label="$W / \\bar{R}_0 = $" + f" {W / mean_radius:.2f}, "
                            #                               f"$A' = $" + f"{A / (vf * np.pi * mean_radius ** 2):5.2f}, "
                            #                                            "$\\phi = " + f"{vf * 100:.1f}$ \\%"
                            if not labelled else None)
            labelled = True

            r_ax.scatter(standoffs, rrs, color=f"C{i}", marker=marker, s=2)

            standoffs, disps, rrs = zip(*sorted(zip(standoffs, disps, rrs)))
            cs = Rbf(standoffs, disps, smooth=1)
            line_stdoffs = np.linspace(np.min(standoffs), np.max(standoffs), 50)
            line_disps = cs(line_stdoffs)

            cs = Rbf(standoffs, rrs, smooth=1)
            line_rrs = cs(line_stdoffs)

            if "On a hole" in root:
                linestyle = (0, (2, 1, 1, 1))
            elif "Between" in root:
                linestyle = "-"
            else:
                linestyle = "dotted"

            disp_ax.plot(line_stdoffs, line_disps, color=f"C{i}", linestyle=linestyle, zorder=5 - i)
            r_ax.plot(line_stdoffs, line_rrs, color=f"C{i}", linestyle=linestyle, zorder=5 - i)

            tags.append(geom_dir)
            vfs.append(vf)

if len(tags) / 2 > 8:
    legend = disp_ax.legend(frameon=False, fontsize="xx-small", ncol=2)
else:
    legend = disp_ax.legend(frameon=False, fontsize=5, ncol=1,
                            title_fontsize="xx-small",
                            title="$W / \\bar{R_0}$ \\hspace{6pt} $A'$ \\hspace{10pt} $\\phi$")

for handle in legend.legendHandles:
    if type(handle) == PathCollection:
        handle.set_sizes(5 * np.array(handle.get_sizes()))

disp_ax.annotate("Above a hole", (2.05, 0.4), (2, 0.25),
                 arrowprops={'width': 0.8, 'headwidth': 3, "headlength": 3, 'facecolor': 'k', 'edgecolor': 'k'},
                 va="top", ha="center", fontsize="xx-small")
disp_ax.annotate("Between-holes", (2.625, 0.62), (4, 0.8),
                 arrowprops={'width': 0.8, 'headwidth': 3, "headlength": 3, 'facecolor': 'k', 'edgecolor': 'k'},
                 va="bottom", ha="center", fontsize="xx-small")

disp_ax.loglog()
disp_ax.set_xlabel("$\\gamma = Y / R_0$")
disp_ax.set_ylabel("$\\Delta / R_0$")
label_subplot(disp_ax, "($a$)")

format_axis_ticks_decimal(disp_ax.xaxis, 0)

r_ax.set_xscale('log')
r_ax.set_xlabel("$\\gamma = Y / R_0$")
r_ax.set_ylabel("$R_1 / R_0$")
label_subplot(r_ax, "($b$)")
format_axis_ticks_decimal(r_ax.xaxis, 0)

print(total_readings)

plt.tight_layout()

for vf, tag in sorted(zip(vfs, tags)):
    print(f"{tag:10s} - {100 * vf:4.1f} %")
plt.show()
