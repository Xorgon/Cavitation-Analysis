import os
import sys
import importlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, Rbf
from experimental.util.analysis_utils import load_readings, Reading
from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal, label_subplot
from common.util.drawing_utils import draw_plate
import numpy as np

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

geoms = os.listdir(root_dir)

initialize_plt()
fig, (disp_ax, r_ax) = plt.subplots(1, 2, figsize=(5, 2.5))


def filter_by(reading: Reading):
    return reading.ecc_at_max < 0.22


def geom_filter(g: str):
    include_filter = [
        "w12vf32circles",
        "solid",
    ]
    exclude_filter = [
    ]
    return (any([f in g for f in include_filter]) or len(include_filter) == 0) \
           and all([f not in g for f in exclude_filter])


tags = []
vfs = []
ws = []

total_readings = 0

trace = False

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

            if hasattr(params, "vf"):
                vf = params.vf
            else:
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

            mean_radius = np.mean([r.get_max_radius(params.mm_per_px) for r in readings if filter_by(r)])
            disp_ax.scatter(standoffs, disps, color=f"C{i}", marker=marker, s=2,
                            label="$W / \\bar{R_0}" + f" = {W / mean_radius:.2f}$,  $\\phi = {vf * 100:.1f}$ %"
                            if not labelled else None)

            r_ax.scatter(standoffs, rrs, color=f"C{i}", marker=marker, s=2,
                         label="$W / \\bar{R_0}" + f" = {W / mean_radius:.2f}$,  $\\phi = {vf * 100:.1f}$ %"
                         if not labelled else None)
            labelled = True

            if trace:
                standoffs, disps = zip(*sorted(zip(standoffs, disps)))
                cs = Rbf(standoffs, disps, smooth=1)
                line_stdoffs = np.linspace(np.min(standoffs), np.max(standoffs), 50)
                line_disps = cs(line_stdoffs)
                disp_ax.plot(line_stdoffs, line_disps, color=f"C{i}")

            tags.append(geom_dir)
            vfs.append(vf)
            ws.append(W / mean_radius)

# if len(tags) / 2 > 8:
#     plt.legend(frameon=False, fontsize="xx-small", ncol=2)
# else:
#     plt.legend(frameon=False, fontsize="xx-small", ncol=1)
disp_ax.loglog()
disp_ax.set_xlabel("$\\gamma = Y / R_0$")
disp_ax.set_ylabel("$\\Delta / R_0$")
disp_ax.set_xlim((1.3, disp_ax.get_xlim()[1]))

r_ax.set_xscale('log')
r_ax.set_xlabel("$\\gamma = Y / R_0$")
r_ax.set_ylabel("$R_1 / R_0$")
r_ax.set_xlim((1.3, r_ax.get_xlim()[1]))
r_ax.set_ylim((r_ax.get_ylim()[0], 0.8))


print(total_readings)

iax = disp_ax.inset_axes([0.7, 0.7, 0.275, 0.275])
draw_plate(iax, vfs[0], ws[0], label="Solid plate", color="C0", linewidth=3)

iax = disp_ax.inset_axes([0.075, 0.2, 0.275, 0.275])
draw_plate(iax, vfs[1], ws[1], label="$W / \\bar{R_0}" + f" = {ws[1]:.2f}$\n$\\phi = {vfs[1] * 100:.1f}$ \\%",
           hole_scale=5, color="C1", linewidth=3)

iax = r_ax.inset_axes([0.7, 0.7, 0.275, 0.275])
draw_plate(iax, vfs[0], ws[0], color="C0", linewidth=3)

iax = r_ax.inset_axes([0.075, 0.075, 0.275, 0.275])
draw_plate(iax, vfs[1], ws[1], hole_scale=5, color="C1", linewidth=3)

format_axis_ticks_decimal(disp_ax.xaxis, 0)
format_axis_ticks_decimal(r_ax.xaxis, 0)

plt.tight_layout()

for vf, tag in sorted(zip(vfs, tags)):
    print(f"{tag:10s} - {100 * vf:4.1f} %")

# plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
#             "paper figures/simple_standoff_vs_displacement.eps", dpi=300)
plt.show()
