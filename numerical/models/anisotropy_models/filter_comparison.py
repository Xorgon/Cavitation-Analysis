import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit
from scipy.stats import anderson
import sys
import importlib

from common.util.plotting_utils import initialize_plt, format_axis_ticks_decimal
from experimental.util.analysis_utils import load_readings, Reading

# label, max_ecc, alpha, marker, m_size, colour, dirs
sets = [["Slots", 0.20, 1, 'o', 1.5, 'C0',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W1H3/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H12/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3a/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H6/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H9/",
          "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W4H12/"]],

        ["Square", 0.205, 1, 's', 1.5, 'C5',  # 0.205 | 0.234
         ["E:/Data/Lebo/Restructured Data/Square/",
          "E:/Data/Lebo/Restructured Data/Square 2/",
          "E:/Data/Lebo/Restructured Data/Square 3/"]],

        ["Triangle", 0.24, 1, '<', 1.5, 'C4',  # 0.22 | 0.257
         ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
          "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]],

        ["$90^\\circ$ corner", 0.22, 1, 'x', 1.5, 'C2',
         ["E:/Data/Ivo/Restructured Data/90 degree corner/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 2/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 3/",
          "E:/Data/Ivo/Restructured Data/90 degree corner 4/"]],

        ["$60^\\circ$ corner", 0.28, 1, (6, 1, 30), 1.5, 'C1',  # (6, 1, 30) makes a 6-pointed star rotated 30 degrees
         ["E:/Data/Ivo/Restructured Data/60 degree corner/"]],

        ["Flat plate", 0.28, 1, '_', 1.7, 'C3',
         ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]],
        ]

show_data_labels = False
plot_additionals = True

if not show_data_labels:
    initialize_plt()  # Do not want to plot thousands of text labels with this on

x_formatter = StrMethodFormatter("{x:.0f}")
y_formatter = StrMethodFormatter("{x:.1f}")

min_anisotropy = np.inf
max_anisotropy = 0

all_anisotropies = []
all_disps = []
all_readings = []

fig_width = 7

fig, (all_ax, filt_ax) = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.5), sharey=True)

total_readings = 0

# all_eccs = []
# all_pct_diffs = []

for j, (label, max_ecc, alpha, marker, m_size, colour, dirs) in enumerate(sets):
    # max_ecc = 0.18
    # max_ecc = 1
    set_readings = 0
    used_readings = 0
    radii = []
    eccs = []
    radii_mm = []


    def r_filter(r: Reading):
        return r.ecc_at_max < max_ecc and r.model_anisotropy is not None and r.get_angle_dif() < np.deg2rad(100)


    for i, dir_path in enumerate(dirs):  # type: int, str
        if i == 0:
            this_label = label
        else:
            this_label = None
        try:

            sys.path.append(dir_path)
            import params

            importlib.reload(params)
            sys.path.remove(dir_path)

            readings = load_readings(dir_path + "readings_dump.csv", include_invalid=True)
            filtered_anisotropies = [np.linalg.norm(r.model_anisotropy) for r in readings if r_filter(r)]
            filtered_displacements = [np.linalg.norm(r.get_normalised_displacement()) for r in readings if r_filter(r)]
            filtered_radius_ratios = [r.get_radius_ratio() for r in readings if r_filter(r)]
            filtered_max_radii = [np.sqrt(r.max_bubble_area / np.pi) for r in readings if r_filter(r)]
            filtered_eccs = [r.ecc_at_max for r in readings if r_filter(r)]
            filtered_labels = [f"{r.idx}:{r.repeat_number}" for r in readings if r_filter(r)]
            radii.extend(filtered_max_radii)
            radii_mm.extend(np.array(filtered_max_radii) * params.mm_per_px)
            eccs.extend(filtered_eccs)

            all_readings.extend(readings)  # ALL readings

            set_readings += len(readings)
            used_readings += len(filtered_anisotropies)
            total_readings += len(filtered_anisotropies)

            if len(filtered_anisotropies) == 0:
                continue

            if show_data_labels:
                for l, x, y in zip(filtered_labels, filtered_anisotropies, filtered_displacements):
                    all_ax.annotate(f'{l}', xy=[x, y], textcoords='data')

            all_ax.scatter([np.linalg.norm(r.model_anisotropy) for r in readings],
                           [np.linalg.norm(r.get_normalised_displacement()) for r in readings],
                           marker=marker, color=colour, label=this_label, alpha=alpha, s=m_size ** 2)

            filt_ax.scatter(filtered_anisotropies, filtered_displacements, marker=marker, color=colour,
                           label=this_label, alpha=alpha, s=m_size ** 2)

            min_anisotropy = np.min([np.min(filtered_anisotropies), min_anisotropy])
            max_anisotropy = np.max([np.max(filtered_anisotropies), max_anisotropy])

            all_anisotropies.extend(filtered_anisotropies)
            all_disps.extend(filtered_displacements)

        except FileNotFoundError:
            print(f"Not yet processed {dir_path}")

    print(f"{label:20s}: {used_readings:4d} of {set_readings:4d} "
          f"({100 * used_readings / set_readings:2.2f} %) | max_ecc = {max_ecc:0.3f}"
          f" | Mean bubble radius {np.mean(radii):2.2f} px ({np.mean(radii_mm):.2f} mm)"
          f" | Mean eccentricity {np.mean(eccs):.2f}")

print(f"Total readings = {total_readings} ({100 * total_readings / len(all_readings):.2f} %)")

all_ax.legend(loc='lower right', frameon=False, fontsize='x-small', markerfirst=False)

all_ax.set_xlabel("$\\zeta$")
all_ax.set_ylabel("$\\Delta / R_0$")
all_ax.loglog()

filt_ax.set_xlabel("$\\zeta$")
all_ax.set_ylabel("$\\Delta / R_0$")
filt_ax.loglog()

fig.axes[0].annotate(f"($a$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
                     textcoords='axes fraction', color="k",
                     horizontalalignment='left', verticalalignment='top')
fig.axes[1].annotate(f"($b$)", xy=(0.01, 0.5), xytext=(0.025, 0.975),
                     textcoords='axes fraction', color="k",
                     horizontalalignment='left', verticalalignment='top')

plt.tight_layout()


plt.show()
