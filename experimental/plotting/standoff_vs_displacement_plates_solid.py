import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt, fig_width
import numpy as np
import matplotlib.pyplot as plt

dirs = [
    ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/",
     "Experiment", "C4", "s"],

    # ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
    #  "1 mm acrylic 50 mm solid plate", "C6", "s"],
]

initialize_plt(font_size=10, line_scale=1)
disp_fig = plt.figure(figsize=(fig_width(), fig_width() * 0.75))

size_ratio_fig = plt.figure()

all_standoffs = []
all_radius_ratios = []

for i, (dir_path, label, colour, marker) in enumerate(dirs):
    sys.path.append(dir_path)
    print(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    y_offset = params.upper_surface_y

    try:
        au.flag_invalid_readings(dir_path)
    except ValueError:
        print(f"Skipping flagging invalid readings for {label}")
    readings = au.load_readings(dir_path + "readings_dump.csv", include_invalid=False)

    norm_disps = []
    standoffs = []
    ys = []
    theta_js = []
    radius_ratios = []
    for j, reading in enumerate(readings):
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        y = pos[1] - y_offset

        radius_ratio = np.sqrt(reading.sec_max_area / reading.max_bubble_area)

        if y >= 0:
            ys.append(y)
            radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
            displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px
            theta_js.append(reading.get_jet_angle())
            standoff = y / radius
            standoffs.append(standoff)
            norm_disps.append(displacement / radius)
            radius_ratios.append(radius_ratio)

            # size_ratio_fig.gca().text(standoff, radius_ratio, f"{reading.idx}:{reading.repeat_number}")

    filt_standoffs, filt_norm_disps = zip(*[(s, n) for s, n in zip(standoffs, norm_disps) if 5.5 > s > 2])

    size_ratio_fig.gca().scatter(standoffs, radius_ratios, label=label, color=colour, marker=marker, alpha=0.5)

    disp_fig.gca().scatter(standoffs, norm_disps, label=label, color=colour, marker=marker, alpha=0.5)
    (a, b) = np.polyfit(np.log10(filt_standoffs), np.log10(filt_norm_disps), 1)
    print(a, b)
    disp_fig.gca().plot(np.linspace(2, 5.5, 2), 10 ** b * np.linspace(2, 5.5, 2) ** a, color=colour,
                        label="$\\Delta / R = %0.3f \\gamma^{%0.3f}$" % (10 ** b, a))

disp_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
disp_fig.gca().legend(frameon=False)
disp_fig.tight_layout()

size_ratio_fig.gca().set_xlabel("$\\gamma = \\frac{Y}{R}$")
size_ratio_fig.gca().set_ylabel("$R_{max2} / R_{max}$")

x_range = size_ratio_fig.gca().get_xlim()
gammas = np.linspace(x_range[0], x_range[1], 50)
size_ratio_fig.gca().plot(gammas, np.power(0.1 * np.log(0.195 * np.power(gammas, -2)) + 0.7, 1 / 3),
                          label="Supponen et al. (2018)")
size_ratio_fig.gca().set_xlim(x_range)
size_ratio_fig.gca().legend(frameon=False)
size_ratio_fig.tight_layout()

plt.show()
