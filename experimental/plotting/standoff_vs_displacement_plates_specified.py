import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
import numpy as np
from scipy.optimize import curve_fit

dirs = [
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/Between 3 holes/",
    #     "1mm acrylic 50mm plate, between 3 holes, 52.5% VF", "C0", "^"],
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/On hole/",
    #     "1mm acrylic 50mm plate, above a hole, 52.5% VF", "C0", "o"],
    #
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/Between 2 holes/",
    #     "1mm acrylic 100mm plate, between 2 holes, 39% VF", "C1", "_"],
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/Between 3 holes/",
    #     "1mm acrylic 100mm plate, between 3 holes, 39% VF", "C1", "^"],
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2ishmm Hole 1mm Acrylic Plate/On hole/",
    #     "1mm acrylic 100mm plate, above a hole, 39% VF", "C1", "o"],
    #
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2mm Hole Steel Plate/Between 3 holes/",
    #     "1mm steel 100mm plate, between 3 holes, 47% VF", "C2", "^"],
    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2mm Hole Steel Plate/On hole/",
    #     "1mm steel 100mm plate, above a hole, 47% VF", "C2", "o"],
    #
    # ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Porous plate/",
    #  "3mm acrylic 100mm quad plate, between 4 holes, 12% VF", "C3", "x"],
    #
    ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/",
     "3mm acrylic 100mm solid plate", "C4", "s"],

    ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/",
     "1mm acrylic 50mm solid plate", "C6", "s"],

    ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/",
     "1mm acrylic 50mm plate, between 3 holes, 24% VF", "black", "^"],  # TODO: Change this back to C5

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Tap water between 3 2.0/",
        "Tap water, 24% VF, repeated", "black", "^"],

    ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/On hole/",
     "1mm acrylic 50mm plate, above a hole, 24% VF", "purple", "o"],

    # ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Varying wait time/",
    #  "1mm acrylic 50mm plate, period variation between 3 holes, 24% VF", "C5", "^"],

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3/",
        "Purer water, 24% VF", "green", "^"],

    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 next day/",
    #     "Purer water, 24% VF, next day", "blue", "^"],

    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 degassed again/",
    #     "Purer water, 24% VF, degassed again", "purple", "^"],

    # [
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 degassed again lower power/",
    #     "Purer water, 24% VF, degassed again, lower power", "black", "^"],

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 2.0/",
        "Purer water, 24% VF, repeated", "green", "^"],

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF 3mm acrylic/Between 3 holes/",
        "3 mm acrylic, between 3 holes", "blue", "^"],
    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF 3mm acrylic/On a hole/",
        "3 mm acrylic, on a hole", "cyan", "o"],
    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 1/",
        "1 mm steel, on a hole", "red", "o"],

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF steel/On a hole 2/",
        "1 mm steel, on a hole", "darkred", "o"],

    [
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Initial Triangle Plate/",
        "Triangles", "pink", "o"],
]

show_fits = True
show_text = True
# fit_range = [1, 5]
fit_range = [1.5, 8]

import matplotlib.pyplot as plt

# initialize_plt(font_size=14, line_scale=2)
disp_fig = plt.figure()

size_ratio_fig = plt.figure()

sd_fig = plt.figure()

ecc_fig = plt.figure()

hist_fig, hist_axes = plt.subplots(len(dirs), 1, sharex="all")

all_standoffs = []
all_radius_ratios = []

total_bubbles = 0
invalid_bubbles = 0

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
    radii = []
    eccs = []
    for j, reading in enumerate(readings):
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        y = pos[1] - y_offset

        radius_ratio = np.sqrt(reading.sec_max_area / reading.max_bubble_area)

        if y >= 0 and reading.ecc_at_max is not None and reading.ecc_at_max < 0.25:
            ys.append(y)
            radius = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px
            radii.append(radius)
            displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px
            theta_js.append(reading.get_jet_angle())
            standoff = y / radius
            standoffs.append(standoff)
            norm_disps.append(displacement / radius)
            radius_ratios.append(radius_ratio)
            eccs.append(reading.ecc_at_max)

            total_bubbles += 1

            if show_text:
                size_ratio_fig.gca().text(standoff, radius_ratio, f"{reading.idx}:{reading.repeat_number}",
                                          color=colour)
                disp_fig.gca().text(standoff, displacement / radius, f"{reading.idx}:{reading.repeat_number}",
                                    color=colour)
                ecc_fig.gca().text(reading.ecc_at_max, 0, f"{reading.idx}:{reading.repeat_number}", color=colour)
        else:
            invalid_bubbles += 1

    filt_standoffs, filt_norm_disps, filt_radii, filt_radius_ratios, filt_eccs = zip(
        *[(s, n, r, rr, e) for s, n, r, rr, e in zip(standoffs, norm_disps, radii, radius_ratios, eccs)
          if fit_range[1] > s > fit_range[0]])

    size_ratio_fig.gca().scatter(standoffs, radius_ratios, label=label, color=colour, marker=marker, alpha=0.5)

    print(f"Mean radius for {label} = {np.mean(radii):.2f} mm")
    print(f"Mean eccentricity for {label} = {np.mean(eccs):.2f}")

    disp_fig.gca().scatter(standoffs, norm_disps, s=np.array(radii) ** 2 * 25, label=label, color=colour, marker=marker,
                           alpha=0.5)
    (a, b), residuals, _, _, _ = np.polyfit(np.log10(filt_standoffs), np.log10(filt_norm_disps), 1, full=True)

    (c, d), _ = curve_fit(lambda x, _c, _d: (10 ** _d) * (x ** _c), filt_standoffs, filt_norm_disps)
    print(c, d)
    print(f"a = {a:.5f}, b = {b:.5f}")
    print(f"c = {c:.5f}, d = {d:.5f}")

    # print(f"Variances = {np.diag(V)}")
    hist_axes[i].hist(filt_norm_disps - (10 ** b) * (filt_standoffs ** a), label=label, color=colour)
    hist_axes[i].legend()
    hist_axes[i].axvline(0, c="k", linestyle="--")
    # print(np.sqrt(np.mean((np.log10(filt_norm_disps) - (a * np.log10(filt_standoffs) + b)) ** 2)))
    # TODO: Adjust how RMSD is calculated to a) be the RMSD and b) relate it to the physical values rather than the logs
    print(f"RMSD = {np.sqrt(np.mean(residuals) / len(filt_standoffs))}")

    (rr_a, rr_b), residuals, _, _, _ = np.polyfit(filt_standoffs, filt_radius_ratios, 1, full=True)

    sd_fig.gca().scatter(np.array(filt_norm_disps) - (10 ** b) * (np.array(filt_standoffs) ** a),
                         np.array(filt_radius_ratios) - (rr_a * np.array(filt_standoffs) + rr_b),
                         label=label, color=colour)
    ecc_fig.gca().scatter(filt_eccs, np.array(filt_norm_disps) - (10 ** b) * (np.array(filt_standoffs) ** a),
                          label=label, color=colour)

    if show_fits:
        disp_fig.gca().plot(np.linspace(fit_range[0], fit_range[1], 2),
                            10 ** (a * np.log10(np.linspace(fit_range[0], fit_range[1], 2)) + b), color=colour,
                            marker=marker, markersize=5)
        # disp_fig.gca().plot(np.linspace(fit_range[0], fit_range[1], 2),
        #                     10 ** (c * np.log10(np.linspace(fit_range[0], fit_range[1], 2)) + d), color=colour,
        #                     marker=marker, markersize=5, linestyle="--")

        size_ratio_fig.gca().plot(np.linspace(fit_range[0], fit_range[1], 2),
                                  rr_a * np.linspace(fit_range[0], fit_range[1], 2) + rr_b, color=colour,
                                  marker=marker, markersize=5)

print(total_bubbles, invalid_bubbles)

disp_fig.gca().set_xlabel("$\\gamma = \\frac{y}{r}$")
disp_fig.gca().set_ylabel("$\\Delta / R$ = displacement / radius")
disp_fig.gca().set_xscale('log')
disp_fig.gca().set_yscale('log')

# disp_fig.gca().axhline(1, linestyle="--", color="grey")
disp_fig.gca().legend(frameon=False)
disp_fig.tight_layout()

size_ratio_fig.gca().set_xlabel("$\\gamma = \\frac{y}{r}$")
size_ratio_fig.gca().set_ylabel("$R_{max2} / R_{max}$")

x_range = size_ratio_fig.gca().get_xlim()
gammas = np.linspace(x_range[0], x_range[1], 50)
# size_ratio_fig.gca().plot(gammas, np.power(0.1 * np.log(0.195 * np.power(gammas, -2)) + 0.7, 1 / 3),
#                           label="Supponen et al. (2018)")
size_ratio_fig.gca().set_xlim(x_range)
size_ratio_fig.gca().legend(frameon=False)
size_ratio_fig.tight_layout()

hist_axes[-1].set_xlabel("Difference from curve fit")
for a in hist_axes:
    a.set_ylabel("Data points")
hist_fig.tight_layout()

sd_fig.gca().set_xlabel("Displacement difference from curve fit")
sd_fig.gca().set_ylabel("Radius ratio difference from curve fit")
sd_fig.gca().legend(frameon=False)

ecc_fig.gca().set_xlabel("Bubble eccentricity at maximum size")
ecc_fig.gca().set_ylabel("Difference from curve fit")
ecc_fig.gca().legend(frameon=False)

plt.show()
