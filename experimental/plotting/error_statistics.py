import math
from typing import List
import sys

import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import fractions
import numpy as np

import common.util.plotting_utils as plt_util
import experimental.util.analysis_utils as au
import common.util.file_utils as file
from experimental.plotting.analyse_slot import SweepData

plt_util.initialize_plt()

# reading_ys = [1.0, 2.0]
# dir_path = "../../../../Data/SlotSweeps/W2H3b/"

# reading_ys = [5.0, 5.5, 6.0, 6.5, 7.0]
# dir_path = "../../../../Data/SlotSweeps/W2H3a/"

dir_path = file.select_dir("../../../../Data/SlotSweeps")

sys.path.append(dir_path)
import params

x_offset = params.left_slot_wall_x + params.slot_width / 2
y_offset = params.upper_surface_y

readings = au.load_readings(dir_path + "readings_dump.csv")
readings = sorted(readings, key=lambda r: r.m_x)  # type: List[au.Reading]

reading_ys = sorted(set([reading.m_y for reading in readings]))

fig_width = 5.31445  # From LaTeX
fig, axes = plt.subplots(1, len(reading_ys), sharex="all", sharey="all", figsize=(fig_width, fig_width / 2))
plt.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.98, wspace=0.05)

x_ranges = []
std_ranges = []
alph = "abcdefg"
sweeps = []
for k, reading_y in enumerate(reading_ys):
    sweep = SweepData(params.title, reading_y, params.slot_width, params.slot_height)
    # Post-process data to get jet angles.
    res_dict = {}
    for reading in readings:
        if reading.m_y != reading_y:
            continue
        theta_j = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
        if reading.m_x in res_dict.keys():
            res_dict[reading.m_x].append(theta_j)
        else:
            res_dict[reading.m_x] = [theta_j]

    xs = []
    means = []
    stds = []
    for x in res_dict.keys():
        # plt.hist(res_dict[x], 20)
        # plt.show()
        xs.append(x)
        means.append(np.mean(res_dict[x]))
        stds.append(np.std(res_dict[x]))

    adjusted_theta_js = []
    for i in range(len(xs)):
        for theta_j in res_dict[xs[i]]:
            adjusted_theta_js.append(theta_j - means[i])

    mu = 0
    variance = 1
    std = float(np.mean(stds))
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)

    print(f"m_y = {reading_y}, Mean = {mu}, Standard deviation = {std}")

    ax = axes[k]
    ax.hist(adjusted_theta_js, 100, density=True, histtype='stepfilled', color="gray")
    ax.hist(adjusted_theta_js, 100, density=True, histtype='step', color="k", label="Data")
    ax.plot(x, stats.norm.pdf(x, mu, std), "k--", label="Normal")
    if k == 0:
        ax.set_ylabel("Probability density", fontsize=10)
        # ax.set_xticklabels(rotation=45)
    if k == len(axes) - 1:
        ax.legend(fancybox=False, edgecolor='k', shadow=False)
    ax.set_xlabel("$\\theta_j$ deviation from mean (rads)", fontsize=10)
    ax.set_xlim([-0.075, 0.075])
    x_ranges.append(xs)
    std_ranges.append(stds)
    ax.annotate(f'$({alph[k]})$', xy=(0, 0), xytext=(0.02, 0.98), textcoords='axes fraction',
                horizontalalignment='left',
                verticalalignment='top')
plt.show()
# for i in range(len(x_ranges)):
#     plt.plot(x_ranges[i], std_ranges[i], label="y={0:.2f}".format(reading_ys[i]))
# plt.legend()
# plt.show()
