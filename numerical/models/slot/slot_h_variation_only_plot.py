import os

import matplotlib.pyplot as plt
import math
import numpy as np

import numerical.bem as bem
import numerical.util.gen_utils as gen
from common.util.plotting_utils import initialize_plt
from common.util.file_utils import csv_to_lists
import matplotlib.patches as patches

initialize_plt()

Hs = np.linspace(1, 5, 5)
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Y = 2

normalize = True

m_0 = 1
n = 20000

fig_width = 5
fig = plt.figure(figsize=(fig_width, 2 * fig_width / 4))

fig.patch.set_facecolor('white')
ax = fig.gca()
xs = Xs / (0.5 * W)

for i, H in enumerate(Hs):
    print(f"Reading H = {H}")
    f_dir = "../model_outputs/slot_h_collapse/"
    f_name = f"h_collapse_n{n}_H{H}.csv"
    xs, thetas = csv_to_lists(f_dir, f_name, True)

    theta_star, x_star = sorted(zip(thetas, xs), key=lambda k: k[0])[-1]
    norm_x_to_plot = np.divide(xs, x_star)
    norm_theta_to_plot = np.divide(thetas, theta_star)

    label_frac = "h"
    ax.plot(xs, thetas, label=f"${label_frac} = {H / W:.2f}$")

label_pad = 0
ax.set_xlabel("$x$", labelpad=label_pad)
ax.set_ylabel("$\\theta$ (rad)", labelpad=label_pad)
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
ax.legend(shadow=False, frameon=False, loc='lower right', labelspacing=0.1)
plt.annotate(f"$y = 1$", xy=(0, 0), xytext=(0.025, 0.97),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.show()
