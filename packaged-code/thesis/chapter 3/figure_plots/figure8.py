import matplotlib.pyplot as plt
import numpy as np

from util.file_utils import csv_to_lists
from util.plotting_utils import initialize_plt

initialize_plt()

Hs = np.linspace(1, 5, 5)
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Y = 2

m_0 = 1
n = 20000

fig_width = 5.31445
fig = plt.figure(figsize=(fig_width, fig_width / 2))

left_pos = 0.11
right_pos = 0.98
wspace = 0.3
plt.subplots_adjust(top=0.95, bottom=0.3, left=left_pos, right=right_pos, wspace=wspace)
fig.patch.set_facecolor('white')
ax = fig.add_subplot(121)
ax.locator_params(nbins=6)
norm_ax = fig.add_subplot(122)
norm_ax.locator_params(nbins=6)
norm_ax.set_xlim(-5.5, 5.5)
norm_ax.set_ylim(-1.1, 1.1)
xs = Xs / (0.5 * W)

for i, H in enumerate(Hs):
    print(f"Reading H = {H}")
    f_dir = "figure8_data/"
    f_name = f"h_collapse_n{n}_H{H}.csv"
    xs, thetas = csv_to_lists(f_dir, f_name, True)

    theta_star, x_star = sorted(zip(thetas, xs), key=lambda k: k[0])[-1]
    norm_x_to_plot = np.divide(xs, x_star)
    norm_theta_to_plot = np.divide(thetas, theta_star)

    label_frac = "h"
    if i == 0:
        ax.plot(xs, thetas, label=f"${label_frac} = {H / W:.2f}$")
        norm_ax.plot(norm_x_to_plot, norm_theta_to_plot, label=f"${label_frac} = {H / W:.2f}$")
    else:
        ax.plot(xs, thetas, label=f"${label_frac} = {H / W:.2f}$", linestyle="--",
                dashes=(i, 2 * i))
        norm_ax.plot(norm_x_to_plot, norm_theta_to_plot, label=f"${label_frac} = {H / W:.2f}$", linestyle="--",
                     dashes=(i, 2 * i))

label_pad = 0
norm_ax.set_xlabel("$\\hat{x}$", labelpad=label_pad)
norm_ax.set_ylabel("$\\hat{\\theta}$", labelpad=label_pad)
ax.set_xlabel("$x$", labelpad=label_pad)
ax.set_ylabel("$\\theta$ (rad)", labelpad=label_pad)
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')

ax.legend(bbox_to_anchor=(0, -0.34, 2.3, .05), loc=10, ncol=len(Hs), mode="expand",
          borderaxespad=0,
          fancybox=False, edgecolor='k', shadow=False, handlelength=1.5, handletextpad=0.5)
ax.annotate('($a$)', xy=(0, 0), xytext=(0.05, 0.95), textcoords='axes fraction', horizontalalignment='left',
            verticalalignment='top')
norm_ax.annotate('($b$)', xy=(0, 0), xytext=(0.05, 0.95), textcoords='axes fraction',
                 horizontalalignment='left', verticalalignment='top')
plt.show()
