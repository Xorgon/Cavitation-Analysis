import os

import matplotlib.pyplot as plt
import math
import numpy as np

import numerical.bem as bem
import numerical.util.gen_utils as gen
import common.util.plotting_utils as pu
import matplotlib.patches as patches

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 10, 'serif': ['cmr10']}
plt.rc('font', **font)
plt.rc('lines', linewidth=1, markersize=3)
plt.rc('axes', linewidth=0.5)
plt.rc('patch', linewidth=0.5)

Hs = np.linspace(1, 5, 5)
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Y = 2

normalize = True

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
    print(f"Testing H = {H}")
    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=12, density_ratio=0.25)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    print(np.mean(centroids, 0))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = np.linalg.inv(R_matrix)

    points = np.empty((len(Xs), 3))
    points[:, 0] = Xs
    points[:, 1] = Y
    points[:, 2] = 0
    vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv, verbose=True)
    theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

    x_to_plot = xs
    theta_j_to_plot = theta_js

    theta_j_star, x_star = sorted(zip(theta_js, xs), key=lambda k: k[0])[-1]
    norm_x_to_plot = np.divide(xs, x_star)
    norm_theta_j_to_plot = np.divide(theta_js, theta_j_star)

    label_frac = "h"
    if i == 0:
        ax.plot(x_to_plot, theta_j_to_plot, label=f"${label_frac} = {H / W:.2f}$")
        norm_ax.plot(norm_x_to_plot, norm_theta_j_to_plot, label=f"${label_frac} = {H / W:.2f}$")
    else:
        ax.plot(x_to_plot, theta_j_to_plot, label=f"${label_frac} = {H / W:.2f}$", linestyle="--",
                dashes=(i, 2 * i))
        norm_ax.plot(norm_x_to_plot, norm_theta_j_to_plot, label=f"${label_frac} = {H / W:.2f}$", linestyle="--",
                     dashes=(i, 2 * i))

# TODO: Convert to data generator + data plotter

label_pad = 0
norm_ax.set_xlabel("$\\hat{x}$", labelpad=label_pad)
norm_ax.set_ylabel("$\\hat{\\theta}$", labelpad=label_pad)
ax.set_xlabel("$x$", labelpad=label_pad)
ax.set_ylabel("$\\theta_j$", labelpad=label_pad)
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
# ax.legend()

ax.legend(bbox_to_anchor=(0, -0.34, 2.3, .05), loc=10, ncol=len(Hs), mode="expand",
          borderaxespad=0,
          fancybox=False, edgecolor='k', shadow=False, handlelength=1.5, handletextpad=0.5)
ax.annotate('($a$)', xy=(0, 0), xytext=(0.115, 0.89), textcoords='figure fraction', horizontalalignment='left',
            verticalalignment='bottom')
norm_ax.annotate('($b$)', xy=(0, 0), xytext=(0.61, 0.89), textcoords='figure fraction',
                 horizontalalignment='left', verticalalignment='bottom')
# ymin, ymax = ax.get_ylim()
# ax.set_yticks(np.round(np.linspace(ymin, ymax, 5), 2))
#
# xmin, xmax = ax.get_xlim()
# ax.set_xticks(np.round(np.linspace(xmin, xmax, 5), 2))
#
# ymin, ymax = norm_ax.get_ylim()
# norm_ax.set_yticks(np.round(np.linspace(ymin, ymax, 5), 2))
#
# xmin, xmax = norm_ax.get_xlim()
# norm_ax.set_xticks(np.round(np.linspace(xmin, xmax, 5), 2))

plt.savefig('model_outputs/h_sweep_plot.pdf')
plt.show()
