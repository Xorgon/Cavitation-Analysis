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

window_title = "Varied Slot 32 bit"

hs = np.linspace(1, 3, 10)
w = 2
ps = np.linspace(-3 * w, 3 * w, 300)
q = 2

normalize = True

m_0 = 1
n = 20000

if not os.path.exists("slot_ms/{0}".format(n)):
    os.makedirs("slot_ms/{0}".format(n))

fig_width = 5.31445
fig = plt.figure(figsize=(fig_width, fig_width / 2), num=window_title)

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
p_bars = ps / (0.5 * w)

for i, h in enumerate(hs):
    centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50, w_thresh=12, density_ratio=0.25)
    # centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    # pu.plot_3d_point_sets([centroids])
    print(np.mean(centroids, 0))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = np.linalg.inv(R_matrix)

    print(f"Testing h = {h}")
    theta_js = []
    for p in ps:
        res_vel, sigma = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv)
        # pu.plot_flow([p, q, 0], cs, ns, sinks, R=R_matrix)
        theta_js.append(math.atan2(res_vel[1], res_vel[0]) + math.pi / 2)

    p_bar_to_plot = p_bars
    theta_j_to_plot = theta_js

    max_theta_j, p_bar_max_theta_j = sorted(zip(theta_js, p_bars), key=lambda k: k[0])[-1]
    norm_p_bar_to_plot = np.divide(p_bars, p_bar_max_theta_j)
    norm_theta_j_to_plot = np.divide(theta_js, max_theta_j)

    label_frac = "\\frac{h}{w}"
    if i == 0:
        ax.plot(p_bar_to_plot, theta_j_to_plot, label=f"${label_frac} = {h / w:.2f}$")
        norm_ax.plot(norm_p_bar_to_plot, norm_theta_j_to_plot, label=f"${label_frac} = {h / w:.2f}$")
    else:
        ax.plot(p_bar_to_plot, theta_j_to_plot, label=f"${label_frac} = {h / w:.2f}$", linestyle="--",
                dashes=(i, 2 * i))
        norm_ax.plot(norm_p_bar_to_plot, norm_theta_j_to_plot, label=f"${label_frac} = {h / w:.2f}$", linestyle="--",
                     dashes=(i, 2 * i))

label_pad = 0
norm_ax.set_xlabel("$P$", labelpad=label_pad)
norm_ax.set_ylabel("$\\hat{\\theta}$", labelpad=label_pad)
ax.set_xlabel("$\\bar{p}$", labelpad=label_pad)
ax.set_ylabel("$\\theta_j$", labelpad=label_pad)
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
# ax.legend()

ax.legend(bbox_to_anchor=(0, -0.34, 2.3, .05), loc=10, ncol=len(hs), mode="expand",
          borderaxespad=0,
          fancybox=False, edgecolor='k', shadow=False, handlelength=1.5, handletextpad=0.5)
ax.annotate('$a)$', xy=(0, 0), xytext=(0.115, 0.89), textcoords='figure fraction', horizontalalignment='left',
            verticalalignment='bottom')
norm_ax.annotate('$b)$', xy=(0, 0), xytext=(0.61, 0.89), textcoords='figure fraction',
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

plt.savefig('h_sweep_plot.pdf')
plt.show()
