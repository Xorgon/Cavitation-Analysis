import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from util.file_utils import csv_to_lists
from util.plotting_utils import initialize_plt

tags = ['W1H3', 'W2H3a']

initialize_plt()
fig, axes = plt.subplots(2, 1, sharex='all', figsize=(5.31445, 4))
legend_handles = []

colors = ['r', 'g', 'm', 'orange', 'b']
for i in range(len(tags)):
    num_ys, num_theta_stars, num_x_stars = csv_to_lists("figure15_data/", f"{tags[i]}_numerical", has_headers=True)
    exp_ys, exp_theta_stars, theta_errs, exp_x_stars, x_star_errs = csv_to_lists("figure15_data/",
                                                                                 f"{tags[i]}_experimental",
                                                                                 has_headers=True)
    color = f'C{i}'
    legend_handles.append(mpatches.Patch(color=color, label=tags[i], linewidth=0.1, capstyle='butt'))
    axes[0].plot(num_ys, num_theta_stars, color, label="Numerical")
    axes[0].errorbar(exp_ys, exp_theta_stars, yerr=theta_errs, color=color, fmt='.',
                     label="Experimental")
    axes[0].set_ylabel("$\\theta^\\star$ (rad)")
    axes[1].plot(num_ys, num_x_stars, color, label="Numerical")
    axes[1].errorbar(exp_ys, exp_x_stars, yerr=x_star_errs, color=color, fmt='.', linestyle=" ",
                     label="Experimental")
    axes[1].set_ylabel("$x^\\star$")
    axes[-1].set_xlabel("$y$")

legend_handles.append(mlines.Line2D([], [], color='k', marker='.', label='Experimental', linestyle=' ',
                                    markersize=5))
legend_handles.append(mlines.Line2D([], [], color='k', label='Numerical'))
axes[0].legend(handles=legend_handles, frameon=False)
axes[0].annotate("($a$)", xy=(np.mean(axes[0].get_xlim()), np.mean(axes[0].get_ylim())), xytext=(0.025, 0.95),
                 textcoords='axes fraction', horizontalalignment='left', verticalalignment='top')
axes[1].annotate("($b$)", xy=(np.mean(axes[1].get_xlim()), np.mean(axes[1].get_ylim())), xytext=(0.025, 0.95),
                 textcoords='axes fraction', horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.show()
