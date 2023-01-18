import matplotlib.pyplot as plt
import numpy as np

from common.util.file_utils import csv_to_lists
from common.util.plotting_utils import initialize_plt

initialize_plt()

H = 2
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Ys = np.linspace(1, 5, 5)

m_0 = 1
n = 20000

fig_width = 5
fig = plt.figure(figsize=(fig_width, 2 * fig_width / 4))

fig.patch.set_facecolor('white')
ax = fig.gca()
xs = Xs / (0.5 * W)

for i, Y in enumerate(Ys):
    print(f"Reading Y = {Y}")
    f_dir = "../model_outputs/slot_y_collapse/"
    f_name = f"y_collapse_n{n}_Y{Y}.csv"
    xs, thetas = csv_to_lists(f_dir, f_name, True)

    theta_star, x_star = sorted(zip(thetas, xs), key=lambda k: k[0])[-1]
    norm_x_to_plot = np.divide(xs, x_star)
    norm_theta_to_plot = np.divide(thetas, theta_star)

    label_frac = "y"
    ax.plot(xs, thetas, label=f"${label_frac} = {Y / W:.2f}$")

label_pad = 0
ax.set_xlabel("$x$", labelpad=label_pad)
ax.set_ylabel("$\\theta$ (rad)", labelpad=label_pad)
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
ax.legend(shadow=False, frameon=False, loc='lower right', labelspacing=0.1)
plt.annotate(f"$h = 1$", xy=(0, 0), xytext=(0.025, 0.97),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')
plt.tight_layout()
plt.show()
