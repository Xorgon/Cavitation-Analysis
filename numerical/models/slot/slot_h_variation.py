import numpy as np
import matplotlib.pyplot as plt

from common.util.plotting_utils import initialize_plt
from numerical.models.slot.slot_opt import find_slot_peak

Hs = np.linspace(1, 10, 10)
W = 2
Y = 2
n = 20000

theta_stars = []
x_stars = []

for H in Hs:
    _, X, theta_j, _ = find_slot_peak(W, Y, H, n, varied_slot_density_ratio=0.25, density_w_thresh=12)
    theta_stars.append(theta_j)
    x_stars.append(2 * X / W)

initialize_plt(font_size=18, line_scale=2)

fig = plt.figure()
ax = fig.gca()
ax.plot(Hs / W, theta_stars, "k")
ax.set_xlabel("$h$")
ax.set_ylabel("$\\theta_j^\\star$ (rad)")
fig.tight_layout()
fig.savefig('../model_outputs/theta_j_star_h_var.svg')

fig = plt.figure()
ax = fig.gca()
# ax.loglog(hs / w, p_bar_stars, "k")
ax.plot(Hs / W, x_stars, "k")
ax.set_xlabel("$h$")
ax.set_ylabel("$x^\\star$")
fig.tight_layout()
fig.savefig('../model_outputs/x_star_h_var.svg')

plt.show()
