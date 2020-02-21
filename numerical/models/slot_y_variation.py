import matplotlib.pyplot as plt
import numpy as np

from common.util.plotting_utils import initialize_plt
from numerical.models.slot_opt import find_slot_peak

H = 2
W = 2
Ys = np.linspace(1, 10, 10)
n = 5000

theta_stars = []
x_stars = []

for Y in Ys:
    _, X, theta_j, _ = find_slot_peak(W, Y, H, n, varied_slot_density_ratio=0.25, density_w_thresh=12)
    theta_stars.append(theta_j)
    x_stars.append(2 * X / W)

initialize_plt(font_size=18, line_scale=2)

fig = plt.figure()
ax = fig.gca()
# ax.loglog(qs / w, theta_stars, "k")
ax.plot(Ys / W, theta_stars, "k")
ax.set_xlabel("$y$")
ax.set_ylabel("$\\theta_j^\\star$")
fig.tight_layout()
fig.savefig('model_outputs/theta_j_star_y_var.svg')

fig = plt.figure()
ax = fig.gca()
ax.plot(Ys / W, x_stars, "k")
ax.set_xlabel("$y$")
ax.set_ylabel("$x^\\star$")
fig.tight_layout()
fig.savefig('model_outputs/x_star_y_var.svg')

plt.show()
