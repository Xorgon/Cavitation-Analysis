import numpy as np
import matplotlib.pyplot as plt

from common.util.plotting_utils import initialize_plt
from numerical.models.slot_opt import find_slot_peak

hs = np.linspace(1, 10, 10)
w = 2
q = 2
n = 20000

theta_stars = []
p_bar_stars = []

for h in hs:
    _, p, theta_j, _ = find_slot_peak(w, q, h, n, varied_slot_density_ratio=0.25, density_w_thresh=12)
    theta_stars.append(theta_j)
    p_bar_stars.append(2 * p / w)

initialize_plt(font_size=18, line_scale=2)

fig = plt.figure()
ax = fig.gca()
ax.plot(hs / w, theta_stars, "k")
ax.set_xlabel("$h / w$")
ax.set_ylabel("$\\theta_j^\\star$")
fig.tight_layout()

fig = plt.figure()
ax = fig.gca()
ax.plot(hs / w, p_bar_stars, "k")
ax.set_xlabel("$h / w$")
ax.set_ylabel("$\\bar{p}^\\star$")
fig.tight_layout()

plt.show()
