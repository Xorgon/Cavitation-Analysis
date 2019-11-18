import numpy as np
import matplotlib.pyplot as plt

from common.util.plotting_utils import initialize_plt
from numerical.models.slot_opt import find_slot_peak

h = 2
w = 2
qs = np.linspace(1, 5, 10)
n = 20000

theta_stars = []
p_bar_stars = []

for q in qs:
    _, p, theta_j, _ = find_slot_peak(w, q, h, n, varied_slot_density_ratio=0.25, density_w_thresh=12)
    theta_stars.append(theta_j)
    p_bar_stars.append(2 * p / w)

initialize_plt(18)

fig = plt.figure()
ax = fig.gca()
ax.plot(qs / w, theta_stars, "k")
ax.set_xlabel("$q / w$")
ax.set_ylabel("$\\theta_j^\\star$")
fig.tight_layout()

fig = plt.figure()
ax = fig.gca()
ax.plot(qs / w, p_bar_stars, "k")
ax.set_xlabel("$q / w$")
ax.set_ylabel("$\\bar{p}^\\star$")
fig.tight_layout()

plt.show()
