import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np

from common.util.file_utils import csv_to_lists
from common.util.plotting_utils import initialize_plt

L = 0.015  # 15 mm
R_init = 1 / 1000  # 1 mm
img_ranges = 32
n = 20000
x, y, z = csv_to_lists(f'../../model_outputs/square_anisotropy_data/',
                       f'R{1000 * R_init:.1f}_L{1000 * L:.0f}_n{n}_i{img_ranges}')

N = int(np.sqrt(len(x)))

X = x.reshape((N, N))
Y = y.reshape((N, N))
Z = z.reshape((N, N))

fig_width = 3

initialize_plt()

plt.figure(figsize=(fig_width, fig_width * 0.75))

plt.gca().patch.set_facecolor("#cbcbcb")
plt.gca().add_patch(Polygon(np.array([[- L / (2 * R_init), L / (2 * R_init), L / (2 * R_init), - L / (2 * R_init)],
                                      [- L / (2 * R_init), - L / (2 * R_init), L / (2 * R_init), L / (2 * R_init)]]).T,
                            facecolor="white", edgecolor="black", linewidth=1))

Z_min = np.min(Z, where=np.invert(np.isnan(Z)), initial=np.inf)
Z_max = np.max(Z, where=np.invert(np.isnan(Z)), initial=0)
cnt = plt.contourf(X / R_init, Y / R_init, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64),
                   locator=LogLocator(10))
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
plt.colorbar(cnt, format=LogFormatterMathtext(10), ticks=np.power(10.0, range(0, -10, -1)), label="$\\zeta$")

# plt.scatter(X / R_init, Y / R_init, s=1)

plt.xlim((-1.2 * L / (2 * R_init), 1.2 * L / (2 * R_init)))
plt.ylim((-1.2 * L / (2 * R_init), 1.2 * L / (2 * R_init)))

plt.xlabel("$x = X / R_0$")
plt.ylabel("$y = Y / R_0$")

plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
