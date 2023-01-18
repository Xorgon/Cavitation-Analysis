import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np

from common.util.file_utils import csv_to_lists
from common.util.plotting_utils import initialize_plt

corner_n = 3
H = 0.016875  # match the plot height of the other two semi-analytic geometries.
L = 0.02  # sets scale for pressure mesh
R_init = 1 / 1000  # 1 mm
n = 50000
x, y, z = csv_to_lists(f'../../model_outputs/corner_anisotropy_data/',
                       f'R{1000 * R_init:.1f}_L{1000 * L:.0f}_n{n}_cn{corner_n}')

N_y = int(0.25 + 0.25 * np.sqrt(1 + 8 * len(x)))
N_x = int(len(x) / N_y)

X = x.reshape((N_y, N_x))
Y = y.reshape((N_y, N_x))
Z = z.reshape((N_y, N_x))

fig_width = 3

initialize_plt()

plt.figure(figsize=(fig_width, fig_width * 0.85))

plt.gca().patch.set_facecolor("#cbcbcb")
plt.gca().add_patch(
    Polygon(np.array([[0, L * np.sin(np.pi / (2 * corner_n)) / R_init, -L * np.sin(np.pi / (2 * corner_n)) / R_init],
                      [0, L * np.cos(np.pi / (2 * corner_n)) / R_init, L * np.cos(np.pi / (2 * corner_n)) / R_init]]).T,
            facecolor="white", edgecolor="white", linewidth=1))
plt.plot([L * np.sin(np.pi / (2 * corner_n)) / R_init, 0, -L * np.sin(np.pi / (2 * corner_n)) / R_init],
         [L * np.cos(np.pi / (2 * corner_n)) / R_init, 0, L * np.cos(np.pi / (2 * corner_n)) / R_init],
         color="k", linewidth=1)

Z_min = np.min(Z, where=np.invert(np.isnan(Z)), initial=np.inf)
Z_max = np.max(Z, where=np.invert(np.isnan(Z)), initial=0)
cnt = plt.contourf(X / R_init, Y / R_init, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64),
                   locator=LogLocator(10))
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
plt.colorbar(cnt, format=LogFormatterMathtext(10), ticks=np.power(10.0, range(0, -10, -1)), label="$\\zeta$")

# plt.scatter(X / R_init, Y / R_init, s=1)

plt.xlabel("$x = X / R_0$")
plt.ylabel("$y = Y / R_0$")

plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
