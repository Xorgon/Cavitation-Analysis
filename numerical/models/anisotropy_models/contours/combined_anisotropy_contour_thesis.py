import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.cm import get_cmap
import numpy as np

from common.util.file_utils import csv_to_lists
from common.util.plotting_utils import initialize_plt, label_subplot

x_lim_factor = 1.5


def load_tri():
    L = 0.015  # 15 mm
    R_init = 1 / 1000  # 1 mm
    img_ranges = 32
    n = 20000
    x, y, z = csv_to_lists(f'../../model_outputs/triangle_anisotropy_data/',
                           f'R{1000 * R_init:.1f}_L{1000 * L:.0f}_n{n}_i{img_ranges}')

    N_x = int(0.25 + 0.25 * np.sqrt(1 + 8 * len(x)))
    N_y = int(2 * N_x - 1)

    X = x.reshape((N_y, N_x))
    Y = y.reshape((N_y, N_x))
    Z = z.reshape((N_y, N_x))

    return X, Y, Z, L, R_init


def plot_tri(X, Y, Z, L, R_init, tri_ax, Z_min, Z_max):
    x_off = L * np.sqrt(3) / (4 * R_init)
    tri_ax.patch.set_facecolor("#cbcbcb")
    tri_ax.plot([L * np.sqrt(3) / (2 * R_init) - x_off, 1 + L * np.sqrt(3) / (2 * R_init) - x_off],
                [-L / (2 * R_init), -L / (2 * R_init)],
                color="gray", linestyle="--", linewidth=0.5)
    tri_ax.plot([L * np.sqrt(3) / (2 * R_init) - x_off, 1 + L * np.sqrt(3) / (2 * R_init) - x_off],
                [L / (2 * R_init), L / (2 * R_init)],
                color="gray", linestyle="--", linewidth=0.5)
    tri_ax.add_patch(Polygon(np.array([[0 - x_off,
                                        L * np.sqrt(3) / (2 * R_init) - x_off,
                                        L * np.sqrt(3) / (2 * R_init) - x_off],
                                       [0, L / (2 * R_init), - L / (2 * R_init)]]).T,
                             facecolor="white", edgecolor="black", linewidth=1, zorder=2))

    cnt = tri_ax.contourf(X / R_init - x_off, Y / R_init, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64),
                          locator=LogLocator(10), zorder=3)

    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.

    tri_ax.set_xlim((-x_lim_factor * L / (2 * R_init) * 0.7, x_lim_factor * L / (2 * R_init)))
    tri_ax.set_ylim((-1.25 * L / (2 * R_init), 1.25 * L / (2 * R_init)))
    tri_ax.set_aspect('equal')

    tri_ax.annotate("", (1 + L * np.sqrt(3) / (2 * R_init) - x_off, L / (2 * R_init)),
                    (1 + L * np.sqrt(3) / (2 * R_init) - x_off, - L / (2 * R_init)),
                    arrowprops={'arrowstyle': '<|-|>', 'fc': f"C3", 'ec': f"C3",
                                'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
    tri_ax.text(1.25 + L * np.sqrt(3) / (2 * R_init) - x_off, 0,
                "$\\frac{L}{R_0}$", ha='left', va='center', color=f"C3", fontsize="medium")

    return cnt


def load_square():
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

    return X, Y, Z, L, R_init


def plot_square(X, Y, Z, L, R_init, sqr_ax, Z_min, Z_max):
    sqr_ax.patch.set_facecolor("#cbcbcb")
    sqr_ax.plot([L / (2 * R_init), 1 + L / (2 * R_init)], [-L / (2 * R_init), -L / (2 * R_init)],
                color="gray", linestyle="--", linewidth=0.5)
    sqr_ax.plot([L / (2 * R_init), 1 + L / (2 * R_init)], [L / (2 * R_init), L / (2 * R_init)],
                color="gray", linestyle="--", linewidth=0.5)
    sqr_ax.add_patch(Polygon(np.array([[- L / (2 * R_init), L / (2 * R_init), L / (2 * R_init), - L / (2 * R_init)],
                                       [- L / (2 * R_init), - L / (2 * R_init), L / (2 * R_init),
                                        L / (2 * R_init)]]).T,
                             facecolor="white", edgecolor="black", linewidth=1, zorder=2))

    cnt = sqr_ax.contourf(X / R_init, Y / R_init, Z, levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64),
                          locator=LogLocator(10), zorder=3)

    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.

    sqr_ax.set_xlim((-x_lim_factor * L / (2 * R_init), x_lim_factor * L / (2 * R_init)))
    sqr_ax.set_ylim((-1.25 * L / (2 * R_init), 1.25 * L / (2 * R_init)))
    sqr_ax.set_aspect('equal')

    sqr_ax.annotate("", (1 + L / (2 * R_init), L / (2 * R_init)),
                    (1 + L / (2 * R_init), - L / (2 * R_init)),
                    arrowprops={'arrowstyle': '<|-|>', 'fc': f"C3", 'ec': f"C3",
                                'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
    sqr_ax.text(1.25 + L / (2 * R_init), 0, "$\\frac{L}{R_0}$", ha='left', va='center', color=f"C3", fontsize="medium")

    return cnt


def load_corner():
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

    return X, Y, Z, L, R_init, corner_n


def plot_corner(X, Y, Z, L, R_init, corner_n, corner_ax, Z_min, Z_max, plot_L, Y_shift):
    corner_ax.patch.set_facecolor("#cbcbcb")
    corner_ax.add_patch(Polygon(
        np.array([[0, L * np.sin(np.pi / (2 * corner_n)) / R_init, -L * np.sin(np.pi / (2 * corner_n)) / R_init],
                  [0 + Y_shift, L * np.cos(np.pi / (2 * corner_n)) / R_init + Y_shift,
                   L * np.cos(np.pi / (2 * corner_n)) / R_init + Y_shift]]).T,
        facecolor="white", edgecolor="white", linewidth=1))

    corner_ax.plot([L * np.sin(np.pi / (2 * corner_n)) / R_init, 0, -L * np.sin(np.pi / (2 * corner_n)) / R_init],
                   [L * np.cos(np.pi / (2 * corner_n)) / R_init + Y_shift, 0 + Y_shift,
                    L * np.cos(np.pi / (2 * corner_n)) / R_init + Y_shift],
                   color="k", linewidth=1)

    cnt = corner_ax.contourf(X / R_init, Y / R_init + Y_shift, Z,
                             levels=np.logspace(np.log10(Z_min), np.log10(Z_max), 64),
                             locator=LogLocator(10))

    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.

    corner_ax.set_xlim((-x_lim_factor * plot_L / (2 * R_init), x_lim_factor * plot_L / (2 * R_init)))
    corner_ax.set_ylim((-1.25 * plot_L / (2 * R_init), 1.25 * plot_L / (2 * R_init)))
    corner_ax.set_aspect('equal')
    return cnt


tri_X, tri_Y, tri_Z, tri_L, tri_R_init = load_tri()
sqr_X, sqr_Y, sqr_Z, sqr_L, sqr_R_init = load_square()
crn_X, crn_Y, crn_Z, crn_L, crn_R_init, crn_n = load_corner()

all_Z = np.concatenate([tri_Z.flatten(), sqr_Z.flatten(), crn_Z.flatten()])
Z_min = np.min(all_Z, where=np.invert(np.isnan(all_Z)), initial=np.inf)
Z_max = np.max(all_Z, where=np.invert(np.isnan(all_Z)), initial=0)

fig_width = 6

initialize_plt()

fig, (triangle_ax, square_ax, corner_ax, c_ax) = plt.subplots(1, 4, figsize=(fig_width, fig_width * 0.3),
                                                              gridspec_kw={"width_ratios": [2.55, 3, 3, 0.125]})
square_ax.set_yticklabels([])
corner_ax.set_yticklabels([])
c_ax.set_aspect(20)
fig.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1)
tri_cnt = plot_tri(tri_X, tri_Y, tri_Z, tri_L, tri_R_init, triangle_ax, Z_min, Z_max)
sqr_cnt = plot_square(sqr_X, sqr_Y, sqr_Z, sqr_L, sqr_R_init, square_ax, Z_min, Z_max)
crn_cnt = plot_corner(crn_X, crn_Y, crn_Z, crn_L, crn_R_init, crn_n, corner_ax, Z_min, Z_max, sqr_L,
                      -sqr_L / (2 * crn_R_init))

plt.colorbar(tri_cnt, format=LogFormatterMathtext(10), ticks=np.power(10.0, range(0, -10, -1)), label="$\\zeta$",
             cax=c_ax)

triangle_ax.set_ylabel("$y = Y / R_0$")
triangle_ax.set_xlabel("$x = X / R_0$")
square_ax.set_xlabel("$x = X / R_0$")
corner_ax.set_xlabel("$x = X / R_0$")

label_subplot(triangle_ax, "($a$)", loc="bl")
label_subplot(square_ax, "($b$)", loc="bl")
label_subplot(corner_ax, "($c$)", loc="bl")

plt.tight_layout()
# for _ in range(10):  # Really hacky but it centers better this way
#     plt.tight_layout()
plt.show()
