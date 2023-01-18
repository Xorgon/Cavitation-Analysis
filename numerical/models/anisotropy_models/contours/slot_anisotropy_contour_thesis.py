import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogLocator, LogFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from common.util.plotting_utils import initialize_plt, label_subplot

W = 4 / 1000
H = 4 / 1000
N = 33
M = 32

n = 23000
density_ratio = 0.1
x_limit = 6
w_thresh = x_limit * 2
length = 100 / 1000

R_init = 0.5 / 1000

Xs = []
Ys = []
us = []
vs = []
speeds = []

data_dir = "../../model_outputs/slot_anisotropy_data/"

file = open(f"{data_dir}anisotropy_sweep_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}"
            f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}_N{N}.csv", 'r')

for line in file.readlines():
    X, Y, u, v = line.split(',')
    Xs.append(float(X))
    Ys.append(float(Y))
    us.append(float(u))
    vs.append(float(v))
    speeds.append(np.linalg.norm([float(u), float(v)]))

Xs, Ys, us, vs, speeds = zip(*sorted(zip(Xs, Ys, us, vs, speeds), key=lambda q: (q[1], q[0])))

if len(Xs) == N ** 2:
    # Equal X and Y length
    X_mat = np.reshape(np.array(Xs), (N, N))
    Y_mat = np.reshape(np.array(Ys), (N, N))
    S = np.reshape(np.array(speeds), (N, N))
    U = np.reshape(np.array(us), X_mat.shape)
    V = np.reshape(np.array(vs), X_mat.shape)
elif M is not None:
    # Must set if X and Y are not equal
    X_mat = np.reshape(np.array(Xs), (M, N))
    Y_mat = np.reshape(np.array(Ys), (M, N))
    S = np.reshape(np.array(speeds), (M, N))
    U = np.reshape(np.array(us), X_mat.shape)
    V = np.reshape(np.array(vs), X_mat.shape)
else:
    # For partially finished grids (for use mid-run)
    X_mat = np.empty((N, N))
    X_mat.fill(np.nan)
    Y_mat = np.empty((N, N))
    Y_mat.fill(np.nan)
    U = np.empty((N, N))
    U.fill(np.nan)
    V = np.empty((N, N))
    V.fill(np.nan)
    S = np.empty((N, N))
    S.fill(np.nan)
    for k in range(len(Xs)):
        i = int(np.floor(k / N))
        j = int(k % N)
        X_mat[i, j] = Xs[k]
        Y_mat[i, j] = Ys[k]
        U[i, j] = us[k]
        V[i, j] = vs[k]
        S[i, j] = speeds[k]

c_xs = []
c_ys = []
c_zs = []
centroids_file = open(f"{data_dir}centroids_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}"
                      f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}.csv", 'r')
for line in centroids_file.readlines():
    split = line.split(",")
    c_xs.append(float(split[0]))
    c_ys.append(float(split[1]))
    c_zs.append(float(split[2]))

n_xs = []
n_ys = []
n_zs = []
normals_file = open(f"{data_dir}normals_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}"
                    f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}.csv", 'r')
for line in normals_file.readlines():
    split = line.split(",")
    n_xs.append(float(split[0]))
    n_ys.append(float(split[1]))
    n_zs.append(float(split[2]))

min_z = np.min([z for z, x in zip(c_zs, c_xs) if min(Xs) <= x <= max(Xs)])
c_xs, c_ys, c_zs, n_xs, n_ys, n_zs = zip(*[o for o in zip(c_xs, c_ys, c_zs, n_xs, n_ys, n_zs)
                                           if o[2] == min_z and min(Xs) <= o[0] <= max(Xs)])
angles = np.arctan2(n_ys, n_xs) - np.pi / 2

zetas = np.sqrt(V ** 2 + U ** 2)

initialize_plt()

fig_width = 6

fig, (ax, lax) = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.5), gridspec_kw={"width_ratios": [1, 0.9]})
ax.set_aspect(2)
min_y_idx = 0
max_y_idx = 10000

ax.set_xlim((2 * min(Xs) / W, 2 * max(Xs) / W))
ax.set_ylim(-H / W - 0.1, np.max(Y_mat[min_y_idx:max_y_idx, :]) / W)
ax.patch.set_facecolor("#cbcbcb")
ax.add_patch(Polygon(np.array([[min(Xs) * 2 / W, min(Xs) * 2 / W, -1, -1, 1, 1, max(Xs) * 2 / W, max(Xs) * 2 / W],
                               [ax.get_ylim()[1], 0, 0, -H / W, -H / W, 0, 0, ax.get_ylim()[1]]]).T,
                     facecolor="white", linewidth=0))

zeta_min = np.min(zetas[min_y_idx:max_y_idx, :], where=np.invert(np.isnan(zetas[min_y_idx:max_y_idx, :])),
                  initial=np.inf)
zeta_max = np.max(zetas[min_y_idx:max_y_idx, :], where=np.invert(np.isnan(zetas[min_y_idx:max_y_idx, :])), initial=0)
cnt = ax.contourf((2 * X_mat / W)[min_y_idx:max_y_idx, :], (Y_mat / W)[min_y_idx:max_y_idx, :],
                  zetas[min_y_idx:max_y_idx, :],
                  np.logspace(np.log10(zeta_min),
                              np.log10(zeta_max), 64),
                  cmap=plt.cm.get_cmap('viridis'),
                  corner_mask=False, locator=LogLocator(10))

# plt.quiver(X_mat / (0.5 * W), Y_mat / (0.5 * W), U / zetas, V / zetas, pivot='mid')
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.plot([min(Xs) * 2 / W, -1, -1, 1, 1, max(Xs) * 2 / W], [0, 0, -H / W, -H / W, 0, 0], 'k')
ax.scatter(np.divide(c_xs, 0.5 * W), np.divide(c_ys, W), marker='.', color='k')
target_y = 1
Y_idx = int(np.argmin(np.abs(np.subtract(np.divide(Ys, W), target_y))))
line_y = Ys[Y_idx] / W

cbar_left = 0.05 * (-1 - (2 * min(Xs) / W)) / (2 * max(Xs) / W - 2 * min(Xs) / W)
cbar_width = 0.9 * (-1 - (2 * min(Xs) / W)) / (2 * max(Xs) / W - 2 * min(Xs) / W)
cbar_bottom = 0.18 / ((np.max(Y_mat[min_y_idx:max_y_idx, :]) / W) - (-H / W - 0.1))
cax = ax.inset_axes([cbar_left, cbar_bottom, cbar_width, 0.05])
cbar = plt.colorbar(cnt, cax=cax, orientation='horizontal', format=LogFormatterMathtext(10),
                    ticks=np.power(10.0, range(0, -10, -1)))
cbar.ax.tick_params(labelsize='small', pad=1)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("$\\zeta$")

# plt.axhline(line_y, color='k', linestyle='--')

# plt.figure(figsize=(fig_width(), 0.75 * fig_width()))
# lax = plt.gca()
min_y_idx = 8
for i in [1, 4, 5, 6, 19]:
    min_y = 0
    ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.axvline(X_mat[0, i] / (0.5 * W),
               (min_y / (0.5 * W) - ax.get_ylim()[0]) / ylim_range, color="gray",
               linestyle="--", linewidth=0.75)
    lax.plot(Y_mat[min_y_idx:, i] / R_init, zetas[min_y_idx:, i], label=f"x = {X_mat[0, i] / (0.5 * W):.2f}")
lax.plot(Y_mat[min_y_idx:, 0] / R_init, 0.195 * (Y_mat[min_y_idx:, 0] / R_init) ** -2, "k--", label="Flat plate")
lax.loglog()

lax.set_ylim((2.5e-4, 0.5))

plt.xlabel("$\\gamma = Y / R_0$")
plt.ylabel("$\\zeta$")
plt.legend(frameon=False, loc='upper right', fontsize='x-small')

label_subplot(ax, "($a$)", color="white")
label_subplot(lax, "($b$)")

dims_ax = ax.inset_axes((0.6, 0.7, 0.375, 0.275))
dims_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)
dims_ax.add_patch(Polygon(np.array([[-2, -2, -1, -1, 1, 1, 2.5, 2.5], [-1.2, 0, 0, -1, -1, 0, 0, -1.2]]).T,
                          color="#cbcbcb"))
dims_ax.plot([1, 1.5], [-1, -1], color="gray", linestyle="--", linewidth=0.5)
dims_ax.plot([-2, -1, -1, 1, 1, 2.5], [0, 0, -1, -1, 0, 0], 'k')
dims_ax.set_xlim((-1.5, 2.5))
dims_ax.set_ylim((-1.2, 0.6))
dims_ax.annotate("", (-1, 0), (1, 0),
                 arrowprops={'arrowstyle': '<|-|>', 'fc': f"C3", 'ec': f"C3",
                             'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
dims_ax.text(0, 0, f"$W$", ha='center', va='bottom', color=f"C3", fontsize="medium")
dims_ax.annotate("", (1.5, 0), (1.5, -1),
                 arrowprops={'arrowstyle': '<|-|>', 'fc': f"C0", 'ec': f"C0",
                             'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
dims_ax.text(1.55, -0.5, f"$H$", ha='left', va='center', color=f"C0", fontsize="medium")

plt.tight_layout()
plt.show()
