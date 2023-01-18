import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from util.plotting_utils import initialize_plt

n = 20000
W = 2
H = 2
density_ratio = 0.1
w_thresh = 12
length = 100

N = 64

Xs = []
Ys = []
us = []
vs = []
speeds = []

data_dir = "figure7_data/"

file = open(f"{data_dir}vel_sweep_n{n}_W{W:.2f}_H{H:.2f}"
            f"_drat{density_ratio}_wthresh{w_thresh}_len{length}_N{N}.csv", 'r')

for line in file.readlines():
    X, Y, u, v = line.split(',')
    Xs.append(float(X))
    Ys.append(float(Y))
    us.append(float(u))
    vs.append(float(v))
    speeds.append(np.linalg.norm([float(u), float(v)]))

Xs, Ys, us, vs, speeds = zip(*sorted(zip(Xs, Ys, us, vs, speeds), key=lambda q: (q[1], q[0])))

if len(Xs) == N ** 2:
    X_mat = np.reshape(np.array(Xs), (N, N))
    Y_mat = np.reshape(np.array(Ys), (N, N))
    S = np.reshape(np.array(speeds), (N, N))
    U = np.reshape(np.array(us), X_mat.shape)
    V = np.reshape(np.array(vs), X_mat.shape)
else:
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
centroids_file = open(f"{data_dir}centroids_n{n}_W{W:.2f}_H{H:.2f}"
                      f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'r')
for line in centroids_file.readlines():
    split = line.split(",")
    c_xs.append(float(split[0]))
    c_ys.append(float(split[1]))
    c_zs.append(float(split[2]))

n_xs = []
n_ys = []
n_zs = []
normals_file = open(f"{data_dir}normals_n{n}_W{W:.2f}_H{H:.2f}"
                    f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'r')
for line in normals_file.readlines():
    split = line.split(",")
    n_xs.append(float(split[0]))
    n_ys.append(float(split[1]))
    n_zs.append(float(split[2]))

min_z = np.min([z for z, x in zip(c_zs, c_xs) if min(Xs) <= x <= max(Xs)])
c_xs, c_ys, c_zs, n_xs, n_ys, n_zs = zip(*[o for o in zip(c_xs, c_ys, c_zs, n_xs, n_ys, n_zs)
                                           if o[2] == min_z and min(Xs) <= o[0] <= max(Xs)])
angles = np.arctan2(n_ys, n_xs) - np.pi / 2

initialize_plt()

############
# LOG PLOT #
############
plot_log = False
if plot_log:
    fig = plt.figure()
    fig.gca().set_aspect('equal', 'box')
    min_y_idx = 0
    cnt = plt.contourf((2 * X_mat / W)[min_y_idx:, :], (Y_mat / W)[min_y_idx:, :],
                       (np.log(np.abs(np.arctan2(V, U) + np.pi / 2)))[min_y_idx:, :], levels=128,
                       cmap=plt.cm.get_cmap('seismic'))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim((2 * min(Xs) / W, 2 * max(Xs) / W))
    plt.ylim(-H - 0.1, max(Ys) / W)
    plt.plot([min(Xs) * 2 / W, -W / 2, -W / 2, W / 2, W / 2, max(Xs) * 2 / W], [0, 0, -H, -H, 0, 0], 'k')
    plt.scatter(c_xs, c_ys, marker='.', color='k')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(cnt, label="$log(|\\theta|)$", cax=cax)
    plt.tight_layout()

###############
# NORMAL PLOT #
###############
# NOTE: In the main plot (ax) all y values are doubled and then corrected in the tick labels to make the scaling work
#       correctly. Other methods cause various axes to become the wrong sizes.
fig, ax = plt.subplots(figsize=(5.31445, 4.3))
ax.set_aspect(1, 'box')
min_y_idx = 2
max_y_idx = -8
cnt = plt.contourf((2 * X_mat / W)[min_y_idx:max_y_idx, :], (2 * Y_mat / W)[min_y_idx:max_y_idx, :],
                   (np.arctan2(V, U) + np.pi / 2)[min_y_idx:max_y_idx, :], levels=64,
                   cmap=plt.cm.get_cmap('seismic'))
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
plt.ylabel("$y$")
plt.xlim((2 * min(Xs) / W, 2 * max(Xs) / W))
plt.ylim(-2 * H / W - 0.1, 2 * np.max(Y_mat[min_y_idx:max_y_idx, :]) / W)
plt.plot([min(Xs) * 2 / W, -1, -1, 1, 1, max(Xs) * 2 / W], [0, 0, -2 * H / W, -2 * H / W, 0, 0], 'k')
Yticks = np.array(range(-2, int(np.max(Y_mat[min_y_idx:max_y_idx, :])) + 1, 2))
ax.set_yticks(Yticks)
ax.set_yticklabels(Yticks / 2)  # Hack to get scaling and sizing right.
plt.scatter(np.divide(c_xs, 0.5 * W), np.divide(c_ys, 0.5 * W), marker='.', color='k')
plt.tick_params(axis='x', which='both', labelbottom=False)
target_y = 1
Y_idx = int(np.argmin(np.abs(np.subtract(np.divide(Ys, W), target_y))))
line_y = Ys[Y_idx] / W

plt.axhline(2 * line_y, color='k', linestyle='--')
plt.annotate(f"($a$)", xy=(0, 0), xytext=(0.025, 0.965),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
cbar = plt.colorbar(cnt, label="$\\theta$ (rad)", cax=cax, ticks=[-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4])
cax.set_yticklabels(["$-\\pi / 4$", "$-\\pi / 8$", "0", "$\\pi / 8$", "$\\pi / 4$"])

lax = divider.append_axes('bottom', 1.2, pad=0.1, sharex=ax)
thetas = np.arctan2(vs, us) + np.pi / 2
line_Xs, line_Ys, line_thetas = zip(*[(X, Y, theta) for X, Y, theta in zip(Xs, Ys, thetas) if Y / W == line_y])
lax.plot(2 * np.divide(line_Xs, W), line_thetas, 'k--', label=f"$y = {line_y:.0f}$")
plt.xlabel("$x$")
plt.ylabel("$\\theta$ (rad)")
plt.legend(frameon=False, loc='lower right')
plt.annotate(f"($b$)", xy=(0, 0), xytext=(0.025, 0.93),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
