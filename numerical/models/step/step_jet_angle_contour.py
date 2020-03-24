import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from common.util.plotting_utils import initialize_plt

n = 15000
H = 2
density_ratio = 0.1
thresh_dist = 12
length = 100

N = 32

Xs = []
Ys = []
us = []
vs = []
speeds = []

file = open(f"../model_outputs/step_vel_data/vel_sweep_n{n}_H{H:.2f}"
            f"_drat{density_ratio}_thresh{thresh_dist}_len{length}_N{N}.csv", 'r')

for line in file.readlines():
    X, Y, u, v = line.split(',')
    Xs.append(float(X))
    Ys.append(float(Y))
    us.append(float(u))
    vs.append(float(v))
    speeds.append(np.linalg.norm([float(u), float(v)]))

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
centroids_file = open(f"../model_outputs/step_vel_data/centroids_n{n}_H{H:.2f}"
                      f"_drat{density_ratio}_thresh{thresh_dist}_len{length}.csv", 'r')
for line in centroids_file.readlines():
    split = line.split(",")
    c_xs.append(float(split[0]))
    c_ys.append(float(split[1]))
    c_zs.append(float(split[2]))

n_xs = []
n_ys = []
n_zs = []
normals_file = open(f"../model_outputs/step_vel_data/normals_n{n}_H{H:.2f}"
                    f"_drat{density_ratio}_thresh{thresh_dist}_len{length}.csv", 'r')
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
    min_q_idx = 0
    cnt = plt.contourf(X_mat[min_q_idx:, :], Y_mat[min_q_idx:, :],
                       (np.log(np.abs(np.arctan2(V, U) + np.pi / 2)))[min_q_idx:, :], levels=128,
                       cmap=plt.cm.get_cmap('seismic'))
    # plt.scatter((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :], color='k', marker='.')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim((min(Xs), max(Xs)))
    plt.ylim(-H - 0.1, max(Ys))
    plt.plot([min(Xs), 0, 0, max(Xs)], [0, 0, -H, -H], 'k')
    plt.scatter(c_xs, c_ys, marker='.', color='k')
    # for x, y, angle in zip(xs, ys, angles):
    #     plt.scatter(x, y, marker=(3, 0, np.degrees(angle)), color='k')

    # plt.axhline(0.5, color='k', linestyle='--')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(cnt, label="$log(|\\theta_j|)$", cax=cax)
    plt.tight_layout()

###############
# NORMAL PLOT #
###############
# NOTE: In the main plot (ax) all y values are doubled and then corrected in the tick labels to make the scaling work
#       correctly. Other methods cause various axes to become the wrong sizes.
fig, ax = plt.subplots(figsize=(5.31445, 5.2))
ax.set_aspect(1, 'box')
min_q_idx = 1
min_x_idx = 1
max_x_idx = -1
cnt = plt.contourf(X_mat[min_q_idx:, min_x_idx:max_x_idx], Y_mat[min_q_idx:, min_x_idx:max_x_idx],
                   (np.arctan2(V, U) + np.pi / 2)[min_q_idx:, min_x_idx:max_x_idx], levels=64,
                   cmap=plt.cm.get_cmap('seismic'))
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
# plt.scatter((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :], color='k', marker='.')
plt.ylabel("$Y$")
# plt.xlim((min(Xs), max(Xs)))
plt.xlim((-10, 10))  # TODO: Don't do this
plt.ylim(-H - 0.1, max(Ys))
plt.plot([min(Xs), 0, 0, max(Xs)], [0, 0, -H, -H], 'k')
plt.scatter(c_xs, c_ys, marker='.', color='k')
# for x, y, angle in zip(xs, ys, angles):
#     plt.scatter(x, y, marker=(3, 0, np.degrees(angle)), color='k')
plt.tick_params(axis='x', which='both', labelbottom=False)
target_y = 1
Y_idx = int(np.argmin(np.abs(np.subtract(Ys, target_y))))
line_y = Ys[Y_idx]

plt.axhline(line_y, color='k', linestyle='--')
plt.annotate(f"($a$)", xy=(0, 0), xytext=(0.025, 0.975),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
plt.colorbar(cnt, label="$\\theta_j$ (rad)", cax=cax, ticks=[-np.pi / 8, 0])
cax.set_yticklabels(["$-\\pi / 8$", "0"])
lax = divider.append_axes('bottom', 1.2, pad=0.1, sharex=ax)
theta_js = np.arctan2(vs, us) + np.pi / 2
line_Xs, line_Ys, line_theta_js = zip(*[(X, Y, theta_j) for X, Y, theta_j in zip(Xs, Ys, theta_js) if Y == line_y])
lax.plot(line_Xs, line_theta_js, 'k--', label=f"$Y = {line_y:.2f}$")
plt.xlabel("$X$")
plt.ylabel("$\\theta_j$")
plt.legend(frameon=False, loc='lower right')
plt.annotate(f"($b$)", xy=(0, 0), xytext=(0.025, 0.95),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
