import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from common.util.plotting_utils import initialize_plt

n = 20000
w = 2
h = 2
density_ratio = 0.1
w_thresh = 12
length = 100

N = 64

ps = []
qs = []
us = []
vs = []
speeds = []

file = open(f"model_outputs/slot_vel_data/vel_sweep_n{n}_w{w:.2f}_h{h:.2f}"
            f"_drat{density_ratio}_wthresh{w_thresh}_len{length}_N{N}.csv", 'r')

for line in file.readlines():
    p, q, u, v = line.split(',')
    ps.append(float(p))
    qs.append(float(q))
    us.append(float(u))
    vs.append(float(v))
    speeds.append(np.linalg.norm([float(u), float(v)]))

if len(ps) == N ** 2:
    P = np.reshape(np.array(ps), (N, N))
    Q = np.reshape(np.array(qs), (N, N))
    S = np.reshape(np.array(speeds), (N, N))
    U = np.reshape(np.array(us), P.shape)
    V = np.reshape(np.array(vs), P.shape)
else:
    P = np.empty((N, N))
    P.fill(np.nan)
    Q = np.empty((N, N))
    Q.fill(np.nan)
    U = np.empty((N, N))
    U.fill(np.nan)
    V = np.empty((N, N))
    V.fill(np.nan)
    S = np.empty((N, N))
    S.fill(np.nan)
    for k in range(len(ps)):
        i = int(np.floor(k / N))
        j = int(k % N)
        P[i, j] = ps[k]
        Q[i, j] = qs[k]
        U[i, j] = us[k]
        V[i, j] = vs[k]
        S[i, j] = speeds[k]

xs = []
ys = []
zs = []
centroids_file = open(f"model_outputs/slot_vel_data/centroids_n{n}_w{w:.2f}_h{h:.2f}"
                      f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'r')
for line in centroids_file.readlines():
    split = line.split(",")
    xs.append(float(split[0]))
    ys.append(float(split[1]))
    zs.append(float(split[2]))

n_xs = []
n_ys = []
n_zs = []
normals_file = open(f"model_outputs/slot_vel_data/normals_n{n}_w{w:.2f}_h{h:.2f}"
                    f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'r')
for line in normals_file.readlines():
    split = line.split(",")
    n_xs.append(float(split[0]))
    n_ys.append(float(split[1]))
    n_zs.append(float(split[2]))

min_z = np.min([z for z, x in zip(zs, xs) if min(ps) <= x <= max(ps)])
xs, ys, zs, n_xs, n_ys, n_zs = zip(*[o for o in zip(xs, ys, zs, n_xs, n_ys, n_zs)
                                     if o[2] == min_z and min(ps) <= o[0] <= max(ps)])
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
    cnt = plt.contourf((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :],
                       (np.log(np.abs(np.arctan2(V, U) + np.pi / 2)))[min_q_idx:, :], levels=128,
                       cmap=plt.cm.get_cmap('seismic'))
    # plt.scatter((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :], color='k', marker='.')
    plt.xlabel("$\\bar{p}$")
    plt.ylabel("$q / w$")
    plt.xlim((2 * min(ps) / w, 2 * max(ps) / w))
    plt.ylim(-h - 0.1, max(qs) / w)
    plt.plot([min(ps) * 2 / w, -w / 2, -w / 2, w / 2, w / 2, max(ps) * 2 / w], [0, 0, -h, -h, 0, 0], 'k')
    plt.scatter(xs, ys, marker='.', color='k')
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
fig, ax = plt.subplots(figsize=(5.31445, 3.9))
ax.set_aspect('equal', 'box')
min_q_idx = 2
cnt = plt.contourf((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :],
                   (np.arctan2(V, U) + np.pi / 2)[min_q_idx:, :], levels=64,
                   cmap=plt.cm.get_cmap('seismic'))
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
# plt.scatter((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :], color='k', marker='.')
plt.ylabel("$q / w$")
plt.xlim((2 * min(ps) / w, 2 * max(ps) / w))
plt.ylim(-h - 0.1, max(qs) / w)
plt.plot([min(ps) * 2 / w, -w / 2, -w / 2, w / 2, w / 2, max(ps) * 2 / w], [0, 0, -h, -h, 0, 0], 'k')
plt.scatter(xs, ys, marker='.', color='k')
# for x, y, angle in zip(xs, ys, angles):
#     plt.scatter(x, y, marker=(3, 0, np.degrees(angle)), color='k')

target_q = 1
q_idx = int(np.argmin(np.abs(np.subtract(qs, 1))))
line_q = qs[q_idx]

plt.axhline(line_q, color='k', linestyle='--')
plt.annotate(f"a)", xy=(0, 0), xytext=(0.025, 0.95),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
plt.colorbar(cnt, label="$\\theta_j$", cax=cax)

lax = divider.append_axes('bottom', 1.2, pad=0.1, sharex=ax)
theta_js = np.arctan2(vs, us) + np.pi / 2
line_ps, line_qs, line_theta_js = zip(*[(p, q, theta_j) for p, q, theta_j in zip(ps, qs, theta_js) if q == line_q])
lax.plot(2 * np.divide(line_ps, w), line_theta_js, 'k--', label=f"$q / w = {line_q:.2f}$")
plt.xlabel("$\\bar{p}$")
plt.ylabel("$\\theta_j$")
plt.legend(frameon=False)
plt.annotate(f"b)", xy=(0, 0), xytext=(0.025, 0.95),
             textcoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
