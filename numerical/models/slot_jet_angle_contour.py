import matplotlib.pyplot as plt
import numpy as np

n = 20000
w = 2
h = 4
density_ratio = 0.15
w_thresh = 15
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

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
min_q_idx = 0
cnt = plt.contourf((2 * P / w)[min_q_idx:, :], (Q / w)[min_q_idx:, :],
                   (np.abs(np.arctan2(V, U) + np.pi / 2))[min_q_idx:, :], levels=128)
plt.xlabel("$\\bar{p}$")
plt.ylabel("$q / w$")
plt.colorbar(label="$|\\theta_j|$")
plt.xlim((min(ps), max(ps)))
plt.ylim(-h - 0.1, max(qs))
plt.plot([min(ps), -w / 2, -w / 2, w / 2, w / 2, max(ps)], [0, 0, -h, -h, 0, 0], 'k')
plt.show()
