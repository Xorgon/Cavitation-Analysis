import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

import numerical.bem as bem
import numerical.util.gen_utils as gen

if not os.path.exists("model_outputs/slot_vel_data"):
    os.makedirs("model_outputs/slot_vel_data")

offset = 0.05
w = 2
h = 4
N = 64
qs = np.concatenate([np.linspace(offset, 3, np.round(3 * N / 4)), np.linspace(3 + 0.1, w * 5, np.ceil(N / 4))])
ps = np.concatenate([np.linspace(-7 * w / 2, -w, np.ceil(N / 8)),
                     np.linspace(-w + 0.1, w - 0.1, np.round(3 * N / 4)),
                     np.linspace(w, 7 * w / 2, np.ceil(N / 8))])

# qs = np.linspace(3 * w, 5 * w, 16)
# ps = np.linspace(0, 5 * w / 2, 16)

m_0 = 1
n = 20000
density_ratio = 0.15
w_thresh = 15
length = 100

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=length, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)

file = open(f"model_outputs/slot_vel_data/vel_sweep_n{n}_w{w:.2f}_h{h:.2f}"
            f"_drat{density_ratio}_wthresh{w_thresh}_len{length}_N{N}.csv", 'w')
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

speeds = []
us = []
vs = []

for q, p in itertools.product(qs, ps):
    print(f"Testing p={p:5.3f}, q={q:5.3f}")
    R_b = bem.get_R_vector([p, q, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    speeds.append(speed)
    us.append(res_vel[0] / speed)
    vs.append(res_vel[1] / speed)
    file.write(f"{p},{q},{res_vel[0]},{res_vel[1]}\n")

file.close()

P, Q = np.meshgrid(ps, qs)
S = np.reshape(np.array(speeds), P.shape)
U = np.reshape(np.array(us), P.shape)
V = np.reshape(np.array(vs), P.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(2 * P / w, Q / w, np.abs(np.arctan2(V, U) + np.pi / 2), levels=128)
plt.xlabel("$\\bar{p}$")
plt.ylabel("$q / w$")
plt.colorbar(label="$|\\theta_j|$")
plt.plot([min(ps), -w / 2, -w / 2, w / 2, w / 2, max(ps)], [0, 0, -h, -h, 0, 0], 'k')
plt.show()
