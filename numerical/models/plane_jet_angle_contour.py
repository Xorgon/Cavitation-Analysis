import numpy as np
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import itertools
import scipy.sparse

offset = 1
w = 2
h = 3
N = 32
qs = np.concatenate([np.linspace(offset, 3, np.round(3 * N / 4)), np.linspace(3 + 0.1, w * 5, np.ceil(N / 4))])
ps = np.concatenate([np.linspace(-5 * w / 2, -w, np.ceil(N / 8)),
                     np.linspace(-w + 0.1, w - 0.1, np.round(3 * N / 4)),
                     np.linspace(w, 5 * w / 2, np.ceil(N / 8))])

# qs = np.linspace(3 * w, 5 * w, 16)
# ps = np.linspace(0, 5 * w / 2, 16)

m_0 = 1
n = 10000

length = 50
depth = 50
centroids, normals, areas = gen.gen_plane([-length / 2, 0, -depth / 2],
                                          [-length / 2, 0, depth / 2],
                                          [length / 2, 0, -depth / 2],
                                          n)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
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

P, Q = np.meshgrid(ps, qs)
S = np.reshape(np.array(speeds), P.shape)
U = np.reshape(np.array(us), P.shape)
V = np.reshape(np.array(vs), P.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(2 * P / w, Q / w, np.arctan2(V, U) + np.pi / 2, levels=128)
plt.xlabel("$\\bar{p}$")
plt.ylabel("$q / w$")
plt.colorbar(label="$\\theta_j$")
plt.plot([min(ps), max(ps)], [0, 0], 'k')
plt.show()
