import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import itertools
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse

offset = 0.2
w = 2
h = 4
N = 16
qs = np.concatenate([np.linspace(offset, 2, np.round(3 * N / 4)), np.linspace(2 + 0.1, 4, np.ceil(N / 4))])
ps = np.concatenate([np.linspace(-3 * w / 2, -w, np.ceil(N / 8)),
                     np.linspace(-w + 0.1, w - 0.1, np.round(3 * N / 4)),
                     np.linspace(w, 3 * w / 2, np.ceil(N / 8))])

inner_qs = np.linspace(offset, -h + offset, np.ceil(N / 2))
inner_ps = np.linspace(-w / 2 + offset, w / 2 - offset, np.ceil(N / 2))

m_0 = 1
n = 15000
density_ratio = 0.25
w_thresh = 6

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
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

P, Q = np.meshgrid(ps, qs)
S = np.reshape(np.array(speeds), P.shape)
U = np.reshape(np.array(us), P.shape)
V = np.reshape(np.array(vs), P.shape)

inner_speeds = []
inner_us = []
inner_vs = []

for q, p in itertools.product(inner_qs, inner_ps):
    print(f"Testing p={p:5.3f}, q={q:5.3f}")
    R_b = bem.get_R_vector([p, q, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    inner_speeds.append(speed)
    inner_us.append(res_vel[0] / speed)
    inner_vs.append(res_vel[1] / speed)

inner_P, inner_Q = np.meshgrid(inner_ps, inner_qs)
inner_S = np.reshape(np.array(inner_speeds), inner_P.shape)
inner_U = np.reshape(np.array(inner_us), inner_P.shape)
inner_V = np.reshape(np.array(inner_vs), inner_P.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(P, Q, S, levels=32)
plt.contourf(inner_P, inner_Q, inner_S, levels=32, vmin=cnt.get_clim()[0], vmax=cnt.get_clim()[1])
plt.quiver(P, Q, U, V)
# plt.quiver(inner_P, inner_Q, inner_U, inner_V)
plt.xlabel("$p$")
plt.ylabel("$q$")
plt.colorbar(label="$|v|$")
plt.plot([min(ps), -w / 2, -w / 2, w / 2, w / 2, max(ps)], [0, 0, -h, -h, 0, 0], 'k')
plt.show()
