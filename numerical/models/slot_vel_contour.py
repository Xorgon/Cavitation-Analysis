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
W = 2
H = 4
N = 16
Ys = np.concatenate([np.linspace(offset, 2, np.round(3 * N / 4)), np.linspace(2 + 0.1, 4, np.ceil(N / 4))])
Xs = np.concatenate([np.linspace(-3 * W / 2, -W, np.ceil(N / 8)),
                     np.linspace(-W + 0.1, W - 0.1, np.round(3 * N / 4)),
                     np.linspace(W, 3 * W / 2, np.ceil(N / 8))])

inner_Ys = np.linspace(offset, -H + offset, np.ceil(N / 2))
inner_Xs = np.linspace(-W / 2 + offset, W / 2 - offset, np.ceil(N / 2))

m_0 = 1
n = 15000
density_ratio = 0.25
w_thresh = 6

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

speeds = []
us = []
vs = []

for Y, X in itertools.product(Ys, Xs):
    print(f"Testing X={X:5.3f}, Y={Y:5.3f}")
    R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([X, Y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    speeds.append(speed)
    us.append(res_vel[0] / speed)
    vs.append(res_vel[1] / speed)

X_mat, Y_mat = np.meshgrid(Xs, Ys)
S = np.reshape(np.array(speeds), X_mat.shape)
U = np.reshape(np.array(us), X_mat.shape)
V = np.reshape(np.array(vs), X_mat.shape)

inner_speeds = []
inner_us = []
inner_vs = []

for Y, X in itertools.product(inner_Ys, inner_Xs):
    print(f"Testing p={X:5.3f}, q={Y:5.3f}")
    R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([X, Y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    inner_speeds.append(speed)
    inner_us.append(res_vel[0] / speed)
    inner_vs.append(res_vel[1] / speed)

inner_X, inner_Y = np.meshgrid(inner_Xs, inner_Ys)
inner_S = np.reshape(np.array(inner_speeds), inner_X.shape)
inner_U = np.reshape(np.array(inner_us), inner_X.shape)
inner_V = np.reshape(np.array(inner_vs), inner_X.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(X_mat, Y_mat, S, levels=32)
plt.contourf(inner_X, inner_Y, inner_S, levels=32, vmin=cnt.get_clim()[0], vmax=cnt.get_clim()[1])
plt.quiver(X_mat, Y_mat, U, V)
# plt.quiver(inner_X, inner_Y, inner_U, inner_V)
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.colorbar(label="$|v|$")
plt.plot([min(Xs), -W / 2, -W / 2, W / 2, W / 2, max(Xs)], [0, 0, -H, -H, 0, 0], 'k')
plt.show()
