import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import itertools
from common.util.vector_utils import dist
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse

W = 2
H = 4
N = 128

R = 1

X = 0
Y = 2

m_0 = 1
n = 20000
density_ratio = 0.25
w_thresh = 6

vxs = np.linspace(-2 * W, 2 * W, N)
vys = np.linspace(-H, 3, N)

centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

speeds = []
us = []
vs = []

R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
sigma = bem.calculate_sigma([X, Y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
for vy, vx in itertools.product(vys, vxs):
    if dist([vx, vy], [X, Y]) < R \
            or vy < 0 and (vx < -W / 2 or vx > W / 2):
        speeds.append(np.nan)
        us.append(np.nan)
        vs.append(np.nan)
    else:
        res_vel = bem.get_vel([vx, vy, 0], centroids, areas, sigma, bubble_pos=[X, Y, 0])
        speed = np.linalg.norm(res_vel)
        speeds.append(speed)
        us.append(res_vel[0] / speed)
        vs.append(res_vel[1] / speed)

vx_mat, vy_mat = np.meshgrid(vxs, vys)
S = np.reshape(np.array(speeds), vx_mat.shape)
U = np.reshape(np.array(us), vx_mat.shape)
V = np.reshape(np.array(vs), vx_mat.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(vx_mat, vy_mat, S, levels=32)
for c in cnt.collections:
    c.set_edgecolor("face")  # Reduce aliasing in output.
plt.quiver(vx_mat, vy_mat, U, V)
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.colorbar(label="$|v|$")
plt.plot([min(vxs), -W / 2, -W / 2, W / 2, W / 2, max(vxs)], [0, 0, -H, -H, 0, 0], 'k')
plt.gca().add_artist(plt.Circle([X, Y], R, color="k", fill=False))
plt.show()
