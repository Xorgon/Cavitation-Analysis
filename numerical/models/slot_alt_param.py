import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse

# ps = [8]
w = 2
h = 2

n_samples = 24
# theta_b = np.pi / 4
# d = w

ds = np.linspace(0, 2 * w, n_samples)
theta_bs = np.linspace(0, np.pi / 2 - 0.2, n_samples)

ds_mat, theta_bs_mat = np.meshgrid(ds, theta_bs)

m_0 = 1
n = 20000

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50, w_thresh=6,
                                                density_ratio=0.25)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

points = np.empty((n_samples, n_samples, 3))
points[:, :, 0] = ds_mat * np.sin(theta_bs_mat)
points[:, :, 1] = ds_mat * np.cos(theta_bs_mat)
points[:, :, 2] = 0
# plot_3d_point_sets([points.reshape(-1, 3)])
vels = bem.get_jet_dirs(points.reshape(-1, 3), centroids, normals, areas, m_0, R_inv)
theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca(projection='3d')

# contour = ax.contour(ds_mat, theta_bs_mat, theta_js.reshape((n_samples, n_samples)))

surf = ax.plot_surface(ds_mat, theta_bs_mat, theta_js.reshape((n_samples, n_samples)),
                       cmap=cm.get_cmap('coolwarm'))

ax.set_xlabel("$d / w$")
ax.set_ylabel("$\\theta_b$")
fig.colorbar(surf)
plt.show()
