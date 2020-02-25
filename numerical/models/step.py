import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse

# Xs = [8]
H = 2
Ys = [1, 2, 4, 8]

x_lim = 15
Xs = np.linspace(-x_lim, x_lim, 200)

calculate_error = False

m_b = 1
n = 15000
density_ratio = 0.25
thresh_dist = x_lim

out_dir = "model_outputs/exp_comparisons"

centroids, normals, areas = gen.gen_varied_step(n=n, H=H, length=100, depth=50, thresh_dist=thresh_dist * 1.5,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca()
for Y in Ys:
    if calculate_error:
        theta_js = []
        for X in Xs:
            print("Testing X =", X)
            R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
            res_vel, sigma = bem.get_jet_dir_and_sigma([X, Y, 0], centroids, normals, areas, m_0=m_b, R_inv=R_inv,
                                                       R_b=R_b)
            theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
            theta_js.append(theta_j)
            print("        theta_j =", theta_j)

            residual = np.abs(m_b * R_b + np.dot(R_matrix, sigma))
            max_err = condition_number_inf * np.linalg.norm(residual, np.inf) / np.linalg.norm(m_b * R_b, np.inf)
            print(f"        Max res = {np.max(residual):.3e}, Mean res = {np.mean(residual):.3e},"
                  f" Max err = {max_err:.3e}")
    else:
        points = np.empty((len(Xs), 3))
        points[:, 0] = Xs
        points[:, 1] = Y
        points[:, 2] = 0
        vels = bem.get_jet_dirs(points, centroids, normals, areas, m_b, R_inv, verbose=True)
        theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

    ax.plot(Xs, theta_js, label=f"$Y={Y}$")

ax.set_xlabel("$X$")
ax.set_ylabel("$\\theta_j$")
ax.axvline(x=0, linestyle='--', color='gray')
ax.legend()
plt.show()
