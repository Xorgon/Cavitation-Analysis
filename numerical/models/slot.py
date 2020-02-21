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
W = 4.2
H = 11.47
Ys = [2.43, 3.43]

x_lim = 5
Xs = np.linspace(-x_lim * W / 2, x_lim * W / 2, 100)

save_to_file = True
calculate_error = False
show_plot = False

if not (save_to_file or show_plot):
    raise ValueError("No output selected for model.")

m_b = 1
n = 20000
density_ratio = 0.25
w_thresh = x_lim

out_dir = "model_outputs/exp_comparisons"

if not os.path.exists(out_dir) and save_to_file:
    os.makedirs(out_dir)

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

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

    xs = Xs / (0.5 * W)
    if show_plot:
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = plt.gca()

        ax.plot(xs, theta_js)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$\\theta_j$")
        ax.axvline(x=-1, linestyle='--', color='gray')
        ax.axvline(x=1, linestyle='--', color='gray')
        plt.show()

    if save_to_file:
        arr = []
        for i in range(len(xs)):
            arr.append([xs[i], theta_js[i]])
        file_path = f"{out_dir}/W{W:.2f}H{H:.2f}Y{Y:.2f}_bem_slot_prediction_{n}_{density_ratio}_{w_thresh}.csv"
        alph = "abcdefgh"
        i = 0
        while os.path.exists(file_path):
            file_path = f"{out_dir}/W{W:.2f}H{H:.2f}Y{Y:.2f}_bem_slot_prediction_{n}_{density_ratio}_{w_thresh}" \
                        f"{alph[i]}.csv"
            i += 1
        file.array_to_csv("", file_path, arr)
