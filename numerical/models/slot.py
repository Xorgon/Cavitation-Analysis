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

# ps = [8]
w = 1.26
h = 3.00
q = 1.89

p_bar_lim = 12
ps = np.linspace(-p_bar_lim * w / 2, p_bar_lim * w / 2, 100)

save_to_file = True
calculate_error = True

m_0 = 1
n = 20000
density_ratio = 0.25
w_thresh = p_bar_lim

if not os.path.exists("model_outputs/slot") and save_to_file:
    os.makedirs("model_outputs/slot")

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

if calculate_error:
    theta_js = []
    for p in ps:
        print("Testing p =", p)
        R_b = bem.get_R_vector([p, q, 0], centroids, normals)
        res_vel, sigma = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
        theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
        theta_js.append(theta_j)
        print("        theta_j =", theta_j)

        residual = np.abs(m_0 * R_b + np.dot(R_matrix, sigma))
        max_err = condition_number_inf * np.linalg.norm(residual, np.inf) / np.linalg.norm(m_0 * R_b, np.inf)
        print(f"        Max res = {np.max(residual):.3e}, Mean res = {np.mean(residual):.3e},"
              f" Max err = {max_err:.3e}")
else:
    points = np.empty((len(ps), 3))
    points[:, 0] = ps
    points[:, 1] = q
    points[:, 2] = 0
    vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv, verbose=True)
    theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca()

p_bars = ps / (0.5 * w)

ax.plot(p_bars, theta_js)
ax.set_xlabel("$\\bar{p}$")
ax.set_ylabel("$\\theta_j$")
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
plt.show()

if save_to_file:
    arr = []
    for i in range(len(p_bars)):
        arr.append([p_bars[i], theta_js[i]])
    file_path = f"model_outputs/slot/w{w:.2f}h{h:.2f}q{q:.2f}_bem_slot_prediction_{n}_{density_ratio}_{w_thresh}.csv"
    alph = "abcdefgh"
    i = 0
    while os.path.exists(file_path):
        file_path = f"model_outputs/slot/w{w:.2f}h{h:.2f}q{q:.2f}_bem_slot_prediction_{n}_{density_ratio}_{w_thresh}" \
                    f"{alph[i]}.csv"
        i += 1
    file.array_to_csv("", file_path, arr)
