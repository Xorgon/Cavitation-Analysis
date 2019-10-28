# import sys
#
# sys.path.insert(0, '../../')

import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import os
import common.util.file_utils as file
import scipy.sparse

# ps = [8]
w = 2.2
q = 3.68
h = 2.9
ps = np.linspace(-10 * w / 2, 10 * w / 2, 500)

save_to_file = True
full_log = False
plot_fitted_function = False
calculate_error = False

m_0 = 1
n = 5000

if not os.path.exists("model_outputs/slot_ms/{0}".format(n)) and full_log:
    os.makedirs("model_outputs/slot_ms/{0}".format(n))

if not os.path.exists("model_outputs/slot".format(n)) and save_to_file:
    os.makedirs("model_outputs/slot".format(n))

centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# pu.plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
# print(sys.getsizeof(R_matrix))
# R_matrix = scipy.sparse.csc_matrix(R_matrix)
# print(sys.getsizeof(R_matrix))
R_inv = scipy.linalg.inv(R_matrix)

# condition_number = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
# print(f"Condition number = {condition_number}")

theta_js = []
counter = 0
for p in ps:
    # print("Testing p =", p)
    res_vel, sigma, R_b = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv,
                                                    ret_R_b=True)
    theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
    theta_js.append(theta_j)
    # print("        theta_j =", theta_j)

    # if calculate_error:
    #     residual = np.abs(m_0 * R_b + np.dot(R_matrix, sigma))
    #     max_err = condition_number * np.linalg.norm(residual, np.inf) / np.linalg.norm(m_0 * R_b, np.inf)
    #     print(f"        Max res = {np.max(residual):.3e}, Mean res = {np.mean(residual):.3e},"
    #           f" Max err = {max_err:.3e}")
    # pu.plot_3d_points(centroids, sigma)

    # if save_to_file and full_log:
    #     jet_dir_file = "model_outputs/slot_ms/{0}/jet_dir_{1}.txt".format(n, counter)
    #     f = open(jet_dir_file, 'w')
    #     f.write("{0},{1},{2},{3},{4},{5}".format(p, q, 0, res_vel[0], res_vel[1], 0))
    #     f.close()
    #
    #     file_path = "model_outputs/slot_ms/{0}/p_counted_{1}.txt".format(n, counter)
    #     counter += 1
    #     f = open(file_path, 'w')
    #     # f.write("{0:.5f},{1:.5f},{2:.5f},{3}\n".format(p, q, 0, -m_0))
    #     for i in range(len(sinks)):
    #         f.write("{0:.5f},{1:.5f},{2:.5f},{3}\n".format(sinks[i][0], sinks[i][1], sinks[i][2], -m[i]))
    #     f.close()


def fit_function(f_x, a, b, c, d):
    return a * f_x * (c + f_x ** 2) * np.exp(- b * f_x ** 2) + d


fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca()

p_bars = ps / (0.5 * w)

if plot_fitted_function:
    opt, _ = curve_fit(fit_function, p_bars, theta_js, maxfev=5000)
    fit_xs = np.linspace(min(p_bars), max(p_bars), 100)
    fit_ys = fit_function(fit_xs, opt[0], opt[1], opt[2], opt[3])
    ax.plot(fit_xs, fit_ys)

ax.plot(p_bars, theta_js)
ax.set_xlabel("$\\bar{p}$")
ax.set_ylabel("$\\theta_j$")
ax.axvline(x=-1, linestyle='--', color='gray')
ax.axvline(x=1, linestyle='--', color='gray')
plt.show()

# plt.savefig('slot_{0}.png'.format(len(centroids)))
if save_to_file:
    arr = []
    for i in range(len(p_bars)):
        arr.append([p_bars[i], theta_js[i]])
    file.array_to_csv("", f"model_outputs/slot/w{w:.2f}h{h:.2f}q{q:.2f}_bem_slot_prediction_{n}.csv", arr)
