import scipy
import numpy as np
import matplotlib.pyplot as plt

import numerical.util.gen_utils as gen
import numerical.bem as bem
from numerical.models.slot_opt import find_slot_peak
from experimental.plotting.analyse_slot import SweepData

m_0 = 1
n = 5000
w_thresh = 15
density_ratio = 0.15
H = 2
W = 2

centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=100, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

xs = np.linspace(-5, 3, 16)
Xs = xs * 0.5 * W
Y = W
points = np.empty((len(Xs), 3))
points[:, 0] = Xs
points[:, 1] = Y
points[:, 2] = 0
vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv)
theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

_, x_star, theta_j_star, _ = find_slot_peak(W, Y, H, n, centroids=centroids, normals=normals, areas=areas,
                                            R_inv=R_inv)

std = 0.015085955056793596
N = 5

num_sweeps = 10000

fit_x_stars = []
fit_theta_j_stars = []
for i in range(num_sweeps):
    fake_sweep = SweepData('test', Y, W, H)
    all_rand_xs = []
    all_rand_theta_js = []
    for x, theta_j in zip(xs, theta_js):
        rand_theta_js = np.random.normal(theta_j, std, N)
        for rand_theta_j in rand_theta_js:
            all_rand_xs.append(x)
            all_rand_theta_js.append(rand_theta_j)
            fake_sweep.add_point(x, x, rand_theta_js, Y)

    try:
        (max_peak_x, max_peak_theta_j, max_poly_coeffs, max_theta_j_err), \
        (min_peak_x, min_peak_theta_j, min_poly_coeffs, min_theta_j_err) = fake_sweep.get_curve_fits(range_fact=0.5)
    except ValueError:
        continue

    fit_p_bar_star = (max_peak_x - min_peak_x) / 2
    fit_theta_j_star = (max_peak_theta_j - min_peak_theta_j) / 2

    fit_x_stars.append(fit_p_bar_star)
    fit_theta_j_stars.append(fit_theta_j_star)

plt.scatter(fit_x_stars, fit_theta_j_stars, marker='x', color='k', label="Randomized Fitted Peaks")
plt.scatter([x_star], [theta_j_star], label="Numerical Peak", color='r', marker='o')

plt.xlabel("$x$")
plt.ylabel("$\\theta_j$")
plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].hist(fit_x_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[0].axvline(x_star, color='r')
axes[0].set_xlabel("$x^\\star$")

axes[1].hist(fit_theta_j_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[1].axvline(theta_j_star, color='r')
axes[1].set_xlabel("$\\theta_j^\\star$")

print(np.std(fit_x_stars))
print(np.std(fit_theta_j_stars))

plt.show()
