import scipy
import numpy as np
import matplotlib.pyplot as plt

import numerical.util.gen_utils as gen
import numerical.bem as bem
from numerical.models.slot.slot_opt import find_slot_peak
from experimental.plotting.analyse_slot import SweepData

m_0 = 1
n = 10000
w_thresh = 15
density_ratio = 0.15
H = 2
W = 2

centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=100, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

xs = np.linspace(-5, 3, 32)
Xs = xs * 0.5 * W
Y = W * 0.25
points = np.empty((len(Xs), 3))
points[:, 0] = Xs
points[:, 1] = Y
points[:, 2] = 0
vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv)
theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

_, x_star, theta_j_star, _ = find_slot_peak(W, Y, H, n, centroids=centroids, normals=normals, areas=areas,
                                            R_inv=R_inv)

std = 0.015085955056793596
N = 10
num_sweeps = 100000

fit_x_stars = []
fit_theta_stars = []
fit_theta_stds = []
max_mean_xs = []
for i in range(num_sweeps):
    if i % 100 == 0:
        print(f"{100 * i / num_sweeps}%")
    fake_sweep = SweepData('test', Y, W, H)
    all_rand_xs = []
    all_rand_theta_js = []
    max_mean_x = 0
    max_mean_theta_j = 0
    for x, theta_j in zip(xs, theta_js):
        rand_theta_js = np.random.normal(theta_j, std, N)
        if np.mean(rand_theta_js) > max_mean_theta_j:
            max_mean_theta_j = np.mean(rand_theta_js)
            max_mean_x = x
        for rand_theta_j in rand_theta_js:
            all_rand_xs.append(x)
            all_rand_theta_js.append(rand_theta_j)
            fake_sweep.add_point(x, x, rand_theta_j, Y)
    try:
        # fit_x_star, fit_theta_star, _, _ = num_prediction_fit(all_rand_xs, all_rand_theta_js, std)
        (max_peak_x, max_peak_theta_j, max_poly_coeffs), \
        (min_peak_x, min_peak_theta_j, min_poly_coeffs), \
        combined_theta_j_std,  _ = fake_sweep.get_curve_fits(range_fact=1.5)
        # fake_sweep.get_curve_fits(range_fact=1.75 * 1.117647058823529 / max_mean_x)
    except ValueError:
        continue

    if np.abs((max_peak_x - min_peak_x) / 2) > 2 * Y:
        continue

    max_mean_xs.append(max_mean_x)
    fit_x_star = (max_peak_x - min_peak_x) / 2
    fit_theta_star = (max_peak_theta_j - min_peak_theta_j) / 2

    fit_x_stars.append(fit_x_star)
    fit_theta_stars.append(fit_theta_star)

    fit_theta_stds.append(combined_theta_j_std)
    # fit_theta_stds.append(max_theta_j_std)
    # fit_x_star_stds.append(max_x_star_std)
plt.scatter(fit_x_stars, fit_theta_stars, marker='.', c=max_mean_xs, label="Randomized Fitted Peaks")
plt.scatter([x_star], [theta_j_star], label="Numerical Peak", color='r', marker='o')

print(f"Mean error: {np.mean(fit_x_stars) - x_star}")

plt.xlabel("$x$")
plt.ylabel("$\\theta_j$ (rad)")
plt.legend()

confidence_interval = 0.99

fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].hist(fit_x_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[0].axvline(x_star, color='r')
# x_star_int = stats.norm.interval(confidence_interval, loc=np.mean(fit_x_stars), scale=np.mean(fit_x_star_stds))
# axes[0].axvline(x_star_int[0], color='g', label="Estimated")
# axes[0].axvline(x_star_int[1], color='g')
axes[0].axvline(np.mean(fit_x_stars) + 0.075, color='g', label="Estimated")
axes[0].axvline(np.mean(fit_x_stars) - 0.075, color='g')
# x_star_int = stats.norm.interval(confidence_interval, loc=np.mean(fit_x_stars), scale=np.std(fit_x_stars))
# axes[0].axvline(x_star_int[0], color='b', label="Measured")
# axes[0].axvline(x_star_int[1], color='b')
axes[0].axvline(np.mean(fit_x_stars) + np.std(fit_x_stars), color='b', label="Measured")
axes[0].axvline(np.mean(fit_x_stars) - np.std(fit_x_stars), color='b')
axes[0].set_xlabel("$x^\\star$")
axes[0].legend()

axes[1].hist(fit_theta_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[1].axvline(theta_j_star, color='r')
# theta_j_star_int = stats.norm.interval(confidence_interval, loc=np.mean(fit_theta_j_stars),
#                                        scale=np.mean(fit_theta_j_stds))
# axes[1].axvline(theta_j_star_int[0], color='g')
# axes[1].axvline(theta_j_star_int[1], color='g')
axes[1].axvline(np.mean(fit_theta_stars) + np.mean(fit_theta_stds), color='g')
axes[1].axvline(np.mean(fit_theta_stars) - np.mean(fit_theta_stds), color='g')
# theta_j_star_int = stats.norm.interval(confidence_interval, loc=np.mean(fit_theta_j_stars),
#                                        scale=np.std(fit_theta_j_stars))
# axes[1].axvline(theta_j_star_int[0], color='b')
# axes[1].axvline(theta_j_star_int[1], color='b')
axes[1].axvline(np.mean(fit_theta_stars) + np.std(fit_theta_stars), color='b')
axes[1].axvline(np.mean(fit_theta_stars - np.std(fit_theta_stars)), color='b')
axes[1].set_xlabel("$\\theta_j^\\star$ (rad)")

# print(f"x_star: measured = {np.std(fit_x_stars)}, estimated = {np.mean(fit_x_star_stds)}")
# print(np.std(fit_x_star_stds))
# print(f"theta_j_star: measured = {np.std(fit_theta_stars)}, estimated = {np.mean(fit_theta_j_stds)}")
# print(np.std(fit_theta_j_stds))

plt.tight_layout()
plt.show()
