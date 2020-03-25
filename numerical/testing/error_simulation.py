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

xs = np.linspace(-12, 8, 32)
Xs = xs * 0.5 * W
Y = W * 1.5
points = np.empty((len(Xs), 3))
points[:, 0] = Xs
points[:, 1] = Y
points[:, 2] = 0
vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv)
thetas = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

_, x_star, theta_j_star, _ = find_slot_peak(W, Y, H, n, centroids=centroids, normals=normals, areas=areas,
                                            R_inv=R_inv)

fake_sweep = SweepData('test', Y, W, H)
std = 0.015085955056793596
N = 10

all_rand_xs = []
all_rand_theta_js = []
for p_bar, theta_j in zip(xs, thetas):
    rand_theta_js = np.random.normal(theta_j, std, N)
    for rand_theta_j in rand_theta_js:
        all_rand_xs.append(p_bar)
        all_rand_theta_js.append(rand_theta_j)
        fake_sweep.add_point(p_bar, p_bar, rand_theta_j, Y)

peak_fit_range = 1.5
(fit_max_peak_x, fit_max_peak_theta_j, max_poly_coeffs), \
(fit_min_peak_x, fit_min_peak_theta_j, min_poly_coeffs), \
combined_tj_std, _ \
    = fake_sweep.get_curve_fits(range_fact=peak_fit_range)
#
fit_x_star = (fit_max_peak_x - fit_min_peak_x) / 2
fit_theta_star = (fit_max_peak_theta_j - fit_min_peak_theta_j) / 2
#
# fit_x_star, fit_theta_star, x_offset, theta_offset = num_prediction_fit(xs, thetas, std)
# num_interp = get_num_prediction_function()
#
# print(x_offset, theta_offset)

# print(max_theta_j_std, min_theta_j_std)

max_peak_x = xs[np.argmax(thetas)]
max_fit_xs = np.linspace(0, peak_fit_range * max_peak_x, 100)
max_fit_ys = np.polyval(max_poly_coeffs, max_fit_xs)

# min_peak_x = xs[np.argmin(thetas)]
# min_fit_xs = np.linspace(0, peak_fit_range * min_peak_x, 100)
# min_fit_ys = np.polyval(min_poly_coeffs, min_fit_xs)

# fit_xs = np.linspace(min(xs), max(xs), 500)
# fit_ys = num_interp(fit_xs, fit_x_star, fit_theta_star, x_offset, theta_offset)

plt.scatter(all_rand_xs, all_rand_theta_js, marker='.', label="Randomized")
plt.scatter(xs, thetas, label="Numerical", marker='x')
plt.scatter([x_star, -x_star], [theta_j_star, -theta_j_star], label="Numerical Peaks", color='g', marker='+')
plt.plot(max_fit_xs, max_fit_ys, label="Curve Fit", color='purple')
confidence_interval = 0.99
# plt.plot(max_fit_xs, np.add(max_fit_ys, stats.norm.interval(confidence_interval, loc=0, scale=max_theta_j_std)[0]),
#          color='gray', linestyle='--')
# plt.plot(max_fit_xs, np.add(max_fit_ys, stats.norm.interval(confidence_interval, loc=0, scale=max_theta_j_std)[1]),
#          color='gray', linestyle='--')
# plt.plot(np.add(max_fit_xs, stats.norm.interval(confidence_interval, loc=0, scale=max_x_star_std)[0]), max_fit_ys,
#          color='gray', linestyle='--')
# plt.plot(np.add(max_fit_xs, stats.norm.interval(confidence_interval, loc=0, scale=max_x_star_std)[1]), max_fit_ys,
#          color='gray', linestyle='--')
# plt.plot(min_fit_xs, min_fit_ys, color='purple')

# plt.axvline(x_star, color='g', linestyle='--')
# plt.axvline(fit_x_star, color='r', linestyle='--')
# plt.axhline(theta_j_star, color='g', linestyle='--')
# plt.axhline(fit_theta_j_star, color='r', linestyle='--')

# plt.errorbar([fit_max_peak_x, fit_min_peak_x], [fit_max_peak_theta_j, fit_min_peak_theta_j], label="Fitted Peaks",
#              yerr=[stats.norm.interval(confidence_interval, loc=0, scale=max_theta_j_std)[0],
#                    stats.norm.interval(confidence_interval, loc=0, scale=min_theta_j_std)[0]],
#              xerr=[stats.norm.interval(confidence_interval, loc=0, scale=max_x_star_std)[0],
#                    stats.norm.interval(confidence_interval, loc=0, scale=min_x_star_std)[0]],
#              color='k', capsize=3, fmt='.')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("$x$")
plt.ylabel("$\\theta_j$ (rad)")
plt.legend()
plt.show()
