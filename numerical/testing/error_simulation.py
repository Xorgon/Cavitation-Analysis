import scipy
import numpy as np
import matplotlib.pyplot as plt

import numerical.util.gen_utils as gen
import numerical.bem as bem
from numerical.models.slot.slot_opt import find_slot_peak
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

fake_sweep = SweepData('test', Y, W, H)
std = 0.015085955056793596
N = 5

all_rand_xs = []
all_rand_theta_js = []
for p_bar, theta_j in zip(xs, theta_js):
    rand_theta_js = np.random.normal(theta_j, std, N)
    for rand_theta_j in rand_theta_js:
        all_rand_xs.append(p_bar)
        all_rand_theta_js.append(rand_theta_j)
        fake_sweep.add_point(p_bar, p_bar, rand_theta_js, Y)

(fit_max_peak_x, fit_max_peak_theta_j, max_poly_coeffs, max_theta_j_std), \
(fit_min_peak_x, fit_min_peak_theta_j, min_poly_coeffs, min_theta_j_std) = fake_sweep.get_curve_fits(range_fact=1.5)

fit_x_star = (fit_max_peak_x - fit_min_peak_x) / 2
fit_theta_j_star = (fit_max_peak_theta_j - fit_min_peak_theta_j) / 2

print(max_theta_j_std, min_theta_j_std)

peak_fit_range = 2.5
max_peak_x = xs[np.argmax(theta_js)]
max_fit_xs = np.linspace(0, peak_fit_range * max_peak_x, 100)
max_fit_ys = np.polyval(max_poly_coeffs, max_fit_xs)

min_peak_x = xs[np.argmin(theta_js)]
min_fit_xs = np.linspace(0, peak_fit_range * min_peak_x, 100)
min_fit_ys = np.polyval(min_poly_coeffs, min_fit_xs)

plt.scatter(all_rand_xs, all_rand_theta_js, marker='.', label="Randomized")
plt.scatter(xs, theta_js, label="Numerical", marker='x')
plt.scatter([x_star, -x_star], [theta_j_star, -theta_j_star], label="Numerical Peaks", color='g', marker='+')
plt.plot(max_fit_xs, max_fit_ys, label="Curve Fits", color='purple')
plt.plot(min_fit_xs, min_fit_ys, color='purple')

plt.axvline(x_star, color='g', linestyle='--')
plt.axvline(fit_x_star, color='r', linestyle='--')
plt.axhline(theta_j_star, color='g', linestyle='--')
plt.axhline(fit_theta_j_star, color='r', linestyle='--')

plt.scatter([fit_max_peak_x, fit_min_peak_x], [fit_max_peak_theta_j, fit_min_peak_theta_j], label="Fitted Peaks", color='k',
            marker='+', s=100)

plt.xlabel("$x$")
plt.ylabel("$\\theta_j$")
plt.legend()
plt.show()
