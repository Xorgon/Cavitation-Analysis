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
h = 2
w = 2

centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=100, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

p_bars = np.linspace(-5, 3, 16)
ps = p_bars * 0.5 * w
q = w
points = np.empty((len(ps), 3))
points[:, 0] = ps
points[:, 1] = q
points[:, 2] = 0
vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv)
theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

_, p_bar_star, theta_j_star, _ = find_slot_peak(w, q, h, n, centroids=centroids, normals=normals, areas=areas,
                                                R_inv=R_inv)

std = 0.015085955056793596
N = 5

num_sweeps = 10000

fit_p_bar_stars = []
fit_theta_j_stars = []
for i in range(num_sweeps):
    fake_sweep = SweepData('test', q, w, h)
    all_rand_p_bars = []
    all_rand_theta_js = []
    for p_bar, theta_j in zip(p_bars, theta_js):
        rand_theta_js = np.random.normal(theta_j, std, N)
        for rand_theta_j in rand_theta_js:
            all_rand_p_bars.append(p_bar)
            all_rand_theta_js.append(rand_theta_j)
            fake_sweep.add_point(p_bar, p_bar, rand_theta_js, q)

    try:
        (max_peak_p_bar, max_peak_theta_j, max_poly_coeffs, max_theta_j_err), \
        (min_peak_p_bar, min_peak_theta_j, min_poly_coeffs, min_theta_j_err) = fake_sweep.get_curve_fits_cubic(range_fact=0.5)
    except ValueError:
        continue

    fit_p_bar_star = (max_peak_p_bar - min_peak_p_bar) / 2
    fit_theta_j_star = (max_peak_theta_j - min_peak_theta_j) / 2

    fit_p_bar_stars.append(fit_p_bar_star)
    fit_theta_j_stars.append(fit_theta_j_star)

plt.scatter(fit_p_bar_stars, fit_theta_j_stars, marker='x', color='k', label="Randomized Fitted Peaks")
plt.scatter([p_bar_star], [theta_j_star], label="Numerical Peak", color='r', marker='o')

plt.xlabel("$\\bar{p}$")
plt.ylabel("$\\theta_j$")
plt.legend()

fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].hist(fit_p_bar_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[0].axvline(p_bar_star, color='r')
axes[0].set_xlabel("$\\bar{p}^\\star$")

axes[1].hist(fit_theta_j_stars, color='k', bins=int(np.ceil(num_sweeps / 10)))
axes[1].axvline(theta_j_star, color='r')
axes[1].set_xlabel("$\\theta_j^\\star$")

print(np.std(fit_p_bar_stars))
print(np.std(fit_theta_j_stars))

plt.show()
