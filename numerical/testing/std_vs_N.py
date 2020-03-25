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

std = 0.015085955056793596
num_sweeps = 50000
Ns = range(10, 50, 5)
theta_j_star_stds = []
x_star_stds = []
mean_Ns = []
for N in Ns:
    print(N)
    fit_x_stars = []
    fit_theta_j_stars = []
    fit_theta_j_stds = []
    fit_Ns = []
    for i in range(num_sweeps):
        fake_sweep = SweepData('test', Y, W, H)
        all_rand_xs = []
        all_rand_theta_js = []
        for x, theta_j in zip(xs, theta_js):
            rand_theta_js = np.random.normal(theta_j, std, N)
            for rand_theta_j in rand_theta_js:
                all_rand_xs.append(x)
                all_rand_theta_js.append(rand_theta_j)
                fake_sweep.add_point(x, x, rand_theta_j, Y)

        try:
            (max_peak_x, max_peak_theta_j, max_poly_coeffs), \
            (min_peak_x, min_peak_theta_j, min_poly_coeffs), \
            combined_theta_j_star_std, N_points = fake_sweep.get_curve_fits(range_fact=1.5)
        except ValueError:
            continue

        fit_Ns.append(N_points)
        fit_x_star = (max_peak_x - min_peak_x) / 2
        fit_theta_j_star = (max_peak_theta_j - min_peak_theta_j) / 2
        # fit_p_bar_star = max_peak_x
        # fit_theta_j_star = max_peak_theta_j

        fit_x_stars.append(fit_x_star)
        fit_theta_j_stars.append(fit_theta_j_star)

        fit_theta_j_stds.append(combined_theta_j_star_std)
        # fit_theta_j_stds.append((max_theta_j_std + min_theta_j_std) / 2)
        # fit_x_star_stds.append((max_x_star_std + min_x_star_std) / 2)
        # fit_theta_j_stds.append(max_theta_j_std)
        # fit_x_star_stds.append(max_x_star_std)
    mean_Ns.append(np.mean(fit_Ns))
    theta_j_star_stds.append(np.std(fit_theta_j_stars))
    x_star_stds.append(np.std(fit_x_stars))

print(np.polyfit(np.log(mean_Ns), np.log(np.divide(theta_j_star_stds, std)), 1))
print(np.polyfit(np.log(mean_Ns), np.log(np.divide(x_star_stds, std)), 1))

plt.plot(mean_Ns, theta_j_star_stds, 'C0')
# plt.plot(np.multiply(Ns, 4), np.multiply(theta_j_star_stds, np.sqrt(np.multiply(Ns, 4))) / std)
plt.ylabel("$\\theta_j^\\star \\sigma$", color='C0')
plt.xlabel("$N$ - Samples per position")
plt.xscale('log')
plt.yscale('log')
plt.twinx()
plt.plot(mean_Ns, x_star_stds, 'C1')
# plt.plot(np.multiply(Ns, 4), np.multiply(x_star_stds, np.sqrt(np.multiply(Ns, 4))) / (25 * std))
plt.ylabel("$x^\\star \\sigma$", color='C1')
plt.yscale('log')
plt.tight_layout()
plt.show()
