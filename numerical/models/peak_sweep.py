import numpy as np
import scipy

from numerical.models.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem
from numerical.models.peak_plot import plot_peak_sweep

# TODO: Run this next

n_points = 16
w = 2
hs = np.linspace(1, 10, n_points)
qs = np.linspace(1, 10, n_points)

n = 15000
w_thresh = 6
density_ratio = 0.25

all_h_over_ws = []
all_q_over_ws = []
theta_stars = []
p_bar_stars = []

save_file = open(f"model_outputs/peak_sweep_{n}_{n_points}x{n_points}.csv", 'a')

for h in hs:
    centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50,
                                                    w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    n = len(centroids)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)
    for q in qs:
        _, p, theta_j, _ = find_slot_peak(w, q, h, n,
                                          varied_slot_density_ratio=density_ratio, density_w_thresh=w_thresh,
                                          centroids=centroids, normals=normals, areas=areas, R_inv=R_inv)
        all_h_over_ws.append(h / w)
        all_q_over_ws.append(q / w)
        theta_stars.append(theta_j)
        p_bar_stars.append(2 * p / w)
        save_file.write(f"{h / w},{q / w},{theta_j},{2 * p / w}\n")

save_file.close()

h_over_w_mat = np.reshape(all_h_over_ws, (n_points, n_points))
q_over_w_mat = np.reshape(all_q_over_ws, (n_points, n_points))
theta_star_mat = np.reshape(theta_stars, (n_points, n_points))
p_bar_star_mat = np.reshape(p_bar_stars, (n_points, n_points))

plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat)
