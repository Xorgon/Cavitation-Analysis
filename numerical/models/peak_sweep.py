import numpy as np
import scipy
import os

from numerical.models.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem
from numerical.models.peak_plot import plot_peak_sweep

n_points = 16
w = 2
hs = np.linspace(1, 10, n_points)
qs = np.linspace(1, 10, n_points)

print(f"hs = {hs}")
print(f"qs = {qs}")

last_h = None
last_q = None

n = 20000
w_thresh = 6
density_ratio = 0.25

all_h_over_ws = []
all_q_over_ws = []
theta_stars = []
p_bar_stars = []

save_file = open(f"model_outputs/peak_sweep_{n}_{n_points}x{n_points}.csv", 'a')

for h in hs:
    if last_h is not None and (h < last_h or (np.isclose(h, last_h) and np.isclose(last_q, np.max(qs)))):
        continue  # Skip already completed slots.
    centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50,
                                                    w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    print("Requested n = {0}, using n = {1} for h = {2}, w = {3}.".format(n, len(centroids), h, w))
    actual_n = len(centroids)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)
    for q in qs:
        if last_h is not None and last_q is not None and np.isclose(h, last_h) and \
                (q < last_q or np.isclose(q, last_q)):
            continue  # Skip already completed positions in the last completed slot.
        _, p, theta_j, _ = find_slot_peak(w, q, h, actual_n,
                                          varied_slot_density_ratio=density_ratio, density_w_thresh=w_thresh,
                                          centroids=centroids, normals=normals, areas=areas, R_inv=R_inv)
        all_h_over_ws.append(h / w)
        all_q_over_ws.append(q / w)
        theta_stars.append(theta_j)
        p_bar_stars.append(2 * p / w)
        save_file.write(f"{h / w},{q / w},{theta_j},{2 * p / w}\n")

        # Make sure the data is saved to disk every time.
        save_file.flush()
        os.fsync(save_file.fileno())

save_file.close()
