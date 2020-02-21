import numpy as np
import scipy
import os

from numerical.models.slot_opt import find_slot_peak
import numerical.util.gen_utils as gen
import numerical.bem as bem

n_points = 16
W = 2
Hs = np.linspace(1, 10, n_points)
Ys = np.linspace(1, 10, n_points)

print(f"Hs = {Hs}")
print(f"Ys = {Ys}")

last_H = None
last_Y = None

n = 20000
w_thresh = 15
density_ratio = 0.15

save_file = open(f"model_outputs/peak_sweep_{n}_{n_points}x{n_points}_plate_500.csv", 'a')

for H in Hs:
    if last_H is not None and (H < last_H or (np.isclose(H, last_H) and np.isclose(last_Y, np.max(Ys)))):
        continue  # Skip already completed slots.
    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=500, depth=50,
                                                    w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    print("Requested n = {0}, using n = {1} for H = {2}, W = {3}.".format(n, len(centroids), H, W))
    actual_n = len(centroids)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)
    for Y in Ys:
        if last_H is not None and last_Y is not None and np.isclose(H, last_H) and \
                (Y < last_Y or np.isclose(Y, last_Y)):
            continue  # Skip already completed positions in the last completed slot.
        _, X, theta_j, _ = find_slot_peak(W, Y, H, actual_n,
                                          varied_slot_density_ratio=density_ratio, density_w_thresh=w_thresh,
                                          centroids=centroids, normals=normals, areas=areas, R_inv=R_inv)

        center_x = 0.05
        near_center_vel, _ = bem.get_jet_dir_and_sigma([center_x, Y, 0], centroids, normals, areas, R_inv=R_inv)
        center_grad = np.arctan2(near_center_vel[1], near_center_vel[0]) + np.pi / 2

        save_file.write(f"{H / W},{Y / W},{theta_j},{2 * X / W},{center_grad / center_x}\n")

        # Make sure the data is saved to disk every time.
        save_file.flush()
        os.fsync(save_file.fileno())

save_file.close()
