import os

import numpy as np

import numerical.bem as bem
import numerical.util.gen_utils as gen
from common.util.file_utils import lists_to_csv

H = 2
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Ys = np.linspace(1, 5, 5)

m_0 = 1
n = 20000

xs = Xs / (0.5 * W)

centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=12, density_ratio=0.25)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
print(np.mean(centroids, 0))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = np.linalg.inv(R_matrix)
for i, Y in enumerate(Ys):
    print(f"Testing Y = {Y}")
    points = np.empty((len(Xs), 3))
    points[:, 0] = Xs
    points[:, 1] = Y
    points[:, 2] = 0
    vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv, verbose=True)
    thetas = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

    x_to_plot = xs
    theta_to_plot = thetas

    theta_star, x_star = sorted(zip(thetas, xs), key=lambda k: k[0])[-1]
    norm_x_to_plot = np.divide(xs, x_star)
    norm_theta_to_plot = np.divide(thetas, theta_star)

    f_dir = "model_outputs/slot_y_collapse/"
    f_name = f"y_collapse_n{n}_Y{Y}.csv"
    if not os.path.exists(f_dir + f_name):
        lists_to_csv(f_dir, f_name, [xs, thetas], ["x", "theta"])
