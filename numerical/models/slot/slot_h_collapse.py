import os

import numpy as np

import numerical.bem as bem
import numerical.util.gen_utils as gen
from common.util.file_utils import lists_to_csv
from common.util.plotting_utils import initialize_plt

initialize_plt()

Hs = np.linspace(1, 5, 5)
W = 2
Xs = np.linspace(-3 * W, 3 * W, 300)
Y = 2

normalize = True

m_0 = 1
n = 20000

xs = Xs / (0.5 * W)

for i, H in enumerate(Hs):
    print(f"Testing H = {H}")
    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=12, density_ratio=0.25)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    print(np.mean(centroids, 0))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = np.linalg.inv(R_matrix)

    points = np.empty((len(Xs), 3))
    points[:, 0] = Xs
    points[:, 1] = Y
    points[:, 2] = 0
    vels = bem.get_jet_dirs(points, centroids, normals, areas, m_0, R_inv, verbose=True)
    thetas = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

    f_dir = "../model_outputs/slot_h_collapse/"
    f_name = f"h_collapse_n{n}_H{H}.csv"
    if not os.path.exists(f_dir + f_name):
        lists_to_csv(f_dir, f_name, [xs, thetas], ["x", "theta"])
