import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file_utils
import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

# ps = [8]
W = 2
Hs = [1, 2, 4, 8, 16]

n_samples = 28

m_0 = 1
n = 20000

for H in Hs:
    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=25, w_thresh=3, density_ratio=0.1)
    print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)

    ds = np.linspace(0, 2 * (W / 2), n_samples)

    theta_b_split = np.pi / 3
    theta_b_lim = (0, np.pi / 2 - 0.1)
    split_ratio = 2  # X:1 ratio
    split_pct = split_ratio / (split_ratio + 1)
    theta_bs = np.concatenate(
        [np.linspace(theta_b_lim[0], theta_b_split * (1 - 1 / np.round(n_samples * (1 - split_pct))),
                     np.round(n_samples * (1 - split_pct)) - 1),
         np.linspace(theta_b_split, theta_b_lim[1], np.round(n_samples * split_pct) + 1)])

    ds_mat, theta_bs_mat = np.meshgrid(ds, theta_bs)

    condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
    condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
    print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")

    points = np.empty((n_samples, n_samples, 3))
    points[:, :, 0] = ds_mat * np.sin(theta_bs_mat)
    points[:, :, 1] = ds_mat * np.cos(theta_bs_mat)
    points[:, :, 2] = 0
    flat_points = points.reshape(-1, 3)

    vels = bem.get_jet_dirs(flat_points, centroids, normals, areas, m_0, R_inv, verbose=True)
    theta_js = np.arctan2(vels[:, 1], vels[:, 0]) + 0.5 * np.pi

    flat_ds = ds_mat.flatten()
    flat_theta_bs = theta_bs_mat.flatten()
    os.makedirs("../model_outputs/slot_alt_param/", exist_ok=True)
    file = open(f"../model_outputs/slot_alt_param/h_{H}_w_{W}_n_{n}.csv", "w")
    file.write("d, theta_b, theta_j\n")
    for d, theta_b, theta_j in zip(flat_ds, flat_theta_bs, theta_js):
        file.write(f"{d}, {theta_b}, {theta_j}\n")
    file.close()
