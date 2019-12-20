import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from common.util.plotting_utils import initialize_plt


def plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat):
    # TODO: Fix colours on these plots.
    # theta_star_mat = np.nan_to_num(theta_star_mat)
    # p_bar_star_mat = np.nan_to_num(p_bar_star_mat)
    initialize_plt(font_size=14, line_scale=2)
    theta_star_fig = plt.figure()
    ax = plt.gca(projection='3d')  # type: Axes3D
    theta_star_surf = ax.plot_surface(h_over_w_mat, q_over_w_mat, theta_star_mat)
    ax.set_xlabel("$h / w$")
    ax.set_ylabel("$q / w$")
    ax.set_zlabel("$\\theta_j^\\star$")
    theta_star_fig.colorbar(theta_star_surf)
    plt.show()

    p_bar_star_fig = plt.figure()
    ax = plt.gca(projection='3d')  # type: Axes3D
    p_bar_star_surf = ax.plot_surface(h_over_w_mat, q_over_w_mat, p_bar_star_mat)
    ax.set_xlabel("$h / w$")
    ax.set_ylabel("$q / w$")
    ax.set_zlabel("$\\bar{p}^\\star$")
    p_bar_star_fig.colorbar(p_bar_star_surf)
    plt.show()


if __name__ == "__main__":
    n = 10000
    n_points = 4
    save_file = open(f"model_outputs/peak_sweep_{n}_{n_points}x{n_points}.csv", 'r')

    all_h_over_ws = []
    all_q_over_ws = []
    theta_stars = []
    p_bar_stars = []
    for line in save_file.readlines():
        split = line.split(',')
        all_h_over_ws.append(float(split[0]))
        all_q_over_ws.append(float(split[1]))
        theta_stars.append(float(split[2]))
        p_bar_stars.append(float(split[3]))

    h_over_w_mat = np.reshape(all_h_over_ws, (n_points, n_points))
    q_over_w_mat = np.reshape(all_q_over_ws, (n_points, n_points))
    theta_star_mat = np.reshape(theta_stars, (n_points, n_points))
    p_bar_star_mat = np.reshape(p_bar_stars, (n_points, n_points))

    plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat)
