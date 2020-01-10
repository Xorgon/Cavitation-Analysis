import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp

from common.util.plotting_utils import initialize_plt


def plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat, plot_3d=False):
    if plot_3d:
        theta_star_mat_no_nan = np.nan_to_num(theta_star_mat)
        tsm_interp = interp.interp2d(h_over_w_mat, q_over_w_mat, theta_star_mat_no_nan, kind='linear', copy=False)
        points_fact = 10
        how_new = np.linspace(np.min(h_over_w_mat), np.max(h_over_w_mat), h_over_w_mat.shape[0] * points_fact)
        qow_new = np.linspace(np.min(q_over_w_mat), np.max(q_over_w_mat), q_over_w_mat.shape[0] * points_fact)
        how_new_mat, qow_new_mat = np.meshgrid(how_new, qow_new)
        tsm_new = tsm_interp(how_new, qow_new)

        initialize_plt(font_size=14, line_scale=2)
        theta_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        theta_star_surf = ax.plot_surface(how_new_mat, qow_new_mat, tsm_new, cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$h / w$")
        ax.set_ylabel("$q / w$")
        ax.set_zlabel("$\\theta_j^\\star$")
        theta_star_surf.set_clim(np.nanmin(theta_star_mat), np.nanmax(theta_star_mat))
        theta_star_fig.colorbar(theta_star_surf)

        log_theta_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        log_theta_star_surf = ax.plot_surface(np.log10(h_over_w_mat), np.log10(q_over_w_mat), np.log10(theta_star_mat),
                                          cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$log_{10}(h / w)$")
        ax.set_ylabel("$log_{10}(q / w)$")
        ax.set_zlabel("$log_{10}(\\theta_j^\\star)$")
        log_theta_star_surf.set_clim(np.nanmin(theta_star_mat), np.nanmax(theta_star_mat))
        log_theta_star_fig.colorbar(log_theta_star_surf)

        p_bar_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        p_bar_star_surf = ax.plot_surface(h_over_w_mat, q_over_w_mat, p_bar_star_mat, cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$h / w$")
        ax.set_ylabel("$q / w$")
        ax.set_zlabel("$\\bar{p}^\\star$")
        p_bar_star_surf.set_clim(np.nanmin(p_bar_star_mat), np.nanmax(p_bar_star_mat))
        p_bar_star_fig.colorbar(p_bar_star_surf)

    initialize_plt()
    fig, axes = plt.subplots(1, 2, figsize=(5.31445, 3.5))
    plt.sca(axes[0])
    cnt = plt.contourf(h_over_w_mat, q_over_w_mat, theta_star_mat, levels=16)
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h / w$")
    plt.ylabel("$q / w$")
    cbar = plt.colorbar(label="$\\theta_j^\\star$", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    plt.sca(axes[1])
    cnt = plt.contourf(h_over_w_mat, q_over_w_mat, p_bar_star_mat, levels=16)
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h / w$")
    plt.ylabel("$q / w$")
    cbar = plt.colorbar(label="$\\bar{p}^\\star$", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    plt.tight_layout()
    # plt.show()

    initialize_plt(font_size=14, line_scale=2)
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(q_over_w_mat, theta_star_mat, c=h_over_w_mat)
    plt.xlabel('$q / w$')
    plt.ylabel('$\\theta_j^\\star$')
    plt.colorbar(label="$h / w$")

    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(h_over_w_mat, theta_star_mat, c=q_over_w_mat)
    plt.xlabel('$h / w$')
    plt.ylabel('$\\theta_j^\\star$')
    plt.colorbar(label="$q / w$")

    plt.show()


if __name__ == "__main__":
    n = 20000
    n_points = 16
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

    plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat, plot_3d=True)