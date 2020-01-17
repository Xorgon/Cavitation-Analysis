import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp

from common.util.plotting_utils import initialize_plt


def plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat, theta_grad_mat,
                    plot_3d=False, plot_logs=False, how_range=None, qow_range=None):
    if plot_3d:
        theta_star_mat_no_nan = np.nan_to_num(theta_star_mat)
        tsm_interp = interp.interp2d(h_over_w_mat, q_over_w_mat, theta_star_mat_no_nan, kind='linear', copy=False)
        points_fact = 1
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
    plt.xticks(range(1, 5 + 1))
    plt.yticks(range(1, 5 + 1))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h / w$")
    plt.ylabel("$q / w$")
    cbar = plt.colorbar(label="$\\theta_j^\\star$", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    plt.sca(axes[1])
    cnt = plt.contourf(h_over_w_mat, q_over_w_mat, p_bar_star_mat, levels=16)
    plt.xticks(range(1, 5 + 1))
    plt.yticks(range(1, 5 + 1))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h / w$")
    plt.ylabel("$q / w$")
    cbar = plt.colorbar(label="$\\bar{p}^\\star$", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    for ax in axes:
        if how_range is not None:
            ax.set_xlim(how_range)
        if qow_range is not None:
            ax.set_ylim(qow_range)

    plt.tight_layout()
    # plt.show()

    initialize_plt(font_size=18, line_scale=2)
    plt.figure()
    cnt = plt.contourf(h_over_w_mat, q_over_w_mat, theta_grad_mat * p_bar_star_mat / theta_star_mat, levels=16)
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h / w$")
    plt.ylabel("$q / w$")
    if how_range is not None:
        plt.gca().set_xlim(how_range)
    if qow_range is not None:
        plt.gca().set_ylim(qow_range)
    plt.colorbar(label="$\\frac{\\theta_j'(0)}{\\theta_j^\\star/\\bar{p}^\\star}$")
    plt.tight_layout()

    plt.figure()
    scat = plt.scatter(p_bar_star_mat, q_over_w_mat, c=h_over_w_mat)
    plt.colorbar(label='$h / w$')
    for p_bars, qows, hows in zip(p_bar_star_mat, q_over_w_mat, h_over_w_mat):
        plt.plot(p_bars, qows, color=scat.to_rgba(hows[0]))
    plt.xlabel('$\\bar{p}$')
    plt.ylabel('$q / w$')

    if plot_logs:
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

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(q_over_w_mat, p_bar_star_mat, c=h_over_w_mat)
        plt.xlabel('$q / w$')
        plt.ylabel('$\\bar{p}^\\star$')
        plt.colorbar(label="$h / w$")

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(h_over_w_mat, p_bar_star_mat, c=q_over_w_mat)
        plt.xlabel('$h / w$')
        plt.ylabel('$\\bar{p}^\\star$')
        plt.colorbar(label="$q / w$")

    plt.show()


if __name__ == "__main__":
    n = 15000
    n_points = 16
    save_file = open(f"model_outputs/peak_sweep_{n}_{n_points}x{n_points}.csv", 'r')

    all_h_over_ws = []
    all_q_over_ws = []
    theta_stars = []
    p_bar_stars = []
    theta_grads = []
    for line in save_file.readlines():
        split = line.split(',')
        all_h_over_ws.append(float(split[0]))
        all_q_over_ws.append(float(split[1]))
        theta_stars.append(float(split[2]))
        p_bar_stars.append(float(split[3]))
        theta_grads.append(float(split[4]))

    if len(all_h_over_ws) == n_points ** 2:
        h_over_w_mat = np.reshape(all_h_over_ws, (n_points, n_points))
        q_over_w_mat = np.reshape(all_q_over_ws, (n_points, n_points))
        theta_star_mat = np.reshape(theta_stars, (n_points, n_points))
        p_bar_star_mat = np.reshape(p_bar_stars, (n_points, n_points))
        theta_grad_mat = np.reshape(theta_grads, (n_points, n_points))
    else:
        h_over_w_mat = np.empty((n_points, n_points))
        h_over_w_mat.fill(np.nan)
        q_over_w_mat = np.empty((n_points, n_points))
        q_over_w_mat.fill(np.nan)
        theta_star_mat = np.empty((n_points, n_points))
        theta_star_mat.fill(np.nan)
        p_bar_star_mat = np.empty((n_points, n_points))
        p_bar_star_mat.fill(np.nan)
        theta_grad_mat = np.empty((n_points, n_points))
        theta_grad_mat.fill(np.nan)
        for k in range(len(all_h_over_ws)):
            i = int(np.floor(k / n_points))
            j = int(k % n_points)
            h_over_w_mat[i, j] = all_h_over_ws[k]
            q_over_w_mat[i, j] = all_q_over_ws[k]
            theta_star_mat[i, j] = theta_stars[k]
            p_bar_star_mat[i, j] = p_bar_stars[k]
            theta_grad_mat[i, j] = theta_grads[k]

    plot_peak_sweep(h_over_w_mat, q_over_w_mat, theta_star_mat, p_bar_star_mat, theta_grad_mat,
                    plot_3d=False, plot_logs=False, how_range=(0.5, 5), qow_range=(0.5, 5))
