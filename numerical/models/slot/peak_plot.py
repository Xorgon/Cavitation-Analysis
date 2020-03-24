import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp

from common.util.plotting_utils import initialize_plt


def plot_peak_sweep(h_mat, y_mat, theta_star_mat, x_star_mat, theta_grad_mat,
                    plot_3d=False, plot_logs=False, h_range=None, y_range=None):
    if plot_3d:
        theta_star_mat_no_nan = np.nan_to_num(theta_star_mat)
        tsm_interp = interp.interp2d(h_mat, y_mat, theta_star_mat_no_nan, kind='linear', copy=False)
        points_fact = 1
        h_new = np.linspace(np.min(h_mat), np.max(h_mat), h_mat.shape[0] * points_fact)
        y_new = np.linspace(np.min(y_mat), np.max(y_mat), y_mat.shape[0] * points_fact)
        h_new_mat, y_new_mat = np.meshgrid(h_new, y_new)
        tsm_new = tsm_interp(h_new, y_new)

        initialize_plt(font_size=14, line_scale=2)
        theta_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        theta_star_surf = ax.plot_surface(h_new_mat, y_new_mat, tsm_new, cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$h$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$\\theta^\\star$ (rad)")
        theta_star_surf.set_clim(np.nanmin(theta_star_mat), np.nanmax(theta_star_mat))
        theta_star_fig.colorbar(theta_star_surf)

        log_theta_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        log_theta_star_surf = ax.plot_surface(np.log10(h_mat), np.log10(y_mat), np.log10(theta_star_mat),
                                              cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$log_{10}(h)$")
        ax.set_ylabel("$log_{10}(y)$")
        ax.set_zlabel("$log_{10}(\\theta^\\star)$")
        log_theta_star_surf.set_clim(np.nanmin(theta_star_mat), np.nanmax(theta_star_mat))
        log_theta_star_fig.colorbar(log_theta_star_surf)

        x_star_fig = plt.figure()
        ax = plt.gca(projection='3d')  # type: Axes3D
        x_star_surf = ax.plot_surface(h_mat, y_mat, x_star_mat, cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel("$h$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$x^\\star$")
        x_star_surf.set_clim(np.nanmin(x_star_mat), np.nanmax(x_star_mat))
        x_star_fig.colorbar(x_star_surf)

    initialize_plt()
    fig, axes = plt.subplots(1, 2, figsize=(5.31445, 3.5))
    plt.sca(axes[0])
    cnt = plt.contourf(h_mat, y_mat, theta_star_mat, levels=16)
    plt.xticks(range(1, 5 + 1))
    plt.yticks(range(1, 5 + 1))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h$", labelpad=0)
    plt.ylabel("$y$")
    cbar = plt.colorbar(label="$\\theta^\\star$ (rad)", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    plt.sca(axes[1])
    cnt = plt.contourf(h_mat, y_mat, x_star_mat, levels=16)
    plt.xticks(range(1, 5 + 1))
    plt.yticks(range(1, 5 + 1))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h$", labelpad=0)
    plt.ylabel("$y$")
    cbar = plt.colorbar(label="$x^\\star$", orientation='horizontal')
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')

    for ax in axes:
        if h_range is not None:
            ax.set_xlim(h_range)
        if y_range is not None:
            ax.set_ylim(y_range)

    axes[0].annotate("($a$)", xy=(1, 1), xytext=(0.05, 0.95),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='top', color='white')
    axes[1].annotate("($b$)", xy=(1, 1), xytext=(0.05, 0.95),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='top')

    plt.tight_layout()
    # plt.show()

    # initialize_plt(font_size=18, line_scale=2)
    # plt.figure()
    # cnt = plt.contourf(h_mat, y_mat, theta_grad_mat * x_star_mat / theta_star_mat, levels=16)
    # for c in cnt.collections:
    #     c.set_edgecolor("face")  # Reduce aliasing in output.
    # plt.xlabel("$h$", labelpad=0)
    # plt.ylabel("$y$")
    # if h_range is not None:
    #     plt.gca().set_xlim(h_range)
    # if y_range is not None:
    #     plt.gca().set_ylim(y_range)
    # plt.colorbar(label="$\\frac{\\theta'(0)}{\\theta^\\star/x^\\star}$")
    # plt.tight_layout()
    #
    # plt.figure()
    # scat = plt.scatter(x_star_mat, y_mat, c=h_mat)
    # plt.colorbar(label='$h$')
    # for xs, qows, hows in zip(x_star_mat, y_mat, h_mat):
    #     plt.plot(xs, qows, color=scat.to_rgba(hows[0]))
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')

    if plot_logs:
        initialize_plt(font_size=14, line_scale=2)
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(y_mat, theta_star_mat, c=h_mat)
        plt.xlabel('$y$')
        plt.ylabel('$\\theta^\\star$ (rad)')
        plt.colorbar(label="$h$")

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(h_mat, theta_star_mat, c=y_mat)
        plt.xlabel('$h$')
        plt.ylabel('$\\theta^\\star$ (rad)')
        plt.colorbar(label="$y$")

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(y_mat, x_star_mat, c=h_mat)
        plt.xlabel('$y$')
        plt.ylabel('$x^\\star$')
        plt.colorbar(label="$h$")

        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(h_mat, x_star_mat, c=y_mat)
        plt.xlabel('$h$')
        plt.ylabel('$x^\\star$')
        plt.colorbar(label="$y$")

    plt.show()


if __name__ == "__main__":
    n = 20000
    n_points = 16
    save_file = open(f"../model_outputs/peak_sweep/peak_sweep_{n}_{n_points}x{n_points}_plate_100.csv", 'r')

    all_hs = []
    all_ys = []
    theta_stars = []
    x_stars = []
    theta_grads = []
    for line in save_file.readlines():
        split = line.split(',')
        all_hs.append(float(split[0]))
        all_ys.append(float(split[1]))
        theta_stars.append(float(split[2]))
        x_stars.append(float(split[3]))
        theta_grads.append(float(split[4]))

    if len(all_hs) == n_points ** 2:
        h_mat = np.reshape(all_hs, (n_points, n_points))
        y_mat = np.reshape(all_ys, (n_points, n_points))
        theta_star_mat = np.reshape(theta_stars, (n_points, n_points))
        x_star_mat = np.reshape(x_stars, (n_points, n_points))
        theta_grad_mat = np.reshape(theta_grads, (n_points, n_points))
    else:
        h_mat = np.empty((n_points, n_points))
        h_mat.fill(np.nan)
        y_mat = np.empty((n_points, n_points))
        y_mat.fill(np.nan)
        theta_star_mat = np.empty((n_points, n_points))
        theta_star_mat.fill(np.nan)
        x_star_mat = np.empty((n_points, n_points))
        x_star_mat.fill(np.nan)
        theta_grad_mat = np.empty((n_points, n_points))
        theta_grad_mat.fill(np.nan)
        for k in range(len(all_hs)):
            i = int(np.floor(k / n_points))
            j = int(k % n_points)
            h_mat[i, j] = all_hs[k]
            y_mat[i, j] = all_ys[k]
            theta_star_mat[i, j] = theta_stars[k]
            x_star_mat[i, j] = x_stars[k]
            theta_grad_mat[i, j] = theta_grads[k]

    plot_peak_sweep(h_mat, y_mat, theta_star_mat, x_star_mat, theta_grad_mat,
                    plot_3d=False, plot_logs=False, h_range=(0.5, 5), y_range=(0.5, 5))
