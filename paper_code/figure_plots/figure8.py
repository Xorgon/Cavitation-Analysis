import matplotlib.pyplot as plt
import numpy as np

from util.plotting_utils import initialize_plt


def plot_peak_sweep(h_mat, y_mat, theta_star_mat, x_star_mat, h_range=None, y_range=None):
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
    cnt = plt.contourf(h_mat, y_mat, x_star_mat, levels=np.arange(0.5, 6.5, 0.25))
    plt.xticks(range(1, 5 + 1))
    plt.yticks(range(1, 5 + 1))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    plt.xlabel("$h$", labelpad=0)
    plt.ylabel("$y$")
    cbar = plt.colorbar(label="$x^\\star$", orientation='horizontal')
    cbar.set_ticks(np.arange(0.5, 7, 0.5))
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
    plt.show()


if __name__ == "__main__":
    n = 20000
    n_points = 16
    save_file = open(f"figure8_data/peak_sweep_{n}_{n_points}x{n_points}_plate_100.csv", 'r')

    all_hs = []
    all_ys = []
    theta_stars = []
    x_stars = []
    for line in save_file.readlines():
        split = line.split(',')
        all_hs.append(float(split[0]))
        all_ys.append(float(split[1]))
        theta_stars.append(float(split[2]))
        x_stars.append(float(split[3]))

    if len(all_hs) == n_points ** 2:
        h_mat = np.reshape(all_hs, (n_points, n_points))
        y_mat = np.reshape(all_ys, (n_points, n_points))
        theta_star_mat = np.reshape(theta_stars, (n_points, n_points))
        x_star_mat = np.reshape(x_stars, (n_points, n_points))
    else:
        h_mat = np.empty((n_points, n_points))
        h_mat.fill(np.nan)
        y_mat = np.empty((n_points, n_points))
        y_mat.fill(np.nan)
        theta_star_mat = np.empty((n_points, n_points))
        theta_star_mat.fill(np.nan)
        x_star_mat = np.empty((n_points, n_points))
        x_star_mat.fill(np.nan)
        for k in range(len(all_hs)):
            i = int(np.floor(k / n_points))
            j = int(k % n_points)
            h_mat[i, j] = all_hs[k]
            y_mat[i, j] = all_ys[k]
            theta_star_mat[i, j] = theta_stars[k]
            x_star_mat[i, j] = x_stars[k]

    plot_peak_sweep(h_mat, y_mat, theta_star_mat, x_star_mat, h_range=(0.5, 5), y_range=(0.5, 5))
