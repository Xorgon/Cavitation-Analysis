import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# unit_arrows = False

# elements = [Vortex(0, 0, 20),
#             Vortex(1, 0, -20),
#             Vortex(1, 1, 20),
#             Vortex(0, 1, -20),
#             Vortex(-1, 1, 20),
#             Vortex(-1, 0, -20)]

# elements = [Vortex(0, 0, 10)]

# elements = [Source(0, 1, -1),
#             Source(0, -1, -1)]

def plot_elements(elements: list, x_bounds=(-1, 1), x_points=20, y_bounds=(-1, 1), y_points=20, pivot='mid',
                  unit_arrows=False, mask_points=(), mask_radii=(), mask_style=()):
    e_xs = []
    e_ys = []

    for element in elements:
        e_xs.append(element.x_0)
        e_ys.append(element.y_0)

    plt.figure(figsize=(8, round(8 * (y_bounds[1] - y_bounds[0]) / (x_bounds[1] - x_bounds[0]))))
    plt.scatter(e_xs, e_ys, marker='+', c='r')
    plot_contour(elements, plt.gca(), x_bounds, x_points * 25, y_bounds, y_points * 25, mask_points, mask_radii)
    plot_arrows(elements, plt.gca(), x_bounds, x_points, y_bounds, y_points,
                pivot, unit_arrows, mask_points, mask_radii)

    for mask_p, r, style in zip(mask_points, mask_radii, mask_style):
        plt.gca().add_patch(Circle(mask_p, r, edgecolor="k", facecolor="white", linewidth=2, linestyle=style))

    plt.gca().set_aspect('equal')
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)


def plot_arrows(elements: list, axes, x_bounds=(-1, 1), x_points=20, y_bounds=(-1, 1), y_points=20, pivot='mid',
                unit_arrows=False, mask_points=(), mask_radii=()):
    xs = np.linspace(x_bounds[0], x_bounds[1], x_points)
    ys = np.linspace(y_bounds[0], y_bounds[1], y_points)

    us = np.zeros((len(ys), len(xs)))
    vs = np.zeros((len(ys), len(xs)))

    for i in range(len(xs)):
        for j in range(len(ys)):
            for mask_p, mask_r in zip(mask_points, mask_radii):
                if (xs[i] - mask_p[0]) ** 2 + (ys[j] - mask_p[1]) ** 2 < mask_r ** 2:
                    us[j, i] = np.NaN
                    vs[j, i] = np.NaN
                    continue

            for element in elements:
                du, dv, _ = element.get_vel(xs[i], ys[j], 0)
                us[j, i] += du
                vs[j, i] += dv

            if unit_arrows:
                speed = math.sqrt(us[j, i] ** 2 + vs[j, i] ** 2)
                if not np.isclose(speed, 0):
                    us[j, i] /= speed
                    vs[j, i] /= speed

    if unit_arrows:
        scale = 20
    else:
        scale = 2.5
    axes.quiver(xs, ys, us, vs, scale=scale, pivot=pivot, scale_units="xy")


def plot_contour(elements: list, axes, x_bounds=(-1, 1), x_points=100, y_bounds=(-1, 1), y_points=100,
                 mask_points=(), mask_radii=(), levels=100, colormap="viridis"):
    xs = np.linspace(x_bounds[0], x_bounds[1], x_points)
    ys = np.linspace(y_bounds[0], y_bounds[1], y_points)

    us = np.zeros((len(ys), len(xs)))
    vs = np.zeros((len(ys), len(xs)))

    speeds = np.zeros((len(ys), len(xs)))

    for i in range(len(xs)):
        for j in range(len(ys)):
            for mask_p, mask_r in zip(mask_points, mask_radii):
                if (xs[i] - mask_p[0]) ** 2 + (ys[j] - mask_p[1]) ** 2 < mask_r ** 2:
                    us[j, i] = np.NaN
                    vs[j, i] = np.NaN
                    continue

            for element in elements:
                du, dv, _ = element.get_vel(xs[i], ys[j], 0)
                us[j, i] += du
                vs[j, i] += dv

            speeds[j, i] = np.sqrt(us[j, i] ** 2 + vs[j, i] ** 2)

    cnt = axes.contourf(xs, ys, np.log10(speeds), levels=levels, cmap=plt.get_cmap(colormap))
    for c in cnt.collections:
        c.set_edgecolor("face")  # Reduce aliasing in output.
    return cnt
