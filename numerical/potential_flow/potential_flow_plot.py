import numpy as np
import math
import matplotlib.pyplot as plt


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
                  unit_arrows=False):
    xs = np.linspace(x_bounds[0], x_bounds[1], x_points)
    ys = np.linspace(y_bounds[0], y_bounds[1], y_points)

    us = np.zeros((len(ys), len(xs)))
    vs = np.zeros((len(ys), len(xs)))

    for i in range(len(xs)):
        for j in range(len(ys)):
            for element in elements:
                du, dv, _ = element.get_vel(xs[i], ys[j], 0)
                us[j, i] += du
                vs[j, i] += dv

                if unit_arrows:
                    speed = math.sqrt(us[j, i] ** 2 + vs[j, i] ** 2)
                    if not np.isclose(speed, 0):
                        us[j, i] /= speed
                        vs[j, i] /= speed

    e_xs = []
    e_ys = []

    for element in elements:
        e_xs.append(element.x_0)
        e_ys.append(element.y_0)

    plt.figure(figsize=(8, round(8 * (y_bounds[1] - y_bounds[0]) / (x_bounds[1] - x_bounds[0]))))
    plt.scatter(e_xs, e_ys, marker='+', c='r')
    plt.quiver(xs, ys, us, vs, scale=2.5, pivot=pivot, scale_units="xy")
    plt.show()
