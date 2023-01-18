import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
from util.file_utils import lists_to_csv
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
from common.util.plotting_utils import plot_3d_point_sets
import random


def sink_positions_equilateral_triangle(l, bubble_position, w=None, x_range=None, y_range=None, print_result=True,
                                        rotate=False):
    """
    Calculate mirror sink positions to give boundary conditions of a bubble collapsing inside an equilateral triangle.
    :param l: triangle side length
    :param bubble_position: bubble position, where bubble is within a triangle with its tip at the origin and two other
    endpoints equidistant from the x-axis in the first and fourth quadrants; a vector
    :param w: range of symmetric reflections in the x-direction and y-direction to calculate; setting this value will
    calculate a square range of symmetric reflections centered at the origin. To set custom values, specify x_range and
    y_range instead.
    :param x_range: range of symmetric reflections in the x-direction to calculate; a vector of the form (x_1, x_2)
    :param y_range: range of symmetric reflections in the y-direction to calculate; a vector of the form (y_1, y_2)
    :param print_result: True or False, if True, prints how many sink positions were calculated
    :param rotate: True or False, if True, rotates the figure by 90 degrees clockwise
    :return: list of sink positions as tuples
    """

    s_x = []
    s_y = []
    s = []
    h = l * np.cos(np.pi / 6)

    if not rotate:
        x_bubble = bubble_position[0]
        y_bubble = bubble_position[1]
    else:
        x_bubble = bubble_position[1]
        y_bubble = bubble_position[0]

    # Make sure bubble is within a triangle with tip at the origin and flat side above the tip.
    if y_bubble <= np.sqrt(3) * x_bubble \
            or y_bubble <= -np.sqrt(3) * x_bubble \
            or y_bubble >= l * np.cos(np.pi / 6):
        raise ValueError('Bubble position must be within equilateral triangle with tip at the origin and flat side '
                         'above the tip (x and y such that y >= sqrt(3) * x; and y >= -sqrt(3) * x); '
                         'and y <= l * cos(pi/6) where l = {0}).'.format(l))

    if x_range is None and y_range is None and w is None:
        raise ValueError('Define the area of reflections to calculate. \n Specify w or x_range and y_range.')
    elif x_range is None and y_range is None and w is not None:
        x_range = (-w, w + 1)
        y_range = (-w, w + 1)
    elif x_range is not None and y_range is not None and w is None:
        x_range = (-x_range, x_range + 1)
        y_range = (-y_range, y_range + 1)
    elif (x_range is None and y_range is not None) \
            or (x_range is not None and y_range is None) \
            or (x_range is not None and y_range is not None and w is not None):
        raise ValueError('Specify BOTH x_range and y_range, or specify w.')

    for i, j in itertools.product(range(x_range[0], x_range[1]), range(y_range[0], y_range[1])):
        # if abs(j) < -np.sqrt(3) / 3 * abs(i) + y_range[1]:
        x_s = x_bubble + i * (3 / 2) * l
        y_s = (-1) ** j * (y_bubble - h / 2) + j * h + (-h + (-1) ** i * h) / 2 + h / 2
        # The second-to-last term of y_s is 0 (even i) or -h (odd i)
        for k in (-1, 0, 1):
            # Avoid floating point precision error for the bubble position.
            if i == 0 and j == 0 and k == 0:
                s_x.append(x_bubble)
                s_y.append(y_bubble)
            else:
                s_x.append((x_s - k ** 2 * x_bubble) * np.cos(k * np.pi / 3) - y_s * np.sin(k * np.pi / 3)
                           + k ** 2 * x_bubble * np.cos(np.pi + k * np.pi / 3))
                s_y.append((x_s - k ** 2 * x_bubble) * np.sin(k * np.pi / 3) + y_s * np.cos(k * np.pi / 3)
                           + k ** 2 * x_bubble * np.sin(np.pi + k * np.pi / 3))
            # The -1 below retrieves the last added element to s_x and s_y.
            if not rotate:
                s.append((s_x[-1], s_y[-1]))
            if rotate:
                s.append((s_y[-1], s_x[-1]))  # This rotates the figure by 90 degrees clockwise.
    number_sinks = len(s)
    if print_result:
        print('Calculated {0} sink positions.'.format(number_sinks))
    return s


L = 1

sinks = sink_positions_equilateral_triangle(L, [0.5 * L, 0], x_range=30, y_range=30, rotate=True)
sinks = np.array(sinks)
plt.scatter(sinks[:, 0], sinks[:, 1], s=2 ** 2)

plt.plot([0, L * np.sqrt(3) / 2, L * np.sqrt(3) / 2, 0], [0, 0.5, -0.5, 0])



# for s in sinks:
#     plt.scatter([s[0]], [s[1]], alpha=0.25)

print(len(sinks))
plt.gca().set_aspect('equal')
plt.show()
