"""
Elements used in potential flow theory.

Equations referenced from https://potentialflow.com/
"""

import math
import numpy as np
import common.util.vector_utils as vect
from numerical.potential_flow import potential_flow_plot as pfp


class Element:
    x_0, y_0, z_0 = None, None, None

    def get_vel_x(self, x, y, _):
        return 0

    def get_vel_y(self, x, y, _):
        return 0


class Element3D:
    def get_vel_x(self, x, y, z):
        return 0

    def get_vel_y(self, x, y, z):
        return 0

    def get_vel_z(self, x, y, z):
        return 0

    def get_vel(self, x, y, z):
        return np.array([self.get_vel_x(x, y, z), self.get_vel_y(x, y, z), self.get_vel_z(x, y, z)])


class Vortex(Element):
    x_0 = None
    y_0 = None
    mag = None

    def __init__(self, x, y, magnitude):
        self.x_0 = x
        self.y_0 = y
        self.mag = magnitude

    def get_vel_x(self, x, y, _):
        if np.isclose(y - self.y_0, 0):
            return 0
        else:
            return - self.mag * (y - self.y_0) / (4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2) ** (3 / 2))

    def get_vel_y(self, x, y, _):
        if np.isclose(x - self.x_0, 0):
            return 0
        else:
            return self.mag * (x - self.x_0) / (4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2) ** (3 / 2))


class Source(Element):
    """ A 3D source on the plane z=0. """
    x_0 = None
    y_0 = None
    mag = None

    def __init__(self, x, y, magnitude):
        self.x_0 = x
        self.y_0 = y
        self.mag = magnitude

    def get_vel_x(self, x, y, _):
        if np.isclose(x - self.x_0, 0):
            return 0
        else:
            return self.mag * (x - self.x_0) / (4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2) ** (3 / 2))

    def get_vel_y(self, x, y, _):
        if np.isclose(y - self.y_0, 0):
            return 0
        else:
            return self.mag * (y - self.y_0) / (4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2) ** (3 / 2))


class Source3D(Element3D):
    x_0 = None
    y_0 = None
    z_0 = None
    mag = None

    def __init__(self, x, y, z, magnitude):
        self.x_0 = x
        self.y_0 = y
        self.z_0 = z
        self.mag = float(magnitude)

    def __str__(self):
        return "Source3D: [{0}, {1}, {2}], {3}".format(self.x_0, self.y_0, self.z_0, self.mag)

    def get_pos(self):
        return np.array([self.x_0, self.y_0, self.z_0])

    def get_vel_x(self, x, y, z):
        if np.isclose(x - self.x_0, 0):
            return 0
        else:
            return self.mag * (x - self.x_0) / (
                    4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2 + (z - self.z_0) ** 2) ** (3 / 2))

    def get_vel_y(self, x, y, z):
        if np.isclose(y - self.y_0, 0):
            return 0
        else:
            return self.mag * (y - self.y_0) / (
                    4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2 + (z - self.z_0) ** 2) ** (3 / 2))

    def get_vel_z(self, x, y, z):
        if np.isclose(z - self.z_0, 0):
            return 0
        else:
            return self.mag * (z - self.z_0) / (
                    4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2 + (z - self.z_0) ** 2) ** (3 / 2))

    def get_vel(self, x, y, z):
        pos = np.array([x, y, z])
        return (pos - self.get_pos()) * self.mag / (
                4 * math.pi * ((x - self.x_0) ** 2 + (y - self.y_0) ** 2 + (z - self.z_0) ** 2) ** (3 / 2))


class Dipole3D(Element3D):
    pos = None  # type: np.ndarray # Position vector
    d = None  # type: np.ndarray  # Direction unit vector
    mu = None  # Magnitude

    def __init__(self, pos, d, mu):
        self.pos = np.array(pos)
        self.x_0, self.y_0, self.z_0 = pos
        self.d = np.array(d) / vect.mag(d)
        self.mu = mu

    def get_vel(self, x, y, z):
        vel_pos = np.array([x, y, z])
        r = vel_pos - self.pos
        r_mag = vect.mag(r)
        return - self.mu * (self.d / r_mag ** 3 - 3 * r * np.dot(self.d, r) / r_mag ** 5)

    def get_vel_x(self, x, y, z):
        return self.get_vel(x, y, z)[0]

    def get_vel_y(self, x, y, z):
        return self.get_vel(x, y, z)[1]

    def get_vel_z(self, x, y, z):
        return self.get_vel(x, y, z)[2]


if __name__ == '__main__':
    elements = [Dipole3D([0, -1, 0], [1, 0, 0], 1),
                Source3D(0, 1, 0, -100)]
    pfp.plot_elements(elements, x_bounds=(-10, 10), y_bounds=(-10, 10), pivot='tail', unit_arrows=True, x_points=30,
                      y_points=30)
