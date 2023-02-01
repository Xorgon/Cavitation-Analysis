import numpy as np
from util.elements import Element3D


def get_all_vel(elements, x, y):
    vel = [0, 0]
    for el in elements:
        vel[0] += el.get_vel_x(x, y, 0)
        vel[1] += el.get_vel_y(x, y, 0)
    return vel


def get_all_vel_3d(elements, x, y, z):
    vel = np.zeros((3,), dtype=np.float64)
    for el in elements:  # type: Element3D
        vel = vel + el.get_vel(x, y, z)
    return vel
