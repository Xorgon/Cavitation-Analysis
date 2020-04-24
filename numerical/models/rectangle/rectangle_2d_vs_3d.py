import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

import numerical.bem as bem
import numerical.util.gen_utils as gen
import numerical.potential_flow.element_utils as eu
from numerical.potential_flow.elements import Source3D, Source2D


def get_rectangle_jet_vel_2d(x, z, length, k=25, m_0=1):
    elements = []
    for i in range(-k, k + 1):
        s_x = i * length + x + (2 * x - length) * ((-1) ** i - 1) / 2
        if i != 0:
            elements.append(Source2D(s_x, z, -m_0))
        elements.append(Source2D(s_x, -z, -m_0))
    return eu.get_all_vel(elements, x, z)


def get_rectangle_jet_vel_3d(x, y, z, length, height, k=100, m_0=1):
    elements = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            s_x = i * length + x + (2 * x - length) * ((-1) ** i - 1) / 2
            s_y = j * height + y + (2 * y - height) * ((-1) ** j - 1) / 2
            if not (i == 0 and j == 0):
                elements.append(Source3D(s_x, s_y, z, -m_0))
            elements.append(Source3D(s_x, s_y, -z, -m_0))
    return eu.get_all_vel_3d(elements, x, y, z)


length = 2
hs = np.linspace(0.1, 2, 5)
theta_j_lists = []
theta_bs = np.linspace(0.01, np.pi / 2)
dist = 1
for h in hs:
    print(h)
    angles = []
    residuals = []
    for theta_b in theta_bs:
        x_b = dist * np.cos(theta_b)
        z_b = dist * np.sin(theta_b)
        vel = get_rectangle_jet_vel_3d(x_b, h / 2, z_b, length, h, k=int(round(50 / np.sqrt(h))))
        residuals.append(abs(vel[1]))
        angle = np.arctan2(vel[2], vel[0]) + np.pi / 2
        angles.append(angle)
    theta_j_lists.append(angles)
    print(f"res = {np.mean(residuals)}")

theta_js_2d = []
for theta_b in theta_bs:
    x_b = dist * np.cos(theta_b)
    z_b = dist * np.sin(theta_b)
    vel_2d = get_rectangle_jet_vel_2d(x_b, z_b, length)
    angle_2d = np.arctan2(vel_2d[1], vel_2d[0]) + np.pi / 2
    theta_js_2d.append(angle_2d)

cmap = plt.cm.get_cmap()
for h, theta_js in zip(hs, theta_j_lists):
    plt.plot(theta_bs, theta_js, label=f"h = {h:.2f}", color=cmap(h / max(hs)))
plt.plot(theta_bs, theta_js_2d, label="2D", color="red")
plt.xlabel("$\\theta_b$")
plt.ylabel("$\\theta_j$")
plt.legend()
plt.show()
