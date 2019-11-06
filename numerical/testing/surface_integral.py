import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from common.util.plotting_utils import plot_3d_point_sets


def get_u(r):
    r = np.array(r)
    length = np.linalg.norm(r)
    r_hat = r / length
    return - r_hat / (4 * np.pi * length ** 2)


def get_avg_u(bubble_pos, sinks, radius, n=int(1e5)):
    vels = []
    positions = []
    offsets = np.random.randn(3, n)
    offsets *= radius / np.linalg.norm(offsets, axis=0)
    offsets = offsets.T
    for offset in offsets:
        pos_1 = bubble_pos + offset
        pos_2 = bubble_pos - offset

        vel_1 = np.array([0., 0., 0.])
        vel_2 = np.array([0., 0., 0.])

        for sink in sinks:
            vel_1 += get_u(pos_1 - sink)
            vel_2 += get_u(pos_2 - sink)
        positions.append(pos_1)
        vels.append(vel_1)
        vels.append(vel_2)
    # plot_3d_point_sets([positions])
    return np.mean(vels, axis=0)


# theta_b = np.pi / 4
# b = np.array([np.cos(theta_b), np.sin(theta_b), 0])
# sink_pos = np.array([[-b[0], b[1], 0],
#                      [-b[0], -b[1], 0],
#                      [b[0], -b[1], 0]])
# v = get_avg_u(b, sink_pos, 0.5)
# print(b)
# print(sink_pos)
# print(v)
# print(np.pi / (- np.arctan2(v[1], v[0]) - np.pi / 2))

# plt.scatter(b[0], b[1])
# plt.scatter(sink_pos[:, 0], sink_pos[:, 1])
# plt.axhline(0)
# plt.axvline(0)
# plt.show()

# radii = [0.1, 0.3, 0.5]
# for radius in radii:
#     print(f"radius = {radius}")
#     theta_bs = np.linspace(np.arcsin(radius), np.pi / 2 - np.arcsin(radius), 16)
#     theta_js = []
#     for theta_b in theta_bs:
#         print(f"    theta_b = {theta_b}")
#         b = np.array([np.cos(theta_b), np.sin(theta_b), 0])
#         sink_pos = [[-b[0], b[1], 0],
#                     [-b[0], -b[1], 0],
#                     [b[0], -b[1], 0]]
#         v = get_avg_u(b, sink_pos, radius, n=int(1e5))
#         theta_js.append(- np.arctan2(v[1], v[0]) - np.pi / 2)
#     plt.plot(theta_bs, theta_js)
#
# plt.axhline(np.pi / 4)
# plt.axvline(np.pi / 4)
# plt.show()

theta_b = np.pi / 3
b = np.array([np.cos(theta_b), np.sin(theta_b), 0])
sink_pos = [[-b[0], b[1], 0],
            [-b[0], -b[1], 0],
            [b[0], -b[1], 0]]
radii = np.linspace(0.01, np.sin(theta_b), 32)
angles = []
for radius in radii:
    print(radius)
    v = get_avg_u(b, sink_pos, radius, n=int(1e5))
    angles.append(- np.arctan2(v[1], v[0]) - np.pi / 2)

plt.plot(radii, angles)
plt.show()

# theta_b = np.pi / 4
# b = np.array([np.cos(theta_b), np.sin(theta_b), 0])
# sink_pos = [[-b[0], b[1], 0],
#             [-b[0], -b[1], 0],
#             [b[0], -b[1], 0]]
# hs = np.logspace(np.log10(0.005), np.log10(0.1), 16)
# angles = []
# for h in hs:
#     print(h)
#     v = get_avg_u(b, sink_pos, 0.5, h=h)
#     angles.append(- np.arctan2(v[1], v[0]) - np.pi / 2)
#
# plt.axhline(np.pi / 4)
# plt.plot(hs, angles)
# plt.show()
