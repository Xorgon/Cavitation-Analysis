from experimental.util.mraw import mraw
import experimental.util.analysis_utils as au
import math
import matplotlib.pyplot as plt
import os
import numpy as np

hor_dist = float(input("Horizontal distance = "))
hor_dist_y = float(input("Horizontal distance measured at y = "))
left_wall_x = float(input("At left wall, when y = {0:.2f}, x = ".format(hor_dist_y)))
left_angle = float(input("Angle of left wall = "))
right_angle = float(input("Angle of right wall = "))
mm_per_px = float(input("mm per pixel = "))

origin_offset_y = hor_dist_y - hor_dist * math.sin(right_angle) * math.sin(math.pi - left_angle) / math.sin(
    left_angle - right_angle)
origin_offset_x = left_wall_x + math.fabs(hor_dist_y - origin_offset_y) / math.tan(math.pi - left_angle)

dir_path = "D:/Data/Test1/HSweep4/"
readings = au.analyse_series(dir_path, mm_per_px=mm_per_px)

# Post-process data to get jet angles.
# Positions relative to origin (corner center)
xs = []
ys = []
theta_bs = []
theta_js = []
for reading in readings:
    x = reading[5][0] - origin_offset_x
    y = reading[5][1] - origin_offset_y
    xs.append(x)
    ys.append(y)

    theta_bs.append(math.atan2(y, x) - (left_angle + math.pi / 2))

    theta_js.append(math.atan2(-reading[3][1], reading[3][0]) + math.pi)

plt.plot(xs, theta_bs)
plt.show()
plt.plot(theta_bs, theta_js)
plt.show()
