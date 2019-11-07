import sys
import importlib
import os
from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from experimental.util.analysis_utils import load_readings, Reading
from experimental.util.file_utils import select_dir
from common.util.plotting_utils import initialize_plt

dir_path = "../../../../Data/SlotErrorMeasurement/"

sys.path.append(dir_path)
import params

importlib.reload(params)
sys.path.remove(dir_path)

w = params.slot_width
x_offset = params.left_slot_wall_x + w / 2
y_offset = params.upper_surface_y

readings = load_readings(dir_path + "readings_dump.csv")  # type: List[Reading]
readings = sorted(readings, key=lambda r: r.m_x)

available_ys = set([reading.m_y for reading in readings])
print(available_ys)
# m_y = float(input("y = "))
m_y = min(available_ys)
readings = [r for r in readings if r.m_y == m_y]

m_xs = set(sorted([r.m_x for r in readings]))

limits = (-4, 4)

xs = []
ys = []
us = []
vs = []
for m_x in m_xs:
    x = np.mean([r.bubble_pos[0] * params.mm_per_px + r.m_x - x_offset for r in readings if r.m_x == m_x])
    y = np.mean([(264 - r.bubble_pos[1]) * params.mm_per_px + r.m_y - y_offset for r in readings if r.m_y == m_y])
    u = np.mean([r.disp_vect[0] * params.mm_per_px for r in readings if r.m_x == m_x])
    v = np.mean([-r.disp_vect[1] * params.mm_per_px for r in readings if r.m_x == m_x])
    u /= np.sqrt(u ** 2 + v ** 2)
    v /= np.sqrt(u ** 2 + v ** 2)
    xs.append(x)
    ys.append(y)
    us.append(u)
    vs.append(v)

# From analyse_slot.py TODO: Get these parameters written to a file during analyse_slot (params.py would be nice)
theta_j_offset = -0.0048
p_bar_offset = -0.0777
xs = np.array(xs) - p_bar_offset
angles = np.arctan2(vs, us) + np.pi / 2 - theta_j_offset

initialize_plt()
plt.figure(figsize=(5.31445, 4.2))
plt.quiver(xs, ys, np.sin(angles), -np.cos(angles),
           np.abs(angles), units='xy', scale=1 / (0.8 * np.mean(ys)), cmap=cm.get_cmap('winter'))  # winter or copper
plt.gca().set_aspect('equal', adjustable='box')

plt.plot([min(xs), -w / 2, -w / 2, w / 2, w / 2, max(xs)],
         [0, 0, -params.slot_height, -params.slot_height, 0, 0],
         "k", linewidth=4)
plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.1)
plt.xlabel("$p$ (mm)")
plt.ylabel("$q$ (mm)")
plt.xlim(limits)

plt.show()
