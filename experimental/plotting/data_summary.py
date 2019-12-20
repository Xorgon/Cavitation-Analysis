import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
import numpy as np
import matplotlib.pyplot as plt

dirs = ["../../../../Data/SlotErrorMeasurement/"]
root_dir = "../../../../Data/SlotSweeps"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")

h_over_ws = []
q_over_ws = []
cs = []

q_dict = {}
colors = ["c", "b", "y", "r", "g", "m", "orange", "purple"]
for i, dir_path in enumerate(dirs):
    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv")
    readings = sorted(readings, key=lambda r: r.m_x)

    reading_ys = set([reading.m_y for reading in readings])

    for reading_y in reading_ys:
        if hasattr(params, 'title'):
            label = f"{params.title}:{reading_y}"
        else:
            label = f"{dir_path}:{reading_y}"

        # Post-process data to get jet angles.
        for reading in readings:
            if reading.m_y != reading_y:
                continue
            pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
            q = pos_mm[1] - y_offset
            if q < 0:
                continue

            if label in q_dict.keys():
                q_dict[label].append(q)
            else:
                q_dict[label] = [q]

        if label in q_dict:
            mean_q = np.mean(q_dict[label])
            h_over_ws.append(params.slot_height / params.slot_width)
            q_over_ws.append(mean_q / params.slot_width)
            cs.append(colors[i])

initialize_plt(font_size=14, line_scale=2)
plt.figure()
plt.scatter(h_over_ws, q_over_ws, c=cs, marker="s")
plt.xlabel("$h / w$")
plt.ylabel("$q / w$")
plt.tight_layout()
plt.show()
