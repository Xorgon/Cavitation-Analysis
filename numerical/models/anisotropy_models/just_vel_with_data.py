import os
import sys
import importlib
import experimental.util.analysis_utils as au
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
import random


dirs = []
root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/"
for root, _, files in os.walk(root_dir):
    if "params.py" in files:
        dirs.append(root + "/")
print(f"Found {len(dirs)} data sets")
print(dirs)

total_readings = 0

for i, dir_path in enumerate(dirs):
    vels = []
    disps = []

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    x_offset = params.left_slot_wall_x + params.slot_width / 2
    y_offset = params.upper_surface_y

    readings = au.load_readings(dir_path + "readings_dump.csv")

    # Convert to meters
    w = params.slot_width / 1000
    h = params.slot_height / 1000

    n = 10000
    density_ratio = 0.25
    w_thresh = 5

    centroids, normals, areas = gen.gen_varied_slot(n=n, H=h, W=w, length=0.1, depth=0.05, w_thresh=w_thresh,
                                                    density_ratio=density_ratio)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float64)
    R_inv = scipy.linalg.inv(R_matrix)

    # random.shuffle(readings)
    for j, reading in enumerate(readings):
        print(f"Directory {i + 1} / {len(dirs)} | Reading {j + 1} / {len(readings)} | Total readings: {total_readings}")
        pos = reading.get_bubble_pos_mm(params.mm_per_px)
        p = (pos[0] - x_offset) / 1000
        q = (pos[1] - y_offset) / 1000

        if q < 0:
            continue

        R_init = np.sqrt(reading.max_bubble_area / np.pi) * params.mm_per_px / 1000
        displacement = np.linalg.norm(reading.disp_vect) * params.mm_per_px / 1000

        bubble_pos = [p, q, 0]

        sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv)
        vel = bem.get_vel(bubble_pos, centroids, areas, sigmas)

        total_readings += 1
        vels.append(np.linalg.norm(vel))
        disps.append(displacement / R_init)

    plt.scatter(vels, disps)

print(f"Total readings = {total_readings}")
plt.xlabel("$vel$")
plt.ylabel("$\\Delta / R_0$")
plt.show()
