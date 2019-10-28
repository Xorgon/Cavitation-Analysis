"""
This is written for a particular reading I was analysing. The structure is reusable, just change the specifics.
"""

import matplotlib.pyplot as plt
import numpy as np

import experimental.util.analysis_utils as au
import experimental.util.file_utils as fu

angle_offset = 0.034206708
vects, _ = au.analyse_reading(fu.select_dir())
vects = np.array(vects)
angles = np.arctan2(vects[:, 1], vects[:, 0]) + np.pi / 2

for i, angle in enumerate(angles):
    if angle > np.pi:
        angles[i] = angle - 2 * np.pi - angle_offset

print(f"Mean = {np.mean(angles)}, SD = {np.std(angles)}, Error = {np.mean(angles) + np.pi / 2}.")

plt.close()
plt.figure()
plt.scatter(range(len(angles)), angles, label="Measured")
plt.axhline(- np.pi / 2, color="gray", linestyle="--", label="Theoretical")
plt.axhline(np.mean(angles), color="r", linestyle="--", label="Mean")
plt.xlabel("Reading")
plt.ylabel("Angle")
plt.legend()
plt.show()
