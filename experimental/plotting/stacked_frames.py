from typing import List

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import sys
import importlib

import experimental.util.file_utils as file
from experimental.util.mraw import mraw
import common.util.plotting_utils as plt_util
import experimental.util.analysis_utils as au

repeat = 1
frame_range = (15, 80)

fig_scale = 2

# f_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/The Lump/normal_angle_C001H001S0001/"
f_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/OAPM - between 3 - sturdier/movie_C001H001S0008/"

mov = file.get_mraw_from_dir(f_dir)  # type: mraw

laser_delay = 151.0e-6
firing_delay = laser_delay  # + mov.trigger_out_delay
fps = mov.get_fps()

fig_width = 5.31445 * fig_scale
ax = plt.gca()

initial_frame = np.int32(mov[repeat * 100])
total_image = None
for idx in range(frame_range[0], frame_range[1] + 1):
    frame = np.int32(mov[repeat * 100 + idx]) - initial_frame

    if total_image is None:
        total_image = frame
    else:
        total_image = total_image + frame

total_image = total_image / (frame_range[1] - frame_range[0])
ax.imshow(total_image, cmap=plt.cm.gray, alpha=1)
ax.set_xticks([])
ax.set_yticks([])

# sbar_x_max = img_scale * (tile_br_corner[0] - 0.1 * width)
# sbar_x_min = sbar_x_max - img_scale * 1 / params.mm_per_px
# sbar_y_pos = img_scale * (tile_tl_corner[1] + 0.12 * height)
# ax.plot([sbar_x_min, sbar_x_max],
#         [sbar_y_pos, sbar_y_pos], 'k', linewidth=2 * fig_scale)
# ax.annotate("$1 mm$", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
#             [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - img_scale * 0.01 * height],
#             color="black", horizontalalignment='center', verticalalignment='bottom')

plt.tight_layout()
plt.show()
