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

black_border = True

# idxs = [37, 44, 48, 53]
# repeats = [5, 0, 0, 3]
# frame_idxs = [45, 43, 42, 49]

idxs = range(1, 54)
repeats = [None for idx in idxs]
frame_idxs = [0] * len(idxs)

scale = 2

dir_path = "../../../../Data/SlotSweeps/w4h12/"

all_readings = au.load_readings(dir_path + "readings_dump.csv")
all_readings = sorted(all_readings, key=lambda r: r.idx)  # type: List[au.Reading]

sys.path.append(dir_path)
import params

importlib.reload(params)

readings = []
for k, index in enumerate(idxs):
    for r in all_readings:
        if r.idx == index and (repeats[k] is None or r.repeat_number == repeats[k]):
            readings.append(r)
            break
    if len(readings) == k:
        raise RuntimeError(f"Could not find reading with idx = {index}, repeat = {repeats[k]}")

width = None
height = None
frames = []
for k, index in enumerate(idxs):
    prefix = file.get_prefix_from_idxs(dir_path, [index])
    f_dir = f"{dir_path}{prefix}{index:04}/"
    mov = file.get_mraw_from_dir(f_dir)  # type: mraw
    if repeats[k] is not None:
        frame = np.int32(mov[repeats[k] * 100 + frame_idxs[k]])
    else:
        frame = np.int32(mov[frame_idxs[k]])
    frame = resize(frame, (frame.shape[0] * scale, frame.shape[1] * scale))
    frames.append(frame)

    if width is None or height is None:
        width = mov.width
        height = mov.height

# # laser_delay = 151.0e-6
# # firing_delay = laser_delay  # + mov.trigger_out_delay
# # fps = mov.get_fps()
#
# times = []
# for idx in frame_idxs:
#     times.append(idx / fps - firing_delay)


min_m_y = min([reading.m_y for reading in readings])
min_m_x = min([reading.m_x for reading in readings])

plt_util.initialize_plt()

fig_width = 5.31445
# fig = plt.figure(figsize=(fig_width, fig_width * height / (len(frame_idxs) * width)))
fig = plt.figure()
ax = fig.gca()

max_right = None
max_top = None
for k, idx in enumerate(frame_idxs):
    # (left, right, bottom, top)
    left = scale * (readings[k].m_x - min_m_x) / params.mm_per_px
    right = left + scale * width
    bottom = scale * (readings[k].m_y - min_m_y) / params.mm_per_px
    top = bottom + scale * height

    if max_right is None or max_right < right:
        max_right = right
    if max_top is None or max_top < top:
        max_top = top

    img_extent = (left,
                  right,
                  bottom,
                  top)
    ax.imshow(frames[k], cmap=plt.cm.gray, extent=img_extent, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # if k == 0:
    #     sbar_x_max = scale * (tile_br_corner[0] - 0.1 * width)
    #     sbar_x_min = sbar_x_max - scale * 1 / params.mm_per_px
    #     sbar_y_pos = scale * (tile_tl_corner[1] + 0.15 * height)
    #     ax.plot([sbar_x_min, sbar_x_max],
    #             [sbar_y_pos, sbar_y_pos], 'w', linewidth=2)
    #     ax.annotate("$1 mm$", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
    #                 [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - scale * 0.01 * height],
    #                 color="white", horizontalalignment='center', verticalalignment='bottom')

surface_y_px = scale * ((params.upper_surface_y - min_m_y) / params.mm_per_px)
floor_y_px = scale * ((params.upper_surface_y - params.slot_height - min_m_y) / params.mm_per_px)
left_x_px = scale * (params.left_slot_wall_x - min_m_x) / params.mm_per_px
right_x_px = scale * (params.left_slot_wall_x + params.slot_width - min_m_x) / params.mm_per_px

style = "k--"
ax.plot([0, left_x_px], [surface_y_px, surface_y_px], style)
ax.plot([left_x_px, left_x_px], [surface_y_px, floor_y_px], style)
ax.plot([left_x_px, right_x_px], [floor_y_px, floor_y_px], style)
ax.plot([right_x_px, right_x_px], [floor_y_px, surface_y_px], style)
ax.plot([right_x_px, max_right], [surface_y_px, surface_y_px], style)

ax.set_xlim((0, max_right))
ax.set_ylim((0, max_top))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
plt.show()
