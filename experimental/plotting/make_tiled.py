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

repeat = 3
index = 24
frame_idxs = [16, 25, 33, 37, 74]

tile_tl_corner = (131, 55)
tile_br_corner = (292, 236)
scale = 2

dir_path = "../../../../Data/SlotSweeps/W2H3a/"

readings = au.load_readings(dir_path + "readings_dump.csv")
readings = sorted(readings, key=lambda r: r.idx)  # type: List[au.Reading]

sys.path.append(dir_path)
import params

importlib.reload(params)

reading = None
for r in readings:
    if r.idx == index and r.repeat_number == repeat:
        reading = r

if reading is None:
    raise RuntimeError("Could not find reading.")

b_pos_abs = reading.get_bubble_pos_mm(params.mm_per_px)
p = b_pos_abs[0] - (params.left_slot_wall_x + params.slot_width / 2)
q = b_pos_abs[1] - params.upper_surface_y
print(f"w = {params.slot_width}\n"
      f"h = {params.slot_height}\n"
      f"p = {p:.2f}\n"
      f"q = {q:.2f}")

prefix = file.get_prefix_from_idxs(dir_path, [index])

f_dir = f"{dir_path}{prefix}{index:04}/"
mov = file.get_mraw_from_dir(f_dir)  # type: mraw

laser_delay = 151.0e-6
firing_delay = laser_delay  # + mov.trigger_out_delay
fps = mov.get_fps()

times = []
for idx in frame_idxs:
    times.append(idx / fps - firing_delay)

width = tile_br_corner[0] - tile_tl_corner[0] + 2
height = tile_br_corner[1] - tile_tl_corner[1]

surface_y_px = scale * (mov.height - (params.upper_surface_y - reading.m_y) / params.mm_per_px)
floor_y_px = scale * (mov.height - (params.upper_surface_y - params.slot_height - reading.m_y) / params.mm_per_px)
left_x_px = scale * (params.left_slot_wall_x - reading.m_x) / params.mm_per_px
right_x_px = scale * (params.left_slot_wall_x + params.slot_width - reading.m_x) / params.mm_per_px

plt_util.initialize_plt()

fig_width = 5.31445
fig = plt.figure(figsize=(fig_width, fig_width * height / (len(frame_idxs) * width)))

num_sub_plots = len(frame_idxs)

total_image = None
for k, idx in enumerate(frame_idxs):
    frame = np.int32(mov[repeat * 100 + idx])
    # frame = frame[tile_tl_corner[1]:tile_br_corner[1],
    #         tile_tl_corner[0]:tile_br_corner[0]]
    frame = resize(frame, (frame.shape[0] * scale, frame.shape[1] * scale))
    plot_num = 100 + 10 * num_sub_plots + (k + 1)
    ax = fig.add_subplot(plot_num)
    ax.imshow(frame, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])

    style = "k--"
    ax.plot([0, left_x_px], [surface_y_px, surface_y_px], style)
    ax.plot([left_x_px, left_x_px], [surface_y_px, floor_y_px], style)
    ax.plot([left_x_px, right_x_px], [floor_y_px, floor_y_px], style)
    ax.plot([right_x_px, right_x_px], [floor_y_px, surface_y_px], style)
    ax.plot([right_x_px, mov.width * scale], [surface_y_px, surface_y_px], style)

    ax.set_xlim((scale * tile_tl_corner[0], scale * tile_br_corner[0]))
    ax.set_ylim((scale * tile_br_corner[1], scale * tile_tl_corner[1]))
    plt.text((tile_br_corner[0] - 0.05 * height) * scale,
             (tile_br_corner[1] - 0.05 * height) * scale,
             f"$t = {times[k] * 1e6:.0f} \mu s$", horizontalalignment='right', verticalalignment='bottom',
             color="white")
    if k == 0:
        sbar_x_max = scale * (tile_br_corner[0] - 0.1 * width)
        sbar_x_min = sbar_x_max - scale * 1 / params.mm_per_px
        sbar_y_pos = scale * (tile_tl_corner[1] + 0.15 * height)
        ax.plot([sbar_x_min, sbar_x_max],
                [sbar_y_pos, sbar_y_pos], 'w', linewidth=2)
        ax.annotate("$1 mm$", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
                    [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - scale * 0.01 * height],
                    color="white", horizontalalignment='center', verticalalignment='bottom')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
plt.show()
