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

repeat = 2
index = 53
frame_idxs = [16, 27, 39, 41, 47, 64]

tile_tl_corner = (147, 8)
tile_br_corner = (272, 148)
img_scale = 8
fig_scale = 3

dir_path = "../../../../Data/SlotSweeps/W1H3/"

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
x = b_pos_abs[0] - (params.left_slot_wall_x + params.slot_width / 2)
y = b_pos_abs[1] - params.upper_surface_y

print(f"w = {params.slot_width}\n"
      f"h = {params.slot_height}\n"
      f"x = {x:.2f}\n"
      f"y = {y:.2f}\n"
      f"theta = {reading.get_jet_angle():.4f} (radians) = {np.degrees(reading.get_jet_angle()):.2f} (degrees)")

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
height = (tile_br_corner[1] - tile_tl_corner[1]) * 1.2

surface_y_px = img_scale * (mov.height - (params.upper_surface_y - reading.m_y) / params.mm_per_px)
floor_y_px = img_scale * (mov.height - (params.upper_surface_y - params.slot_height - reading.m_y) / params.mm_per_px)
left_x_px = img_scale * (params.left_slot_wall_x - reading.m_x) / params.mm_per_px
right_x_px = img_scale * (params.left_slot_wall_x + params.slot_width - reading.m_x) / params.mm_per_px

plt_util.initialize_plt(font_size=10*fig_scale, line_scale=fig_scale, capsize=3*fig_scale)

fig_width = 5.31445 * fig_scale
fig = plt.figure(figsize=(fig_width, fig_width * height / (len(frame_idxs) * width)))

num_sub_plots = len(frame_idxs)

total_image = None
for k, idx in enumerate(frame_idxs):
    frame = np.int32(mov[repeat * 100 + idx])
    # frame = frame[tile_tl_corner[1]:tile_br_corner[1],
    #         tile_tl_corner[0]:tile_br_corner[0]]
    frame = resize(frame, (frame.shape[0] * img_scale, frame.shape[1] * img_scale))
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
    ax.plot([right_x_px, mov.width * img_scale], [surface_y_px, surface_y_px], style)

    ax.set_xlim((img_scale * tile_tl_corner[0], img_scale * tile_br_corner[0]))
    ax.set_ylim((img_scale * tile_br_corner[1], img_scale * tile_tl_corner[1]))
    plt.text((tile_br_corner[0] - 0.05 * height) * img_scale,
             (tile_br_corner[1] + 0.05 * height) * img_scale,
             f"$t = {times[k] * 1e6:.0f} \mu s$", horizontalalignment='right', verticalalignment='top',
             color="black")
    if k == 0:
        sbar_x_max = img_scale * (tile_br_corner[0] - 0.1 * width)
        sbar_x_min = sbar_x_max - img_scale * 1 / params.mm_per_px
        sbar_y_pos = img_scale * (tile_tl_corner[1] + 0.12 * height)
        ax.plot([sbar_x_min, sbar_x_max],
                [sbar_y_pos, sbar_y_pos], 'k', linewidth=2*fig_scale)
        ax.annotate("$1 mm$", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
                    [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - img_scale * 0.01 * height],
                    color="black", horizontalalignment='center', verticalalignment='bottom')
plt.subplots_adjust(left=0.005, right=0.995, top=1, bottom=0.15, wspace=0.05, hspace=0.05)
# plt.savefig("fig2.eps", dpi=300)
plt.show()

