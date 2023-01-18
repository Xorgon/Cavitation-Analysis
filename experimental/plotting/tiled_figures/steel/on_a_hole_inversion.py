from typing import List

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import sys
import importlib

from common.util.drawing_utils import draw_porous_overlay

import experimental.util.file_utils as file
from experimental.util.mraw import mraw
from experimental.util.pixel_correction import load_norm_mat, safe_correct
import common.util.plotting_utils as plt_util
import experimental.util.analysis_utils as au

black_border = True
include_scale = True

repeat = 1
frame_idxs = [14, 22, 30, 38, 44, 55]

tile_tl_corner = (126, 48)
tile_br_corner = (272, 264)
img_scale = 8
fig_scale = 1

f_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/w12vf24circles/On a hole/movie_C001H001S0001/"

mov = file.get_mraw_from_dir(f_dir)  # type: mraw
norm_mat = load_norm_mat(f_dir + "../")

laser_delay = 140.0e-6
firing_delay = laser_delay  # + mov.trigger_out_delay
fps = mov.get_fps()

times = []
for idx in frame_idxs:
    times.append(idx / fps - firing_delay)

width = tile_br_corner[0] - tile_tl_corner[0] + 2
height = (tile_br_corner[1] - tile_tl_corner[1]) * 1.2

plt_util.initialize_plt(font_size=10 * fig_scale, line_scale=fig_scale, capsize=3 * fig_scale)

fig_width = 5.31445 * fig_scale
fig = plt.figure(figsize=(fig_width, fig_width * height / (len(frame_idxs) * width)))

num_sub_plots = len(frame_idxs)

total_image = None
for k, idx in enumerate(frame_idxs):
    frame = np.int32(mov[repeat * 100 + idx])
    frame = safe_correct(frame, norm_mat)
    # frame = frame[tile_tl_corner[1]:tile_br_corner[1],
    #         tile_tl_corner[0]:tile_br_corner[0]]
    frame = resize(frame, (frame.shape[0] * img_scale, frame.shape[1] * img_scale))
    plot_num = 100 + 10 * num_sub_plots + (k + 1)
    ax = fig.add_subplot(plot_num)
    ax.imshow(frame, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])

    style = "k--"

    ax.set_xlim((img_scale * tile_tl_corner[0], img_scale * tile_br_corner[0]))
    ax.set_ylim((img_scale * tile_br_corner[1], img_scale * tile_tl_corner[1]))
    plt.text(((tile_br_corner[0] + tile_tl_corner[0]) / 2) * img_scale,
             (tile_br_corner[1] + 0.05 * height) * img_scale,
             f"$t = {times[k] * 1e6:.0f}$ $\mu$s", horizontalalignment='center', verticalalignment='top',
             color="black")
    if k == len(frame_idxs) - 1 and include_scale:
        params = file.load_params(f_dir + "../")
        print("W =", params.W, "mm")
        print("vf =", params.vf * 100, "%")
        print("mm/px = ", params.mm_per_px)
        sbar_x_max = img_scale * (tile_br_corner[0] - 0.1 * width)
        sbar_x_min = sbar_x_max - img_scale * 1 / params.mm_per_px
        sbar_y_pos = img_scale * (tile_tl_corner[1] + 0.12 * height)
        ax.plot([sbar_x_min, sbar_x_max],
                [sbar_y_pos, sbar_y_pos], 'k', linewidth=2 * fig_scale)
        ax.annotate("$1$ mm", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
                    [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - img_scale * 0.01 * height],
                    color="black", horizontalalignment='center', verticalalignment='bottom')
    draw_porous_overlay(ax, f_dir, 0.075, img_scale=img_scale, thickness=1)
plt.subplots_adjust(left=0.005, right=0.995, top=1, bottom=0.15, wspace=0.05, hspace=0.05)
# plt.savefig("fig2.eps", dpi=300)
plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
            "paper figures/on_hole_inversion.eps", dpi=300)
plt.show()
