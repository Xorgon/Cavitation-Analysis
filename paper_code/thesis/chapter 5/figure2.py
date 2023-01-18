import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from skimage.transform import resize

import util.plotting_utils as plt_util
from util.mp4 import MP4

black_border = True
include_scale = True

frame_idxs = [16, 19, 26, 32, 35]

tile_tl_corner = (117, 96)
tile_br_corner = (237, 231)
img_scale = 8
fig_scale = 1

f_dir = "fig_data/"
mm_per_px = 0.0250  # estimate

# mov = file.get_mraw_from_dir(f_dir)  # type: mraw
mov = MP4(f_dir + f"surface_nucleation.mp4")

laser_delay = 160.0e-6
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

target_sum = 3.3e6


def adjust_brightness(f, exp=1):
    f_min = np.min(f)
    f_max = np.max(f)
    return ((f - f_min) / (f_max - f_min)) ** exp


def get_img_sum(exp, f):
    print(exp)
    return np.sum(adjust_brightness(f, exp))


total_image = None
for k, idx in enumerate(frame_idxs):
    frame = np.int32(mov[idx])
    # frame = frame[tile_tl_corner[1]:tile_br_corner[1],
    #         tile_tl_corner[0]:tile_br_corner[0]]
    frame = resize(frame, (frame.shape[0] * img_scale, frame.shape[1] * img_scale))

    opt_exp = minimize(lambda x: np.abs(get_img_sum(x, frame) - target_sum), [1], method="Nelder-Mead",
                       bounds=[(0.1, 10)], tol=target_sum / 1e4).x[0]

    frame = adjust_brightness(frame, opt_exp)

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
        sbar_x_max = img_scale * (tile_br_corner[0] - 0.1 * width)
        sbar_x_min = sbar_x_max - img_scale * 1 / mm_per_px
        sbar_y_pos = img_scale * (tile_tl_corner[1] + 0.12 * height)
        ax.plot([sbar_x_min, sbar_x_max],
                [sbar_y_pos, sbar_y_pos], 'k', linewidth=2 * fig_scale)
        ax.annotate("$1$ mm", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
                    [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - img_scale * 0.01 * height],
                    color="black", horizontalalignment='center', verticalalignment='bottom')
plt.subplots_adjust(left=0.005, right=0.995, top=1, bottom=0.15, wspace=0.05, hspace=0.05)
plt.show()
