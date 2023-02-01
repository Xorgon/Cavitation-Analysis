import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage.transform import resize

import util.plotting_utils as plt_util
import util.analysis_utils as au
from util.mp4 import MP4

black_border = True
include_scale = True

repeat = 0
frame_idxs = [15, 29, 42, 49, 57]
comp_frames = [29, 49]

tile_tl_corner = (118, 55)
tile_br_corner = (266, 260)
img_scale = 1
fig_scale = 1

f_dir = "../Data/w12vf32circles/On a hole/movie_C001H001S0006/"
mm_per_px = 0.0265

mov = MP4(f_dir + f"video_{repeat}.mp4")

laser_delay = 150.0e-6
firing_delay = laser_delay  # + mov.trigger_out_delay
fps = mov.get_fps()
bg_frame = np.int32(mov[0])

valid_idxs = []
xs = []
ys = []
areas = []
times = []
eccs = []

for idx in range(100):
    x, y, area, ecc, _, _ = au.analyse_frame(np.int32(mov[idx]), bg_frame, debug=True)
    time = idx / mov.frame_rate - firing_delay

    if area is not None:
        valid_idxs.append(idx)
        xs.append(x)
        ys.append(y)
        areas.append(area)
        times.append(time)
        eccs.append(ecc)

width = tile_br_corner[0] - tile_tl_corner[0] + 2
height = (tile_br_corner[1] - tile_tl_corner[1]) * 1.2

plt_util.initialize_plt(font_size=10 * fig_scale, line_scale=fig_scale * 2, capsize=3 * fig_scale)
plt.rc("axes", linewidth=fig_scale / 2)

fig_width = 5 * fig_scale
fig = plt.figure(figsize=(fig_width, 2.55 * fig_width * height / (len(frame_idxs) * width)))

(upper_fig, lower_fig) = fig.subfigures(2, 1, height_ratios=[2, 3])

u_gspec = upper_fig.add_gridspec(ncols=5, nrows=1,  # left=0.02,  # right=0.975,
                                 top=0.95, bottom=0.15, wspace=0.05, hspace=0.05)

num_sub_plots = len(frame_idxs)

total_image = None
for k, idx in enumerate(frame_idxs):
    frame = np.int32(mov[idx])
    # frame = frame[tile_tl_corner[1]:tile_br_corner[1],
    #         tile_tl_corner[0]:tile_br_corner[0]]

    frame = resize(frame, (frame.shape[0] * img_scale, frame.shape[1] * img_scale))

    ax = upper_fig.add_subplot(u_gspec[0, k])
    ax.imshow(frame, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])

    if idx in valid_idxs:
        data_idx = valid_idxs.index(idx)
    else:
        data_idx = None
    if data_idx is not None and idx in comp_frames:
        ax.add_patch(Circle((img_scale * xs[data_idx], img_scale * ys[data_idx]),
                            img_scale * np.sqrt(areas[data_idx] / np.pi),
                            fill=False, color=f"C{comp_frames.index(idx)}"))

    style = "k--"

    ax.set_xlim((img_scale * tile_tl_corner[0], img_scale * tile_br_corner[0]))
    ax.set_ylim((img_scale * tile_br_corner[1], img_scale * tile_tl_corner[1]))
    plt.text(((tile_br_corner[0] + tile_tl_corner[0]) / 2) * img_scale,
             (tile_br_corner[1] + 0.05 * height) * img_scale,
             f"$t = {(idx / mov.frame_rate - firing_delay) * 1e6:.0f}$ $\\mu$s", horizontalalignment='center',
             verticalalignment='top', color="black")
    if k == len(frame_idxs) - 1 and include_scale:
        sbar_x_max = img_scale * (tile_br_corner[0] - 0.15 * width)
        sbar_x_min = sbar_x_max - img_scale * 1 / mm_per_px
        sbar_y_pos = img_scale * (tile_tl_corner[1] + 0.125 * height)
        ax.plot([sbar_x_min, sbar_x_max],
                [sbar_y_pos, sbar_y_pos], 'k', linewidth=2 * fig_scale)
        ax.annotate("1 mm", [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos],
                    [0.5 * (sbar_x_min + sbar_x_max), sbar_y_pos - img_scale * 0.01 * height],
                    color="black", horizontalalignment='center', verticalalignment='bottom')
    if k == 0:
        plt_util.label_subplot(ax, "($a$)", loc='out')

radius_ax, comp_ax = lower_fig.subplots(1, 2, gridspec_kw={'width_ratios': [2, 3]})

radius_ax.plot(np.array(times) * 1e6, mm_per_px * np.sqrt(np.array(areas) / np.pi), "k.-", linewidth=1)
radius_ax.set_xlabel("$t$ ($\\mu$s)")
radius_ax.set_ylabel("$R$ (mm)")

comp_frame = np.minimum(np.int32(mov[comp_frames[0]]), np.int32(mov[comp_frames[1]]))
comp_ax.imshow(comp_frame, cmap=plt.cm.gray)
comp_ax.set_xticks([])
comp_ax.set_yticks([])
comp_ax.set_ylim((258, 50))
comp_ax.set_xlim((70, 314))

c_xs = []
c_ys = []
rs = []

for i, idx in enumerate(comp_frames):
    comp_ax.imshow(np.int32(mov[idx]), cmap=plt.cm.gray, alpha=0.5)
    data_idx = valid_idxs.index(idx)
    x = xs[data_idx]
    y = ys[data_idx]
    r = np.sqrt(areas[data_idx] / np.pi)

    print(f"{times[data_idx] * 1e6}, {eccs[data_idx]}")

    c_xs.append(x)
    c_ys.append(y)
    rs.append(r)

    comp_ax.add_patch(Circle((x, y), r, fill=False, color=f"C{i}"))
    radius_ax.axvline(times[data_idx] * 1e6, color=f"C{i}", linestyle="dashed", linewidth=1)
    plt.scatter([x], [y], color=f"C{i}", s=1)

    if i == 0:
        comp_ax.annotate("", (x - 1.2 * r, y + r), (x - 1.2 * r, y - r),
                         arrowprops={'arrowstyle': '<|-|>', 'fc': f"C{i}", 'ec': f"C{i}",
                                     'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
        comp_ax.text(x - 1.2 * r - 24, y, f"$2R_{i}$", ha='center', va='center', color=f"C{i}",
                     fontsize="x-large")
        plt.plot([x, x - 1.2 * r], [y - r, y - r], color=f"C{i}", linestyle="dotted", linewidth=0.75)
        plt.plot([x, x - 1.2 * r], [y + r, y + r], color=f"C{i}", linestyle="dotted", linewidth=0.75)
    if i == 1:
        comp_ax.annotate("", (x - r, y + 1.2 * r), (x + r, y + 1.2 * r),
                         arrowprops={'arrowstyle': '<|-|>', 'fc': f"C{i}", 'ec': f"C{i}",
                                     'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
        comp_ax.text(x, y + 1.2 * r + 18, f"$2R_{i}$", ha='center', va='center', color=f"C{i}",
                     fontsize="x-large")
        plt.plot([x - r, x - r], [y, y + 1.2 * r], color=f"C{i}", linestyle="dotted", linewidth=0.75)
        plt.plot([x + r, x + r], [y, y + 1.2 * r], color=f"C{i}", linestyle="dotted", linewidth=0.75)

plt.plot([c_xs[0], c_xs[0] + 1.5 * np.max(rs)], [c_ys[0], c_ys[0]], color=f"C2", linestyle="dotted", linewidth=0.75)
plt.plot([c_xs[1], c_xs[1] + 1.5 * np.max(rs)], [c_ys[1], c_ys[1]], color=f"C2", linestyle="dotted", linewidth=0.75)
comp_ax.annotate("", (c_xs[0] + 1.5 * np.max(rs), c_ys[0]), (c_xs[1] + 1.5 * np.max(rs), c_ys[1]),
                 arrowprops={'arrowstyle': '<|-|>', 'fc': f"C2", 'ec': f"C2",
                             'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 10})
comp_ax.text(c_xs[0] + 1.5 * np.max(rs) + 16, np.mean(c_ys), f"$\\Delta$", ha='center', va='center', color=f"C2",
             fontsize="x-large")

plt_util.label_subplot(radius_ax, "($b$)")
plt_util.label_subplot(comp_ax, "($c$)")

plt.tight_layout()

# plt.savefig('figure4.eps', dpi=300)
plt.show()
