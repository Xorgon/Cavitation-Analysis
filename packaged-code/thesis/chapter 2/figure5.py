import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from util.raytrace import Ray, Mirror
from util.plotting_utils import initialize_plt, label_subplot
from util.mp4 import MP4

initialize_plt(dpi=200)

fig = plt.figure(figsize=(5, 3.1))
main_ax = fig.gca()
sub_ax1 = main_ax.inset_axes([0.25, 0.1, 0.3, 0.3])
sub_ax2 = main_ax.inset_axes([0.475, 0.1, 0.3, 0.3])
sub_ax3 = main_ax.inset_axes([0.7, 0.1, 0.3, 0.3])


def run_trace(big_ax, zoom_ax, angle):
    mirs = []
    ys = np.linspace(50.8 - 25.4, 50.8 + 25.4, 10000)
    xs = (1 / (4 * 25.4)) * ys ** 2
    for i in range(len(xs) - 1):
        mirs.append(Mirror([xs[i], ys[i]], [xs[i + 1], ys[i + 1]]))

    # Plotted like this to avoid aliasing. Would need to adjust for angled mirrors.
    if big_ax is not None:
        big_ax.plot(xs - 6.35, ys - 25.4, color="grey")
    if zoom_ax is not None:
        zoom_ax.plot(xs - 6.35, ys - 25.4, color="grey")

    for m in mirs:
        m.translate([-6.35, -25.4])
        m.rotate(np.deg2rad(angle), [0, 25.4])

    beam_height = 16
    beam_width = 30
    rays = []
    for y in np.linspace(beam_height - beam_width / 2, beam_height + beam_width / 2, 200):
        rays.append(Ray([200, y], [-1, 0], mirrors=mirs))

    for i, ray in enumerate(rays):
        print(f"{100 * i / len(rays):.2f}%")
        ray.trace()
        ray.extend_ray(200)
        if big_ax is not None:
            big_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=min(1, 10 / len(rays)))
            # if i == 0 or i + 1 == len(rays):
            #     big_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=0.1)

        if zoom_ax is not None:
            zoom_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00",
                         alpha=min(1, 4 / len(rays)))


run_trace(main_ax, sub_ax1, 0)
run_trace(None, sub_ax2, 1)

main_ax.set_aspect('equal')
main_ax.set_xlim((-5, 150))
main_ax.set_ylim((-50, 40))
label_subplot(main_ax, "($a$)")
main_ax.set_xlabel("$X$ (mm)")
main_ax.set_ylabel("$Y$ (mm)")

frame_w = 384 * 0.025
frame_h = 264 * 0.025
main_ax.add_patch(Rectangle((19 - frame_w / 2, -25.4 - frame_h / 2), frame_w, frame_h, facecolor="#aaaaff"))
main_ax.annotate("Typical camera frame", (19 + frame_w / 4, -25.4 + frame_h / 2), (30, -7), color="#7777ff",
                 arrowprops={"width": 2, "edgecolor": "#7777ff", "facecolor": "#7777ff",
                             "headwidth": 8, "headlength": 8, "shrink": 0.05})

sub_ax1.set_aspect('equal')
sub_ax1.set_xlim((18, 20))
sub_ax1.set_ylim((-26.4, -24.4))
sub_ax1.set_xticks([])
sub_ax1.set_yticks([])
label_subplot(sub_ax1, "($b$)", loc='tr')
main_ax.indicate_inset_zoom(sub_ax1, edgecolor="black")

sub_ax2.set_aspect('equal')
sub_ax2.set_xlim((20, 22))
sub_ax2.set_ylim((-26.4, -24.4))
sub_ax2.set_xticks([])
sub_ax2.set_yticks([])
label_subplot(sub_ax2, "($c$)", loc='tr')

mraw = MP4("fig_data/plasma.mp4")

sub_frame = np.int32(mraw[14]) - np.int32(mraw[13])
sub_ax3.imshow(np.abs(sub_frame) ** 0.75, cmap=plt.cm.gray)
sub_ax3.set_aspect('equal')
sub_ax3.set_xlim((160, 270))
sub_ax3.set_ylim((175, 65))
sub_ax3.set_xticks([])
sub_ax3.set_yticks([])
label_subplot(sub_ax3, "($d$)", loc='tr', color="white")

plt.tight_layout()
plt.show()
