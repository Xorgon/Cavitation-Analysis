import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from numerical.raytrace.raytrace import Ray, Mirror
from common.util.plotting_utils import initialize_plt, label_subplot

target_dpi = 1000

initialize_plt(dpi=target_dpi)

right_lim = 150
fig_width = 5 * (right_lim + 5) / 155

fig = plt.figure(figsize=(fig_width, 5))
main_ax = fig.gca()

# trans = big_ax.transData.transform((0, beam_width))[1] - big_ax.transData.transform((0, 0))[1]
# width_pts = 72 * trans / target_dpi  # 72 points per inch * pixels / pixels per inch
width_pts = 61


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
            big_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=min(1, 40 / len(rays)),
                        linewidth=width_pts / len(rays))
            # if i == 0 or i + 1 == len(rays):
            #     big_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=0.1)

        if zoom_ax is not None:
            zoom_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00",
                         alpha=min(1, 4 / len(rays)))


beam_height = 16
beam_width = 6.35

for y in np.linspace(beam_height - beam_width / 2, beam_height + beam_width / 2, 423):
    ray = Ray([700, y], [-1, 0], mirrors=[])
    ray.extend_ray(700 - 354)
    main_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=min(1, 40 / 200),
                 linewidth=width_pts / 200)

beam_height = 16
beam_width = 3

for y in np.linspace(beam_height - beam_width / 2, beam_height + beam_width / 2, 200):
    ray = Ray([353, y], [-1, 0], mirrors=[])
    ray.extend_ray(17)
    main_ax.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="#65ff00", alpha=min(1, 40 / 200),
                 linewidth=width_pts / 200)

run_trace(main_ax, None, 0)

main_ax.set_aspect('equal')
main_ax.set_xlim((-5, right_lim))
main_ax.set_ylim((-95, 60))
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.set_frame_on(False)

frame_w = 384 * 0.025
frame_h = 264 * 0.025
# main_ax.add_patch(Rectangle((19 - frame_w / 2, -25.4 - frame_h / 2), frame_w, frame_h, facecolor="#ffaaaa"))
# main_ax.annotate("Typical camera frame", (19 + frame_w / 4, -25.4 + frame_h / 2), (30, -7), color="#ff7777",
#                  arrowprops={"width": 2, "edgecolor": "#ff7777", "facecolor": "#ff7777",
#                              "headwidth": 8, "headlength": 8, "shrink": 0.05})

plt.tight_layout()

fig_path = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/" \
           "Porous Materials/Conferences/APS DFD 2022/GFM/Images/raytrace.png"
plt.savefig(fig_path, dpi=target_dpi, transparent=True)
