import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import os

hs = [1, 2, 4, 8, 16]
w = 2
n = 20000

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca(projection='3d')  # type: Axes3D
ax.set_zlim((0, 0.9))
ax.set_xlabel("$2 d / w$")
ax.set_ylabel("$\\theta_b$")

surfs = []

for h in hs:
    file = open(f"../model_outputs/slot_alt_param/h_{h}_w_{w}_n_{n}.csv", "r")

    flat_ds = []
    flat_theta_bs = []
    flat_theta_js = []
    for line in file.readlines()[1:]:
        s = line.split(",")
        flat_ds.append(float(s[0]))
        flat_theta_bs.append(float(s[1]))
        flat_theta_js.append(float(s[2]))

    n_samples = int(np.sqrt(len(flat_ds)))
    ds_mat = np.array(flat_ds).reshape((n_samples, n_samples))
    theta_bs_mat = np.array(flat_theta_bs).reshape((n_samples, n_samples))
    theta_js = np.array(flat_theta_js).reshape((n_samples, n_samples))

    surf = ax.plot_surface(ds_mat / (w / 2), theta_bs_mat, theta_js,
                           cmap=cm.get_cmap('coolwarm'))
    surf.set_visible(False)
    surfs.append(surf)
    # ax.set_title(f"h / w = {h / w:.2f}")


def show_surf(idx, ax: Axes3D):
    ax.title.set_text(f"h / w = {hs[idx] / w:.2f}")
    for surf in surfs:
        surf.set_visible(False)
    surfs[idx].set_visible(True)


frames = list(range(len(surfs)))
frames.extend([len(surfs) - 1] * 4)  # Hold at top
frames.extend(range(len(surfs) - 2, -1, -1))  # Play backwards

anim = animation.FuncAnimation(fig, show_surf, frames=frames, interval=1000, fargs=[ax])

writer = animation.FFMpegWriter()
anim.save('../model_outputs/slot_alt_param/animated_surfaces.mp4', writer=writer, dpi=250)

plt.show()
