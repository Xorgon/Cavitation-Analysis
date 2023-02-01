import itertools
import sys
import importlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Circle

from util.file_utils import csv_to_lists


def draw_plate(ax: Axes, vf, W=2, p_size=50, border=3, label=None, hole_scale=1, color="k", linewidth=5):
    W = W * hole_scale

    ax.set_aspect('equal')
    ax.set_xlim((-p_size / 2, p_size / 2))
    ax.set_ylim((-p_size / 2, p_size / 2))

    ax.set_xticks([])
    ax.set_yticks([])

    if label is not None:
        ax.set_xlabel(label, fontsize="x-small")

    ax.add_patch(Rectangle((-p_size / 2, - p_size / 2), p_size, p_size, facecolor='gray',
                           edgecolor=color, linewidth=linewidth))

    plt.setp(ax.spines.values(), color=color)

    if vf == 0 or W == 0:
        return

    S = W * np.sqrt(np.pi * np.sqrt(3) / (6 * vf))

    for i, j in itertools.product(range(-50, 50), range(-50, 50)):
        x = i * S + np.cos(np.pi / 3) * j * S
        y = np.sin(np.pi / 3) * j * S

        if x - W / 2 < - p_size / 2 + border or x + W / 2 > p_size / 2 - border \
                or y - W / 2 < - p_size / 2 + border or y + W / 2 > p_size / 2 - border:
            continue

        ax.add_patch(Circle((x, y), W / 2, facecolor='white', edgecolor="k"))


def draw_porous_overlay(ax, movie_dir, shift, img_scale=1, thickness=0.9):
    super_dir = movie_dir + "../"

    sys.path.append(super_dir)
    import params

    importlib.reload(params)
    sys.path.remove(super_dir)

    if hasattr(params, "W") and hasattr(params, "vf") and hasattr(params, "mm_per_px") \
            and hasattr(params, "upper_surface_y"):
        W = params.W
        vf = params.vf
        mm_per_px = params.mm_per_px
        surf_y = params.upper_surface_y
    elif hasattr(params, "mm_per_px") and hasattr(params, "upper_surface_y"):
        W = 0
        vf = 0
        mm_per_px = params.mm_per_px
        surf_y = params.upper_surface_y
    else:
        raise RuntimeError("params.py is not complete")

    idx = int(movie_dir.split("movie_C001H001S")[-1].replace("/", ""))

    m_xs, m_ys, idxs = csv_to_lists(super_dir, "index.csv", has_headers=True)
    m_y = [m_y for m_x, m_y, m_idx in zip(m_xs, m_ys, idxs) if m_idx == idx][0]  # There ~can~ should only be one.

    upper_y_px = 264 - (surf_y - m_y) / mm_per_px
    lower_y_px = 264 - (surf_y - m_y - thickness) / mm_per_px
    ax.axhline(img_scale * upper_y_px, color="black", linewidth=0.5)
    ax.axhline(img_scale * lower_y_px, color="black", linewidth=0.5)

    if "circles" in movie_dir:
        S = W * np.sqrt(np.pi * np.sqrt(3) / (6 * vf))
    elif "triangles" in movie_dir:
        print("NOT DONE YET")
        S = None
    elif "squares" in movie_dir:
        S = W * np.sqrt(1 / vf)
    else:
        S = 50  # whole plate
        shift = 25  # force center on the plate

    plt.rc("hatch", linewidth=0.5)

    mid_x = 384 / 2
    for i in range(-10, 10):
        ax.add_patch(Rectangle((img_scale * (mid_x + (shift + W / 2 + i * S) / mm_per_px), img_scale * lower_y_px),
                               img_scale * (S - W) / mm_per_px, -img_scale * thickness / mm_per_px,
                               hatch='////', edgecolor="k", facecolor="dimgrey", linewidth=0.5, alpha=1))


if __name__ == "__main__":
    draw_plate(plt.gca(), 0.2)
    plt.show()
