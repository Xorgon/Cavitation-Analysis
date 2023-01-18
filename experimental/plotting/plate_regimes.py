import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from common.util.plotting_utils import initialize_plt, fig_width

# void fraction, hole size, name, vertical text offset
geometries = [[0.525, 2.13, "1 mm acrylic", 0],  # 50 mm plate
              [0.39, 1.84, "1 mm acrylic (100 mm plate)", 0.05],  # 100 mm plate
              [0.47, 2, "0.7 mm steel", 0],  # 100 mm plate
              [0.12, 2, "3 mm acrylic", 0],  # 100 mm plate
              [0.24, 1.44, "1 mm acrylic", 0]]  # 50 mm plate

next_geometries = [[0.08, 0.83, "1 mm acrylic", 0],  # 50 mm plate
                   [0.12, 1.02, "1 mm acrylic", 0],  # 50 mm plate
                   [0.16, 1.18, "1 mm acrylic", 0],  # 50 mm plate
                   [0.38, 1.81, "1 mm acrylic (50 mm plate)", -0.05]]  # 50 mm plate

initialize_plt()
plt.figure(figsize=(fig_width(), fig_width() * 0.75))
max_w = 3
min_gap = 0.5
min_w = 0.1

min_w_vf = (min_w / (min_w + min_gap)) ** 2 * np.pi * np.sqrt(3) / 6

max_vf = np.pi * np.sqrt(3) / 6
plt.axvline(100 * max_vf, color="grey", linestyle="--", linewidth=0.5)
plt.text(100 * max_vf, max_w / 2, "Limit for hexagonal grid of circles",
         rotation=90, ha="right", va="center", color="grey")

plt.scatter([a[0] * 100 for a in geometries], [a[1] for a in geometries], c="C0", label="Already complete")
for g in geometries:
    plt.text(g[0] * 100, g[1] + g[3], "-  " + g[2], va="center", zorder=0.5)

plt.scatter([a[0] * 100 for a in next_geometries], [a[1] for a in next_geometries], c="C1", label="Next experiments")
for g in next_geometries:
    plt.text(g[0] * 100, g[1] + g[3], "-  " + g[2], va="center", zorder=0.5)

vfs = np.arange(min_w_vf, max_vf, 0.01)

min_gap_ws = min_gap / (np.sqrt(np.pi * np.sqrt(3) / (6 * vfs)) - 1)
# plt.plot(100 * vfs, min_gap_ws)

poly = Polygon(np.append([[0, max_w], [0, min_w], [min_w_vf, min_w]], np.array([100 * vfs, min_gap_ws]).T, axis=0),
               color="#bbFFbb", zorder=0, alpha=1)
plt.gca().add_patch(poly)
text_pos_idx = len(vfs) // 3
plt.text(100 * vfs[text_pos_idx] + 5, min_gap_ws[text_pos_idx], "$S - W > 0.5$",
         rotation=36, va="bottom", ha="center", c="#009900", alpha=0.75)
plt.legend(fancybox=False)
plt.xlabel("\\% Void Fraction")
plt.ylabel("Hole Size (mm)")
plt.xlim(0, 100)
plt.ylim(0, max_w)
plt.tight_layout()
plt.show()
