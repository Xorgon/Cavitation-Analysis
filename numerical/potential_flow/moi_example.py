import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from common.util.plotting_utils import initialize_plt

from numerical.potential_flow.elements import Source3D
from numerical.potential_flow.potential_flow_plot import plot_elements, plot_arrows, plot_contour

initialize_plt(dpi=200)

m = 0.1
bubble = Source3D(0, 1, 0, -m)
mirror = Source3D(0, -1, 0, -m)

elements = [bubble, mirror]
x_bounds = [-1.5, 1.5]
y_bounds = [-1.5, 1.5]
x_points = 24
y_points = 24
mask_points = [[0, 1], [0, -1]]
mask_radii = [0.2, 0.2]
mask_style = ("-", "--")
unit_arrows = True

fig_width = 4

plt.figure(figsize=(fig_width, round(fig_width * (y_bounds[1] - y_bounds[0]) / (x_bounds[1] - x_bounds[0]))))
plot_contour(elements, plt.gca(), x_bounds, x_points * 25, [0, y_bounds[1]], round(y_points * 25 / 2),
             mask_points, mask_radii, colormap="viridis", levels=200)
plot_contour(elements, plt.gca(), x_bounds, x_points * 25, [y_bounds[0], 0], round(y_points * 25 / 2),
             mask_points, mask_radii, colormap="gray", levels=200)

plot_arrows(elements, plt.gca(), x_bounds, x_points, y_bounds, y_points,
            'mid', unit_arrows, mask_points, mask_radii)

for mask_p, r, style in zip(mask_points, mask_radii, mask_style):
    plt.gca().add_patch(Circle(mask_p, r, edgecolor="k", facecolor="white", linewidth=1, linestyle=style))

plt.gca().set_aspect('equal')
plt.xlim(x_bounds)
plt.ylim(y_bounds)

plt.plot([-2, 2], [0, 0], color="black", linewidth=2)
plt.xticks([])
plt.yticks([])

plt.text(0, 1, "Bubble", ha='center', va='center', fontsize="small")
plt.text(0, -1, "Image", ha='center', va='center', fontsize="small")

plt.tight_layout()

plt.show()
