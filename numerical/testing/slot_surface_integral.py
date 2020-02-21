import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse


def get_avg_u(bubble_pos, centroids, areas, source_densities, radius, n=int(1e5)):
    vels = []
    offsets = np.random.randn(3, n)
    offsets *= radius / np.linalg.norm(offsets, axis=0)
    offsets = offsets.T
    for offset in offsets:
        pos_1 = bubble_pos + offset
        pos_2 = bubble_pos - offset

        vel_1 = np.array([0., 0., 0.])
        vel_2 = np.array([0., 0., 0.])

        vel_1 += bem.get_vel(pos_1, centroids, areas, source_densities, bubble_pos)
        vel_2 += bem.get_vel(pos_2, centroids, areas, source_densities, bubble_pos)
        vels.append(vel_1)
        vels.append(vel_2)
    return np.mean(vels, axis=0)


# ps = [8]
W = 4.2
Y = 2.43
H = 11.47
X = W / 2

m_0 = 1
n = 5000

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=6,
                                                density_ratio=0.25)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
plot_3d_point_sets([[[X, Y, 0]], centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

theta_js = []
R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
sigma = bem.calculate_sigma([X, Y, 0], centroids, normals, areas, m_0, R_inv, R_b)

radii = np.linspace(0.1, 2, 16)
for radius in radii:
    print(f"Radius = {radius}")
    res_vel = get_avg_u(np.array([X, Y, 0]), centroids, areas, sigma, radius, n=int(1e5))
    theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
    theta_js.append(theta_j)

plt.plot(radii, theta_js)
plt.xlabel("Radius")
plt.ylabel("$\\theta_j$")
plt.show()
