import matplotlib.pyplot as plt
import numpy as np
import math

import numerical.bem as bem
import numerical.util.gen_utils as gen

length = 50
depth = 25
n = 5000
centroids, normals, areas = gen.gen_plane([- length / 2, 0, - depth / 2],
                                          [- length / 2, 0, depth / 2],
                                          [length / 2, 0, - depth / 2],
                                          n)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas)
R_inv = np.linalg.inv(R_matrix)
# pu.plot_3d_point_sets([cs, sinks])

ps = np.linspace(-4, 4, 150)
q = 2
m_0 = 1
theta_js = []
for p in ps:
    res_vel, sigma = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv)
    theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
    theta_js.append(theta_j)

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca()
ax.plot(ps, theta_js, label="Numeric")
ax.set_xlabel("$\\theta_b$", fontsize=18)
ax.set_ylabel("$\\theta_j$ (rad)", fontsize=18)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)
ax.axvline(0, linestyle='--', color='gray')
plt.show()
