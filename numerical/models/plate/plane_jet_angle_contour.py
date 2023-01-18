import numpy as np
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import itertools
import scipy.sparse

offset = 1
span = 5
heights = [1, 2]
N = 32

xs = np.linspace(0, span, 16)

m_0 = 1
n = 10000

length = 50
depth = 50
centroids, normals, areas = gen.gen_plane([-length / 2, 0, -depth / 2],
                                          [-length / 2, 0, depth / 2],
                                          [length / 2, 0, -depth / 2],
                                          n)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

speeds = []
us = []
vs = []

for y, x in itertools.product(heights, xs):
    print(f"Testing p={x:5.3f}, q={y:5.3f}")
    R_b = bem.get_R_vector([x, y, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    speeds.append(speed)
    us.append(res_vel[0] / speed)
    vs.append(res_vel[1] / speed)

X, Y = np.meshgrid(xs, heights)
S = np.reshape(np.array(speeds), X.shape)
U = np.reshape(np.array(us), X.shape)
V = np.reshape(np.array(vs), X.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(2 * X / span, Y / span, np.arctan2(V, U) + np.pi / 2, levels=128)
plt.xlabel("$\\bar{x}$")
plt.ylabel("$y / w$")
plt.colorbar(label="$\\theta_j$")
plt.plot([min(xs), max(xs)], [0, 0], 'k')
plt.show()
