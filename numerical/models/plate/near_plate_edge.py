import numpy as np
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
import scipy.sparse
import math

offset = 1
span = 5
y = 10
N = 64

lengths = [10, 15, 20, 25, 50, 75, 100]

xs = np.linspace(0, span, N)

m_0 = 1
n_per_length = 140 / max(lengths)

fig = plt.figure()

for length in lengths:
    n = (n_per_length * length) ** 2
    depth = length
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
    thetas = []

    for x in xs:
        print(f"Testing x={x:5.3f}, y={y:5.3f}")
        R_b = bem.get_R_vector([x, y, 0], centroids, normals)
        res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
        speed = np.linalg.norm(res_vel)
        speeds.append(speed)
        us.append(res_vel[0] / speed)
        vs.append(res_vel[1] / speed)
        thetas.append(math.atan2(res_vel[1], res_vel[0]) + np.pi / 2)

    plt.plot(xs, thetas, label=f"plate length = {length}")
plt.legend()
plt.xlabel("x")
plt.ylabel("$\\theta$")
plt.gca().set_yticks([-np.pi / 2, -3 *np.pi / 8, -np.pi / 4, -np.pi / 8, 0])
plt.gca().set_yticklabels(["$-\\pi / 2$", "$- 3 \\pi / 8$", "$-\\pi / 4$", "$- \\pi / 8$", "0"])
plt.show()
