import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

import numerical.bem as bem
import numerical.util.gen_utils as gen

if not os.path.exists("model_outputs/slot_vel_data"):
    os.makedirs("model_outputs/slot_vel_data")

offset = 0.05
W = 2
H = 2
N = 64
Ys = np.concatenate([np.linspace(offset, 3, np.round(3 * N / 4) - 1), [1], np.linspace(3 + 0.1, W * 5, np.ceil(N / 4))])
Ys = sorted(Ys)
Xs = np.concatenate([np.linspace(-7 * W / 2, -W, np.ceil(N / 8)),
                     np.linspace(-W + 0.1, W - 0.1, np.round(3 * N / 4)),
                     np.linspace(W, 7 * W / 2, np.ceil(N / 8))])

# qs = np.linspace(3 * w, 5 * w, 16)
# ps = np.linspace(0, 5 * w / 2, 16)

m_0 = 1
n = 20000
density_ratio = 0.1
w_thresh = 12
length = 100

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=length, depth=50, w_thresh=w_thresh,
                                                density_ratio=density_ratio)

centroids_file = open(f"model_outputs/slot_vel_data/centroids_n{n}_W{W:.2f}_H{H:.2f}"
                      f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'w')
for c in centroids:
    centroids_file.write(f"{c[0]},{c[1]},{c[2]}\n")
centroids_file.close()

normals_file = open(f"model_outputs/slot_vel_data/normals_n{n}_W{W:.2f}_H{H:.2f}"
                    f"_drat{density_ratio}_wthresh{w_thresh}_len{length}.csv", 'w')
for normal in normals:
    normals_file.write(f"{normal[0]},{normal[1]},{normal[2]}\n")
normals_file.close()

output_path = f"model_outputs/slot_vel_data/vel_sweep_n{n}_W{W:.2f}_H{H:.2f}" \
              f"_drat{density_ratio}_wthresh{w_thresh}_len{length}_N{N}.csv"
if os.path.exists(output_path):
    print("Output path already exists!")
    exit()
file = open(output_path, 'w')
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

speeds = []
us = []
vs = []

for Y, X in itertools.product(Ys, Xs):
    print(f"Testing p={X:5.3f}, q={Y:5.3f}")
    R_b = bem.get_R_vector([X, Y, 0], centroids, normals)
    res_vel, sigma = bem.get_jet_dir_and_sigma([X, Y, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)
    speed = np.linalg.norm(res_vel)
    speeds.append(speed)
    us.append(res_vel[0] / speed)
    vs.append(res_vel[1] / speed)
    file.write(f"{X},{Y},{res_vel[0]},{res_vel[1]}\n")
    file.flush()
    os.fsync(file.fileno())

file.close()

P, Q = np.meshgrid(Xs, Ys)
S = np.reshape(np.array(speeds), P.shape)
U = np.reshape(np.array(us), P.shape)
V = np.reshape(np.array(vs), P.shape)

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
cnt = plt.contourf(2 * P / W, Q / W, np.abs(np.arctan2(V, U) + np.pi / 2), levels=128)
plt.xlabel("$\\bar{p}$")
plt.ylabel("$q / w$")
plt.colorbar(label="$|\\theta_j|$")
plt.plot([min(Xs), -W / 2, -W / 2, W / 2, W / 2, max(Xs)], [0, 0, -H, -H, 0, 0], 'k')
plt.show()
