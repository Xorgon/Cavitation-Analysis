import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

import numerical.bem as bem
import numerical.util.gen_utils as gen

if not os.path.exists("../model_outputs/step_vel_data"):
    os.makedirs("../model_outputs/step_vel_data")

offset = 0.05
H = 2
N = 32
Ys = np.concatenate([np.linspace(offset, 3, np.round(3 * N / 4) - 1), [1], np.linspace(3 + 0.1, 5, np.ceil(N / 4))])
Ys = sorted(Ys)
Xs = np.concatenate([np.linspace(-15, -5.1, np.ceil(N / 8)),
                     np.linspace(-5, 5, np.round(3 * N / 4)),
                     np.linspace(5.1, 15, np.ceil(N / 8))])

# qs = np.linspace(3 * w, 5 * w, 16)
# ps = np.linspace(0, 5 * w / 2, 16)

m_0 = 1
n = 15000
density_ratio = 0.1
thresh_dist = 12
length = 100

centroids, normals, areas = gen.gen_varied_step(n=n, H=H, length=length, depth=50, thresh_dist=thresh_dist,
                                                density_ratio=density_ratio)

centroids_file = open(f"../model_outputs/step_vel_data/centroids_n{n}_H{H:.2f}"
                      f"_drat{density_ratio}_thresh{thresh_dist}_len{length}.csv", 'w')
for c in centroids:
    centroids_file.write(f"{c[0]},{c[1]},{c[2]}\n")
centroids_file.close()

normals_file = open(f"../model_outputs/step_vel_data/normals_n{n}_H{H:.2f}"
                    f"_drat{density_ratio}_thresh{thresh_dist}_len{length}.csv", 'w')
for normal in normals:
    normals_file.write(f"{normal[0]},{normal[1]},{normal[2]}\n")
normals_file.close()

output_path = f"../model_outputs/step_vel_data/vel_sweep_n{n}_H{H:.2f}" \
              f"_drat{density_ratio}_thresh{thresh_dist}_len{length}_N{N}.csv"
if os.path.exists(output_path):
    print("Output path already exists!")
    exit()
file = open(output_path, 'w')
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
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
