import numpy as np
import itertools
import numerical.util.gen_utils as gen
from common.util.plotting_utils import initialize_plt
import numerical.bem as bem
import scipy
import matplotlib.pyplot as plt

W = 2
Hs = [2]
Xs = np.linspace(0, 10, 16)
Ys = np.linspace(1, 4, 16)

n = 10000
m_0 = 66

all_Xs = []
all_Ys = []
disps = []
counter = 0
total = len(Hs) * len(Xs) * len(Ys)
for H in Hs:
    centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=50, depth=50, w_thresh=6,
                                                    density_ratio=0.25)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)
    for X, Y in itertools.product(Xs, Ys):
        print(f"{100 * counter / total:.2f}% complete...")
        counter += 1
        vel, _ = bem.get_jet_dir_and_sigma([X, Y, 0], centroids, normals, areas, m_0, R_inv)
        all_Xs.append(X)
        all_Ys.append(Y)
        disps.append(np.linalg.norm(vel))

initialize_plt()
plt.scatter(all_Ys, disps, c=2 * np.array(all_Xs) / W)
plt.xlabel("Y")
plt.ylabel("vel")
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label="$x$")
plt.show()
