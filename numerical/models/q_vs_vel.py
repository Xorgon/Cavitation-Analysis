import numpy as np
import itertools
import numerical.util.gen_utils as gen
from common.util.plotting_utils import initialize_plt
import numerical.bem as bem
import scipy
import matplotlib.pyplot as plt

w = 2
hs = [2]
ps = np.linspace(0, 10, 16)
qs = np.linspace(1, 4, 16)

n = 10000
m_0 = 1

all_ps = []
all_qs = []
disps = []
counter = 0
total = len(hs) * len(ps) * len(qs)
for h in hs:
    centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=50, depth=50, w_thresh=6,
                                                    density_ratio=0.25)
    R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
    R_inv = scipy.linalg.inv(R_matrix)
    for p, q in itertools.product(ps, qs):
        print(f"{100 * counter / total:.2f}% complete...")
        counter += 1
        vel, _ = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0, R_inv)
        all_ps.append(p)
        all_qs.append(q)
        disps.append(np.linalg.norm(vel))

initialize_plt()
plt.scatter(all_qs, disps, c=2 * np.array(all_ps) / w)
plt.xlabel("q")
plt.ylabel("vel")
plt.xscale('log')
plt.yscale('log')
plt.colorbar(label="$\\bar{p}$")
plt.show()
