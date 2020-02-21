import math
import numpy as np

from scipy.optimize import minimize_scalar
import numerical.util.gen_utils as gen
import numerical.bem as bem
import scipy.sparse
import matplotlib.pyplot as plt
from common.util.plotting_utils import initialize_plt


def find_slot_peak(theta_b, R_inv, centroids, normals, areas, length):
    m_0 = 1

    def get_neg_theta_j(d):
        res_vel, _ = bem.get_jet_dir_and_sigma([d * np.sin(theta_b),
                                                d * np.cos(theta_b), 0], centroids, normals, areas, m_0=m_0,
                                               R_inv=R_inv)
        return -(math.atan2(res_vel[1], res_vel[0]) + math.pi / 2)

    res = minimize_scalar(get_neg_theta_j, bounds=(0, length / 2))
    print(f"Optimization finished with {res.nfev} evaluations, theta_b={theta_b}  d={res.x}  n={n}")
    return res.x


n = 20000
W = 2
H = 2
length = 50

centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=length, depth=25, w_thresh=3,
                                                density_ratio=0.1)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

theta_bs = np.linspace(0, np.pi / 2 - 0.1, 16)
ds = []
for t_b in theta_bs:
    d = find_slot_peak(t_b, R_inv, centroids, normals, areas, length)
    ds.append(d)

initialize_plt(font_size=18, line_scale=2)
plt.plot(theta_bs, 2 * np.array(ds) / W)
plt.xlabel("$\\theta_b$")
plt.ylabel("$(2d / w)^\\star$")
plt.show()