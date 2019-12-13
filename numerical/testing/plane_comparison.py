import numpy as np
import math
from scipy.optimize import curve_fit
import numerical.util.gen_utils as gen
import numerical.bem as bem
import matplotlib.pyplot as plt
from numerical.potential_flow.elements import Source3D
from common.util.plotting_utils import plot_3d_point_sets
import os
import common.util.file_utils as file
import scipy.sparse

length = 50
n = 2500
centroids, normals, areas = gen.gen_plane([-length / 2, 0, -length / 2],
                                          [length / 2, 0, -length / 2],
                                          [-length / 2, 0, length / 2], n)

R = bem.get_R_matrix(centroids, normals, areas)
R_inv = scipy.linalg.inv(R)

rs = np.linspace(1, 10, 64)
pos = np.array([[0] * len(rs), rs, [0] * len(rs)]).T

m_0 = 1
jet_dirs = bem.get_jet_dirs(pos, centroids, normals, areas, m_0=m_0, R_inv=R_inv)
bem_speeds = np.linalg.norm(jet_dirs, axis=1)

mir_speeds = []
for r in rs:
    mir = Source3D(0, -r, 0, m_0)
    mir_speeds.append(np.linalg.norm(mir.get_vel(0, r, 0)))

plt.plot(rs, bem_speeds, label="BEM")
plt.plot(rs, mir_speeds, label="Mirror")
plt.xlabel("r")
plt.ylabel("Speed")
plt.show()
