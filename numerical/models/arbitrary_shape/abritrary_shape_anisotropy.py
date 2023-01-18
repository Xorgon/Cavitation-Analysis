import numerical.util.stl_utils as stl
import numpy as np
import scipy
import numerical.bem as bem

import numerical.util.gen_utils as gen


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
R_init = 0.0015
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

centroids, normals, areas = stl.load_cna_from_mesh(
    "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Abritrary Shape/CAD/Shape 1/surface.STL")

centroids = centroids + np.array([-0.025, -0.001, -0.025])
# centroids, normals, areas = stl.load_cna_from_mesh(
#     'C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/CAD/n4-corner/n4-corner-surface.STL')
# centroids, normals, areas = gen.gen_corner(4900, 50, 50, angle=np.pi / 4)
# centroids, normals, areas = stl.load_cna_from_mesh(
# "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Prisms/CAD/equi-triangle/equi-triangle.STL")
print(centroids.shape, normals.shape, areas.shape)

print(f"n = {len(centroids)}.")
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float64)
R_inv = scipy.linalg.inv(R_matrix)

condition_number_1 = np.linalg.norm(R_inv, 1) * np.linalg.norm(R_matrix, 1)
condition_number_inf = np.linalg.norm(R_inv, np.inf) * np.linalg.norm(R_matrix, np.inf)
print(f"Condition numbers: 1 norm = {condition_number_1}, inf norm = {condition_number_inf}")
