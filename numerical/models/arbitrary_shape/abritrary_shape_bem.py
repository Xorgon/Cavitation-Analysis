import numerical.util.stl_utils as stl
import numpy as np
import scipy
import numerical.bem as bem

import numerical.util.gen_utils as gen

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
