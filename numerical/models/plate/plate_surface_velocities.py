import matplotlib.pyplot as plt
import numpy as np
import math

import numerical.bem as bem
import numerical.util.gen_utils as gen


def get_analytical_velocities(centroids, bubble_pos, m_0=1):
    vels = []
    for pos in centroids:
        pos, bubble_pos = np.array(pos), np.array(bubble_pos)

        bubble_vel = (m_0 * (pos - bubble_pos) / (4 * np.pi * np.linalg.norm(pos - bubble_pos) ** 3))

        mirror_pos = np.array([bubble_pos[0], -bubble_pos[1], bubble_pos[2]])
        mirror_vel = (m_0 * (pos - mirror_pos) / (4 * np.pi * np.linalg.norm(pos - mirror_pos) ** 3))
        vels.append(bubble_vel + mirror_vel)

    return vels


def get_bem_velocities(bubble_pos, centroids, areas, sigmas):
    # Expand centroids into two n x n matrices.
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(centroids, 0))

    # p - q
    dif_mat = np.subtract(ps, qs)  # dif_mat
    # |p - q| , equals zero if p = q
    r_mat = np.linalg.norm(dif_mat, axis=2)
    r_mat = r_mat.reshape((len(r_mat[0]), len(r_mat[0]), 1))
    norm_dif_mat = np.zeros((len(centroids), len(centroids), 3))
    dif_mat = np.divide(dif_mat, r_mat ** 3, out=norm_dif_mat, where=r_mat != 0)

    # centroids - bubble pos
    bub_dif = centroids - bubble_pos
    # |centroids - bubble pos|
    bub_r = np.linalg.norm(bub_dif, axis=1)
    bub_dif = np.divide(bub_dif, bub_r.reshape((len(bub_r), 1)) ** 3)

    # sigma * area
    sig_as = np.multiply(areas.reshape((len(areas), 1)), sigmas)

    grad_phi_prime = (bub_dif + np.einsum("ijk,jl->ik", dif_mat, sig_as)) / (4 * np.pi)

    grad_phi_prime[:, 1] = np.zeros((len(grad_phi_prime)))

    return grad_phi_prime


bubble_radius = 0.002
bubble_pos = np.array([0, bubble_radius * 5, 0])

length = 0.1
depth = 0.1
n = 10000
centroids, normals, areas = gen.gen_plane([- length / 2, 0, - depth / 2],
                                          [- length / 2, 0, depth / 2],
                                          [length / 2, 0, - depth / 2],
                                          n)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas)
R_inv = np.linalg.inv(R_matrix)
# pu.plot_3d_point_sets([cs, sinks])

sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, 1, R_inv=R_inv)

analytical_vels = get_analytical_velocities(centroids, bubble_pos)
analytical_speeds = np.linalg.norm(analytical_vels, axis=1)
density = 997
force_prime = 0.5 * density * np.matmul(normals.T, areas * analytical_speeds ** 2)
print(force_prime)


bem_vels = get_bem_velocities(bubble_pos, centroids, areas, sigmas)
bem_speeds = np.linalg.norm(bem_vels, axis=1)

min_speed = np.min([analytical_speeds, bem_speeds])
max_speed = np.max([analytical_speeds, bem_speeds])

fig, axs = plt.subplots(1, 3)
axs[0].scatter(centroids[:, 0], centroids[:, 2],
               c=np.linalg.norm(analytical_vels, axis=1), vmin=min_speed, vmax=max_speed, marker='s')
axs[0].axis('equal')

axs[1].scatter(centroids[:, 0], centroids[:, 2],
               c=np.linalg.norm(bem_vels, axis=1), vmin=min_speed, vmax=max_speed, marker='s')
axs[1].axis('equal')

axs[2].scatter(centroids[:, 0], centroids[:, 2],
               c=np.linalg.norm(bem_vels - analytical_vels, axis=1), marker='s')#, vmin=min_speed, vmax=max_speed)
axs[2].axis('equal')
plt.show()
