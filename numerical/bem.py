import numpy as np
from memory_profiler import profile

import numerical.util.gen_utils as gen
import common.util.plotting_utils as pu
from numerical.util.linalg import gauss_seidel


def get_vel(point, centroids, areas, source_densities, bubble_pos=None, m_0=1):
    """
    Calculate the fluid velocity at a point.

    :param point: 3D point, [x, y, z]
    :param centroids:
    :param areas:
    :param source_densities:
    :param bubble_pos: Bubble position vector, if None bubble is not included.
    :param m_0:
    :return:
    """
    s_mat = np.array(source_densities)
    if len(s_mat.shape) == 1:
        s_mat = np.expand_dims(s_mat, 1)
    ps, qs = np.array(centroids), np.array(point)
    a_mat = np.expand_dims(areas, 1)

    dif_mat = np.subtract(ps, qs, dtype=np.float32)
    r_mat = np.expand_dims(np.linalg.norm(dif_mat, axis=1), 1)
    res_mat = np.divide(s_mat * a_mat * dif_mat, (4 * np.pi * r_mat ** 3), where=r_mat != 0)
    boundary_vel_sum = np.sum(res_mat, 0)

    if bubble_pos is None:
        vel = boundary_vel_sum
    else:
        point = np.array(point)
        bubble_pos = np.array(bubble_pos)
        bubble_vel = -(point - bubble_pos) * m_0 / (
                4 * np.pi * ((point[0] - bubble_pos[0]) ** 2
                             + (point[1] - bubble_pos[1]) ** 2
                             + (point[2] - bubble_pos[2]) ** 2) ** (3 / 2))
        vel = boundary_vel_sum + bubble_vel
    return vel


# @profile
def get_R_matrix(centroids, normals, areas, dtype=np.float32):
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(centroids, 0))
    n_mat = np.broadcast_to(np.expand_dims(normals, 1), (len(ps), len(qs), 3))
    a_mat = np.broadcast_to(np.expand_dims(areas, 1), (len(ps), len(qs)))

    res_mat = np.subtract(ps, qs, dtype=dtype)  # dif_mat
    r_mat = np.linalg.norm(res_mat, axis=2)
    res_mat = np.einsum('...k,...k', n_mat, res_mat)  # n_dot_dif
    res_mat = np.divide(a_mat * res_mat, (4 * np.pi * r_mat ** 3), where=r_mat != 0, dtype=dtype)
    for i in range(len(centroids)):
        res_mat[i, i] = 0.5
    return res_mat


def get_R_vector(point, centroids, normals):
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(point, 0))
    n_mat = np.broadcast_to(np.expand_dims(normals, 1), (len(centroids), 1, 3))

    dif_mat = np.subtract(ps, qs, dtype=np.float64)
    n_dot_dif = np.einsum('...k,...k', n_mat, dif_mat)
    r_mat = np.linalg.norm(dif_mat, axis=2)
    res_mat = np.divide(n_dot_dif, (4 * np.pi * r_mat ** 3), where=r_mat != 0)
    return res_mat


def calculate_sigma(bubble_pos, centroids, normals, areas, m_0, R_inv=None, ret_R_b=False):
    R_b = get_R_vector(bubble_pos, centroids, normals)

    if R_inv is None:
        R = get_R_matrix(centroids, normals, areas)
        sigma = gauss_seidel(R, -m_0 * R_b, max_res=1e-12)
        # sigma = linalg.solve(R, m_0 * R_b)
    else:
        sigma = -m_0 * np.dot(R_inv, R_b)

    if ret_R_b:
        return sigma, R_b
    else:
        return sigma


def get_jet_dir_and_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=None, ret_R_b=False):
    if ret_R_b:
        sigma, R_b = calculate_sigma(bubble_pos, centroids, normals, areas, m_0, R_inv, ret_R_b)
    else:
        sigma = calculate_sigma(bubble_pos, centroids, normals, areas, m_0, R_inv, ret_R_b)
    vel = get_vel(bubble_pos, centroids, areas, sigma)

    if ret_R_b:
        return vel, sigma, R_b
    else:
        return vel, sigma


def test_run_analysis():
    centroids, normals, areas = gen.gen_slot(500)
    print(len(centroids))
    bubble = np.array([0, 1, 0])
    m_0 = 1

    R_b = get_R_vector(bubble, centroids, normals)
    R_mat = get_R_matrix(centroids, normals, areas)
    R_inv = np.linalg.inv(R_mat)
    source_densities = m_0 * np.dot(R_inv, R_b)
    pu.plot_3d_points(centroids, source_densities)
    print(source_densities)
    res_vel = get_vel(bubble, centroids, areas, source_densities, m_0)
    print("Resultant velocity = ", res_vel)


if __name__ == "__main__":
    test_run_analysis()
