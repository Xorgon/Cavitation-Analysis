import math
import numpy as np
import scipy.linalg

import util.gen_utils as gen
from util.vector_utils import mag


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
    # Expand centroids into two n x n matrices.
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(centroids, 0))

    # Expand normals into an n x n x 3 matrix.
    n_mat = np.broadcast_to(np.expand_dims(normals, 1), (len(ps), len(qs), 3))

    # Expand areas into an n x n matrix.
    a_mat = np.broadcast_to(np.expand_dims(areas, 0), (len(ps), len(qs)))

    # p - q
    res_mat = np.subtract(ps, qs, dtype=dtype)  # dif_mat

    # |p - q| , equals zero if p = q
    r_mat = np.linalg.norm(res_mat, axis=2)

    # n . (p - q)
    res_mat = np.einsum('...k,...k', n_mat, res_mat)  # n_dot_dif

    # a_q * (n_p . (p - q)) / (4 * pi * |p - q| ^ 3)
    res_mat = np.divide(a_mat * res_mat, (4 * np.pi * r_mat ** 3), where=r_mat != 0, dtype=dtype)

    # Set diagonal to value obtained from integrating over the singularity in a panel (the effect on itself).
    np.fill_diagonal(res_mat, 0.5)
    return res_mat


def get_R_factor(arr_p, arr_q):
    """
    :param arr_p: [p, (centroid, normal)]
    :param arr_q: [q, (centroid, area)]
    :return: R_factor
    """
    p = arr_p[0]
    p_cent = arr_p[1][0]
    p_norm = arr_p[1][1]

    q = arr_q[0]
    q_cent = arr_q[1][0]
    q_area = arr_q[1][1]

    if p != q:
        r = p_cent - q_cent
        return q_area * np.dot(p_norm, r) / (4 * np.pi * mag(r) ** 3)
        # return 0
    else:
        return 0.5


def get_R_vector(point, centroids, normals):
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(point, 0))
    n_mat = np.broadcast_to(np.expand_dims(normals, 1), (len(centroids), 1, 3))

    dif_mat = np.subtract(ps, qs, dtype=np.float64)
    n_dot_dif = np.einsum('...k,...k', n_mat, dif_mat)
    r_mat = np.linalg.norm(dif_mat, axis=2)
    res_mat = np.divide(n_dot_dif, (4 * np.pi * r_mat ** 3), where=r_mat != 0)
    return res_mat


def calculate_sigma(bubble_pos, centroids, normals, areas, m_0, R_inv=None, R_b=None) -> np.ndarray:
    if R_b is None:
        R_b = get_R_vector(bubble_pos, centroids, normals)

    if R_inv is None:
        R = get_R_matrix(centroids, normals, areas)
        # sigma = gauss_seidel(R, -m_0 * R_b, max_res=1e-12)
        sigma = scipy.linalg.solve(R, -m_0 * R_b)  # Faster to do this than inverting the matrix
        # sigma = svd_solve(R, -m_0 * R_b)
    else:
        sigma = -m_0 * np.dot(R_inv, R_b)

    return sigma


def get_jet_dir_and_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=None, R_b=None) \
        -> [np.ndarray, np.ndarray]:
    sigma = calculate_sigma(bubble_pos, centroids, normals, areas, m_0, R_inv, R_b)
    vel = get_vel(bubble_pos, centroids, areas, sigma)
    return vel, sigma


def get_jet_dirs(bubble_pos_list, centroids, normals, areas, m_0=1, R_inv=None, R_b=None, verbose=False):
    vels = np.empty((len(bubble_pos_list), 3))

    for i, pos in enumerate(bubble_pos_list):
        if verbose:
            print(f"{100 * i / len(bubble_pos_list):.2f}% complete...")
        vels[i] = get_jet_dir_and_sigma(pos, centroids, normals, areas, m_0=m_0, R_inv=R_inv, R_b=R_b)[0]

    return vels


def test_run_analysis():
    centroids, normals, areas = gen.gen_slot(500)
    bubble = np.array([0, 1, 0])
    m_0 = 1

    R_b = get_R_vector(bubble, centroids, normals)
    R_mat = get_R_matrix(centroids, normals, areas)
    R_inv = np.linalg.inv(R_mat)
    source_densities = calculate_sigma(bubble, centroids, normals, areas, m_0, R_inv, R_b)
    res_vel = get_vel(bubble, centroids, areas, source_densities, m_0=m_0)
    print("Resultant velocity = ", res_vel)
    assert (np.all([math.isclose(res_vel[0], 0, abs_tol=1e-16), math.isclose(res_vel[2], 0, abs_tol=1e-16)]))
    assert (res_vel[1] < 0)
