import numpy as np
import math
# from memory_profiler import profile
import multiprocessing as mp
import itertools

import numerical.util.gen_utils as gen
import common.util.plotting_utils as pu
from common.util.vector_utils import mag
from numerical.util.linalg import gauss_seidel, svd_solve
import scipy.linalg
import matplotlib.pyplot as plt


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


def get_R_matrix_low_mem(centroids, normals, areas, dtype=np.float32):
    centroids = np.array(centroids)
    normals = np.array(normals)

    it = itertools.product(enumerate(zip(centroids, normals)), enumerate(zip(centroids, areas)))
    # TODO: Use shared memory (Array)

    R_mat = np.empty((len(centroids), len(centroids)), dtype=dtype)
    for p_arr, q_arr in it:
        p = p_arr[0]
        q = q_arr[0]
        R_mat[p, q] = get_R_factor(p_arr, q_arr)

    # pool = mp.Pool()
    # results = pool.starmap(get_R_factor, it)
    # R_arr = np.array(results, dtype=dtype)
    # R_mat = R_arr.reshape((len(centroids), len(centroids)))

    return R_mat


def get_R_vector(point, centroids, normals):
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(point, 0))
    n_mat = np.broadcast_to(np.expand_dims(normals, 1), (len(centroids), 1, 3))

    dif_mat = np.subtract(ps, qs, dtype=np.float32)
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


def calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density,
                          dtype=np.float32, plot_dist=False) -> np.ndarray:
    # Expand centroids into two n x n matrices.
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(centroids, 0))

    # p - q
    dif_mat = np.subtract(ps, qs, dtype=dtype)  # dif_mat
    # |p - q| , equals zero if p = q
    r_mat = np.linalg.norm(dif_mat, axis=2)
    r_mat = r_mat.reshape((len(r_mat[0]), len(r_mat[0]), 1))
    dif_mat = np.divide(dif_mat, r_mat ** 3, where=r_mat != 0, dtype=dtype)  # Excluding effects of sinks on themselves

    # centroids - bubble pos
    bub_dif = centroids - bubble_pos
    # |centroids - bubble pos|
    bub_r = np.linalg.norm(bub_dif, axis=1)
    bub_dif = np.divide(bub_dif, bub_r.reshape((len(bub_r), 1)) ** 3)

    # sigma * area
    sig_as = np.multiply(areas.reshape((len(areas), 1)), sigmas)

    grad_phi_prime = (bub_dif + np.einsum("ijk,jl->ik", dif_mat, sig_as)) / (4 * np.pi) + 0.5 * sigmas * normals

    speed_squares = np.linalg.norm(grad_phi_prime, axis=1) ** 2

    if plot_dist:
        plt.figure()
        plt.scatter(centroids[:, 0], centroids[:, 2], c=np.sqrt(speed_squares), alpha=0.5)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()

    # Negative due to opposite convention for normals direction in the maths, see Blake and Cerone (1982) figure 1
    force_prime = - 0.5 * density * np.matmul(normals.T, areas * speed_squares)

    return force_prime


def calculate_force_prime_semi_analytic(sinks, centroids, normals, areas, density,
                                        dtype=np.float32, plot_dist=False, output_pressures=False) -> np.ndarray:
    # Expand sinks and centroids into two matrices.
    ps, qs = np.broadcast_arrays(np.expand_dims(centroids, 1), np.expand_dims(sinks, 0))

    # p - q
    dif_mat = np.subtract(ps, qs, dtype=dtype)  # dif_mat
    # |p - q| , equals zero if p = q
    r_mat = np.linalg.norm(dif_mat, axis=2)
    r_mat = r_mat.reshape((dif_mat.shape[0], dif_mat.shape[1], 1))
    dif_mat = np.divide(dif_mat, r_mat ** 3, where=r_mat != 0, dtype=dtype)

    grad_phi_prime = np.sum(dif_mat, axis=1) / (4 * np.pi)

    speed_squares = np.linalg.norm(grad_phi_prime, axis=1) ** 2

    if plot_dist:
        plt.figure()
        plt.scatter(centroids[:, 0], centroids[:, 2], c=np.sqrt(speed_squares), alpha=0.5)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()

    pressures = 0.5 * density * speed_squares

    # Negative due to opposite convention for normals direction in the maths, see Blake and Cerone (1982) figure 1
    force_prime = -0.5 * density * np.matmul(normals.T, areas * speed_squares)

    if output_pressures:
        return force_prime, pressures
    else:
        return force_prime


def calculate_phi_prime(bubble_pos, centroids, areas, sigmas) -> np.ndarray:
    bubble_pos = np.array(bubble_pos)
    centroids = np.array(centroids)
    areas = np.array(areas)

    difs = - centroids + bubble_pos
    return np.sum(- np.multiply(areas, sigmas.reshape((len(sigmas)))) / (4 * np.pi * np.linalg.norm(difs, axis=1)))


def get_jet_dir_and_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=None, R_b=None) \
        -> [np.ndarray, np.ndarray]:
    bubble_pos = np.array(bubble_pos)
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
    R_mat = get_R_matrix_low_mem(centroids, normals, areas)
    R_inv = np.linalg.inv(R_mat)
    source_densities = calculate_sigma(bubble, centroids, normals, areas, m_0, R_inv, R_b)
    # pu.plot_3d_points(centroids, source_densities)
    # print(source_densities)
    res_vel = get_vel(bubble, centroids, areas, source_densities, m_0=m_0)
    print("Resultant velocity = ", res_vel)
    assert (np.all([math.isclose(res_vel[0], 0, abs_tol=1e-16), math.isclose(res_vel[2], 0, abs_tol=1e-16)]))
    assert (res_vel[1] < 0)


if __name__ == "__main__":
    centroids, normals, areas = gen.gen_varied_slot(2500, H=3, W=2, w_thresh=6, density_ratio=0.25)
    m_0 = 1
    print("Making fast matrix...")
    R_mat_fast = get_R_matrix(centroids, normals, areas, dtype=np.float64)
    R_inv_fast = scipy.linalg.inv(R_mat_fast)
    print(R_mat_fast)
    print("Making small matrix...")
    R_mat_small = get_R_matrix_low_mem(centroids, normals, areas, dtype=np.float64)
    R_inv_small = scipy.linalg.inv(R_mat_small)
    print(R_mat_small)
    print(f"Maximum difference = {np.max(R_mat_small - R_mat_fast)}")

    n_points = 32
    bubble_pos_list = np.zeros((n_points, 3))
    ps = np.linspace(-3, 3, n_points)
    bubble_pos_list[:, 0] = ps
    bubble_pos_list[:, 1] = 1
    vels_fast = get_jet_dirs(bubble_pos_list, centroids, normals, areas, m_0, R_inv_fast)
    vels_small = get_jet_dirs(bubble_pos_list, centroids, normals, areas, m_0, R_inv_small)
    theta_js_fast = np.arctan2(vels_fast[:, 1], vels_fast[:, 0]) + 0.5 * np.pi
    theta_js_small = np.arctan2(vels_small[:, 1], vels_small[:, 0]) + 0.5 * np.pi
    plt.plot(ps, theta_js_fast, label="fast")
    plt.plot(ps, theta_js_small, label="small")
    plt.legend()
    plt.show()
