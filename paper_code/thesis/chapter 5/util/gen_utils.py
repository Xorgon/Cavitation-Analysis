import itertools as it
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np

from util.plotting_utils import plot_3d_point_sets
import util.vector_utils as vect


def gen_plane(corner_1, corner_2, corner_3, n, filename=None):
    """
    Generates the control points, normals, and sinks for a plane.
    :param corner_1: First corner.
    :param corner_2: Second corner (adjacent to corner_1).
    :param corner_3: Third corner (adjacent to corner_1).
    :param n: Number of panels to generate.
    :param filename: File name root for saving mesh.
    :return: centroids, normals, areas
    """
    corner_1 = np.array(corner_1)
    corner_2 = np.array(corner_2)
    corner_3 = np.array(corner_3)

    # Define the two vectors to move in the plane.
    vect_1 = corner_2 - corner_1
    vect_2 = corner_3 - corner_1
    normal = np.cross(vect_1, vect_2)
    normal = normal / vect.mag(normal)

    ratio = vect.mag(vect_1) / vect.mag(vect_2)
    side_1_n = (n * ratio) ** 0.5
    side_2_n = side_1_n / ratio

    side_1_n = int(round(side_1_n))
    side_2_n = int(round(side_2_n))

    side_1_panel_h = vect.mag(vect_1) / side_1_n
    side_2_panel_h = vect.mag(vect_2) / side_2_n

    vect_1_hat = vect_1 / vect.mag(vect_1)
    vect_2_hat = vect_2 / vect.mag(vect_2)

    panel_area = side_1_panel_h * side_2_panel_h  # Assuming perpendicular

    centroids = []
    normals = []
    areas = []

    if filename is not None:
        f = open(f"{filename[:-4]}-{side_1_n}x{side_2_n}{filename[-4:]}", "w")
        f.write("x,y,z,0\n")
        to_add = []  # To save for writing at the end.
        for i, j in it.product(range(side_1_n), range(side_2_n)):
            panel_c1 = corner_1 + i * side_1_panel_h * vect_1_hat + j * side_2_panel_h * vect_2_hat
            panel_c2 = panel_c1 + side_1_panel_h * vect_1_hat
            panel_c3 = panel_c1 + side_2_panel_h * vect_2_hat
            panel_c4 = panel_c1 + side_1_panel_h * vect_1_hat + side_2_panel_h * vect_2_hat

            # Adding a zero so that it can be used in paraview properly
            f.write(f"{panel_c1[0]},{panel_c1[1]},{panel_c1[2]},0\n")
            if i == side_1_n - 1:
                to_add.append(f"{panel_c2[0]},{panel_c2[1]},{panel_c2[2]},0\n")
            if j == side_2_n - 1:
                f.write(f"{panel_c3[0]},{panel_c3[1]},{panel_c3[2]},0\n")
            if i == side_1_n - 1 and j == side_2_n - 1:
                to_add.append(f"{panel_c4[0]},{panel_c4[1]},{panel_c4[2]},0\n")
        for s in to_add:
            f.write(s)
        f.close()

    for i, j in it.product(np.linspace(side_1_panel_h / 2, vect.mag(vect_1) - side_1_panel_h / 2, side_1_n),
                           np.linspace(side_2_panel_h / 2, vect.mag(vect_2) - side_2_panel_h / 2, side_2_n)):
        c = corner_1 + i * vect_1_hat + j * vect_2_hat
        centroids.append(c)
        normals.append(normal)
        areas.append(panel_area)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_varied_simple_plane(dense_length, n_dense, sparse_length, n_sparse):
    centroids = []
    normals = []
    areas = []

    sparse_area = sparse_length ** 2 - dense_length ** 2
    npa = n_sparse / sparse_area

    # CENTRAL PANEL
    p_centroids, p_normals, p_areas = gen_plane([- dense_length / 2, 0, - dense_length / 2],
                                                [- dense_length / 2, 0, dense_length / 2],
                                                [dense_length / 2, 0, - dense_length / 2],
                                                n_dense)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # CORNER -,-
    p_centroids, p_normals, p_areas = gen_plane([- sparse_length / 2, 0, - sparse_length / 2],
                                                [- sparse_length / 2, 0, -dense_length / 2],
                                                [-dense_length / 2, 0, - sparse_length / 2],
                                                npa * (sparse_length - dense_length) ** 2 / 4)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # CORNER -,+
    p_centroids, p_normals, p_areas = gen_plane([-sparse_length / 2, 0, dense_length / 2],
                                                [- sparse_length / 2, 0, sparse_length / 2],
                                                [-dense_length / 2, 0, dense_length / 2],
                                                npa * (sparse_length - dense_length) ** 2 / 4)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # CORNER +,-
    p_centroids, p_normals, p_areas = gen_plane([dense_length / 2, 0, -sparse_length / 2],
                                                [dense_length / 2, 0, -dense_length / 2],
                                                [sparse_length / 2, 0, -sparse_length / 2],
                                                npa * (sparse_length - dense_length) ** 2 / 4)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # CORNER +,+
    p_centroids, p_normals, p_areas = gen_plane([dense_length / 2, 0, dense_length / 2],
                                                [dense_length / 2, 0, sparse_length / 2],
                                                [sparse_length / 2, 0, dense_length / 2],
                                                npa * (sparse_length - dense_length) ** 2 / 4)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # EDGE Z-
    p_centroids, p_normals, p_areas = gen_plane([-dense_length / 2, 0, -sparse_length / 2],
                                                [-dense_length / 2, 0, -dense_length / 2],
                                                [dense_length / 2, 0, -sparse_length / 2],
                                                npa * dense_length * (sparse_length - dense_length) / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # EDGE Z+
    p_centroids, p_normals, p_areas = gen_plane([-dense_length / 2, 0, dense_length / 2],
                                                [-dense_length / 2, 0, sparse_length / 2],
                                                [dense_length / 2, 0, dense_length / 2],
                                                npa * dense_length * (sparse_length - dense_length) / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # EDGE X-
    p_centroids, p_normals, p_areas = gen_plane([-sparse_length / 2, 0, -dense_length / 2],
                                                [-sparse_length / 2, 0, dense_length / 2],
                                                [-dense_length / 2, 0, -dense_length / 2],
                                                npa * dense_length * (sparse_length - dense_length) / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    # EDGE X+
    p_centroids, p_normals, p_areas = gen_plane([dense_length / 2, 0, -dense_length / 2],
                                                [dense_length / 2, 0, dense_length / 2],
                                                [sparse_length / 2, 0, -dense_length / 2],
                                                npa * dense_length * (sparse_length - dense_length) / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_slot(n=3000, H=5, W=5, length=50, depth=50):
    centroids = []
    normals = []
    areas = []

    total_surface_length = W + 2 * H + (length - W - 2)
    n_slot_floor = int(round(n * W / total_surface_length))
    n_slot_wall = int(round(n * H / total_surface_length))
    print("n_slot_wall = {0}".format(n_slot_wall))
    n_surface_boundary = int(round(n * (length - W) / 2) / total_surface_length)

    ######################
    # Surface boundaries #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([W / 2, 0, - depth / 2],
                                                [W / 2, 0, depth / 2],
                                                [length / 2, 0, - depth / 2],
                                                n_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                [-W / 2, 0, - depth / 2],
                                                n_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot floor         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, -H, - depth / 2],
                                                [-W / 2, -H, depth / 2],
                                                [W / 2, -H, - depth / 2],
                                                n_slot_floor)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot walls         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, 0, - depth / 2],
                                                [-W / 2, 0, depth / 2],
                                                [-W / 2, -H, - depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([W / 2, -H, - depth / 2],
                                                [W / 2, -H, depth / 2],
                                                [W / 2, 0, - depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_varied_slot(n=3000, H=3, W=2, length=50, depth=50, w_thresh=6, density_ratio=0.25, save_to_files=False):
    """
    Generates a slot with varying panel density.

    :param n: Approximate number of panels
    :param H: Slot height
    :param W: Slot width
    :param length: Geometry length
    :param depth: Geometry depth
    :param w_thresh: w threshold at which to reduce density (density reduced starting after position w_thresh * w / 2)
    :param density_ratio: Ratio of densities
    :return: centroids, normals, areas
    """
    if w_thresh < 0:
        raise ValueError(f"gen_varied_slot w_thresh cannot be less than zero ({w_thresh})")
    if W * w_thresh > length:
        warnings.warn("w threshold too high, reverting to gen_slot")
        return gen_slot(n, H, W, length, depth)

    centroids = []
    normals = []
    areas = []

    total_surface_length = length + 2 * H
    dense_surface_length = 2 * H + W * w_thresh
    sparse_surface_length = total_surface_length - dense_surface_length

    zeta = dense_surface_length / (density_ratio * sparse_surface_length)
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    n_slot_floor = int(round(n_dense * W / dense_surface_length))
    n_slot_wall = int(round(n_dense * H / dense_surface_length))
    n_dense_surface_boundary = int(round(n_dense * (W * (w_thresh - 1)) / 2) / dense_surface_length)
    n_sparse_surface_boundary = int(round(n_sparse / 2))

    ######################
    # Surface boundaries #
    ######################
    filename = "model_outputs/left_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W / 2, 0, - depth / 2],
                                                [W / 2, 0, depth / 2],
                                                [W * w_thresh / 2, 0, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/left_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W * w_thresh / 2, 0, - depth / 2],
                                                [W * w_thresh / 2, 0, depth / 2],
                                                [length / 2, 0, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W * w_thresh / 2, 0, - depth / 2],
                                                [-W * w_thresh / 2, 0, depth / 2],
                                                [-W / 2, 0, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                [-W * w_thresh / 2, 0, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot floor         #
    ######################
    filename = "model_outputs/floor.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, -H, - depth / 2],
                                                [-W / 2, -H, depth / 2],
                                                [W / 2, -H, - depth / 2],
                                                n_slot_floor, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot walls         #
    ######################
    filename = "model_outputs/left_wall.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, 0, - depth / 2],
                                                [-W / 2, 0, depth / 2],
                                                [-W / 2, -H, - depth / 2],
                                                n_slot_wall, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_wall.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W / 2, -H, - depth / 2],
                                                [W / 2, -H, depth / 2],
                                                [W / 2, 0, - depth / 2],
                                                n_slot_wall, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_asymmetric_varied_slot(n=3000, H=3, W=2, length=50, depth=50, w_thresh=6, density_ratio=0.25,
                               asym_angle=np.pi / 8, save_to_files=False):
    """
    Generates a slot with varying panel density.

    :param n: Approximate number of panels
    :param H: Slot height
    :param W: Slot width
    :param length: Geometry length
    :param depth: Geometry depth
    :param w_thresh: w threshold at which to reduce density (density reduced starting after position w_thresh * w / 2)
    :param density_ratio: Ratio of densities
    :return: centroids, normals, areas
    """
    if w_thresh < 0:
        raise ValueError(f"gen_varied_slot w_thresh cannot be less than zero ({w_thresh})")
    if W * w_thresh > length:
        warnings.warn("w threshold too high, reverting to gen_slot")
        return gen_slot(n, H, W, length, depth)

    centroids = []
    normals = []
    areas = []

    total_asym_offset = H * np.tan(asym_angle)
    if total_asym_offset / 2 >= W or total_asym_offset / 2 >= W * w_thresh / 2 - W:
        raise ValueError(f"asym_angle is too high ({asym_angle}), "
                         f"width is too low ({W}), "
                         f"or w_thresh is too low ({w_thresh})")

    total_surface_length = length + 2 * H
    dense_surface_length = 2 * H + W * w_thresh
    sparse_surface_length = total_surface_length - dense_surface_length

    zeta = dense_surface_length / (density_ratio * sparse_surface_length)
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    n_slot_floor = int(round(n_dense * W / dense_surface_length))
    n_slot_wall = int(round(n_dense * H / dense_surface_length))
    n_dense_surface_boundary = int(round(n_dense * (W * (w_thresh - 1)) / 2) / dense_surface_length)
    n_sparse_surface_boundary = int(round(n_sparse / 2))

    ######################
    # Surface boundaries #
    ######################
    filename = "model_outputs/left_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W / 2 + total_asym_offset / 2, 0, - depth / 2],
                                                [W / 2 + total_asym_offset / 2, 0, depth / 2],
                                                [W * w_thresh / 2, 0, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/left_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W * w_thresh / 2, 0, - depth / 2],
                                                [W * w_thresh / 2, 0, depth / 2],
                                                [length / 2, 0, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W * w_thresh / 2, 0, - depth / 2],
                                                [-W * w_thresh / 2, 0, depth / 2],
                                                [-W / 2, 0, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                [-W * w_thresh / 2, 0, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot floor         #
    ######################
    filename = "model_outputs/floor.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, -H, - depth / 2],
                                                [-W / 2, -H, depth / 2],
                                                [W / 2 - total_asym_offset / 2, -H, - depth / 2],
                                                n_slot_floor, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot walls         #
    ######################
    filename = "model_outputs/left_wall.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-W / 2, 0, - depth / 2],
                                                [-W / 2, 0, depth / 2],
                                                [-W / 2, -H, - depth / 2],
                                                n_slot_wall, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/right_wall.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([W / 2 - total_asym_offset / 2, -H, - depth / 2],
                                                [W / 2 - total_asym_offset / 2, -H, depth / 2],
                                                [W / 2 + total_asym_offset / 2, 0, - depth / 2],
                                                n_slot_wall, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_varied_step(n=3000, H=3, length=50, depth=50, thresh_dist=6, density_ratio=0.25, save_to_files=False):
    """
    Generates a step with varying panel density.

    :param n: Approximate number of panels
    :param H: Step height
    :param length: Geometry length
    :param depth: Geometry depth
    :param thresh_dist: threshold at which to reduce density
    :param density_ratio: Ratio of densities
    :return: centroids, normals, areas
    """
    if thresh_dist < 0:
        raise ValueError(f"gen_varied_step w_thresh cannot be less than zero ({thresh_dist})")
    if thresh_dist > length / 2:
        warnings.warn("threshold distance too high.")
        raise ValueError

    centroids = []
    normals = []
    areas = []

    total_surface_length = length + H
    dense_surface_length = H + thresh_dist * 2
    sparse_surface_length = total_surface_length - dense_surface_length

    zeta = dense_surface_length / (density_ratio * sparse_surface_length)
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    n_wall = int(round(n_dense * H / dense_surface_length))
    n_dense_surface_boundary = int(round((n_dense - n_wall) / 2))
    n_sparse_surface_boundary = int(round(n_sparse / 2))

    ######################
    # Surface boundaries #
    ######################
    filename = "model_outputs/step_left_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([0, -H, - depth / 2],
                                                [0, -H, depth / 2],
                                                [thresh_dist, -H, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/step_left_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([thresh_dist, -H, - depth / 2],
                                                [thresh_dist, -H, depth / 2],
                                                [length / 2, -H, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/step_right_dense.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-thresh_dist, 0, - depth / 2],
                                                [-thresh_dist, 0, depth / 2],
                                                [0, 0, - depth / 2],
                                                n_dense_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    filename = "model_outputs/step_right_sparse.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                [-thresh_dist, 0, - depth / 2],
                                                n_sparse_surface_boundary, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Step wall         #
    ######################
    filename = "model_outputs/step_wall.csv" if save_to_files else None
    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                [0, -H, - depth / 2],
                                                n_wall, filename=filename)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_corner(n=3000, length=25, depth=50, angle=np.pi / 2):
    centroids = []
    normals = []
    areas = []

    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                [length, 0, - depth / 2],
                                                n / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([length * math.cos(angle), length * math.sin(angle), - depth / 2],
                                                [length * math.cos(angle), length * math.sin(angle), depth / 2],
                                                [0, 0, - depth / 2],
                                                n / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_varied_corner(n=3000, length=25, depth=50, angle=np.pi / 2, thresh=10, density_ratio=0.5):
    centroids = []
    normals = []
    areas = []

    zeta = 2 * thresh / (density_ratio * 2 * (length - thresh))
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                [thresh, 0, - depth / 2],
                                                n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([thresh, 0, - depth / 2],
                                                [thresh, 0, depth / 2],
                                                [length, 0, - depth / 2],
                                                n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([thresh * math.cos(angle), thresh * math.sin(angle), - depth / 2],
                                                [thresh * math.cos(angle), thresh * math.sin(angle), depth / 2],
                                                [0, 0, - depth / 2],
                                                n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([length * math.cos(angle), length * math.sin(angle), - depth / 2],
                                                [length * math.cos(angle), length * math.sin(angle), depth / 2],
                                                [thresh * math.cos(angle), thresh * math.sin(angle), - depth / 2],
                                                n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_vertical_varied_corner(n=3000, length=25, depth=50, angle=np.pi / 2, thresh=10, density_ratio=0.5):
    """ Same as varied corner, except that the plane of symmetry is vertical. """
    centroids = []
    normals = []
    areas = []

    zeta = 2 * thresh / (density_ratio * 2 * (length - thresh))
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                [thresh * math.sin(angle / 2), thresh * math.cos(angle / 2),
                                                 - depth / 2],
                                                n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane(
        [thresh * math.sin(angle / 2), thresh * math.cos(angle / 2), - depth / 2],
        [thresh * math.sin(angle / 2), thresh * math.cos(angle / 2), depth / 2],
        [length * math.sin(angle / 2), length * math.cos(angle / 2), - depth / 2],
        n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane(
        [- thresh * math.sin(angle / 2), thresh * math.cos(angle / 2), - depth / 2],
        [- thresh * math.sin(angle / 2), thresh * math.cos(angle / 2), depth / 2],
        [0, 0, - depth / 2],
        n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane(
        [- length * math.sin(angle / 2), length * math.cos(angle / 2), - depth / 2],
        [- length * math.sin(angle / 2), length * math.cos(angle / 2), depth / 2],
        [- thresh * math.sin(angle / 2), thresh * math.cos(angle / 2), - depth / 2],
        n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_rectangle(n=3000, h=5, w=5, depth=50):
    """
    Generates a rectangle with the bottom left corner at the origin.
    :param n: Requested number of control points.
    :param h: Height
    :param w: Width
    :param depth: Depth (z)
    :return: cs, ns, sinks
    """
    centroids = []
    normals = []
    areas = []

    total_surface_length = 2 * w + 2 * h
    n_bottom = int(round(n * w / total_surface_length))
    n_side = int(round(n * h / total_surface_length))

    ######################
    # Bottom             #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                [w, 0, - depth / 2],
                                                n_bottom)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Top                #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w, h, - depth / 2],
                                                [w, h, depth / 2],
                                                [0, h, - depth / 2],
                                                n_bottom)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Left Side          #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([0, h, - depth / 2],
                                                [0, h, depth / 2],
                                                [0, 0, - depth / 2],
                                                n_side)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Right Side          #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w, 0, - depth / 2],
                                                [w, 0, depth / 2],
                                                [w, h, - depth / 2],
                                                n_side)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_triangular_prism(n, corner_1, corner_2, corner_3, depth):
    """
    Generates a triangular prism with corners defined anti-clockwise for inward-facing normals.
    :param corner_1: [x, y] for corner 1
    :param corner_2: [x, y] for corner 2
    :param corner_3: [x, y] for corner 3
    :param depth: Depth of the prism
    :return: centroids, normals, areas
    """
    corner_1, corner_2, corner_3 = np.array(corner_1), np.array(corner_2), np.array(corner_3)

    centroids = []
    normals = []
    areas = []

    bottom_length = vect.mag(corner_2 - corner_1)
    right_length = vect.mag(corner_3 - corner_2)
    top_length = vect.mag(corner_1 - corner_3)
    total_length = bottom_length + right_length + top_length

    ##############################################
    # 'Bottom' Side - corner 1 -> corner 2       #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_1[0], corner_1[1], - depth / 2],
                                                [corner_1[0], corner_1[1], depth / 2],
                                                [corner_2[0], corner_2[1], - depth / 2],
                                                int(round(n * bottom_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ##############################################
    # 'Right' Side - corner 2 -> corner 3        #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_2[0], corner_2[1], - depth / 2],
                                                [corner_2[0], corner_2[1], depth / 2],
                                                [corner_3[0], corner_3[1], - depth / 2],
                                                int(round(n * right_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ###########################################
    # 'Top' Side - corner 1 -> corner 2       #
    ###########################################
    p_centroids, p_normals, p_areas = gen_plane([corner_3[0], corner_3[1], - depth / 2],
                                                [corner_3[0], corner_3[1], depth / 2],
                                                [corner_1[0], corner_1[1], - depth / 2],
                                                int(round(n * top_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_clipped_triangular_prism(n, corner_1, corner_2, corner_3, depth, clip_fact):
    """
    Generates a triangular prism with corners defined anti-clockwise for inward-facing normals.
    :param corner_1: [x, y] for corner 1
    :param corner_2: [x, y] for corner 2
    :param corner_3: [x, y] for corner 3
    :param depth: Depth of the prism
    :param clip_fact: Fraction of the corner to clip
    :return: centroids, normals, areas
    """
    corner_1, corner_2, corner_3 = np.array(corner_1), np.array(corner_2), np.array(corner_3)

    centroids = []
    normals = []
    areas = []

    bottom_length = vect.mag(corner_2 - corner_1)
    right_length = vect.mag(corner_3 - corner_2)
    top_length = vect.mag(corner_1 - corner_3)
    total_length = bottom_length + right_length + top_length

    ##############################################
    # 'Bottom' Side - corner 1 -> corner 2       #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_1[0] + (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_1[1] + (corner_2[1] - corner_1[1]) * clip_fact, - depth / 2],

                                                [corner_1[0] + (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_1[1] + (corner_2[1] - corner_1[1]) * clip_fact, depth / 2],

                                                [corner_2[0] - (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_2[1] - (corner_2[1] - corner_1[1]) * clip_fact, - depth / 2],
                                                int(round(n * (1 - 2 * clip_fact) * bottom_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ##############################################
    # Corner between 'Bottom' and 'Right'        #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_2[0] - (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_2[1] - (corner_2[1] - corner_1[1]) * clip_fact, - depth / 2],

                                                [corner_2[0] - (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_2[1] - (corner_2[1] - corner_1[1]) * clip_fact, depth / 2],

                                                [corner_2[0] + (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_2[1] + (corner_3[1] - corner_2[1]) * clip_fact, - depth / 2],
                                                int(round(
                                                    n * clip_fact * (bottom_length + right_length) / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ##############################################
    # 'Right' Side - corner 2 -> corner 3        #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_2[0] + (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_2[1] + (corner_3[1] - corner_2[1]) * clip_fact, - depth / 2],

                                                [corner_2[0] + (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_2[1] + (corner_3[1] - corner_2[1]) * clip_fact, depth / 2],

                                                [corner_3[0] - (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_3[1] - (corner_3[1] - corner_2[1]) * clip_fact, - depth / 2],
                                                int(round(n * (1 - 2 * clip_fact) * right_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ##############################################
    # Corner between 'Right' and 'Top'           #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_3[0] - (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_3[1] - (corner_3[1] - corner_2[1]) * clip_fact, - depth / 2],

                                                [corner_3[0] - (corner_3[0] - corner_2[0]) * clip_fact,
                                                 corner_3[1] - (corner_3[1] - corner_2[1]) * clip_fact, depth / 2],

                                                [corner_3[0] + (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_3[1] + (corner_1[1] - corner_3[1]) * clip_fact, - depth / 2],
                                                int(round(n * clip_fact * (right_length + top_length) / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ###########################################
    # 'Top' Side - corner 1 -> corner 2       #
    ###########################################
    p_centroids, p_normals, p_areas = gen_plane([corner_3[0] + (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_3[1] + (corner_1[1] - corner_3[1]) * clip_fact, - depth / 2],

                                                [corner_3[0] + (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_3[1] + (corner_1[1] - corner_3[1]) * clip_fact, depth / 2],

                                                [corner_1[0] - (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_1[1] - (corner_1[1] - corner_3[1]) * clip_fact, - depth / 2],
                                                int(round(n * (1 - 2 * clip_fact) * top_length / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ##############################################
    # Corner between 'Top' and 'Bottom'          #
    ##############################################
    p_centroids, p_normals, p_areas = gen_plane([corner_1[0] - (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_1[1] - (corner_1[1] - corner_3[1]) * clip_fact, - depth / 2],

                                                [corner_1[0] - (corner_1[0] - corner_3[0]) * clip_fact,
                                                 corner_1[1] - (corner_1[1] - corner_3[1]) * clip_fact, depth / 2],

                                                [corner_1[0] + (corner_2[0] - corner_1[0]) * clip_fact,
                                                 corner_1[1] + (corner_2[1] - corner_1[1]) * clip_fact, - depth / 2],
                                                int(round(n * clip_fact * (top_length + bottom_length) / total_length)))
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return np.array(centroids), np.array(normals), np.array(areas)


if __name__ == "__main__":
    cs, ns, _ = gen_clipped_triangular_prism(1000, [1, 1], [0, 0], [1, 0], 2, 0.15)
    plot_3d_point_sets([cs, cs + 0.05 * ns])
    plt.show()
