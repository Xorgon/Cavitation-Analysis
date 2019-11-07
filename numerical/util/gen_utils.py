import itertools as it
import math
import warnings

import numpy as np

import common.util.vector_utils as vect


def gen_plane(corner_1, corner_2, corner_3, n):
    """
    Generates the control points, normals, and sinks for a plane.
    :param corner_1: First corner.
    :param corner_2: Second corner (adjacent to corner_1).
    :param corner_3: Third corner (adjacent to corner_1).
    :param n: Number of panels to generate.
    :return: centroids, normals, areas
    """
    corner_1 = np.array(corner_1)
    corner_2 = np.array(corner_2)
    corner_3 = np.array(corner_3)

    # Define the two vectors to move in the plane.
    vect_1 = corner_2 - corner_1
    vect_2 = corner_3 - corner_1
    normal = np.cross(vect_2, vect_1)
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

    for i, j in it.product(np.linspace(side_1_panel_h / 2, vect.mag(vect_1) - side_1_panel_h / 2, side_1_n),
                           np.linspace(side_2_panel_h / 2, vect.mag(vect_2) - side_2_panel_h / 2, side_2_n)):
        c = corner_1 + i * vect_1_hat + j * vect_2_hat
        centroids.append(c)
        normals.append(normal)
        areas.append(panel_area)

    return np.array(centroids), np.array(normals), np.array(areas)


def gen_slot(n=3000, h=5, w=5, length=50, depth=50):
    centroids = []
    normals = []
    areas = []

    total_surface_length = w + 2 * h + (length - w - 2)
    n_slot_floor = int(round(n * w / total_surface_length))
    n_slot_wall = int(round(n * h / total_surface_length))
    print("n_slot_wall = {0}".format(n_slot_wall))
    n_surface_boundary = int(round(n * (length - w) / 2) / total_surface_length)

    ######################
    # Surface boundaries #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w / 2, 0, - depth / 2],
                                                [length / 2, 0, - depth / 2],
                                                [w / 2, 0, depth / 2],
                                                n_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-w / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                n_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot floor         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-w / 2, -h, - depth / 2],
                                                [w / 2, -h, - depth / 2],
                                                [-w / 2, -h, depth / 2],
                                                n_slot_floor)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot walls         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-w / 2, 0, - depth / 2],
                                                [-w / 2, -h, - depth / 2],
                                                [-w / 2, 0, depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([w / 2, -h, - depth / 2],
                                                [w / 2, 0, - depth / 2],
                                                [w / 2, -h, depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_varied_slot(n=3000, h=5, w=5, length=50, depth=50, w_thresh=2, density_ratio=0.5):
    """
    Generates a slot with varying panel density.

    :param n: Approximate number of panels
    :param h: Slot height
    :param w: Slot width
    :param length: Geometry length
    :param depth: Geometry depth
    :param w_thresh: w threshold at which to reduce density (density reduced starting after position w * w_thresh)
    :param density_ratio: Ratio of densities
    :return: centroids, normals, areas
    """
    if w * w_thresh > length:
        warnings.warn("w threshold too high, reverting to gen_slot")
        return gen_slot(n, h, w, length, depth)

    centroids = []
    normals = []
    areas = []

    total_surface_length = length + 2 * h
    dense_surface_length = 2 * h + w * w_thresh
    sparse_surface_length = total_surface_length - dense_surface_length

    zeta = dense_surface_length / (density_ratio * sparse_surface_length)
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    n_slot_floor = int(round(n_dense * w / dense_surface_length))
    n_slot_wall = int(round(n_dense * h / dense_surface_length))
    n_dense_surface_boundary = int(round(n_dense * (w * (w_thresh - 1)) / 2) / dense_surface_length)
    n_sparse_surface_boundary = int(round(n_sparse / 2))

    ######################
    # Surface boundaries #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w / 2, 0, - depth / 2],
                                                [w * w_thresh / 2, 0, - depth / 2],
                                                [w / 2, 0, depth / 2],
                                                n_dense_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([w * w_thresh / 2, 0, - depth / 2],
                                                [length / 2, 0, - depth / 2],
                                                [w * w_thresh / 2, 0, depth / 2],
                                                n_sparse_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([-w * w_thresh / 2, 0, - depth / 2],
                                                [-w / 2, 0, - depth / 2],
                                                [-w * w_thresh / 2, 0, depth / 2],
                                                n_dense_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([-length / 2, 0, - depth / 2],
                                                [-w * w_thresh / 2, 0, - depth / 2],
                                                [-length / 2, 0, depth / 2],
                                                n_sparse_surface_boundary)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot floor         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-w / 2, -h, - depth / 2],
                                                [w / 2, -h, - depth / 2],
                                                [-w / 2, -h, depth / 2],
                                                n_slot_floor)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Slot walls         #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([-w / 2, 0, - depth / 2],
                                                [-w / 2, -h, - depth / 2],
                                                [-w / 2, 0, depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)
    p_centroids, p_normals, p_areas = gen_plane([w / 2, -h, - depth / 2],
                                                [w / 2, 0, - depth / 2],
                                                [w / 2, -h, depth / 2],
                                                n_slot_wall)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_corner(n=3000, length=25, depth=50, angle=np.pi / 2):
    centroids = []
    normals = []
    areas = []

    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [length, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                n / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([length * math.cos(angle), length * math.sin(angle), - depth / 2],
                                                [0, 0, - depth / 2],
                                                [length * math.cos(angle), length * math.sin(angle), depth / 2],
                                                n / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_varied_corner(n=3000, length=25, depth=50, angle=np.pi / 2, thresh=10, density_ratio=0.5):
    centroids = []
    normals = []
    areas = []

    zeta = 2 * thresh / (density_ratio * 2 * (length - thresh))
    n_dense = n * zeta / (1 + zeta)
    n_sparse = n - n_dense

    p_centroids, p_normals, p_areas = gen_plane([0, 0, - depth / 2],
                                                [thresh, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([thresh, 0, - depth / 2],
                                                [length, 0, - depth / 2],
                                                [thresh, 0, depth / 2],
                                                n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([thresh * math.cos(angle), thresh * math.sin(angle), - depth / 2],
                                                [0, 0, - depth / 2],
                                                [thresh * math.cos(angle), thresh * math.sin(angle), depth / 2],
                                                n_dense / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    p_centroids, p_normals, p_areas = gen_plane([length * math.cos(angle), length * math.sin(angle), - depth / 2],
                                                [thresh * math.cos(angle), thresh * math.sin(angle), - depth / 2],
                                                [length * math.cos(angle), length * math.sin(angle), depth / 2],
                                                n_sparse / 2)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas


def gen_rectangle(n=3000, h=5, w=5, depth=50):
    """
    Generates a rectangle with one corner at the origin.
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
                                                [w, 0, - depth / 2],
                                                [0, 0, depth / 2],
                                                n_bottom)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Top                #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w, h, - depth / 2],
                                                [0, h, - depth / 2],
                                                [w, h, depth / 2],
                                                n_bottom)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Left Side          #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([0, h, - depth / 2],
                                                [0, 0, - depth / 2],
                                                [0, h, depth / 2],
                                                n_side)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    ######################
    # Right Side          #
    ######################
    p_centroids, p_normals, p_areas = gen_plane([w, 0, - depth / 2],
                                                [w, h, - depth / 2],
                                                [w, 0, depth / 2],
                                                n_side)
    centroids.extend(p_centroids)
    normals.extend(p_normals)
    areas.extend(p_areas)

    return centroids, normals, areas
