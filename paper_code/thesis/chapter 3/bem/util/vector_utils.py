def mag(v):
    if len(v) == 2:
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    if len(v) == 3:
        return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def unit(v):
    m = mag(v)
    if len(v) == 2:
        return [v[0] / m, v[1] / m]

    if len(v) == 3:
        return [v[0] / m, v[1] / m, v[2] / m]


def mag_sqrd(v):
    if len(v) == 2:
        return v[0] ** 2 + v[1] ** 2

    if len(v) == 3:
        return v[0] ** 2 + v[1] ** 2 + v[2] ** 2


def dist(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vector lengths do not match (v1: {0}, v2: {1}).".format(len(v1), len(v2)))

    if len(v1) == 2:
        return mag([v2[0] - v1[0], v2[1] - v1[1]])

    if len(v1) == 3:
        return mag([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]])


def get_line_intersect(pos, vect, l_a, l_b):
    """
    Calculates the intersect (if any) between a line and a vector from a point.
    :param pos: Position from which to trace.
    :param vect: Vector along which to trace.
    :param l_a: Point a of the line.
    :param l_b: Point b of the line.
    :return: Position (x, y) of the intersect or None
    """
    # Intersect = pos + zeta * vect
    zeta = (((pos[0] - l_a[0]) * (l_b[1] - l_a[1])) - (pos[1] - l_a[1]) * (l_b[0] - l_a[0])) / (
            vect[1] * (l_b[0] - l_a[0]) - vect[0] * (l_b[1] - l_a[1]))

    # Intersect = l_a + kappa * l_b
    kappa = (vect[0] * (l_a[1] - pos[1]) - vect[1] * (l_a[0] - pos[0])) / (
            vect[1] * (l_b[0] - l_a[0]) - vect[0] * (l_b[1] - l_a[1]))

    if zeta >= 0 and 0 <= kappa <= 1:
        return [pos[0] + zeta * vect[0], pos[1] + zeta * vect[1]]
    else:
        return None
