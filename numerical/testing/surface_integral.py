from scipy.integrate import quad, dblquad
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math


def get_u(r):
    r = np.array(r)
    length = np.linalg.norm(r)
    r_hat = r / length
    return - r_hat / (4 * np.pi * length ** 2)


def get_r(t, p, d, radius):
    return np.array([
        radius * np.cos(t) * np.cos(p),
        d + radius * np.sin(t),
        radius * np.cos(t) * np.sin(p)
    ])


def get_vel(t, p, d, radius):
    r = get_r(t, p, d, radius)
    return get_u(r)


# def get_vel_x(t, p, d, radius):
#     return get_vel(t, p, d, radius)[0]


def get_vel_y(t, p, d, radius):
    """ Velocity in the axial direction (positive is from the sink to the bubble) """
    return get_vel(t, p, d, radius)[1]


# def get_vel_z(t, p, d, radius):
#     return get_vel(t, p, d, radius)[2]


if __name__ == "__main__":
    h = 0.05
    bubble_pos = np.array([1, 2, 0])
    bubble_radius = min([i for i in bubble_pos if i != 0]) / 1.5

    sink_pos = [[-bubble_pos[0], bubble_pos[1], 0],
                [-bubble_pos[0], -bubble_pos[1], 0],
                [bubble_pos[0], -bubble_pos[1], 0]]
    # sink_pos = [[bubble_pos[0], - bubble_pos[1], 0]]

    total_vel = np.array([0., 0., 0.])
    test_vel = np.array([0., 0., 0.])
    center_vel = np.array([0., 0., 0.])

    ts = np.arange(0, np.pi, h)
    ps = np.arange(0, 2 * np.pi, h)
    for sink in sink_pos:
        # Surface integral
        sink = np.array(sink)
        r = sink - bubble_pos
        dist = np.linalg.norm(r)
        r_hat = r / dist

        # get_vel_y is the axial velocity component for the sink-bubble combination.
        # Multiply by the unit direction vector from the bubble to the sink to get
        total_vel -= r_hat * dblquad(get_vel_y, 0, np.pi, lambda _: 0, lambda _: 2 * np.pi,
                                     args=(dist, bubble_radius))[0]

        # Center
        center_vel -= get_u(r)

        # Crude approximation to check integration
        count = 0
        last_print = 0
        for t, p in itertools.product(ts, ps):
            if count - last_print > 0.05 * len(ts) * len(ps):
                print(f"{count / (len(ts) * len(ps)) * 100:.2f}%")
                last_print = count
            count += 1
            v = r_hat * np.linalg.norm(get_vel(t, p, dist, bubble_radius))
            test_vel += v

    print(total_vel)
    total_vel /= 4 * np.pi * bubble_radius ** 2

    test_vel *= (4 * np.pi * bubble_radius ** 2) / (len(ts) * len(ps))  # Area per point
    test_vel /= 4 * np.pi * bubble_radius ** 2
    print(test_vel)

    print(center_vel * 4 * np.pi * bubble_radius ** 2)

    print(f"Surface integration:\n"
          f"    Velocity: {total_vel}\n"
          f"    Angle: {math.atan2(total_vel[1], total_vel[0])}")

    print(f"Test:\n"
          f"    Velocity: {test_vel}\n"
          f"    Angle: {math.atan2(test_vel[1], test_vel[0])}")

    print(f"Bubble center:\n"
          f"    Velocity: {center_vel}\n"
          f"    Angle: {math.atan2(center_vel[1], center_vel[0])}")
