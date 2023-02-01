import numpy as np
import matplotlib.pyplot as plt


class Mirror:
    p1, p2 = None, None

    def __init__(self, p1, p2):
        self.p1, self.p2 = np.array(p1), np.array(p2)

    def get_unit_vect(self):
        return (self.p2 - self.p1) / ((self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[1]) ** 2) ** 0.5

    def get_vect(self):
        return self.p2 - self.p1

    def rotate(self, angle, centre):
        """ Rotates the mirror anti-clockwise around the given centre by the given radians. """
        centre = np.array(centre)

        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.p1 = np.matmul(rot_mat, self.p1 - centre) + centre
        self.p2 = np.matmul(rot_mat, self.p2 - centre) + centre

    def translate(self, offset):
        offset = np.array(offset)
        self.p1 += offset
        self.p2 += offset


class Ray:
    ps, dirs = None, None

    mirrors = None

    nearest_intercept_coeff = np.inf
    nearest_intercept_mirror = None  # type: Mirror

    def __init__(self, start, d, mirrors):
        self.ps = []
        self.dirs = []

        self.ps.append(np.array(start))
        self.dirs.append(np.array(d))
        self.mirrors = mirrors

    def calc_intercept(self, m: Mirror):
        p = self.ps[-1]
        d = self.dirs[-1]
        a = (p[1] - m.p1[1] - d[1] * (p[0] - m.p1[0]) / d[0]) / (m.p2[1] - m.p1[1] - d[1] * (m.p2[0] - m.p1[0]) / d[0])
        if 0 <= a <= 1:
            # Hit
            gamma = (a * (m.p2[0] - m.p1[0]) + m.p1[0] - p[0]) / d[0]

            # Check that it's a forward intercept and not immediately after previous intercept
            if 0 < gamma < self.nearest_intercept_coeff and not np.isclose(0, gamma):
                self.nearest_intercept_coeff = gamma
                self.nearest_intercept_mirror = m

    def iterate(self, mirrors) -> bool:
        for m in mirrors:
            self.calc_intercept(m)

        if self.nearest_intercept_mirror is not None:
            intercept = self.ps[-1] + self.nearest_intercept_coeff * self.dirs[-1]
            self.ps.append(intercept)

            m_t = self.nearest_intercept_mirror.get_unit_vect()
            m_n = np.array([1 / np.sqrt(1 + m_t[0] ** 2 / m_t[1] ** 2),
                            - m_t[0] / (m_t[1] * np.sqrt(1 + m_t[0] ** 2 / m_t[1] ** 2))])

            dir_t = np.dot(self.dirs[-1], m_t) * m_t
            dir_n = - np.dot(self.dirs[-1], m_n) * m_n

            self.dirs.append(dir_t + dir_n)

            self.nearest_intercept_coeff = np.inf
            self.nearest_intercept_mirror = None
            return True
        else:
            return False

    def trace(self):
        while self.iterate(self.mirrors):
            continue

    def extend_ray(self, dist):
        self.ps.append(self.ps[-1] + self.dirs[-1] * dist)

    def set_mirrors(self, mirrors):
        self.mirrors = mirrors


if __name__ == "__main__":
    mirs = []
    ys = np.linspace(50.8 - 25.4, 50.8 + 25.4, 5000)
    xs = (1 / (4 * 25.4)) * ys ** 2
    for i in range(len(xs) - 1):
        mirs.append(Mirror([xs[i], ys[i]], [xs[i + 1], ys[i + 1]]))

    for m in mirs:
        m.translate([-6.35, -25.4])
        m.rotate(np.deg2rad(1), [0, 25.4])

    beam_height = 16
    beam_width = 30
    rays = []
    for y in np.linspace(beam_height - beam_width / 2, beam_height + beam_width / 2, 100):
        rays.append(Ray([200, y], [-1, 0], mirrors=mirs))

    for i, ray in enumerate(rays):
        print(f"{100 * i / len(rays):.2f}%")
        ray.trace()
        ray.extend_ray(200)
        plt.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="lightgreen", alpha=min(1, 5 / len(rays)))

        if i == 0 or i + 1 == len(rays):
            plt.plot([p[0] for p in ray.ps], [p[1] for p in ray.ps], color="red", alpha=min(1, 5 / len(rays)))

    ray_0_d = rays[0].dirs[-1]
    ray_1_d = rays[-1].dirs[-1]
    print(np.rad2deg(np.arccos(np.dot(ray_0_d, ray_1_d) / (np.linalg.norm(ray_0_d) * np.linalg.norm(ray_1_d)))))

    for mir in mirs:
        plt.plot([mir.p1[0], mir.p2[0]], [mir.p1[1], mir.p2[1]], color="grey")
    plt.gca().set_aspect('equal')
    plt.show()
