import matplotlib.pyplot as plt
import numpy as np
import math

import numerical.bem as bem
import numerical.util.gen_utils as gen
import numerical.potential_flow.element_utils as eu
import common.util.vector_utils as vect
from numerical.potential_flow.elements import Source3D


def get_rectangle_jet_vel(x, y, length, height, k=25, m_0=1):
    elements = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if i == 0 and j == 0:
                continue  # Don't include the bubble.
            s_x = i * length + x + (2 * x - length) * ((-1) ** i - 1) / 2
            s_y = j * height + y + (2 * y - height) * ((-1) ** j - 1) / 2
            elements.append(Source3D(s_x, s_y, 0, -m_0))
    return eu.get_all_vel_3d(elements, x, y, 0)


w = 5
h = 5
r = 2
n = 4000
error_points = 200

centroids, normals, areas = gen.gen_rectangle(n=n, w=w, h=h, depth=50)
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = np.linalg.inv(R_matrix)
# pu.plot_3d_point_sets([cs, sinks])

theta_bs = np.linspace(0, 2 * math.pi, 50)
theta_js_numeric = []
theta_js_analytic = []
errors = []
for theta_b in theta_bs:
    print("Testing ", theta_b)
    x = r * math.cos(theta_b - math.pi / 2) + w / 2
    y = r * math.sin(theta_b - math.pi / 2) + h / 2
    res_vel, sigma = bem.get_jet_dir_and_sigma([x, y, 0], centroids, normals, areas, R_inv=R_inv)
    theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
    if theta_j <= 0:
        theta_j += 2 * math.pi
    theta_js_numeric.append(theta_j)

    res_vel = get_rectangle_jet_vel(x, y, w, h)
    theta_j = math.atan2(res_vel[1], res_vel[0]) + math.pi / 2
    if theta_j <= 0:
        theta_j += 2 * math.pi
    theta_js_analytic.append(theta_j)

    point_xs = []
    point_ys = []
    point_errors = []
    error_sum = 0
    # Top
    for i in np.linspace(-0.5, 0.5, int(round(error_points / 4))):
        p_x = i * w
        p_y = h / 2
        err = abs(np.dot(bem.get_vel([p_x, p_y, 0], centroids, areas, sigma), [0, 1, 0]))
        point_xs.append(p_x)
        point_ys.append(p_y)
        point_errors.append(err)
        error_sum += err
    # Bottom
    for i in np.linspace(-0.5, 0.5, int(round(error_points / 4))):
        p_x = i * w
        p_y = - h / 2
        err = abs(np.dot(bem.get_vel([p_x, p_y, 0], centroids, areas, sigma), [0, 1, 0]))
        point_xs.append(p_x)
        point_ys.append(p_y)
        point_errors.append(err)
        error_sum += err
    # Left
    for i in np.linspace(-0.5, 0.5, int(round(error_points / 4))):
        p_x = - w / 2
        p_y = i * h
        err = abs(np.dot(bem.get_vel([p_x, p_y, 0], centroids, areas, sigma), [1, 0, 0]))
        point_xs.append(p_x)
        point_ys.append(p_y)
        point_errors.append(err)
        error_sum += err
    # Right
    for i in np.linspace(-0.5, 0.5, int(round(error_points / 4))):
        p_x = w / 2
        p_y = i * h
        err = abs(np.dot(bem.get_vel([p_x, p_y, 0], centroids, areas, sigma), [1, 0, 0]))
        point_xs.append(p_x)
        point_ys.append(p_y)
        point_errors.append(err)
        error_sum += err
    errors.append(error_sum)

    point_errors = np.array(point_errors) / vect.mag(res_vel)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_xs, point_ys, point_errors, c=point_errors)
    plt.show()

for i in range(len(theta_js_numeric)):
    if theta_bs[i] < math.pi and theta_js_numeric[i] > 3 * math.pi / 2:
        theta_js_numeric[i] = theta_js_numeric[i] - 2 * math.pi
    if theta_bs[i] > math.pi and theta_js_numeric[i] < math.pi / 2:
        theta_js_numeric[i] = theta_js_numeric[i] + 2 * math.pi

for i in range(len(theta_js_analytic)):
    if theta_bs[i] < math.pi and theta_js_analytic[i] > 3 * math.pi / 2:
        theta_js_analytic[i] = theta_js_analytic[i] - 2 * math.pi
    if theta_bs[i] > math.pi and theta_js_analytic[i] < math.pi / 2:
        theta_js_analytic[i] = theta_js_analytic[i] + 2 * math.pi

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = plt.gca()
ax.plot(theta_bs, theta_js_numeric, label="Numeric")
ax.plot(theta_bs, theta_js_analytic, label="Analytic (approx)")
ax.set_xlabel("$\\theta_b$", fontsize=18)
ax.set_ylabel("$\\theta_j$", fontsize=18)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)
ax.axvline(x=math.pi / 2, linestyle='--', color='gray')
ax.axvline(x=math.pi, linestyle='--', color='gray')
ax.axvline(x=3 * math.pi / 2, linestyle='--', color='gray')

plt.figure()
plt.plot(theta_bs, errors)
plt.show()
