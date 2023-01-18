import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks

import numerical.bem as bem
import numerical.util.gen_utils as gen

if not os.path.exists("../../model_outputs/slot_anisotropy_data"):
    os.makedirs("../../model_outputs/slot_anisotropy_data")


def rp_inertial(t, lhs, density=997, delta_P=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_P(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def delta_P(t, R):
    return P_init * (R_init / R) ** (3 * polytropic_const) + P_vapour - P_inf


density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
P_vapour = 2.3388e3
P_inf = 100e3
P_init = P_vapour
polytropic_const = 1.33  # ratio of specific heats of water vapour

R_init = 0.5 / 1000

offset = R_init  # 0.05 / 1000
W = 4 / 1000
H = 4 / 1000
N = 33
Ys = np.concatenate([np.linspace(-H + offset, 0, int(np.round(N / 4))),
                     np.linspace(offset, H, int(np.round(N / 2))),
                     np.linspace(H * 1.1, H * 2.5, int(np.round(N / 4)))])
Ys = sorted(Ys)
x_limit = 6
Xs = np.concatenate([np.linspace(-6 * W / 2, -2 * W / 2, 5),
                     [-1.5 * W / 2, - W / 2, - W / 2 + offset / 2],
                     np.linspace(-W / 2 + offset, W / 2 - offset, int(np.round(N - 10))),
                     [W / 2, 1.5 * W / 2]])
Xs = sorted(Xs)

print(len(Ys), len(Xs))
print(Xs)

n = 23000
density_ratio = 0.1
w_thresh = x_limit * 2
length = 100 / 1000

# centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=50, depth=50)
centroids, normals, areas = gen.gen_varied_slot(n=n, H=H, W=W, length=length, depth=50 / 1000, w_thresh=w_thresh,
                                                density_ratio=density_ratio)

centroids_file = open(f"../../model_outputs/slot_anisotropy_data/centroids_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}"
                      f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}.csv", 'w')
for c in centroids:
    centroids_file.write(f"{c[0]},{c[1]},{c[2]}\n")
centroids_file.close()

normals_file = open(f"../../model_outputs/slot_anisotropy_data/normals_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}"
                    f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}.csv", 'w')
for normal in normals:
    normals_file.write(f"{normal[0]},{normal[1]},{normal[2]}\n")
normals_file.close()

output_path = f"../../model_outputs/slot_anisotropy_data/anisotropy_sweep_n{n}_W{W * 1000:.2f}_H{H * 1000:.2f}" \
              f"_drat{density_ratio}_wthresh{w_thresh:.2f}_len{length * 1000:.1f}_N{N}.csv"
if os.path.exists(output_path):
    print("Output path already exists!")
    exit()
file = open(output_path, 'w')
print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
# plot_3d_point_sets([centroids])
R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
R_inv = scipy.linalg.inv(R_matrix)

us = []
vs = []

ray_col_time = 0.915 * R_init * (density / (P_inf - P_vapour)) ** 0.5
sim_length = 4 * ray_col_time

out = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                args=(density, delta_P))

for Y, X in itertools.product(Ys, Xs):
    print(f"Testing X={X * 1000:5.3f} mm, Y={Y * 1000:5.3f} mm")
    bubble_pos = [X, Y, 0]

    if Y < offset and not (- W / 2 + offset * 0.98 <= X <= W / 2 - offset * 0.98):
        # Inside the boundary, not inside the slot
        us.append(np.nan)
        vs.append(np.nan)
        file.write(f"{X},{Y},{np.nan},{np.nan}\n")
        file.flush()
        os.fsync(file.fileno())
        continue

    R_b = bem.get_R_vector(bubble_pos, centroids, normals)
    sigmas = bem.calculate_sigma(bubble_pos, centroids, normals, areas, m_0=1, R_inv=R_inv, R_b=R_b)

    force_prime = bem.calculate_force_prime(bubble_pos, centroids, normals, areas, sigmas, density)

    peaks = find_peaks(-out.y[0])[0]
    if len(peaks) >= 2:
        kelvin_impulse = trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, peaks[0]:peaks[1]] ** 4 * out.y[1,
                                                                                               peaks[0]:peaks[
                                                                                                   1]] ** 2,
            x=out.t[peaks[0]:peaks[1]])
    else:
        # Under vacuum cavity conditions the solution stops at the first collapse point so cannot continue, but does
        # helpfully still cover exactly the right period for half an expansion-collapse cycle.
        kelvin_impulse = 2 * trapezoid(
            16 * np.pi ** 2 * np.linalg.norm(force_prime) * out.y[0, :] ** 4 * out.y[1, :] ** 2,
            x=out.t[:])
    anisotropy = kelvin_impulse / (4.789 * R_init ** 3 * np.sqrt(density * (P_inf - P_vapour)))

    vect_anisotropy = anisotropy * force_prime / np.linalg.norm(force_prime)

    us.append(vect_anisotropy[0])
    vs.append(vect_anisotropy[1])
    file.write(f"{X},{Y},{vect_anisotropy[0]},{vect_anisotropy[1]}\n")
    file.flush()
    os.fsync(file.fileno())

file.close()

P, Q = np.meshgrid(Xs, Ys)
U = np.reshape(np.array(us), P.shape)
V = np.reshape(np.array(vs), P.shape)
