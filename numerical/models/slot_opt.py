import math
import numpy as np

from scipy.optimize import minimize_scalar
import numerical.util.gen_utils as gen
import numerical.bem as bem
import scipy.sparse
import matplotlib.pyplot as plt


def find_slot_peak(w, q, h, n=10000, length=50, depth=50, varied_slot_density_ratio=1.0, density_w_thresh=12,
                   centroids=None, normals=None, areas=None, R_inv=None, m_0=1):
    if centroids is None or normals is None or areas is None or R_inv is None:
        if varied_slot_density_ratio == 1:
            centroids, normals, areas = gen.gen_slot(n=n, h=h, w=w, length=length, depth=depth)
        else:
            centroids, normals, areas = gen.gen_varied_slot(n=n, h=h, w=w, length=length, depth=depth,
                                                            w_thresh=density_w_thresh,
                                                            density_ratio=varied_slot_density_ratio)
        print("Requested n = {0}, using n = {1}.".format(n, len(centroids)))
        n = len(centroids)
        R_matrix = bem.get_R_matrix(centroids, normals, areas, dtype=np.float32)
        R_inv = scipy.linalg.inv(R_matrix)

    def get_theta_j(p):
        res_vel, _ = bem.get_jet_dir_and_sigma([p, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv)
        return math.atan2(res_vel[1], res_vel[0]) + math.pi / 2

    # Very close to middle
    mid_res_vel, _ = bem.get_jet_dir_and_sigma([0.05, q, 0], centroids, normals, areas, m_0=m_0, R_inv=R_inv)
    mid_theta_j = math.atan2(mid_res_vel[1], mid_res_vel[0]) + math.pi / 2
    mid_grad = 2 * mid_theta_j / 0.1  # Approximate gradient of middle

    try:
        res = minimize_scalar(get_theta_j, bracket=(-length / 2, -w, 0),
                              method='brent')  # Any correct peak must occur on the left half
    except ValueError:
        print(f"Peak not found in interval, returning NaN. w={w}  q={q}  h={h}  n={n}")
        return np.nan, np.nan, np.nan, np.nan
    if res.success:
        print(f"Optimization finished with {res.nfev} evaluations, w={w}  q={q}  h={h}  n={n}, "
              f"theta_star={-res.fun:.3f} p_bar_star={-res.x:.3f}")
        return n, -res.x, -res.fun, mid_grad  # Take the right peak rather than the left peak that was found.
    else:
        print(f"Optimization failed, w={w}  q={q}  h={h}  n={n}")
        return None


if __name__ == "__main__":
    w = 2.2
    q = 2.81
    h = 2.7

    ns = np.round(np.linspace(5000, 20000, 10))
    ps = []
    ps_conv = [1]
    theta_js = []
    theta_js_conv = [1]
    mid_grads = []
    mid_grads_conv = [1]

    density_ratio = 0.25
    for i, n in enumerate(ns):
        real_n, p, theta_j, mid_grad = find_slot_peak(w=w, q=q, h=h, n=n, varied_slot_density_ratio=density_ratio)
        ns[i] = real_n
        ps.append(p)
        theta_js.append(theta_j)
        mid_grads.append(mid_grad)
        if len(ps) > 1:
            ps_conv.append(abs(ps[-2] - p) / p)
            theta_js_conv.append(abs(theta_js[-2] - theta_j) / theta_j)
            mid_grads_conv.append(abs(mid_grads[-2] - mid_grad) / mid_grad)

    fig = plt.figure()
    fig.suptitle(f"q / w = {w / q:.2f}, h / w = {h / w:.2f}, density_ratio = {density_ratio}")
    ax1 = fig.add_subplot(311)
    ax1.plot(ns, ps, 'k')
    ax1.set_ylabel("$p^\\star$")
    ax1.set_xlabel("$N$")
    ax1_conv = ax1.twinx()
    ax1_conv.plot(ns, ps_conv, 'k--')
    ax1_conv.set_yscale('log')
    ax1_conv.set_ylabel("Convergence Residual")

    ax2 = fig.add_subplot(312)
    ax2.plot(ns, theta_js, 'k')
    ax2.set_ylabel("$\\theta_j^\\star$")
    ax2.set_xlabel("$N$")
    ax2_conv = ax2.twinx()
    ax2_conv.plot(ns, theta_js_conv, 'k--')
    ax2_conv.set_yscale('log')
    ax2_conv.set_ylabel("Convergence Residual")

    ax3 = fig.add_subplot(313)
    ax3.plot(ns, mid_grads, 'k')
    ax3.set_ylabel("Middle $d\\theta_j/dp$")
    ax3.set_xlabel("$N$")
    ax3_conv = ax3.twinx()
    ax3_conv.plot(ns, mid_grads_conv, 'k--')
    ax3_conv.set_yscale('log')
    ax3_conv.set_ylabel("Convergence Residual")

    plt.show()
