import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width
import numerical.bem as bem
import numerical.util.gen_utils as gen
from experimental.util.analysis_utils import get_collapse_variations
import itertools as it


def rp_complete(t, lhs, density=997, kin_visc=0.10533e-5, surf_tension=0, delta_p=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            (delta_p(t, R) / density
             - R_dot ** 2 * (3 / 2)
             - 4 * kin_visc * R_dot / R
             - 2 * surf_tension / (density * R))
            / R]


def rp_inertial(t, lhs, density=997, delta_p=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot, (delta_p(t, R) / density - R_dot ** 2 * (3 / 2)) / R]


def rp_wall_model(t, lhs, phi_prime, density=997, delta_p=lambda t, R: 1000):
    """ Rayleigh-Plesset formulation as two first order ODEs. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            (delta_p(t, R) / density
             - R_dot ** 2 * (3 / 2 - 8 * np.pi * phi_prime * R))
            / (R - 4 * np.pi * phi_prime * R ** 2)]


def rp_no_internal_gas(t, lhs, p_vapour, p_inf, density=997):
    """ Rayleigh-Plesset formulation as two first order ODEs but entirely simplified. """
    R = lhs[0]
    R_dot = lhs[1]
    return [R_dot,
            ((p_vapour - p_inf) / density
             - R_dot ** 2 * (3 / 2))
            / R]


def km_inertial(t, lhs, p_v, p_inf, c, density, R_init, p_g0, gamma):
    """ Based on equation 6 in Tinguely et al. (2012). """
    R = lhs[0]
    R_dot = lhs[1]

    v = R_dot / c
    p_g = p_g0 * (R_init / R) ** (3 * gamma)
    p_g_dot = -3 * gamma * p_g0 * R_init ** (3 * gamma) * R ** (-3 * gamma - 1) * R_dot  # Chain rule

    R_dotdot = ((p_g - (p_inf - p_v)) * (1 + v)
                + R * p_g_dot / c - (3 - v) * R_dot ** 2 * density / 2) / ((1 - v) * R * density)

    return [R_dot, R_dotdot]


def km_pg0_opt(p_v, p_inf, c, density, R_init, radius_ratio, gamma):
    def peak_event(t, lhs, *args):
        return np.nan if lhs[0] == R_init else lhs[1]  # R_dot

    peak_event.terminal = True  # End the integration at the peak
    peak_event.direction = -1  # Only detect the maximum

    ray_col_time = 0.915 * R_init * (density / (p_inf - p_v)) ** 0.5
    max_length = 5 * ray_col_time

    def get_rr_diff(p_g0):
        ivp_out = solve_ivp(km_inertial, (0, max_length), (R_init, 0), events=peak_event,
                            args=(p_v, p_inf, c, density, R_init, p_g0, gamma))
        rr = ivp_out.y[0, -1] / ivp_out.y[0, 0]
        return np.abs(rr - radius_ratio)

    res = minimize(get_rr_diff, p_v, method="Nelder-Mead", bounds=[(1e-5, 5 * p_v)])
    return res.x


def delta_p(t, R):
    return p_init * (R_init / R) ** (3 * polytropic_const) + p_vapour - p_inf


initialize_plt()
plt.figure(figsize=(4, 2.75))

R_inits = []
radius_ratios = []
init_xs = []
init_ys = []

w = 2.2 / 1000
h = 2.9 / 1000

density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
p_vapour = 2.3388e3
p_inf = 100e3
p_init = p_vapour  # 560
polytropic_const = 1.33  # ratio of specific heats of water vapour
c = 1480  # speed of sound in water

interp_points = 1000
all_exp_plot_xs = np.linspace(0, 3, interp_points)
all_exp_plot_ys_max = np.full((interp_points,), np.nan)
all_exp_plot_ys_min = np.full((interp_points,), np.nan)

mid_min_idxs = []
mid_min_vals = []

end_min_idxs = []
end_min_vals = []

for idx, rep in it.product(range(9, 10), range(5)):
    exp_xs, exp_ys, exp_radii, exp_ts = get_collapse_variations(
        "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b/",
        idx, rep)

    exp_xs, exp_ys, exp_radii = exp_xs / 1000, exp_ys / 1000, exp_radii / 1000  # Convert to meters

    peak_idxs, _ = find_peaks(exp_radii)

    if len(peak_idxs) < 2:
        continue

    R_init = exp_radii[peak_idxs[0]]
    ray_col_time = 0.915 * R_init * (density / (p_inf - p_vapour)) ** 0.5

    R_inits.append(R_init)
    radius_ratios.append(exp_radii[peak_idxs[1]] / R_init)
    init_xs.append(exp_xs[peak_idxs[0]])
    init_ys.append(exp_ys[peak_idxs[0]])

    norm_ts = (exp_ts - exp_ts[peak_idxs[0]]) / ray_col_time
    norm_Rs = exp_radii / R_init

    plt.plot(norm_ts, norm_Rs, color="#CAE7CA")

    interp_ys = interp1d(norm_ts, norm_Rs, bounds_error=False, fill_value=np.nan)(all_exp_plot_xs)

    # Find the index of the minimum value for each array for normalised times less than 1.5
    interp_ys_ma = np.ma.masked_where(np.logical_or(np.isnan(interp_ys), all_exp_plot_xs > 1.5), interp_ys)
    min_interp_idx = np.ma.argmin(interp_ys_ma)
    mid_min_idxs.append(min_interp_idx)
    mid_min_vals.append(interp_ys[min_interp_idx])

    # Find the index of the minimum value for each array for normalised times greater than 1.5 (hopefully the ends)
    interp_ys_ma = np.ma.masked_where(np.logical_or(np.isnan(interp_ys), all_exp_plot_xs < 1.5), interp_ys)
    end_interp_idx = np.ma.argmin(interp_ys_ma)
    end_min_idxs.append(end_interp_idx)
    end_min_vals.append(interp_ys[end_interp_idx])

    # Interpolate between the minimums
    if len(mid_min_idxs) > 1:
        min_smooth_ys = interp1d(all_exp_plot_xs[mid_min_idxs], mid_min_vals, bounds_error=False)(all_exp_plot_xs)
        end_smooth_ys = interp1d(all_exp_plot_xs[end_min_idxs], end_min_vals, bounds_error=False)(all_exp_plot_xs)
    else:
        min_smooth_ys = np.full((interp_points,), np.nan)
        end_smooth_ys = np.full((interp_points,), np.nan)

    # Find the min and max values at each interpolation point and include smoothing arrays
    all_exp_plot_ys_max = np.nanmax([interp_ys, all_exp_plot_ys_max, min_smooth_ys, end_smooth_ys], axis=0)
    all_exp_plot_ys_min = np.nanmin([interp_ys, all_exp_plot_ys_min, min_smooth_ys, end_smooth_ys], axis=0)

R_init = np.mean(R_inits)
radius_ratio = np.mean(radius_ratios)

print(f"R_0 varies between {np.min(R_inits)} and {np.max(R_inits)}, with a mean of {R_init}.")

ray_col_time = 0.915 * R_init * (density / (p_inf - p_vapour)) ** 0.5
sim_length = 5 * ray_col_time

u = R_init / ray_col_time
print("Rayleigh Collapse Time = ", ray_col_time)
pressure_ratio = p_init / (p_inf - p_vapour)
We = density * R_init * u ** 2 / surf_tension
Re = u * R_init / kin_visc

n = 12000
density_ratio = 0.25
w_thresh = 5

centroids, normals, areas = gen.gen_varied_slot(n=n, H=h, W=w, length=0.1, depth=0.05, w_thresh=w_thresh,
                                                density_ratio=density_ratio)

print(f"Minimum element area = {np.min(areas) / R_init ** 2}, maximum element area = {np.max(areas) / R_init ** 2}")

R_matrix = bem.get_R_matrix(centroids, normals, areas)

bubbles_pos = np.array([np.mean(init_xs), np.mean(init_ys), 0])
sigmas = bem.calculate_sigma(bubbles_pos, centroids, normals, areas, m_0=1)

phi_prime = bem.calculate_phi_prime(bubbles_pos, centroids, areas, sigmas=sigmas)
print(f"phi_prime = {phi_prime}")

force_prime = bem.calculate_force_prime(bubbles_pos, centroids, normals, areas, sigmas, density)

standoff = bubbles_pos[1] / R_init

out_complete = solve_ivp(rp_complete, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                         args=(density, kin_visc, surf_tension, delta_p))
plt.plot(out_complete.t / ray_col_time, out_complete.y[0, :] / R_init, label="Complete", color="C0")

out_inertial = solve_ivp(rp_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                         args=(density, delta_p))
plt.plot(out_inertial.t / ray_col_time, out_inertial.y[0, :] / R_init, label="Inertial", color="C1", linestyle="--")

out_wall_model = solve_ivp(rp_wall_model, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                           args=(phi_prime, density, delta_p))
plt.plot(out_wall_model.t / ray_col_time, out_wall_model.y[0, :] / R_init, label="Wall model",
         color="C2", linestyle="dashdot")

out_no_internal_gas = solve_ivp(rp_no_internal_gas, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                                args=(p_vapour, p_inf, density))
plt.plot(out_no_internal_gas.t / ray_col_time, out_no_internal_gas.y[0, :] / R_init, label="No internal gas",
         color="C3", linestyle="dotted")

# p_g0 = km_pg0_opt(p_vapour, p_inf, c, density, R_init, radius_ratio, polytropic_const)[0]
# print(f"p_g0 = {p_g0:.2f}")
# out_km_inertial = solve_ivp(km_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
#                             args=(p_vapour, p_inf, c, density, R_init, p_g0, polytropic_const))
# plt.plot(out_km_inertial.t / ray_col_time, out_km_inertial.y[0, :] / R_init, label="Keller-Miksis",
#          color="C5", linestyle=(0, (5, 2, 1, 2, 1, 2)))  # dashdotdottedish

plt.xlabel("$t / t_{TC}$")
plt.ylabel("$R / R_0$")

xs_bounds = np.concatenate([all_exp_plot_xs, all_exp_plot_xs[::-1]])
ys_bounds = np.concatenate([all_exp_plot_ys_min, all_exp_plot_ys_max[::-1]])
plt.gca().add_patch(Polygon(np.array([xs_bounds, ys_bounds]).T, color="#D5ECD5",
                            linewidth=1, edgecolor="#D5ECD5", capstyle='round', label="Experiment"))

plt.xlim((0, 2.5))
plt.ylim((-0.05, 1.05))
plt.legend(frameon=False, fontsize='small')
plt.tight_layout()
plt.show()
