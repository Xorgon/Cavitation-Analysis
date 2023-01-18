import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
from common.util.plotting_utils import initialize_plt, fig_width, format_axis_ticks_decimal
import numerical.bem as bem
import numerical.util.gen_utils as gen
from experimental.util.analysis_utils import get_collapse_variations
import itertools as it


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


density = 997
kin_visc = 1.003e-6
surf_tension = 0.0728
p_vapour = 2.3388e3
p_inf = 100e3
polytropic_const = 1.33  # ratio of specific heats of water vapour

R_init = 1e-3
ray_col_time = 0.915 * R_init * (density / (p_inf - p_vapour)) ** 0.5
sim_length = 5 * ray_col_time

initialize_plt()
plt.figure()

all_p_inits = np.logspace(np.log10(0.01 * p_vapour), np.log10(10 * p_vapour), 25)
rebound_ratios = []

for i, p_init in enumerate(all_p_inits):
    out_km_inertial = solve_ivp(km_inertial, (0, sim_length), (R_init, 0), max_step=sim_length / 5000,
                                args=(p_vapour, p_inf, 1480, density, R_init, p_init, polytropic_const))

    peaks = find_peaks(out_km_inertial.y[0, :])[0]
    ratio = out_km_inertial.y[0, peaks[0]] / R_init
    rebound_ratios.append(ratio)

    if i % 5 == 0:
        plt.plot(out_km_inertial.t / ray_col_time, out_km_inertial.y[0, :] / R_init,
                 label="$p_{g0} =$ " + f"{p_init:.0f} Pa = {p_init / p_vapour:.2f} $p_v$")

plt.legend(fancybox=False, fontsize="x-small", loc="upper right", framealpha=1)
plt.xlabel("$t / t_{TC}$")
plt.ylabel("$R / R_0$")
plt.tight_layout()

plt.figure()

filt_p_inits, filt_rrs = zip(*[(p, r) for p, r in zip(all_p_inits, rebound_ratios) if 0.025 < p / p_vapour < 0.5])
(a, b), _ = curve_fit(lambda x, a, b: a * x ** b, filt_p_inits, filt_rrs)
filt_p_inits = np.array(filt_p_inits)

plt.plot(all_p_inits / p_vapour, rebound_ratios)
plt.plot(filt_p_inits / p_vapour, a * np.array(filt_p_inits) ** b, "k--",
         label=f"${a:.2f}" + "p_{g0}^{" + f"{b:.2f}" + "}$")
plt.xlabel("$p_{g0} / p_v$")
plt.ylabel("$R_1 / R_0$")
plt.loglog()
plt.legend(frameon=False)
format_axis_ticks_decimal(plt.gca().yaxis)
plt.tight_layout()

plt.show()
