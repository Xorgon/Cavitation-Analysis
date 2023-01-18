import os
import sys
import importlib
import experimental.util.analysis_utils as au
from common.util.plotting_utils import initialize_plt
from util.file_utils import lists_to_csv, csv_to_lists
import numpy as np
import matplotlib.pyplot as plt
import numerical.util.gen_utils as gen
import scipy
import numerical.bem as bem
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import find_peaks
import random

R_init = 3 / 1000
bubble_pos = [0, R_init * 2.5, 0]

for plane_length in [0.01, 0.02, 0.03, 0.04, 0.05, 0.15, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5]:
    print(f"L = {plane_length:.2f}", end="    ")
    try:
        filename = f"uniform_hs_vs_zetas_L{100 * plane_length:.0f}"
        print(filename)
        hs, anisotropies = csv_to_lists("", filename, has_headers=True)
        plt.scatter(np.array(hs) / R_init, anisotropies, label=f"$L / R = {plane_length / R_init:.2f}$")
    except FileNotFoundError:
        print("    not done yet")

plt.axhline(0.195 * (bubble_pos[1] / R_init) ** -2, color="grey", linestyle="--", label="Analytic")
plt.legend(frameon=False)
plt.xlabel("Normalised smallest element length $h / R$")
plt.ylabel("Anisotropy $\\zeta$")
plt.show()
