from common.util.file_utils import csv_to_lists
from scipy.optimize import curve_fit
import numpy as np


def get_num_prediction_function():
    file_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Code/Cavitation-Analysis/numerical/models/model_outputs/slot/"
    file_name = "W2.00H2.00Y2.00_bem_slot_prediction_20000_0.25_24.0_normalised.csv"
    x_hats, theta_hats = csv_to_lists(file_dir, file_name, has_headers=True)
    x_hats, theta_hats = zip(*sorted(zip(x_hats, theta_hats)))

    x_hats = np.array(x_hats)
    theta_hats = np.array(theta_hats)

    def interp_pred(x, x_star, theta_star, x_offset, theta_offset):
        fit_xs = x_hats * x_star + x_offset
        fit_thetas = theta_hats * theta_star + theta_offset

        return np.interp(x, fit_xs, fit_thetas)

    return interp_pred


def num_prediction_fit(xs, thetas, std):
    interp_pred = get_num_prediction_function()
    (x_star, theta_star, x_offset, theta_offset), cov = \
        curve_fit(interp_pred, xs, thetas, [1, 0.5, 0, 0], sigma=[std] * len(xs))

    return x_star, theta_star, x_offset, theta_offset
