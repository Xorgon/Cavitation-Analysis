import importlib
import math
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

import experimental.util.analysis_utils as au
import common.util.file_utils as file
import experimental.util.config_utils as cu
import common.util.plotting_utils as plt_util


class SweepData:
    """
    A class to contain data for an individual experiment sweep. Also provides useful functions for the usage of this
    data.
    """
    geometry_label = None  # Geometry label.
    m_y = None  # Measured y value for all sweep points.
    W = None  # Slot width
    H = None  # Slot height
    m_xs = None  # Measured x values.
    xs = None  # Calculated x values.
    theta_js = None  # Calculated theta_j values.
    is_shifted = False  # Whether the data has been shifted to correct for offset
    Ys = None  # Calculated y values.

    def __init__(self, geometry_label, y, w, h):
        self.geometry_label = geometry_label
        self.m_y = y
        self.W = w
        self.H = h
        self.m_xs = []
        self.xs = []
        self.theta_js = []
        self.Ys = []

    def __str__(self):
        return f"{self.geometry_label}: y={self.m_y}, q={np.mean(self.Ys):.2f}"

    def add_point(self, X, x, theta_j, q):
        """ Adds a data point to the sweep. """
        self.m_xs.append(X)
        self.xs.append(x)
        self.theta_js.append(theta_j)
        self.Ys.append(q)

    def get_grouped_data(self):
        """ Groups data based on x position. """
        m_xs_set = set(self.m_xs)
        m_xs_xs = []
        m_xs_theta_js = []
        for m_x in m_xs_set:
            xs, theta_js = zip(*[(x, theta_j) for res_x, x, theta_j
                                 in zip(self.m_xs, self.xs, self.theta_js)
                                 if res_x == m_x])
            m_xs_xs.append(xs)
            m_xs_theta_js.append(theta_js)
        return m_xs_set, np.array(m_xs_xs), np.array(m_xs_theta_js)

    def get_mean_data(self):
        """ Computes the mean data for each group from :func:`SweepData.get_grouped_data`. """
        m_xs_set, xs_groups, theta_j_groups = self.get_grouped_data()
        mean_xs = []
        mean_theta_js = []
        for xs, theta_js in zip(xs_groups, theta_j_groups):
            mean_xs.append(np.mean(xs))
            mean_theta_js.append(np.mean(theta_js))
        return m_xs_set, mean_xs, mean_theta_js

    def get_curve_fits(self, range_fact=1.5, std=0.015085955056793596):
        """
        Computes and returns curve fits for the two peaks of a sweep. Returns two tuples, one each for the maximum and
        minimum peaks. Each tuple contains the position of the peak, value at the peak, and curve fit polynomial
        coefficients.
        :param range_fact: How far beyond the highest mean value to use for peak fitting.
        :param std: Standard deviation for data set.
        :returns: (max_peak_x, max_peak_theta_j, max_poly_coeffs, max_std_theta_j),
                 (min_peak_x, min_peak_theta_j, min_poly_coeffs, min_std_theta_j)
        """
        _, mean_xs, mean_theta_js = self.get_mean_data()
        srtd_mean_xs, srtd_mean_theta_js = zip(*sorted(zip(mean_xs, mean_theta_js),
                                                       key=lambda k: k[1]))  # Sorted by theta_j
        max_peak_x = srtd_mean_xs[-1]
        min_peak_x = srtd_mean_xs[0]

        x_range = range_fact * (max_peak_x - min_peak_x) / 2

        max_xs, max_theta_js = zip(*[(x, theta_j) for x, theta_j in zip(mean_xs, mean_theta_js)
                                     if 0 < x < x_range])
        max_weights = [1 / std ** 2] * len(max_xs)
        max_poly_coeffs, max_cov = np.polyfit(max_xs, max_theta_js, 2, cov='unscaled', w=max_weights)
        max_peak_theta_j = - max_poly_coeffs[1] ** 2 / (4 * max_poly_coeffs[0]) + max_poly_coeffs[2]  # -b^2 / (4a) + c
        max_peak_x = - max_poly_coeffs[1] / (2 * max_poly_coeffs[0])  # -b / (2a)

        a, b, c = max_poly_coeffs
        s_a, s_b, s_c = max_cov.diagonal()
        max_std_theta_j = 3 * std / len(max_xs)  # M * std / N

        min_xs, min_theta_js = zip(*[(x, theta_j) for x, theta_j in zip(mean_xs, mean_theta_js)
                                     if - x_range < x < 0])
        min_weights = [1 / std ** 2] * len(min_xs)
        min_poly_coeffs, min_cov = np.polyfit(min_xs, min_theta_js, 2, cov='unscaled', w=min_weights)
        min_peak_theta_j = - min_poly_coeffs[1] ** 2 / (4 * min_poly_coeffs[0]) + min_poly_coeffs[2]  # -b^2 / (4a) + c
        min_peak_x = - min_poly_coeffs[1] / (2 * min_poly_coeffs[0])

        a, b, c = max_poly_coeffs
        s_a, s_b, s_c = max_cov.diagonal()
        min_std_theta_j = 3 * std / len(min_xs)  # M * std / N

        return (max_peak_x, max_peak_theta_j, max_poly_coeffs, max_std_theta_j), \
               (min_peak_x, min_peak_theta_j, min_poly_coeffs, min_std_theta_j)

    def get_curve_fits_cubic(self, range_fact=1.5, std=0.015085955056793596):
        """
        Computes and returns curve fits for the two peaks of a sweep. Returns two tuples, one each for the maximum and
        minimum peaks. Each tuple contains the position of the peak, value at the peak, and curve fit polynomial
        coefficients.
        :param range_fact: How far beyond the highest mean value to use for peak fitting.
        :param std: Standard deviation for data set.
        :returns: (max_peak_x, max_peak_theta_j, max_poly_coeffs, max_std_theta_j),
                 (min_peak_x, min_peak_theta_j, min_poly_coeffs, min_std_theta_j)
        """
        _, mean_xs, mean_theta_js = self.get_mean_data()
        srtd_mean_xs, srtd_mean_theta_js = zip(*sorted(zip(mean_xs, mean_theta_js),
                                                       key=lambda k: k[1]))  # Sorted by theta_j
        max_peak_x = srtd_mean_xs[-1]
        min_peak_x = srtd_mean_xs[0]

        x_range = range_fact * (max_peak_x - min_peak_x) / 2

        ################
        # Maximum peak #
        ################
        max_xs, max_theta_js = zip(*[(x, theta_j) for x, theta_j in zip(mean_xs, mean_theta_js)
                                     if 0 < x < x_range])
        max_weights = [1 / std ** 2] * len(max_xs)
        max_poly_coeffs, max_cov = np.polyfit(max_xs, max_theta_js, 3, cov='unscaled', w=max_weights)

        a, b, c, d = max_poly_coeffs
        if 4 * b ** 2 - 12 * a * c < 0:
            raise ValueError("Invalid curve fit.")  # TODO: Handle this better.
        max_peak_x_pos = (-2 * b + np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * a)
        pos_deriv = 6 * a * max_peak_x_pos + 2 * b

        max_peak_x_neg = (-2 * b - np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * a)
        neg_deriv = 6 * a * max_peak_x_neg + 2 * b

        if pos_deriv <= 0:
            max_peak_x = max_peak_x_pos
        elif neg_deriv <= 0:
            max_peak_x = max_peak_x_neg
        else:
            raise ValueError("Invalid curve fit.")
        max_peak_theta_j = np.polyval(max_poly_coeffs, max_peak_x)
        max_std_theta_j = 4 * std / len(max_xs)  # M * std / N

        ################
        # Minimum peak #
        ################
        min_xs, min_theta_js = zip(*[(x, theta_j) for x, theta_j in zip(mean_xs, mean_theta_js)
                                     if -x_range < x < 0])
        min_weights = [1 / std ** 2] * len(min_xs)
        min_poly_coeffs, min_cov = np.polyfit(min_xs, min_theta_js, 3, cov='unscaled', w=min_weights)

        a, b, c, d = min_poly_coeffs
        if 4 * b ** 2 - 12 * a * c < 0:
            raise ValueError("Invalid curve fit.")  # TODO: Handle this better.
        min_peak_x_pos = (-2 * b + np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * a)
        pos_deriv = 6 * a * min_peak_x_pos + 2 * b

        min_peak_x_neg = (-2 * b - np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * a)
        neg_deriv = 6 * a * min_peak_x_neg + 2 * b

        if pos_deriv >= 0:
            min_peak_x = min_peak_x_pos
        elif neg_deriv >= 0:
            min_peak_x = min_peak_x_neg
        else:
            raise ValueError("Invalid curve fit.")
        min_peak_theta_j = np.polyval(min_poly_coeffs, min_peak_x)
        min_std_theta_j = 4 * std / len(min_xs)  # M * std / N

        return (max_peak_x, max_peak_theta_j, max_poly_coeffs, max_std_theta_j), \
               (min_peak_x, min_peak_theta_j, min_poly_coeffs, min_std_theta_j)

    def check_curve_fits(self, max_peak_x, max_peak_theta_j, max_poly_coeffs,
                         min_peak_x, min_peak_theta_j, min_poly_coeffs,
                         range_fact=1.5, verbose=False):  # TODO: Use range_fact rather than x_range
        """
        Shifts the data contained in SweepData based on the curve fit peaks supplied.
        :param max_peak_x: x position of maximum peak.
        :param max_peak_theta_j: theta_j value at maximum peak.
        :param max_poly_coeffs: polynomial coefficients for the maximum peak fit.
        :param min_peak_x: x position of minimum peak.
        :param min_peak_theta_j: theta_j value at miminum peak.
        :param min_poly_coeffs: polynomial coefficients for the minimum peak fit.
        :param range_fact: How far beyond the highest mean value to use for peak fitting.
        :param verbose: debug outputs.
        :return:
        """
        theta_j_offset = (max_peak_theta_j + min_peak_theta_j) / 2
        x_star = (max_peak_x - min_peak_x) / 2
        x_offset = (max_peak_x + min_peak_x) / 2

        x_range = range_fact * (max_peak_x - min_peak_x) / 2  # The range of x over which the peak is fitted

        if max_poly_coeffs[0] > 0 or min_poly_coeffs[0] < 0:
            print(f"WARNING: Incorrect curve fit on {self.geometry_label}:{self.m_y}.")
            return False

        shifted_x = self.xs if self.is_shifted else np.subtract(self.xs, x_offset)
        shifted_theta_j = self.theta_js if self.is_shifted else np.subtract(self.theta_js, theta_j_offset)

        r2_max_to_max = r2_score([theta_j for x, theta_j in zip(shifted_x, shifted_theta_j) if
                                  0 < x < x_range],
                                 np.subtract(np.polyval(max_poly_coeffs,
                                                        [x for x in shifted_x if
                                                         0 < x < x_range]),
                                             theta_j_offset),
                                 multioutput='uniform_average')

        r2_min_to_min = r2_score([theta_j for x, theta_j in zip(shifted_x, shifted_theta_j) if
                                  - x_range < x < 0],
                                 np.subtract(np.polyval(min_poly_coeffs,
                                                        [x for x in shifted_x if
                                                         - x_range < x < 0]),
                                             theta_j_offset),
                                 multioutput='uniform_average')

        r2_min_to_max = r2_score([theta_j for x, theta_j in zip(shifted_x, shifted_theta_j) if
                                  0 < x < x_range],
                                 # Minimum curve fit reflected
                                 -np.subtract(np.polyval(min_poly_coeffs,
                                                         [-x for x in shifted_x if
                                                          0 < x < x_range]),
                                              theta_j_offset),
                                 multioutput='uniform_average')
        r2_max_to_min = r2_score([theta_j for x, theta_j in zip(shifted_x, shifted_theta_j) if
                                  - x_range < x < 0],
                                 # Maximum curve fit reflected
                                 -np.subtract(np.polyval(max_poly_coeffs,
                                                         [-x for x in shifted_x if
                                                          - x_range < x < 0]),
                                              theta_j_offset),
                                 multioutput='uniform_average')

        if verbose:
            print(f"{self.geometry_label}: q={np.mean(self.Ys)}")
            print(f"    Maximum fit on maximum data, r2 = {r2_max_to_max:.3f}")
            print(f"    Minimum fit on maximum data, r2 = {r2_min_to_max:.3f}")
            print(f"    Minimum fit on minimum data, r2 = {r2_min_to_min:.3f}")
            print(f"    Maximum fit on minimum data, r2 = {r2_max_to_min:.3f}")

        low_values = np.mean([r2_max_to_max, r2_max_to_min, r2_min_to_min, r2_min_to_max]) < 0.25
        badly_matched = r2_max_to_max / r2_min_to_max < 0.75 or r2_min_to_min / r2_max_to_min < 0.75
        if low_values:
            print(f"WARNING: Poor correlation between fits {self.geometry_label}:{self.m_y}, q={np.mean(self.Ys):.2f}.")
            return False

        return True

    def shift_data(self, max_peak_x, max_peak_theta_j, min_peak_x, min_peak_theta_j, keep_shifted=True):
        """
        Shifts the data contained in SweepData based on the curve fit peaks supplied.
        :param max_peak_x: x position of maximum peak.
        :param max_peak_theta_j: theta_j value at maximum peak.
        :param min_peak_x: x position of minimum peak.
        :param min_peak_theta_j: theta_j value at miminum peak.
        :param keep_shifted: whether to keep the shifted data rather than the original data.
        :return:
        """
        theta_j_max = (max_peak_theta_j - min_peak_theta_j) / 2
        theta_j_offset = (max_peak_theta_j + min_peak_theta_j) / 2
        x_star = (max_peak_x - min_peak_x) / 2
        x_offset = (max_peak_x + min_peak_x) / 2

        # Correct the offset
        if keep_shifted:
            self.theta_js = np.subtract(self.theta_js, theta_j_offset)
            self.xs = np.subtract(self.xs, x_offset)
            self.is_shifted = True

        return x_star, x_offset, theta_j_max, theta_j_offset

    def get_error_bars(self, confidence_interval, std):
        """ Computes error bar values for each group from :func:`SweepData.get_grouped_data`. """
        _, x_groups, theta_j_groups = self.get_grouped_data()
        errors = []
        for xs, theta_js in zip(x_groups, theta_j_groups):
            # https://stackoverflow.com/a/28243282/5270376
            interval = stats.norm.interval(confidence_interval, loc=np.mean(theta_js),
                                           scale=std / math.sqrt(len(xs)))
            errors.append(interval[1] - np.mean(theta_js))
        return errors


def select_data_series(use_all_dirs=True, num_series=None, use_defaults=True, verbose=False, create_window=True):
    dirs = []
    if use_all_dirs:
        if use_defaults:
            root_dir = "../../../../Data/SlotSweeps"
        else:
            root_dir = file.select_dir("../../../../Data/SlotSweeps", create_window=create_window)
            if root_dir == "/":
                exit()
        for root, _, files in os.walk(root_dir):
            if "params.py" in files:
                dirs.append(root + "/")
        if verbose:
            print(f"Found {len(dirs)} data sets")
    else:
        if num_series is None:
            num_series = int(input("Number of data sets to load = "))
        for i in range(num_series):
            dirs.append(file.select_dir("../../../../Data/SlotSweeps", create_window=create_window))
    return dirs


def plot_prediction_file(prediction_file, ax, normalize=False, x_star=None, theta_star=None, label="Numerical", c='k'):
    x_min, x_max = ax.get_xlim()  # Keep track of original x limits.
    predicted_xs, predicted_theta_js = file.csv_to_lists("", prediction_file, has_headers=True)
    if x_star is None or theta_star is None:
        x_star, theta_star = sorted(zip(predicted_xs, predicted_theta_js), key=lambda k: k[1])[-1]
    if normalize:
        predicted_xs = np.divide(predicted_xs, x_star)
        predicted_theta_js = np.divide(predicted_theta_js, theta_star)
    ax.plot([x for x in predicted_xs if x_min < x < x_max],
            [theta_j for x, theta_j in zip(predicted_xs, predicted_theta_js) if x_min < x < x_max],
            color=c, label=label, zorder=9001)
    ax.set_xlim((x_min, x_max))  # Reset x limits to their original values.


def plot_prediction_files(prediction_files, ax, normalize=False, coloured_lines=True, x_stars=None, theta_stars=None):
    x_min, x_max = ax.get_xlim()  # Keep track of original x limits.
    for i, f_name in enumerate(prediction_files):
        x_star = None
        theta_star = None
        if x_stars is not None and len(x_stars) == len(prediction_files):
            x_star = x_stars[i]
        if theta_stars is not None and len(theta_stars) == len(prediction_files):
            theta_star = theta_stars[i]
        label = "Numerical"
        if len(prediction_files) > 0:
            label += f" {i}"
        c = 'k'
        if coloured_lines:
            c = f"C{i + 1}"
        plot_prediction_file(f_name, ax, normalize, x_star, theta_star, label=label, c=c)
    ax.set_xlim((x_min, x_max))  # Reset x limits to their original values.


def analyse_slot(ax, set_y_label=True, set_x_label=True, use_defaults=False, config=None, num_series=None):
    create_window = not mpl.get_backend() == "Qt5Agg"
    default_config = {
        "use_all_series": True,
        "use_all_dirs": True,
        "normalize": True,
        "plot_fits": False,
        "skip_bad_data": True,
        "plot_means": False,
        "labelled": True,
        "label_tags": False,
        "colours": True,
        "error_bars": True,
        "do_shift": True,
        "verbose": False,
        "plot_predicted": False
    }

    if config is None:
        config = default_config
        if not use_defaults:
            config = cu.get_config(config, create_window=create_window)
    else:
        for key in default_config.keys():
            if key not in config:
                config[key] = default_config[key]

    use_all_series = config["use_all_series"]  # Use all reading_y values for selected data sets.
    use_all_dirs = config["use_all_dirs"]  # Use all directories that contain params.py.
    normalize = config["normalize"]  # Normalize the plot (theta_j = theta_j*, x = x / x*).
    plot_fits = config["plot_fits"]  # Plot the fitted peaks.
    skip_bad_data = config["skip_bad_data"]  # Do not plot data sets that have bad data detected.
    plot_means = config["plot_means"]  # Plot a line through all of the means.
    labelled = config["labelled"]  # Label the graph with the Y value(s).
    label_tags = config["label_tags"]  # Include geometry tags in labels.
    colours = config["colours"]  # Plot each series in different colours.
    error_bars = config["error_bars"]  # Plot as a mean and error bars rather than data points.
    do_shift = config["do_shift"]  # Shift the data according to peak positions.
    verbose = config["verbose"]  # Display additional messages.
    plot_predicted = config["plot_predicted"]  # Plot BEM predictions.
    y_max = None  # Set a fixed y-axis maximum value.
    confidence_interval = 0.99  # Set the confidence interval for the error bars.
    std = 0.015085955056793596  # Set the standard deviation for the error bars (from error_statistics.py).

    prediction_file_dir = "../../numerical/models/model_outputs/exp_comparisons/"
    prediction_files = []
    if plot_predicted:
        prediction_files = file.select_multiple_files(prediction_file_dir, create_window=create_window)

    dirs = select_data_series(use_all_dirs, num_series, use_defaults, verbose, create_window)

    sweeps = []
    for dir_path in dirs:
        sys.path.append(dir_path)
        import params
        importlib.reload(params)
        sys.path.remove(dir_path)

        x_offset = params.left_slot_wall_x + params.slot_width / 2
        y_offset = params.upper_surface_y

        if os.path.exists(dir_path + "invalid_readings.txt") and os.path.exists(dir_path + "readings_dump.csv"):
            au.flag_invalid_readings(dir_path)
        readings = au.load_readings(dir_path + "readings_dump.csv")
        readings = sorted(readings, key=lambda r: r.m_x)

        # Select y values to use
        available_ys = set([reading.m_y for reading in readings])
        if use_all_series:
            reading_ys = available_ys
        else:
            reading_ys = []
            ys_config_dict = {}
            for this_y in sorted(available_ys):
                ys_config_dict[str(this_y)] = False
            ys_config_dict = cu.get_config(ys_config_dict, create_window=False)
            for key in ys_config_dict.keys():
                if ys_config_dict[key]:
                    reading_ys.append(float(key))

        for reading_y in reading_ys:
            if hasattr(params, 'title'):
                label = f"{params.title}"
            else:
                label = f"{dir_path}"

            sweep_data = SweepData(label, reading_y, params.slot_width, params.slot_height)
            sweep_readings = [reading for reading in readings if reading.m_y == reading_y]
            # Post-process data to get jet angles.
            for reading in sweep_readings:
                theta_j = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
                pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
                x = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
                Y = pos_mm[1] - y_offset
                sweep_data.add_point(reading.m_x, x, theta_j, Y)
            if np.mean(sweep_data.Ys) < 0:
                continue  # Ignore bubbles generated inside the slot.
            sweeps.append(sweep_data)

    num_rejected_sets = 0
    print(f"Found {len(sweeps)} sweeps")
    if verbose:
        for s in sweeps:
            print(s)
    markers = [".", "v", "s", "x", "^", "+", "D", "1", "*", "P", "X", "4", "2", "<", "3", ">", "H", "o", "p", "|"]
    if len(markers) < len(sweeps):
        raise ValueError("Too few markers are available for the data sets.")
    for i, sweep in enumerate(sorted(sweeps, key=lambda j: (j.geometry_label, j.m_y))):
        is_bad_data = False

        m_x_set, mean_xs, means = sweep.get_mean_data()
        y_errs = sweep.get_error_bars(confidence_interval, std)

        sorted_mean_xs = sorted(zip(mean_xs, means), key=lambda k: k[1])
        max_peak_x = sorted_mean_xs[-1][0]
        min_peak_x = sorted_mean_xs[0][0]

        range_fact = 1.5

        (max_fitted_peak_p, max_fitted_peak, max_poly_coeffs, _), \
        (min_fitted_peak_p, min_fitted_peak, min_poly_coeffs, _) \
            = sweep.get_curve_fits(range_fact)

        is_bad_data = not sweep.check_curve_fits(max_fitted_peak_p, max_fitted_peak, max_poly_coeffs,
                                                 min_fitted_peak_p, min_fitted_peak, min_poly_coeffs,
                                                 range_fact, verbose) or is_bad_data

        x_star, x_offset, theta_star, theta_j_offset = sweep.shift_data(max_fitted_peak_p,
                                                                        max_fitted_peak,
                                                                        min_fitted_peak_p,
                                                                        min_fitted_peak,
                                                                        keep_shifted=do_shift)
        m_x_set, mean_xs, means = sweep.get_mean_data()

        if abs(theta_j_offset) > 0.05:  # Any larger than this would mean very noticeable tilt of the frame.
            print(f"WARNING: Large jet angle offset detected on {sweep.geometry_label}:{sweep.m_y}."
                  f" Y={np.mean(sweep.Ys)}, theta_j_offset={theta_j_offset:.5f}")
            is_bad_data = True

        Y = np.mean(sweep.Ys)
        if verbose:
            print(f"{sweep.geometry_label}:{sweep.m_y}\n"
                  f"    Y = {Y :.4f}\n"
                  f"    Max peak = {max_fitted_peak:.4f} (at x={max_fitted_peak_p:.4f})\n"
                  f"    Min peak = {min_fitted_peak:.4f} (at x={min_fitted_peak_p:.4f})\n"
                  f"    Average peak = {theta_star:.4f} (at x={x_star:.4f})\n"
                  f"    Offset = {theta_j_offset:.4f} (x_offset={x_offset:.4f})")

        # Curve fit plot data
        x_range = range_fact * (max_peak_x - min_peak_x) / 2
        max_fit_xs = np.linspace(0, x_range, 100)
        max_fit_ys = np.polyval(max_poly_coeffs, max_fit_xs)
        shifted_max_fit_xs = np.subtract(max_fit_xs, x_offset)
        shifted_max_fit_ys = np.subtract(max_fit_ys, theta_j_offset)
        if do_shift:
            max_fit_xs = shifted_max_fit_xs
            max_fit_ys = shifted_max_fit_ys

        min_fit_xs = np.linspace(-x_range, 0, 100)
        min_fit_ys = np.polyval(min_poly_coeffs, min_fit_xs)
        shifted_min_fit_xs = np.subtract(min_fit_xs, x_offset)
        shifted_min_fit_ys = np.subtract(min_fit_ys, theta_j_offset)
        if do_shift:
            min_fit_xs = shifted_min_fit_xs
            min_fit_ys = shifted_min_fit_ys

        if normalize:
            sweep.theta_js = np.divide(sweep.theta_js, theta_star)
            max_fit_ys = np.divide(max_fit_ys, theta_star)
            min_fit_ys = np.divide(min_fit_ys, theta_star)
            y_errs = np.divide(y_errs, theta_star)
            means = np.divide(means, theta_star)

            sweep.xs = np.divide(sweep.xs, x_star)
            mean_xs = np.divide(mean_xs, x_star)
            max_fit_xs = np.divide(max_fit_xs, x_star)
            min_fit_xs = np.divide(min_fit_xs, x_star)

        mean_xs, means = zip(*sorted(zip(mean_xs, means), key=lambda k: k[0]))

        if skip_bad_data and is_bad_data:
            num_rejected_sets += 1
            continue

        # Plot the data
        if plot_fits:
            ax.plot(max_fit_xs, max_fit_ys, color="r")
            ax.plot(min_fit_xs, min_fit_ys, color="r")
        if plot_means:
            ax.plot(mean_xs, means, color="g")
        marker = markers.pop(0)
        c = f"C{i}"
        if labelled:
            zorder = Y
            if len(sweeps) > 1 or colours:
                legend_label = f"Y = {Y :.2f}"
            else:
                legend_label = "Experimental"
                c = "k"
            if label_tags:
                legend_label = "{0}, ".format(sweep.geometry_label) + legend_label
            if not colours:
                c = 'k'
            if is_bad_data:
                c = 'lightgray'
                zorder = -1
            if error_bars:
                ax.errorbar(mean_xs, means, yerr=y_errs, fmt=marker, label=legend_label, color=c,
                            zorder=zorder)
            else:
                ax.scatter(sweep.xs, sweep.theta_js, c, marker=marker, label=legend_label, zorder=zorder)
        else:
            c = 'k' if not colours else c
            zorder = Y
            if is_bad_data:
                c = 'lightgray'
                zorder = -1
            if error_bars:
                ax.errorbar(mean_xs, means, yerr=y_errs, fmt=marker, color=c, zorder=zorder)
                if verbose:
                    print(f"{sweep.geometry_label}:{sweep.m_y}, Mean Y = {Y:.3f}\n")
            else:
                ax.scatter(sweep.xs, sweep.theta_js, marker=marker, color=c, zorder=zorder)
                if verbose:
                    print(f"{sweep.geometry_label}:{sweep.m_y}, Mean Y = {Y:.3f}\n")
        if len(sweeps) == len(prediction_files):
            label = None
            if len(sweeps) == 1:
                label = "Numerical"
            plot_prediction_file(prediction_files[i], ax, normalize, x_star=x_star, theta_star=theta_star, label=label,
                                 c=c)

    print(f"Number of rejected data sets = {num_rejected_sets}")

    # Plot predictions.
    if plot_predicted and len(sweeps) != len(prediction_files):
        plot_prediction_files(prediction_files, ax, normalize, not colours)

    # Set axis labels.
    if normalize:
        if set_x_label:
            ax.set_xlabel("$\\hat{x}$", labelpad=2)
        if set_y_label:
            ax.set_ylabel("$\\hat{\\theta}$", labelpad=2)
    else:
        if set_x_label:
            ax.set_xlabel("$x$", labelpad=0)
        if set_y_label:
            ax.set_ylabel("$\\theta_j$", labelpad=-5)

    # Create legend.
    if labelled:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda k: k[1]))
        ax.legend(handles, labels, loc='upper left', fancybox=False, edgecolor='k', shadow=False, handletextpad=0.1,
                  borderpad=0.3, frameon=False)

    # Add slot boundary lines.
    if not normalize:
        ax.axvline(x=-1, linestyle='--', color='gray')
        ax.axvline(x=1, linestyle='--', color='gray')

    if y_max is not None:
        ax.set_ylim(-y_max, y_max)


if __name__ == "__main__":
    font_size = 10
    plt_util.initialize_plt(font_size=font_size, capsize=3)

    fig_width = 5.31445
    fig_height = 5.31445 * 2 / 3
    left_padding = 0.4 + font_size / 35
    right_padding = 0.08
    top_padding = 0.08
    bottom_padding = 0.2 + font_size / 50
    v_spacing = 0.3
    h_spacing = 0.1
    ax_width = fig_width - left_padding - right_padding

    this_fig = plt.figure(figsize=(fig_width, fig_height))
    plt.subplots_adjust(left=left_padding / fig_width,
                        right=(fig_width - right_padding) / fig_width,
                        top=(fig_height - top_padding) / fig_height,
                        bottom=bottom_padding / fig_height,
                        wspace=h_spacing / ax_width,
                        hspace=v_spacing / ax_width)
    this_ax = this_fig.gca()
    this_ax = analyse_slot(this_ax)
    # plt.tight_layout()
    plt.show()
