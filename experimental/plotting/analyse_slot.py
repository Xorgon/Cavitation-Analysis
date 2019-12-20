import importlib
import math
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

import experimental.util.analysis_utils as au
import experimental.util.file_utils as file
import experimental.util.config_utils as cu
import common.util.plotting_utils as plt_util


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
        "show_title": False,
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
    normalize = config["normalize"]  # Normalize the plot (theta_j = theta_j*, p_bar = p_bar / p_bar*).
    plot_fits = config["plot_fits"]  # Plot the fitted peaks.
    skip_bad_data = config["skip_bad_data"]  # Do not plot data sets that have bad data detected.
    plot_means = config["plot_means"]  # Plot a line through all of the means.
    labelled = config["labelled"]  # Label the graph with the q value(s).
    label_tags = config["label_tags"]  # Include geometry tags in labels.
    colours = config["colours"]  # Plot each series in different colours.
    error_bars = config["error_bars"]  # Plot as a mean and error bars rather than data points.
    show_title = config["show_title"]  # Show a title on the graph.
    do_shift = config["do_shift"]  # Shift the data according to peak positions.
    verbose = config["verbose"]  # Display additional messages.
    plot_predicted = config["plot_predicted"]  # Plot BEM predictions.
    y_max = None  # Set a fixed y-axis maximum value.
    confidence_interval = 0.99  # Set the confidence interval for the error bars.
    std = 0.015085955056793596  # Set the standard deviation for the error bars (from error_statistics.py).

    # prediction_files = ['bem_slot_prediction_20000']
    # prediction_files = ['w2h3y5.5_bem_slot_prediction_20000',
    #                     'w2h3y5.5_bem_slot_prediction_15000',
    #                     'w2h3y5.5_bem_slot_prediction_10000']
    prediction_file_dir = "../../numerical/models/model_outputs/slot/"
    # prediction_files = ['w2.20h2.70q3.84_bem_slot_prediction_20000']
    prediction_files = ['w4.20h11.47q2.43_bem_slot_prediction_20000']
    # prediction_files = []

    dirs = []
    if use_all_dirs:
        if use_defaults:
            root_dir = "../../../Data/SlotSweeps"
        else:
            root_dir = file.select_dir(create_window=create_window)
        for root, _, files in os.walk(root_dir):
            if "params.py" in files:
                dirs.append(root + "/")
        if verbose:
            print(f"Found {len(dirs)} data sets")
    else:
        if num_series is None:
            num_series = int(input("Number of data sets to load = "))
        for i in range(num_series):
            dirs.append(file.select_dir(create_window=create_window))

    res_dict = {}
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

        available_ys = set([reading.m_y for reading in readings])
        if use_all_series:
            reading_ys = available_ys
        else:
            reading_ys = []
            ys_config_dict = {}
            for this_y in available_ys:
                ys_config_dict[str(this_y)] = False
            ys_config_dict = cu.get_config(ys_config_dict, create_window=False)
            for key in ys_config_dict.keys():
                if ys_config_dict[key]:
                    reading_ys.append(float(key))

        for reading_y in reading_ys:
            if hasattr(params, 'title'):
                label = f"{params.title}:{reading_y}"
            else:
                label = f"{dir_path}:{reading_y}"

            # Post-process data to get jet angles.
            for reading in readings:
                if reading.m_y != reading_y:
                    continue
                theta_j = math.atan2(-reading.disp_vect[1], reading.disp_vect[0]) + math.pi / 2
                pos_mm = reading.get_bubble_pos_mm(params.mm_per_px)
                m_x = (pos_mm[0] - x_offset) / (0.5 * params.slot_width)
                q = pos_mm[1] - y_offset
                if q < 0:
                    continue

                if label in res_dict.keys():
                    res_dict[label][0].append(m_x)
                    res_dict[label][1].append(theta_j)
                    res_dict[label][2].append(q)
                    res_dict[label][3].append(reading.max_bubble_area)
                    res_dict[label][4].append(reading.m_x)
                else:
                    res_dict[label] = [[m_x], [theta_j], [q], [reading.max_bubble_area], [reading.m_x]]

    if y_max is not None:
        ax.set_ylim(-y_max, y_max)

    num_rejected_sets = 0
    print(f"Found {len(res_dict.keys())} reading_y values")
    markers = [".", "v", "s", "x", "^", "+", "D", "1", "*", "P", "X", "4", "2", "<", "3", ">", "H", "o", "p", "|"]
    if len(markers) < len(res_dict.keys()):
        raise ValueError("Too few markers are available for the data sets.")
    for label in res_dict.keys():
        # Use a second order polynomial to approximate both peaks.
        m_x_set = set(res_dict[label][4])  # Set of measurement x values
        mean_xs = []
        means = []
        y_errs = []
        for m_x in m_x_set:
            # Collect all theta_js where the measurement x value is m_x
            this_theta_js = [theta_j for res_x, theta_j in zip(res_dict[label][4], res_dict[label][1]) if res_x == m_x]
            this_mean = np.mean(this_theta_js)
            num_repeats = len(this_theta_js)

            # https://stackoverflow.com/a/28243282/5270376
            interval = stats.norm.interval(confidence_interval, loc=this_mean, scale=std / math.sqrt(num_repeats))
            y_errs.append(interval[1] - this_mean)

            means.append(this_mean)
            mean_xs.append(np.mean([b_x for res_x, b_x in zip(res_dict[label][4], res_dict[label][0]) if res_x == m_x]))

        sorted_mean_xs = sorted(zip(mean_xs, means), key=lambda k: k[1])
        max_peak_x = sorted_mean_xs[-1][0]
        min_peak_x = sorted_mean_xs[0][0]

        x_range = 1  # The range of x over which the peak is fitted

        max_poly_coeffs = np.polyfit([x for x in res_dict[label][0] if 0 < x < max_peak_x + x_range],
                                     [theta_j for x, theta_j in zip(res_dict[label][0], res_dict[label][1]) if
                                      0 < x < max_peak_x + x_range], 2)
        max_fitted_peak = - max_poly_coeffs[1] ** 2 / (4 * max_poly_coeffs[0]) + max_poly_coeffs[2]  # -b^2 / (4a) + c
        max_fitted_peak_p = - max_poly_coeffs[1] / (2 * max_poly_coeffs[0])  # -b / (2a)

        min_poly_coeffs = np.polyfit([x for x in res_dict[label][0] if min_peak_x - x_range < x < 0],
                                     [theta_j for x, theta_j in zip(res_dict[label][0], res_dict[label][1]) if
                                      min_peak_x - x_range < x < 0], 2)
        min_fitted_peak = - min_poly_coeffs[1] ** 2 / (4 * min_poly_coeffs[0]) + min_poly_coeffs[2]  # -b^2 / (4a) + c
        min_fitted_peak_p = - min_poly_coeffs[1] / (2 * min_poly_coeffs[0])

        if max_poly_coeffs[0] > 0 or min_poly_coeffs[0] < 0:
            print(f"WARNING: Incorrect curve fit on {label}.")
            if skip_bad_data:
                num_rejected_sets += 1
                continue

        theta_j_max = (max_fitted_peak - min_fitted_peak) / 2
        theta_j_offset = (max_fitted_peak + min_fitted_peak) / 2
        p_bar_theta_j_max = (max_fitted_peak_p - min_fitted_peak_p) / 2
        p_bar_offset = (max_fitted_peak_p + min_fitted_peak_p) / 2

        if abs(theta_j_offset) > 0.05:  # Any larger than this would mean very noticeable tilt of the frame.
            print(f"WARNING: Large jet angle offset detected on {label}.")
            if skip_bad_data:
                num_rejected_sets += 1
                continue

        if verbose:
            print(f"{label}\n"
                  f"    q = {np.mean(res_dict[label][2]):.4f}\n"
                  f"    Max peak = {max_fitted_peak:.4f} (at p_bar={max_fitted_peak_p:.4f})\n"
                  f"    Min peak = {min_fitted_peak:.4f} (at p_bar={min_fitted_peak_p:.4f})\n"
                  f"    Average peak = {theta_j_max:.4f} (at p_bar={p_bar_theta_j_max:.4f})\n"
                  f"    Offset = {theta_j_offset:.4f} (p_bar_offset={p_bar_offset:.4f})")

        # Correct any offset
        shifted_theta_j = np.subtract(res_dict[label][1], theta_j_offset)
        shifted_p_bar = np.subtract(res_dict[label][0], p_bar_offset)
        shifted_means = np.subtract(means, theta_j_offset)
        shifted_mean_xs = np.subtract(mean_xs, p_bar_offset)

        if do_shift:
            res_dict[label][1] = shifted_theta_j
            res_dict[label][0] = shifted_p_bar
            means = shifted_means
            mean_xs = shifted_mean_xs

        # Curve fit plot data
        max_fit_xs = np.linspace(0, max_peak_x + x_range, 100)
        max_fit_ys = np.polyval(max_poly_coeffs, max_fit_xs)
        shifted_max_fit_xs = np.subtract(max_fit_xs, p_bar_offset)
        shifted_max_fit_ys = np.subtract(max_fit_ys, theta_j_offset)
        if do_shift:
            max_fit_xs = shifted_max_fit_xs
            max_fit_ys = shifted_max_fit_ys

        min_fit_xs = np.linspace(min_peak_x - x_range, 0, 100)
        min_fit_ys = np.polyval(min_poly_coeffs, min_fit_xs)
        shifted_min_fit_xs = np.subtract(min_fit_xs, p_bar_offset)
        shifted_min_fit_ys = np.subtract(min_fit_ys, theta_j_offset)
        if do_shift:
            min_fit_xs = shifted_min_fit_xs
            min_fit_ys = shifted_min_fit_ys

        r2_min_to_max = r2_score([theta_j for x, theta_j in zip(shifted_p_bar, shifted_theta_j) if
                                  0 < x < max_peak_x + x_range],
                                 # Minimum curve fit reflected
                                 -np.subtract(np.polyval(min_poly_coeffs,
                                                         [-x for x in shifted_p_bar if
                                                          0 < x < max_peak_x + x_range]),
                                              theta_j_offset),
                                 multioutput='uniform_average')
        r2_max_to_min = r2_score([theta_j for x, theta_j in zip(shifted_p_bar, shifted_theta_j) if
                                  min_peak_x - x_range < x < 0],
                                 # Maximum curve fit reflected
                                 -np.subtract(np.polyval(max_poly_coeffs,
                                                         [-x for x in shifted_p_bar if
                                                          min_peak_x - x_range < x < 0]),
                                              theta_j_offset),
                                 multioutput='uniform_average')
        r2_cross = r2_score(np.flip(-shifted_min_fit_ys), shifted_max_fit_ys)

        if verbose:
            print(f"    Minimum fit on maximum data, r2 = {r2_min_to_max}")
            print(f"    Maximum fit on minimum data, r2 = {r2_max_to_min}")
            print(f"    Correlation between fits, r2 = {r2_cross}")

        if r2_min_to_max < 0.5 and r2_max_to_min < 0.5 and r2_cross < 0.5:  # Essentially arbitrary values
            print(f"WARNING: Poor correlation between fits {label}.")
            if skip_bad_data:
                num_rejected_sets += 1
                continue

        if normalize:
            res_dict[label][1] = np.divide(res_dict[label][1], theta_j_max)
            max_fit_ys = np.divide(max_fit_ys, theta_j_max)
            min_fit_ys = np.divide(min_fit_ys, theta_j_max)
            y_errs = np.divide(y_errs, theta_j_max)
            means = np.divide(means, theta_j_max)

            res_dict[label][0] = np.divide(res_dict[label][0], p_bar_theta_j_max)
            mean_xs = np.divide(mean_xs, p_bar_theta_j_max)
            max_fit_xs = np.divide(max_fit_xs, p_bar_theta_j_max)
            min_fit_xs = np.divide(min_fit_xs, p_bar_theta_j_max)

        mean_xs, means = zip(*sorted(zip(mean_xs, means), key=lambda k: k[0]))

        # Plot the data
        if plot_fits:
            ax.plot(max_fit_xs, max_fit_ys, color="r")
            ax.plot(min_fit_xs, min_fit_ys, color="r")
        if plot_means:
            ax.plot(mean_xs, means, color="g")
        marker = markers.pop(0)
        if labelled:
            c = None
            if len(res_dict.keys()) > 1 or colours:
                legend_label = f"q = {np.mean(res_dict[label][2]):.2f}"
            else:
                legend_label = "Experimental"
                c = "k"
            if label_tags:
                legend_label = "{0}, ".format(label.split(":")[0]) + legend_label
            if not colours:
                c = 'k'
            if error_bars:
                ax.errorbar(mean_xs, means, yerr=y_errs, capsize=3, fmt=marker, label=legend_label, color=c)
            else:
                ax.scatter(res_dict[label][0], res_dict[label][1], c, marker=marker, label=legend_label)
        else:
            c = None if colours else 'k'
            if error_bars:
                ax.errorbar(mean_xs, means, yerr=y_errs, capsize=3, fmt=marker, color=c)
                if verbose:
                    print(f"{label}, Mean q = {np.mean(res_dict[label][2])}\n")
            else:
                ax.scatter(res_dict[label][0], res_dict[label][1], marker=marker, color=c)
                if verbose:
                    print(f"{label}, Mean q = {np.mean(res_dict[label][2])}\n")

    print(f"Number of rejected data sets = {num_rejected_sets}")
    if plot_predicted:
        x_min, x_max = ax.get_xlim()
        colors = ['r', 'g', 'm', 'orange', 'b']
        for i, f_name in enumerate(prediction_files):
            predicted_xs = []
            predicted_theta_js = []
            f = open(f"{prediction_file_dir}{f_name}.csv")
            for line in f.readlines():
                split = line.strip().split(",")
                predicted_xs.append(float(split[0]))
                predicted_theta_js.append(float(split[1]))
            p_bar_theta_j_max, theta_j_max = sorted(zip(predicted_xs, predicted_theta_js), key=lambda k: k[1])[-1]
            if normalize:
                predicted_xs = np.divide(predicted_xs, p_bar_theta_j_max)
                predicted_theta_js = np.divide(predicted_theta_js, theta_j_max)
            label = f"Numerical {i}" if len(prediction_files) > 1 else "Numerical"
            c = colors[i] if not colours else 'k'
            ax.plot([x for x in predicted_xs if x_min < x < x_max],
                    [theta_j for x, theta_j in zip(predicted_xs, predicted_theta_js) if x_min < x < x_max],
                    c, label=label, zorder=9001)
            ax.set_xlim((x_min, x_max))
    if normalize:
        if set_x_label:
            ax.set_xlabel("$P$", labelpad=2)
        if set_y_label:
            ax.set_ylabel("$\\hat{\\theta}$", labelpad=2)
    else:
        if set_x_label:
            ax.set_xlabel("$\\bar{p}$", labelpad=0)
        if set_y_label:
            ax.set_ylabel("$\\theta_j$", labelpad=-5)
    if hasattr(params, 'title') and show_title:
        ax.set_title(params.title)
    # if reading_y is not None and labelled and show_title:
    #     if not hasattr(params, 'title'):
    #         ax.set_title("Mean $q = {0:.2f}$ mm".format(np.mean(ys)))
    #     else:
    #         ax.set_title(params.title + " (Mean $q = {0:.2f}$ mm)".format(np.mean(ys)))
    if labelled:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda k: k[1]))
        ax.legend(handles, labels, loc='upper left', fancybox=False, edgecolor='k', shadow=False, handletextpad=0.1,
                  borderpad=0.3)
    if not normalize:
        ax.axvline(x=-1, linestyle='--', color='gray')
        ax.axvline(x=1, linestyle='--', color='gray')


if __name__ == "__main__":
    font_size = 18
    plt_util.initialize_plt(font_size=font_size)

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
