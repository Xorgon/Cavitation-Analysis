import matplotlib.pyplot as plt
import numpy as np
from util.analyse_slot import analyse_slot, SweepData
import util.plotting_utils as plt_util

font_size = 10
plt_util.initialize_plt(font_size=font_size, capsize=2, line_scale=0.5)

fig_width = 0.75 * 5.31445
fig_height = fig_width * 2 / 3
left_padding = 0.4 + font_size / 35
right_padding = 0.08
top_padding = 0.08
bottom_padding = 0.2 + font_size / 50
v_spacing = 0.3
h_spacing = 0.1
ax_width = fig_width - left_padding - right_padding

sweeps = [SweepData("W1H3", 1.94, 1.23, 2.74, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W1H3_Y1.94.csv"),
          SweepData("W1H3", 2.91, 1.23, 2.74, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W1H3_Y2.91.csv"),
          SweepData("W1H3", 3.89, 1.23, 2.74, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W1H3_Y3.89.csv"),
          SweepData("W2H3a", 1.77, 2.20, 2.70, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3a_Y1.77.csv"),
          SweepData("W2H3a", 2.29, 2.20, 2.70, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3a_Y2.29.csv"),
          SweepData("W2H3a", 2.81, 2.20, 2.70, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3a_Y2.81.csv"),
          SweepData("W2H3a", 3.32, 2.20, 2.70, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3a_Y3.32.csv"),
          SweepData("W2H3a", 3.84, 2.20, 2.70, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3a_Y3.84.csv"),
          SweepData("W2H3b", 2.66, 2.20, 2.90, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3b_Y2.66.csv"),
          SweepData("W2H3b", 3.68, 2.20, 2.90, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H3b_Y3.68.csv"),
          SweepData("W2H6", 1.52, 2.20, 5.40, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H6_Y1.52.csv"),
          SweepData("W2H6", 1.99, 2.20, 5.40, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H6_Y1.99.csv"),
          SweepData("W2H9", 1.66, 2.14, 8.21, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H9_Y1.66.csv"),
          SweepData("W2H9", 2.66, 2.14, 8.21, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H9_Y2.66.csv"),
          SweepData("W2H12", 2.63, 2.20, 11.50, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W2H12_Y2.63.csv"),
          SweepData("W4H12", 2.43, 4.20, 11.47, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W4H12_Y2.43.csv"),
          SweepData("W4H12", 3.43, 4.20, 11.47, "experiment_data/shifted_data/",
                    sweep_file="shifted_data_sweep_W4H12_Y3.43.csv")]

predictions = ["model_predictions/W1.23H2.74Y1.94_bem_slot_prediction_20000_0.25_15.csv",
               "model_predictions/W1.23H2.74Y2.91_bem_slot_prediction_20000_0.25_15.csv",
               "model_predictions/W1.23H2.74Y3.89_bem_slot_prediction_20000_0.25_15.csv",
               "model_predictions/W2.20H2.70Y1.77_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H2.70Y2.29_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H2.70Y2.81_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H2.70Y3.32_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H2.70Y3.84_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H2.90Y2.66_bem_slot_prediction_20000_0.25_8.csv",
               "model_predictions/W2.20H2.90Y3.68_bem_slot_prediction_20000_0.25_8.csv",
               "model_predictions/W2.20H5.40Y1.52_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H5.40Y1.99_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.14H8.21Y1.66_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.14H8.21Y2.66_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W2.20H11.50Y2.63_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W4.20H11.47Y2.43_bem_slot_prediction_20000_0.25_5.csv",
               "model_predictions/W4.20H11.47Y3.43_bem_slot_prediction_20000_0.25_5.csv"]

config = {
    "use_all_series": False,
    "use_all_dirs": False,
    "normalize": False,
    "plot_fits": False,
    "skip_bad_data": True,
    "plot_means": False,
    "labelled": False,
    "label_tags": False,
    "colours": False,
    "error_bars": True,
    "do_shift": True,
    "verbose": True,
    "plot_predicted": True
}

for sweep, pred_file in zip(sweeps, predictions):
    this_fig = plt.figure(figsize=(fig_width, fig_height), num=f"{sweep.geometry_label} - Y = {np.mean(sweep.Ys):.2f}")
    plt.subplots_adjust(left=left_padding / fig_width,
                        right=(fig_width - right_padding) / fig_width,
                        top=(fig_height - top_padding) / fig_height,
                        bottom=bottom_padding / fig_height,
                        wspace=h_spacing / ax_width,
                        hspace=v_spacing / ax_width)
    analyse_slot(this_fig.gca(), sweeps=[sweep], prediction_files=[pred_file], config=config, num_series=1)
    plt.tight_layout()
plt.show()
