import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

this_fig = plt.figure(figsize=(fig_width, fig_height))
plt.subplots_adjust(left=left_padding / fig_width,
                    right=(fig_width - right_padding) / fig_width,
                    top=(fig_height - top_padding) / fig_height,
                    bottom=bottom_padding / fig_height,
                    wspace=h_spacing / ax_width,
                    hspace=v_spacing / ax_width)
this_ax = this_fig.gca()

config = {
    "use_all_series": True,
    "use_all_dirs": True,
    "normalize": True,
    "plot_fits": False,
    "skip_bad_data": True,
    "plot_means": False,
    "labelled": False,
    "label_tags": False,
    "colours": False,
    "error_bars": True,
    "do_shift": True,
    "verbose": False,
    "plot_predicted": True
}

sweep_lists = [SweepData("W1H3", 1.94, 1.23, 2.74, "experiment_data/shifted_data/",
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

prediction_files = ["model_predictions/W1.23H2.74Y1.94_bem_slot_prediction_20000_0.25_15.csv"]

this_ax = analyse_slot(this_ax, sweeps=sweep_lists, config=config, num_series=1, prediction_files=prediction_files)
exp_line = mlines.Line2D([], [], color="black", linestyle=" ", marker=".", label='Experimental')
pred_line = mlines.Line2D([], [], color="C1", label='Numerical', linewidth=1)
this_ax.legend(handles=[exp_line, pred_line], frameon=False, loc='lower left')
plt.show()
