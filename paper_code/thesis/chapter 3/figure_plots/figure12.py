import matplotlib.pyplot as plt
import itertools
from util.analyse_slot import analyse_slot, SweepData
import util.plotting_utils as plt_util

num_cols = 3
num_rows = 3
num_plots = 7

plt_util.initialize_plt(line_scale=0.5, capsize=1.5)

config = {
    "use_all_series": True,
    "use_all_dirs": False,
    "normalize": False,
    "plot_fits": False,
    "skip_bad_data": True,
    "plot_means": False,
    "labelled": False,
    "label_tags": False,
    "colours": True,
    "error_bars": True,
    "do_shift": True,
    "verbose": False,
    "plot_predicted": False
}

fig_width = 5.31445  # From LaTeX

left_padding = 0.7
right_padding = 0.08
top_padding = 0.08
bottom_padding = 0.5
v_spacing = 0.3
h_spacing = 0.1
ax_width = (fig_width - left_padding - right_padding - (num_cols - 1) * h_spacing) / num_cols

fig_height = ax_width * num_rows + bottom_padding + top_padding + (num_rows - 1) * v_spacing

fig, axes = plt.subplots(num_rows, num_cols, sharex='none', sharey="all", figsize=(fig_width, fig_height))
labels = ["($a$)", "($b$)", "($c$)", "($d$)", "($e$)", "($f$)", "($g$)"]

sweep_lists = [[SweepData("W1H3", 1.94, 1.23, 2.74, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W1H3_Y1.94.csv"),
                SweepData("W1H3", 2.91, 1.23, 2.74, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W1H3_Y2.91.csv"),
                SweepData("W1H3", 3.89, 1.23, 2.74, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W1H3_Y3.89.csv")],

               [SweepData("W2H3a", 1.77, 2.20, 2.70, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3a_Y1.77.csv"),
                SweepData("W2H3a", 2.29, 2.20, 2.70, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3a_Y2.29.csv"),
                SweepData("W2H3a", 2.81, 2.20, 2.70, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3a_Y2.81.csv"),
                SweepData("W2H3a", 3.32, 2.20, 2.70, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3a_Y3.32.csv"),
                SweepData("W2H3a", 3.84, 2.20, 2.70, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3a_Y3.84.csv")],

               [SweepData("W2H3b", 2.66, 2.20, 2.90, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3b_Y2.66.csv"),
                SweepData("W2H3b", 3.68, 2.20, 2.90, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H3b_Y3.68.csv")],

               [SweepData("W2H6", 1.52, 2.20, 5.40, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H6_Y1.52.csv"),
                SweepData("W2H6", 1.99, 2.20, 5.40, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H6_Y1.99.csv")],

               [SweepData("W2H9", 1.66, 2.14, 8.21, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H9_Y1.66.csv"),
                SweepData("W2H9", 2.66, 2.14, 8.21, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H9_Y2.66.csv")],

               [SweepData("W2H12", 2.63, 2.20, 11.50, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W2H12_Y2.63.csv")],

               [SweepData("W4H12", 2.43, 4.20, 11.47, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W4H12_Y2.43.csv"),
                SweepData("W4H12", 3.43, 4.20, 11.47, "experiment_data/shifted_data/",
                          sweep_file="shifted_data_sweep_W4H12_Y3.43.csv")],
               ]

for j, i in itertools.product(range(num_rows), range(num_cols)):
    if num_cols * j + i >= num_plots:
        axes[j, i].set_visible(False)
    else:
        axes[j, i].tick_params(axis='both', which='major', labelsize=10)
        analyse_slot(axes[j, i], sweeps=sweep_lists[i + num_cols * j], set_y_label=(i == 0),
                     set_x_label=(j == num_rows - 1 or (j == num_rows - 2 and i >= num_plots % num_cols)),
                     use_defaults=False, config=config, num_series=1)
        axes[j, i].xaxis.label.set_size(10)
        axes[j, i].yaxis.label.set_size(10)
        axes[j, i].annotate(f"{labels[i + num_cols * j]}", xy=(0, 0), xytext=(0.05, 0.95),
                            textcoords='axes fraction',
                            horizontalalignment='left', verticalalignment='top')

plt.tight_layout()
plt.show()
