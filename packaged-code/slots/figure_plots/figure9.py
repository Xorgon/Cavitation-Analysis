import matplotlib.pyplot as plt
from util.analyse_slot import analyse_slot, SweepData
import util.plotting_utils as plt_util

font_size = 10
plt_util.initialize_plt(font_size=font_size, capsize=2, line_scale=1)

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
    "use_all_series": False,
    "use_all_dirs": False,
    "normalize": False,
    "plot_fits": True,
    "skip_bad_data": False,
    "plot_means": False,
    "labelled": False,
    "label_tags": False,
    "colours": False,
    "error_bars": False,
    "do_shift": False,
    "verbose": False,
    "plot_predicted": False
}

sweep = SweepData("W2H3a", 2.81, 2.2, 2.7)
sweep.add_points_from_csv("experiment_data/raw_data/", "raw_data_sweep_W2H3a_Y2.81.csv")

this_ax = analyse_slot(this_ax, sweeps=[sweep], config=config, num_series=1)
plt.show()
