import matplotlib.pyplot as plt
from experimental.plotting.analyse_slot import analyse_slot
import common.util.plotting_utils as plt_util

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
this_ax = analyse_slot(this_ax, config=config, num_series=1, sweep_save_dir="../../../experimental/plotting/sweeps/",
                       prediction_file_dir="../../../numerical/models/model_outputs/exp_comparisons/",
                       data_dir="../../../../../Data/SlotSweeps")
# exp_line = mlines.Line2D([], [], color="black", linestyle=" ", marker=".", label='Experimental')
# pred_line = mlines.Line2D([], [], color="C1", label='Numerical', linewidth=1)
# this_ax.legend(handles=[exp_line, pred_line], frameon=False, loc='lower left')
# plt.tight_layout()
plt.show()
