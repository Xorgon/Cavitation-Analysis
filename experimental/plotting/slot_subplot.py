import matplotlib.pyplot as plt
from experimental.plotting.analyse_slot import analyse_slot
import itertools
import common.util.plotting_utils as plt_util

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

# config["use_all_series"] = False
# config["colours"] = False
# config["plot_predicted"] = True

fig_width = 5.31445  # From LaTeX

left_padding = 0.7
right_padding = 0.08
top_padding = 0.08
bottom_padding = 0.5
v_spacing = 0.3
h_spacing = 0.1
ax_width = (fig_width - left_padding - right_padding - (num_cols - 1) * h_spacing) / num_cols

# fig_width = ax_width * num_cols + left_padding + right_padding + (num_cols - 1) * spacing
fig_height = ax_width * num_rows + bottom_padding + top_padding + (num_rows - 1) * v_spacing

fig, axes = plt.subplots(num_rows, num_cols, sharex='none', sharey="all", figsize=(fig_width, fig_height))
# labels = "abcdefghijk"
# labels = ["w1h3\nq=1.94", "w2h3a\nq=2.29", "w2h6\nq=1.52", "w4h12\nq=3.43"]
# labels = ["w1h3", "w2h3a", "w2h3b", "w2h6", "w2h9", "w2h12", "w4h12"]
labels = ["($a$)", "($b$)", "($c$)", "($d$)", "($e$)", "($f$)", "($g$)"]

for j, i in itertools.product(range(num_rows), range(num_cols)):
    if num_cols * j + i >= num_plots:
        axes[j, i].set_visible(False)
    else:
        axes[j, i].tick_params(axis='both', which='major', labelsize=10)
        analyse_slot(axes[j, i], set_y_label=(i == 0),
                     set_x_label=(j == num_rows - 1 or (j == num_rows - 2 and i >= num_plots % num_cols)),
                     use_defaults=False, config=config, num_series=1)
        axes[j, i].xaxis.label.set_size(10)
        axes[j, i].yaxis.label.set_size(10)
        # text_pos = ((left_padding + i * (h_spacing + ax_width) + 0.075) / fig_width,
        #             (bottom_padding + (num_rows - j) * (ax_width + v_spacing) - 0.1 - v_spacing) / fig_height)
        axes[j, i].annotate(f"{labels[i + num_cols * j]}", xy=(0, 0), xytext=(0.05, 0.95),
                            textcoords='axes fraction',
                            horizontalalignment='left', verticalalignment='top')

# plt.subplots_adjust(left=left_padding / fig_width,
#                     right=(fig_width - right_padding) / fig_width,
#                     top=(fig_height - top_padding) / fig_height,
#                     bottom=bottom_padding / fig_height,
#                     wspace=h_spacing / ax_width,
#                     hspace=v_spacing / ax_width)
plt.tight_layout()
plt.show()
