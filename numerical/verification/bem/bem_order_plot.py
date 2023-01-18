import numpy as np
import matplotlib.pyplot as plt
from util.file_utils import csv_to_lists
from util.plotting_utils import initialize_plt, fig_width

# filenames = ["uniform_corner_rms_50.csv",
#              "uniform_corner_rms_100.csv",
#              "uniform_corner_rms_150.csv",
#              "uniform_corner_rms_200.csv",
#              "uniform_corner_rms_250.csv"]

filenames = ["uniform_corner_rms_between_panels_10.csv",
             "uniform_corner_rms_between_panels_20.csv",
             "uniform_corner_rms_between_panels_30.csv",
             "uniform_corner_rms_between_panels_40.csv",
             "uniform_corner_rms_between_panels_50.csv",
             "uniform_corner_rms_between_panels_100.csv",
             "uniform_corner_rms_between_panels_150.csv",
             "uniform_corner_rms_between_panels_200.csv",
             "uniform_corner_rms_between_panels_250.csv"
             ]

# filenames = ["uniform_corner_rms_bp_limited_10.csv",
#              "uniform_corner_rms_bp_limited_30.csv",
#              "uniform_corner_rms_bp_limited_50.csv",
#              "uniform_corner_rms_bp_limited_100.csv",
#              "uniform_corner_rms_bp_limited_150.csv",
#              "uniform_corner_rms_bp_limited_250.csv"]

initialize_plt()
plt.figure(figsize=(fig_width(), fig_width() * 0.75))

for f in filenames:
    ns, areas, rmsds = csv_to_lists("", f)

    rmsds = np.sqrt(np.power(rmsds, 2) * (np.pi / 2 - 0.2) / ((np.pi / 2) / 2))  # Correcting for slight mistake in bem_order_corner:116

    a, b = np.polyfit(np.log(1 / np.array(areas)), np.log(rmsds), 1)
    L = int(f.replace('uniform_corner_rms_between_panels_', '')[:-4])
    plt.loglog(np.sqrt(areas), rmsds, "o", label=f"$L = {L}$")
    # plt.loglog(areas, np.exp(b) * np.power(np.divide(1, areas), a),
    #            label=f"RMSD = ${b:.2f}N^{{{a:.2f}}}$ ({f.replace('_', ' ')[:-4]})")

plt.xlabel("Panel length")
plt.ylabel("Root Mean Squared Difference")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.show()

# initialize_plt()
plt.figure(figsize=(fig_width(), fig_width() * 0.75))

for f in filenames:
    ns, areas, rmsds = csv_to_lists("", f)
    plt.scatter(np.sqrt(areas), rmsds, label=f"{f.replace('_', ' ')[:-4]}",
                c=np.mod(5, np.sqrt(areas)) / np.sqrt(areas))
    break
plt.colorbar()

plt.xlabel("Panel length")
plt.ylabel("Root Mean Squared Difference")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(fig_width(), fig_width() * 0.75))

for f in filenames:
    ns, areas, rmsds = csv_to_lists("", f)
    plt.scatter(ns, np.log(rmsds), label=f"{f.replace('_', ' ')[:-4]}", c=np.mod(5, np.sqrt(areas)) / np.sqrt(areas))
    break
plt.colorbar()

plt.xlabel("N")
plt.ylabel("Root Mean Squared Difference")
plt.legend()
plt.tight_layout()

plt.show()
