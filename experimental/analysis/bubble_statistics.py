import os
import sys
import numpy as np
import importlib
import matplotlib.pyplot as plt

from experimental.util.analysis_utils import load_readings
from common.util.plotting_utils import initialize_plt

root_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/",
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"]

labels = ["Slots (MO)", "Porous plates (OAPM)"]

totals = []

initialize_plt(font_size=10, dpi=300)
fig, ((r_ax, r_off_ax), (disp_ax, ecc_ax)) = plt.subplots(2, 2, figsize=(5.5, 4.2))

for n, root_dir in enumerate(root_dirs):
    print(root_dir)
    total = 0
    all_radii = []
    all_radii_offsets = []
    all_eccs = []

    all_mm_per_px = []
    all_disp_stds = []
    all_radii_stds = []

    all_disp_offsets = []
    all_grpd_radii_offsets = []

    for root, dirs, files in os.walk(root_dir):
        if "readings_dump.csv" in files:
            readings = load_readings(root + "/readings_dump.csv", include_invalid=False)

            sys.path.append(root)
            import params

            importlib.reload(params)
            sys.path.remove(root)

            all_mm_per_px.append(params.mm_per_px)

            radii = np.array([r.get_max_radius(params.mm_per_px) for r in readings])
            all_radii.extend(radii)

            radii_offsets = radii - np.mean(radii)
            all_radii_offsets.extend(radii_offsets)

            eccs = [float(r.ecc_at_max) for r in readings]
            all_eccs.extend(eccs)

            total += len(readings)

            max_idx = np.max([r.idx for r in readings])
            sorted_readings = []
            for i in range(max_idx + 1):
                sorted_readings.append([])

            for r in readings:
                sorted_readings[r.idx].append(r)

            for i, group in enumerate(sorted_readings):
                if len(group) == 0:
                    continue
                group_disps = np.array([np.linalg.norm(r.get_disp_mm(params.mm_per_px)) for r in group])
                all_disp_offsets.extend(group_disps - np.mean(group_disps))
                all_disp_stds.append(np.std(group_disps))

                group_radii = np.array([r.get_max_radius(params.mm_per_px) for r in group])
                all_grpd_radii_offsets.extend(group_radii - np.mean(group_radii))

    totals.append(total)
    print("Mean radius = ", np.mean(all_radii), "Std = ", np.std(all_radii_offsets))
    print("Grouped displacement std = ", np.std(all_disp_offsets))
    print("Grouped radius std = ", np.std(all_grpd_radii_offsets))
    print("Mean eccentricity = ", np.mean(all_eccs))
    print("Mean mm/px = ", np.mean(all_mm_per_px))

    # r_ax.hist(all_radii_offsets, bins=100, label=labels[n], density=True,
    #           alpha=0.5, histtype='stepfilled', edgecolor=f"C{n}")

    r_ax.hist(all_radii, bins=100, label=labels[n], density=True,
              alpha=0.5, histtype='stepfilled', edgecolor=f"C{n}")
    r_off_ax.hist(all_grpd_radii_offsets, bins=100, label=labels[n], density=True,
                  alpha=0.5, histtype='stepfilled', edgecolor=f"C{n}")
    disp_ax.hist(all_disp_offsets, bins=100, label=labels[n], density=True,
                 alpha=0.5, histtype='stepfilled', edgecolor=f"C{n}")
    ecc_ax.hist(all_eccs, bins=100, label=labels[n], density=True,
                alpha=0.5, histtype='stepfilled', edgecolor=f"C{n}")

for d, t in zip(root_dirs, totals):
    print(d, t)

print(sum(totals))
r_ax.legend(frameon=False, loc="upper left", fontsize="x-small")

r_ax.set_ylabel("Probability density")
r_ax.set_xlabel("Radius (mm)")

# r_off_ax.set_ylabel("Probability density")
r_off_ax.set_xlabel("Radius offset from mean (mm)")

disp_ax.set_ylabel("Probability density")
disp_ax.set_xlabel("Displacement offset from mean (mm)")

# ecc_ax.set_ylabel("Probability density")
ecc_ax.set_xlabel("Eccentricity")

plt.tight_layout()

plt.savefig("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
            "paper figures/supp_mat/bubble_statistics.pdf")
plt.show()
