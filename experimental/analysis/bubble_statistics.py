import os
import sys
import numpy as np
import importlib

from experimental.util.analysis_utils import load_readings

root_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/",
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/"]

totals = []

for root_dir in root_dirs:
    print(root_dir)
    total = 0
    all_radii = []
    all_radii_offsets = []
    all_eccs = []

    for root, dirs, files in os.walk(root_dir):
        if "readings_dump.csv" in files:
            readings = load_readings(root + "/readings_dump.csv", include_invalid=True)

            sys.path.append(root)
            import params

            importlib.reload(params)
            sys.path.remove(root)

            radii = np.array([r.get_max_radius(params.mm_per_px) for r in readings])
            all_radii.extend(radii)

            radii_offsets = radii - np.mean(radii)
            all_radii_offsets.extend(radii_offsets)

            eccs = [r.ecc_at_max for r in readings]
            all_eccs.extend(eccs)

            total += len(readings)

    totals.append(total)
    print("Mean radius = ", np.mean(all_radii), "Std = ", np.std(all_radii_offsets))
    print("Mean eccentricity = ", np.mean(all_eccs))

for d, t in zip(root_dirs, totals):
    print(d, t)

print(sum(totals))
