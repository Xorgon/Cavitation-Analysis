import os
from experimental.util.analysis_utils import load_readings

root_dirs = ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/",
             "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/"]

totals = []

for root_dir in root_dirs:
    total = 0

    for root, dirs, files in os.walk(root_dir):
        if "readings_dump.csv" in files:
            print(root)
            readings = load_readings(root + "/readings_dump.csv", include_invalid=True)
            total += len(readings)

    totals.append(total)

for d, t in zip(root_dirs, totals):
    print(d, t)

print(sum(totals))
