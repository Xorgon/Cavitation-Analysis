import os
import numpy as np
import matplotlib.pyplot as plt

from experimental.util.analysis_utils import analyse_frame
import experimental.util.file_utils as file
from experimental.util.determineDisplacement import findPeaks

root_path = file.select_dir()
if root_path == "/":
    exit()

indexed_dirs = []
for root, dirs, files in os.walk(root_path):
    if "index.csv" in files:
        indexed_dirs.append(root + "/")

laser_delay = 151.0e-6

start_frames = []
frame_arrs = []
area_arrs = []
for dir_path in indexed_dirs:
    index_file = open(dir_path + "index.csv")
    index_lines = index_file.readlines()
    index_file.close()

    input_data = []  # x, y, index

    for i in range(1, len(index_lines)):  # Start at 1 to skip header.
        split = index_lines[i].strip().split(",")
        input_data.append([float(split[0]), float(split[1]), split[2]])  # x, y, idx

    reading_prefix = file.get_prefix_from_idxs(dir_path, np.array(input_data)[:, 2])
    for i in range(5):
        print(f"{input_data[i][2]}")
        reading_path = dir_path + reading_prefix + str(input_data[i][2]).rjust(4, "0") + "/"

        movie = file.get_mraw_from_dir(reading_path)
        if movie.image_count % 100 != 0:
            print(
                "Warning: {0} does not have a multiple of 100 frames.".format(dir_path))

        frame_rate = movie.get_fps()
        trigger_out_delay = movie.trigger_out_delay
        total_laser_delay = laser_delay + movie.trigger_out_delay  # determine total delay in firing the laser
        laser_frame_idx = int(round(frame_rate * total_laser_delay))  # delay in frames
        first_frame_idx = laser_frame_idx + 1  # first frame we will analyse

        repeats = int(movie.image_count / 100)
        for i in range(repeats):
            frames = list(movie[i * 100: (i + 1) * 100])
            bg_frame = np.int32(frames[0])
            positions = []
            areas = []
            frame_idxs = []

            minima_found = 0
            for idx in range(first_frame_idx, len(frames)):
                if minima_found >= 2:
                    continue
                x, y, area = analyse_frame(frames[idx], frames[0])
                if area is None:
                    continue
                if len(areas) >= 2 and area > areas[-1] and areas[-2] > areas[-1]:
                    minima_found += 1
                positions.append([x, y])
                areas.append(area)
                frame_idxs.append(idx)
            if len(areas) > 0:
                start_frames.append(laser_frame_idx)
                frame_arrs.append(frame_idxs)
                area_arrs.append(areas)
        movie.close()

plt.figure()
for frames, areas in zip(frame_arrs, area_arrs):
    areas = np.array(areas)
    radii = np.power(areas / np.pi, 1 / 2)

    peak_idxs, radii_peaks = findPeaks(frames, radii)

    if len(peak_idxs) < 2:
        continue

    frames = np.array(frames) - frames[peak_idxs[0]]
    times = frames / 1e5
    times = times / times[peak_idxs[1]]
    radii = radii / radii_peaks[0]
    plt.plot(times, radii)
plt.xlabel("$(t - t_{max}) / t_{peak2}$")
plt.ylabel("$R / R_{max}$")
plt.show()
