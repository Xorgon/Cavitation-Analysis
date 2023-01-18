import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from skimage.measure import label, regionprops
from scipy import ndimage

import experimental.util.file_utils as file
from experimental.util.mraw import mraw
from experimental.util.analysis_utils import calculate_displacement, analyse_frame
import experimental.util.determineDisplacement as dd
from common.util.plotting_utils import initialize_plt, fig_width

initialize_plt()
fig, axes = plt.subplots(2, 3, figsize=(5, 2.8))
plt.subplots_adjust(0.025, 0.05, 0.975, 1, 0.075, 0.1)

# dir_path = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/movie_C001H001S0006/"
# i = 1  # repeat number

# dir_path = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/movie_C001H001S0011/"
# i = 0

# dir_path = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 2.0/movie_C001H001S0001/"
# i = 0

dir_path = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/w12vf40circles/Between 3 holes/movie_C001H001S0004/"
i = 0

movie = file.get_mraw_from_dir(dir_path)

frame_rate = movie.get_fps()
trigger_out_delay = movie.trigger_out_delay

frames = list(movie[i * 100: (i + 1) * 100])
# disp_out = calculate_displacement(frames, frame_rate, trigger_out_delay, save_path=None, repeat_num=i)
# disp, pos, area, sec_area, imf, sup_disp = disp_out

laser_delay = 151.0e-6
total_laser_delay = laser_delay + trigger_out_delay  # determine total delay in firing the laser
laser_frame_idx = int(round(frame_rate * total_laser_delay))  # delay in frames
first_frame_idx = laser_frame_idx + 1  # first frame we will analyse

bg_frame = np.int32(frames[0])

xs = []
ys = []
areas = []
idxs = []
for idx in range(first_frame_idx, len(frames)):
    x, y, area = analyse_frame(frames[idx], bg_frame)
    if area is None:
        continue  # No bubble found.
    xs.append(x)
    ys.append(y)
    areas.append(area)
    idxs.append(idx)

xs = np.array(xs)
ys = np.array(ys)
areas = np.array(areas)
idxs = np.array(idxs)

peak_idxs, peak_areas = dd.findPeaks(idxs, areas, kernel=5)
frame_idx = 29  # peak_idxs[1]
I = frames[frame_idx]
axes[0, 0].imshow(I, cmap=plt.cm.gray)
axes[0, 0].set_xlabel("($a$) Original frame")
axes[0, 1].imshow(bg_frame - np.int32(I), cmap=plt.cm.gray)
axes[0, 1].set_xlabel("($b$) Subtracted background")
# find a reasonable threshold
thresh = threshold_otsu(bg_frame - np.int32(I))
# make a binary version of the image
binary = bg_frame - np.int32(I) > thresh
axes[0, 2].imshow(binary, cmap=plt.cm.gray)
axes[0, 2].set_xlabel("($c$) Threshold applied")
# try to separate small objects that are attached to the main bubble
morph.binary_opening(binary, footprint=None, out=binary)
axes[1, 0].imshow(binary, cmap=plt.cm.gray)
axes[1, 0].set_xlabel("($d$) Binary opening")
# fill up holes to make the bright spot in the center of the bubble black
ndimage.binary_fill_holes(binary, output=binary)
axes[1, 1].imshow(binary, cmap=plt.cm.gray)
axes[1, 1].set_xlabel("($e$) Holes filled")
# remove small objects (noise and object that we separated a few steps ago)
binary = morph.remove_small_objects(binary, min_size=256)
axes[1, 2].imshow(binary, cmap=plt.cm.gray)
axes[1, 2].set_xlabel("($f$) Small objects removed")

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.figure(figsize=(fig_width(), fig_width() * 0.75))
time_per_frame = 1 / frame_rate
times = np.array(idxs) * time_per_frame
plt.plot(times * 1e6, areas)
plt.xlabel("$t$ $(\\mu s)$")
plt.ylabel("Bubble area (square pixels)")

plt.annotate("First maximum area", (times[peak_idxs[0]] * 1e6, areas[peak_idxs[0]]), ha='left', va='bottom')
plt.annotate("Second maximum area", (times[peak_idxs[1]] * 1e6, areas[peak_idxs[1]]), ha='left', va='bottom')
plt.scatter([times[peak_idxs[0]] * 1e6, times[peak_idxs[1]] * 1e6],
            [areas[peak_idxs[0]], areas[peak_idxs[1]]], color="C1")

plt.tight_layout()
plt.show()
