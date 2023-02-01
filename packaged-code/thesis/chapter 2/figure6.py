import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from scipy import ndimage
from skimage.filters import threshold_otsu

import util.determineDisplacement as dd
from util.plotting_utils import initialize_plt
from util.analysis_utils import analyse_frame
from util.mp4 import MP4

initialize_plt()
fig, axes = plt.subplots(2, 3, figsize=(5, 2.8))
plt.subplots_adjust(0.025, 0.05, 0.975, 1, 0.075, 0.1)

dir_path = "../../Chapter 5/Data/w12vf40circles/Between 3 holes/movie_C001H001S0004/"
i = 0

movie = MP4(dir_path + "video_0.mp4")

frame_rate = movie.get_fps()
trigger_out_delay = movie.trigger_out_delay

frames = list(movie[:100])

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

plt.show()
