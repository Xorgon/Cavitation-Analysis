import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from skimage.measure import label, regionprops
from scipy import ndimage
from scipy.interpolate import interp1d
import sys
import importlib

import experimental.util.file_utils as file
from experimental.util.mraw import mraw
from experimental.util.analysis_utils import calculate_displacement, analyse_frame, load_readings
import experimental.util.determineDisplacement as dd
from common.util.plotting_utils import initialize_plt, fig_width

# initialize_plt()

params_dir = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/"
dir_path = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/50mm solid plate/movie_C001H001S0006/"
movie_idx = 6
rep = 1  # repeat number

# params_dir = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"
# dir_path = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/movie_C001H001S0006/"
# movie_idx = 6
# rep = 3

# params_dir = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/"
# dir_path = "C:/Users/Elijah/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Between 3 holes/movie_C001H001S0011/"
# movie_idx = 11
# rep = 0
# TODO: Fix this such that when one of the minimum gradients is lower than the inbetween gradient it doesn't find it

movie = file.get_mraw_from_dir(dir_path)

frame_rate = movie.get_fps()
trigger_out_delay = movie.trigger_out_delay

frames = list(movie[rep * 100: (rep + 1) * 100])
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

plt.figure(figsize=(fig_width(), fig_width() * 0.75))
top_ax = plt.subplot(311)
time_per_frame = 1 / frame_rate
times = np.array(idxs) * time_per_frame
plt.plot(times * 1e6, areas)
plt.xticks([])
plt.ylabel("Bubble area (px$^2$)")

plt.annotate("First maximum area", (times[peak_idxs[0]] * 1e6, areas[peak_idxs[0]]), ha='left', va='bottom')
plt.annotate("Second maximum area", (times[peak_idxs[1]] * 1e6, areas[peak_idxs[1]]), ha='left', va='bottom')
plt.scatter([times[peak_idxs[0]] * 1e6, times[peak_idxs[1]] * 1e6],
            [areas[peak_idxs[0]], areas[peak_idxs[1]]], color="C1")

sys.path.append(params_dir)
params = importlib.import_module("params")
importlib.reload(params)
sys.path.remove(params_dir)
readings = load_readings(params_dir + "readings_dump.csv", include_invalid=False)

gradients = np.abs((areas[1:] - areas[:-1]) / (times[1:] - times[:-1]))
grad_change_of_sign = np.sign(gradients[1:] - gradients[:-1])
abs_d_grad_cos = np.abs(grad_change_of_sign[1:] - grad_change_of_sign[:-1])

for i in range(len(abs_d_grad_cos)):
    if all(abs_d_grad_cos[i:i + 3] == 2):
        min_grad_idx = i + 2
        break

first_collapse_end_idx = min_grad_idx
scnd_collapse_start_idx = min_grad_idx + 1

coeffs1 = np.polyfit(times[:first_collapse_end_idx + 1], areas[:first_collapse_end_idx + 1], 2)
coeffs2 = np.polyfit(times[scnd_collapse_start_idx:-1], areas[scnd_collapse_start_idx:-1], 2)

ts1 = np.linspace(times[0], times[scnd_collapse_start_idx])
ts2 = np.linspace(times[first_collapse_end_idx], times[-1])

plt.plot(ts1 * 1e6, np.poly1d(coeffs1)(ts1))
plt.plot(ts2 * 1e6, np.poly1d(coeffs2)(ts2))

intersection = np.roots(coeffs1 - coeffs2)
plt.scatter(intersection[-1] * 1e6, np.poly1d(coeffs1)(intersection[-1]))
plt.scatter([times[first_collapse_end_idx] * 1e6, times[scnd_collapse_start_idx] * 1e6],
            [areas[first_collapse_end_idx], areas[scnd_collapse_start_idx]], color="C1")

plt.subplot(312, sharex=top_ax)
plt.plot(times[:-1] * 1e6, gradients / np.max(gradients))
plt.plot(times[:-2] * 1e6, grad_change_of_sign)
plt.plot(times[1:-2] * 1e6, abs_d_grad_cos)
plt.xticks([])

this_reading = None
for r in readings:
    if r.idx == movie_idx and r.repeat_number == rep:
        this_reading = r
        break

max_radius = np.sqrt(this_reading.max_bubble_area / np.pi) * params.mm_per_px
y_offset = params.upper_surface_y
y = this_reading.get_bubble_pos_mm(params.mm_per_px)[1] - y_offset
standoff = y / max_radius
anisotropy = 0.195 * standoff ** (-2)
supp_disp = 2.5 * anisotropy ** (3 / 5)

plt.subplot(313, sharex=top_ax)
disp_over_rad_max = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2) * params.mm_per_px / max_radius
plt.plot(times * 1e6, disp_over_rad_max)
plt.xlabel("$t$ $(\\mu s)$")
plt.ylabel("$\\Delta / R$")
plt.axhline(supp_disp, color="grey", linestyle="--")
plt.axvline(intersection[-1] * 1e6, color="grey", linestyle="--")
interp_disp = interp1d(times, disp_over_rad_max, "linear")(intersection[-1])
plt.axhline(interp_disp, color="red")
plt.text(plt.gca().get_xlim()[0] + 10, supp_disp, "Supponen \\textit{et al.} (2016) prediction", ha="left", va="bottom")
plt.tight_layout()

plt.show()
