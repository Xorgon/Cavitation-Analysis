"""
This code is based on code by Ivo Peters. The original code can be found in util/determineDisplacement.

TODO:
- Improve error detection.
    -> Void fraction? (of both frames)
    -> Frames too far along (>50?)
- Improve handling of sideways measurements (separate analysis from processing i.e. just x, y, z, idx, vectors etc.)
"""
import os
from typing import List
import sys
import importlib

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

import util.determineDisplacement as dd
import util.file_utils as file
from util.mp4 import MP4


class Reading:
    m_x, m_y, m_z = None, None, None  # Measurement coordinates
    idx = None
    repeat_number = None

    disp_vect = None  # Bubble displacement vector (pixels)
    sup_disp_vect = None  # Displacement vector between minima as in Supponen et al. (2016)
    bubble_pos = None  # Bubble position in the frame (px coords)
    max_bubble_area = None  # Maximum bubble area (pixels)
    sec_max_area = None  # Second maximum of bubble area (pixels)
    inter_max_frames = None  # Time between
    ecc_at_max = None  # Eccentricity at maximum area
    model_anisotropy = None  # Anisotropy vector generated from BEM model

    def __init__(self, idx, repeat_number, m_x=None, m_y=None, m_z=None):
        self.idx = idx
        self.repeat_number = repeat_number
        self.m_x = m_x
        self.m_y = m_y
        self.m_z = m_z

    def __str__(self):
        string = f"{self.idx}:{self.repeat_number},{self.m_x},{self.m_y},{self.m_z}," \
                 f"{self.disp_vect[0]},{self.disp_vect[1]}," \
                 f"{self.bubble_pos[0]},{self.bubble_pos[1]}," \
                 f"{self.max_bubble_area},{self.sec_max_area},{self.inter_max_frames}," \
                 f"{self.sup_disp_vect[0]},{self.sup_disp_vect[1]}," \
                 f"{self.ecc_at_max}"
        if self.model_anisotropy is not None:  # Only include anisotropy if available
            string += f",{self.model_anisotropy[0]},{self.model_anisotropy[1]},{self.model_anisotropy[2]}"
        return string

    def get_bubble_pos_mm(self, mm_per_px, frame_height=264):
        return np.array([self.bubble_pos[0] * mm_per_px + self.m_x,
                         (frame_height - self.bubble_pos[1]) * mm_per_px + self.m_y])

    def is_complete(self):
        return self.disp_vect and self.bubble_pos and self.max_bubble_area \
               and self.sec_max_area and self.inter_max_frames

    def get_jet_angle(self):
        return np.arctan2(-self.disp_vect[1], self.disp_vect[0]) + np.pi / 2

    def get_angle_dif(self):
        """ Computes the angle difference between the experimental measurement and the predicted anisotropy. """
        if self.model_anisotropy is not None:
            # mm_per_px scaling doesn't matter here, just using this function to get the right direction conventions
            return np.arccos(np.dot(self.get_disp_mm(1), self.model_anisotropy) /
                             (np.linalg.norm(self.get_disp_mm(1)) * np.linalg.norm(self.model_anisotropy)))

    def get_normalised_displacement(self):
        return self.disp_vect / np.sqrt(self.max_bubble_area / np.pi)

    def get_disp_mm(self, mm_per_px):
        return np.array([self.disp_vect[0] * mm_per_px, - self.disp_vect[1] * mm_per_px, 0])

    def get_radius_ratio(self):
        return np.sqrt(self.sec_max_area / self.max_bubble_area)

    def get_max_radius(self, mm_per_px):
        return mm_per_px * np.sqrt(self.max_bubble_area / np.pi)

    def get_scalar_anisotropy(self):
        if self.model_anisotropy is not None:
            return np.linalg.norm(self.model_anisotropy)

    @staticmethod
    def from_str(string: str):
        """ Deserializes and returns a Reading object. """
        string = string.strip()
        split = string.split(",")
        if split[1] == "None":
            x = None
        else:
            x = float(split[1])
        if split[2] == "None":
            y = None
        else:
            y = float(split[2])
        if split[3] == "None":
            z = None
        else:
            z = float(split[3])
        reading = Reading(int(split[0].split(":")[0]), int(split[0].split(":")[1]), x, y, z)
        reading.disp_vect = np.array([float(split[4]), float(split[5])])
        reading.bubble_pos = np.array([float(split[6]), float(split[7])])
        reading.max_bubble_area = np.array(int(split[8]))
        if len(split) > 9:  # Only include newer metrics if available
            reading.sec_max_area = np.array(int(split[9]))
            reading.inter_max_frames = np.array(int(split[10]))
        if len(split) > 11:  # Only include even newer metrics if available
            reading.sup_disp_vect = np.array([float(split[11]), float(split[12])])
        if len(split) > 13:  # Only include even even newer metric (eccentricity) if available
            reading.ecc_at_max = np.array(float(split[13]))
        if len(split) > 14:  # Only include blah blah blah (anisotropy vector)
            reading.model_anisotropy = np.array([float(split[14]), float(split[15]), float(split[16])])
        return reading


def load_readings(filename, include_invalid=False, do_flag_invalid=True):
    if do_flag_invalid:
        flag_invalid_readings(filename.rpartition("/")[0] + "/")
    dump_file = open(filename)
    lines = dump_file.readlines()

    readings = []

    for line in lines[1:]:
        if line[0] == "#" and not include_invalid:
            continue
        if line[0] == "#":
            line = line[1:]
        readings.append(Reading.from_str(line))

    dump_file.close()

    return readings


def save_readings(dir_path, readings):
    if os.path.exists(dir_path + "readings_dump.csv"):
        bkp_num = 0
        while os.path.exists(dir_path + f"readings_dump.csv.bkp.{bkp_num}"):
            bkp_num += 1
        os.rename(dir_path + "readings_dump.csv", dir_path + f"readings_dump.csv.bkp.{bkp_num}")

    dump_file = open(dir_path + "readings_dump.csv", "a")
    dump_file.write("index:repeat number, measured x (mm), measured y (mm), measured z (mm), "
                    "peak-to-peak x displacement (px), peak-to-peak y displacement (px), "
                    "in-frame bubble position x (px), in-frame bubble position y (px), "
                    "maximum bubble area (px^2), second maximum of bubble area (px^2), frames between maxima, "
                    "minimum-to-minimum x displacement (px), minimum-to-minimum y displacement (px),"
                    "eccentricity at maximum size,"
                    "model anisotropy x, model anisotropy y, model anisotropy z\n")

    for reading in readings:
        dump_file.write(str(reading) + "\n")

    dump_file.close()


def plot_analysis(areas, xs, ys, dx, dy, ns, path, frames, frame_rate, repeat_number=0, show_plot=False):
    areas = np.array(areas)
    xs = np.array(xs)
    ys = np.array(ys)
    ns = np.array(ns)

    # calculate time (s)
    t = ns / frame_rate

    # close the previous plot
    plt.close()
    # find and plot peaks
    plt.figure()
    plt.subplot(2, 3, 3)
    plt.plot(1000 * t, areas)
    plt.scatter(1000 * t, areas, c="red", s=1)
    plt.xlabel('t (ms)')
    plt.ylabel('area (sq. pix.)')
    peak_idxs, yPeak = dd.findPeaks(ns, areas, kernel=4)
    idx = ns[peak_idxs]
    plt.plot(1000 * t[peak_idxs], yPeak, 'o')

    # plot raw images
    if len(idx) > 0:
        plt.subplot(2, 3, 1)
        plt.imshow(frames[idx[0]], cmap=plt.cm.gray)
        plt.plot(xs[peak_idxs[0]], ys[peak_idxs[0]], 'x')
        plt.xticks([])
        plt.yticks([])
        plt.title("Frame {0}".format(idx[0]))
    if len(idx) > 1:
        plt.subplot(2, 3, 4)
        plt.imshow(frames[idx[1]], cmap=plt.cm.gray)
        plt.plot(xs[peak_idxs[1]], ys[peak_idxs[1]], 'x')
        plt.xticks([])
        plt.yticks([])
        plt.title("Frame {0}".format(idx[1]))

    # plot binary processed bubble images
    if len(idx) > 0:
        plt.subplot(2, 3, 2)
        binary1 = dd.makeBinary(frames[0] - np.int32(frames[idx[0]]))
        plt.imshow(binary1, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
    if len(idx) > 1:
        plt.subplot(2, 3, 5)
        binary2 = dd.makeBinary(frames[0] - np.int32(frames[idx[1]]))
        plt.imshow(binary2, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

    # calculate and plot displacement on background image
    if len(idx) > 1:
        plt.subplot(2, 3, 6)
        plt.imshow(frames[0], cmap=plt.cm.gray)
        plt.plot(xs[peak_idxs[0]:peak_idxs[1] + 1], ys[peak_idxs[0]:peak_idxs[1] + 1], 'x')

        plt.quiver(xs[peak_idxs[0]], ys[peak_idxs[0]], dx, -dy,
                   scale_units='x',
                   scale=0.33,
                   width=0.02)

    plt.tight_layout()
    if show_plot:
        plt.show()

    # save figure for checking proper working of algorithms
    plt.savefig(path + "analysis_plot_r{0}.png".format(repeat_number), dpi=150)


def analyse_frame(frame, bg_frame, max_ecc=0.6, min_sol=0.9, debug=False):
    """
    Analyses a single frame to find the coordinates and area of the bubble region. Additional parameters returned if
    debug is True.

    :param frame: Image frame.
    :param bg_frame: Background image frame.
    :return: x, y, area, [eccentricity, solidity]
    """
    x, y, area, ecc, sol, jt = None, None, None, None, None, None

    img = np.int32(bg_frame) - np.int32(frame)
    binary = dd.makeBinary(img)
    label_img = label(binary)
    props = regionprops(label_img, cache=False)

    if len(props) > 0:
        if len(props) > 1:
            max_area = 0.0
            for j in range(len(props)):
                if props[j].area > max_area:
                    max_area = props[j].area
                    max_idx = j
            y, x = props[max_idx].centroid
            area = props[max_idx].area
            ecc = props[max_idx].eccentricity
            sol = props[max_idx].solidity

            coords = np.array(props[max_idx].coords)
            dists = np.linalg.norm(np.subtract(coords, [y, x]), axis=1)
            max_dist_idx = np.argmax(dists)
            jt = coords[max_dist_idx]  # np.arctan2(coords[max_dist_idx][1] - y, coords[max_dist_idx][0] - x)

            if props[max_idx].eccentricity > max_ecc or props[max_idx].solidity < min_sol:
                area = None
        else:
            y, x = props[0].centroid
            area = props[0].area
            ecc = props[0].eccentricity
            sol = props[0].solidity

            coords = np.array(props[0].coords)
            dists = np.linalg.norm(np.subtract(coords, [y, x]), axis=1)
            max_dist_idx = np.argmax(dists)
            jt = coords[max_dist_idx]  # np.arctan2(coords[max_dist_idx][1] - y, coords[max_dist_idx][0] - x)

            if props[0].eccentricity > max_ecc or props[0].solidity < min_sol:
                area = None

    if not debug:
        return x, y, area
    else:
        if jt is not None:
            jet_tip = (jt[1], jt[0])
        else:
            jet_tip = None
        return x, y, area, ecc, sol, jet_tip


def estimate_supponen_disp(areas, idxs, frame_rate, xs, ys):
    time_per_frame = 1 / frame_rate
    times = np.array(idxs) * time_per_frame

    gradients = np.abs((areas[1:] - areas[:-1]) / (times[1:] - times[:-1]))
    grad_change_of_sign = np.sign(gradients[1:] - gradients[:-1])
    abs_d_grad_cos = np.abs(grad_change_of_sign[1:] - grad_change_of_sign[:-1])

    min_grad_idx = None
    for i in range(len(abs_d_grad_cos)):
        if all(abs_d_grad_cos[i:i + 3] == 2):
            min_grad_idx = i + 2
            break

    if min_grad_idx is None:  # TODO: Properly analyse these failures
        return 0, 0

    first_collapse_end_idx = min_grad_idx
    scnd_collapse_start_idx = min_grad_idx + 1

    try:
        coeffs1 = np.polyfit(times[:first_collapse_end_idx + 1], areas[:first_collapse_end_idx + 1], 2)
        coeffs2 = np.polyfit(times[scnd_collapse_start_idx:-1], areas[scnd_collapse_start_idx:-1], 2)
        intersection = np.roots(coeffs1 - coeffs2)
        interp_x = interp1d(times, xs - xs[0], "linear")(intersection[-1])
        interp_y = interp1d(times, ys - ys[0], "linear")(intersection[-1])
    except:
        return 0, 0

    return interp_x, interp_y


def calculate_displacement(frames, frame_rate, trigger_out_delay, save_path=None, repeat_num=0, laser_delay=100e-6,
                           warn=False):
    """
    Calculate bubble displacement vector for a series of frames.
    :param frames: Frames
    :param frame_rate: Frame rate (fps)
    :param trigger_out_delay: Camera trigger out delay
    :param save_path: Path in which to save the analysis plot, if None does not save.
    :param repeat_num: Number of the repeat, used to save the analysis plot for movies with multiple repeats.
    :param laser_delay: Laser delay
    :return: displacement vector [x, y], initial bubble centroid [x, y], max area, second max area, inter-max frames,
        Supponen displacement vector [x, y]
    """
    xs = []
    ys = []
    areas = []
    eccs = []
    idxs = []

    bg_frame = np.int32(frames[0])

    total_laser_delay = laser_delay + trigger_out_delay  # determine total delay in firing the laser
    laser_frame_idx = int(round(frame_rate * total_laser_delay))  # delay in frames
    first_frame_idx = laser_frame_idx + 1  # first frame we will analyse

    for idx in range(first_frame_idx, len(frames)):
        x, y, area, ecc, sol, _ = analyse_frame(frames[idx], bg_frame, debug=True)
        if area is None:
            continue  # No bubble found.
        xs.append(x)
        ys.append(y)
        areas.append(area)
        idxs.append(idx)
        eccs.append(ecc)

    xs = np.array(xs)
    ys = np.array(ys)
    areas = np.array(areas)
    idxs = np.array(idxs)

    peak_idxs, peak_areas = dd.findPeaks(idxs, areas, kernel=4)

    if len(peak_idxs) > 1:
        dx = xs[peak_idxs[1]] - xs[peak_idxs[0]]
        dy = ys[peak_idxs[1]] - ys[peak_idxs[0]]

        sup_dx, sup_dy = estimate_supponen_disp(areas, idxs, frame_rate, xs, ys)

        if save_path is not None:
            plot_analysis(areas, xs, ys, dx, dy, idxs, save_path, frames, frame_rate, repeat_number=repeat_num)

        return [dx, dy], [xs[peak_idxs[0]], ys[peak_idxs[0]]], areas[peak_idxs[0]], areas[peak_idxs[1]], \
               peak_idxs[1] - peak_idxs[0], [sup_dx, sup_dy], eccs[peak_idxs[0]]
    else:
        if save_path is not None:
            plot_analysis(areas, xs, ys, 0, 0, idxs, save_path, frames, frame_rate, repeat_number=repeat_num)
        if warn:
            print("Warning: Less than two area peaks found.")
        return None


def analyse_reading(dir_path, return_mean=False, warn=False, short_status=False):
    """
    Analyses a reading in specified file path.

    :param dir_path: Reading directory path.
    :param return_mean: Whether to return the mean displacement or a list of displacements.
    :return: mean displacement vector [x, y], mean bubble position vector [x, y]
    """
    if not os.path.exists(dir_path):
        if warn:
            print("Warning: " + dir_path + " does not exist.")
        return

    if not short_status:
        print("Analysing " + dir_path)
    movie = file.get_mraw_from_dir(dir_path)
    if movie.image_count % 100 != 0:
        if warn:
            print("Warning: {0} does not have a multiple of 100 frames. Assuming single collapse.".format(dir_path))
        repeats = 1
    else:
        repeats = int(movie.image_count / 100)

    frame_rate = movie.get_fps()
    trigger_out_delay = movie.trigger_out_delay

    disps = []
    positions = []
    areas = []
    sec_areas = []
    inter_max_frames = []
    sup_disps = []
    eccs = []
    for i in range(repeats):
        frames = list(movie[i * 100: (i + 1) * 100])
        disp_out = calculate_displacement(frames, frame_rate, trigger_out_delay, save_path=dir_path, repeat_num=i,
                                          warn=warn)

        if disp_out is not None:
            disp, pos, area, sec_area, imf, sup_disp, ecc_at_max = disp_out
            disps.append(disp)
            positions.append(pos)
            areas.append(area)
            sec_areas.append(sec_area)
            inter_max_frames.append(imf)
            sup_disps.append(sup_disp)
            eccs.append(ecc_at_max)
        else:
            disps.append(None)
            positions.append(None)
            areas.append(None)
            sec_areas.append(None)
            inter_max_frames.append(None)
            sup_disps.append(None)
            eccs.append(None)

    movie.close()

    if short_status:
        print("|", end="", flush=True)  # TODO: Make this work

    if return_mean:
        mean_disp = np.mean([d for d in disps if d is not None], axis=1)
        mean_pos = np.mean([p for p in positions if p is not None], axis=1)
        return mean_disp, mean_pos
    else:
        return disps, positions, areas, sec_areas, inter_max_frames, sup_disps, eccs


def analyse_series(dir_path, frame_shape=(384, 264), warn=False):
    index_file = open(dir_path + "index.csv")
    index_lines = index_file.readlines()
    index_file.close()

    if os.path.exists(dir_path + "readings_dump.csv"):
        bkp_num = 0
        while os.path.exists(dir_path + f"readings_dump.csv.bkp.{bkp_num}"):
            bkp_num += 1
        os.rename(dir_path + "readings_dump.csv", dir_path + f"readings_dump.csv.bkp.{bkp_num}")

    for root, _, files in os.walk(dir_path):
        for name in files:
            if "analysis_plot" in name and ".png" in name:
                os.remove(os.path.join(root, name))

    dump_file = open(dir_path + "readings_dump.csv", "a")
    dump_file.write("index:repeat number, measured x (mm), measured y (mm), measured z (mm), "
                    "peak-to-peak x displacement (px), peak-to-peak y displacement (px), "
                    "in-frame bubble position x (px), in-frame bubble position y (px), "
                    "maximum bubble area (px^2), second maximum of bubble area (px^2), frames between maxima, "
                    "minimum-to-minimum x displacement (px), minimum-to-minimum y displacement (px),"
                    "eccentricity at maximum size\n")

    sideways = False
    # Identify system:
    headers = index_lines[0].strip().split(",")
    for i, h in enumerate(headers):
        headers[i] = h.strip()
    if headers[0][0] == "x" and headers[1][0] == "y" and headers[2] == "idx":
        sideways = False
        frame_height = frame_shape[1]
    elif headers[0][0] == "x" and headers[1][0] == "z" and headers[2] == "idx":
        sideways = True
        frame_height = frame_shape[0]
    else:
        raise ValueError("Index file format was not recognized.")

    # At code end this is formatted:
    # x, y, index, displacement vector, bubble position vector (px), bubble position vector (mm)
    # readings = []  # type: List[List[Union[float, float, str, np.ndarray, np.ndarray, np.ndarray]]]
    readings = []  # type: List[Reading]

    input_data = []  # x, y, index

    for i in range(1, len(index_lines)):  # Start at 1 to skip header.
        split = index_lines[i].strip().split(",")
        if sideways:
            # In sideways configuration the actual x is the measured -z, actual y is measured x.
            input_data.append([-float(split[1]), float(split[0]), split[2]])  # x, y, idx converted from x, z, idx
            pass
        else:
            input_data.append([float(split[0]), float(split[1]), split[2]])  # x, y, idx

    reading_prefix = file.get_prefix_from_idxs(dir_path, np.array(input_data)[:, 2])
    for i in range(len(input_data)):
        reading_path = dir_path + reading_prefix + str(input_data[i][2]).rjust(4, "0") + "/"
        to_write = 0

        disps, positions, areas, sec_areas, inter_max_frames, sup_disps, eccs = analyse_reading(reading_path, False,
                                                                                                warn=warn)
        for d in range(len(disps)):
            reading = Reading(input_data[i][2], d, m_x=input_data[i][0], m_y=input_data[i][1])
            # reading = input_data[i].copy()

            if sideways and disps[d] is not None and positions[d] is not None:
                # Rotate into correct orientation.
                disps[d] = [disps[d][1], -disps[d][0]]
                positions[d] = [positions[d][1], frame_height - positions[d][0]]

            reading.disp_vect = disps[d]
            reading.bubble_pos = positions[d]
            reading.max_bubble_area = areas[d]
            reading.sec_max_area = sec_areas[d]
            reading.inter_max_frames = inter_max_frames[d]
            reading.sup_disp_vect = sup_disps[d]
            reading.ecc_at_max = eccs[d]

            readings.append(reading)
            to_write += 1

        for j in range(1, to_write + 1):
            reading = readings[-j]
            if reading.is_complete():
                dump_file.write(str(reading) + "\n")

    dump_file.close()

    try:
        flag_invalid_readings(dir_path)
    except ValueError:
        if warn:
            print(f"Could not flag invalid readings in {dir_path}")
        else:
            pass
    return readings


def flag_invalid_readings(dir_path, error_on_not_found=False):
    r_path = dir_path + "readings_dump.csv"
    i_path = dir_path + "invalid_readings.txt"
    if os.path.exists(i_path) and os.path.exists(r_path):
        r_file = open(r_path, "r")
        r_lines = r_file.readlines()
        i_file = open(i_path, "r")
        i_lines = i_file.readlines()
        new_lines = []
        for r_line in r_lines:
            flagged = False
            for i_line in i_lines:
                if r_line.split(",")[0] == i_line.strip() and r_line[0] != "#":
                    new_lines.append("#" + r_line)
                    flagged = True
                    break
            if not flagged:
                new_lines.append(r_line)
        r_file.close()
        i_file.close()

        r_file = open(r_path, "w")
        r_file.writelines(new_lines)
        r_file.close()
    elif error_on_not_found:
        raise ValueError("Path does not contain the required files.")


def get_collapse_variations(dir_path, idx, repeat, laser_delay=151.0e-6, geometry_type=None):
    reading_prefix = file.get_prefix_from_idxs(dir_path, [idx])
    reading_path = dir_path + reading_prefix + str(idx).rjust(4, "0") + "/"

    movie = MP4(reading_path + f"video_{repeat}.mp4")

    frame_rate = movie.get_fps()
    trigger_out_delay = movie.trigger_out_delay

    frames = list(movie[:100])

    xs = []
    ys = []
    areas = []
    eccs = []
    idxs = []

    bg_frame = np.int32(frames[0])

    total_laser_delay = laser_delay + trigger_out_delay  # determine total delay in firing the laser
    laser_frame_idx = int(round(frame_rate * total_laser_delay))  # delay in frames
    first_frame_idx = laser_frame_idx + 1  # first frame we will analyse

    for idx in range(first_frame_idx, len(frames)):
        x, y, area, ecc, sol, _ = analyse_frame(frames[idx], bg_frame, debug=True)
        if area is None:
            continue  # No bubble found.
        xs.append(x)
        ys.append(y)
        areas.append(area)
        idxs.append(idx)
        eccs.append(ecc)

    xs = np.array(xs)
    ys = np.array(ys)
    areas = np.array(areas)
    idxs = np.array(idxs)

    sys.path.append(dir_path)
    import params

    importlib.reload(params)
    sys.path.remove(dir_path)

    if geometry_type == 'slot':
        xs = xs - (params.left_slot_wall_x + params.slot_width / 2)
        ys = ys - params.upper_surface_y

    xs = xs * params.mm_per_px
    ys = ys * params.mm_per_px
    radii = np.sqrt(areas / np.pi) * params.mm_per_px
    ts = idxs / frame_rate

    return xs, ys, radii, ts


if __name__ == "__main__":
    flag_invalid_readings(file.select_dir("../../../../../"))
