import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tf
from skimage.feature import register_translation
from skimage.filters import sobel, threshold_triangle
from skimage.restoration import denoise_bilateral

import common.util.plotting_utils as pu
import common.util.vector_utils as vect
import experimental.util.file_utils as fu
from experimental.util.mraw import mraw


def calculate_offset(frame_1, frame_2, plot_compared_frames=False):
    """
    Calculates a required offset of frame_1 in direction dir such that it maps onto frame_2 as closely as possible.

    :param frame_1: Frame to be moved.
    :param frame_2: Frame to be mapped onto.
    :param plot_compared_frames: Whether to plot the final frame comparison.
    :return: Offset required in direction.
    """
    frame_1 = denoise_bilateral(frame_1, multichannel=False, bins=2)
    frame_2 = denoise_bilateral(frame_2, multichannel=False, bins=2)
    # frame_1 = sobel(frame_1)
    # frame_2 = sobel(frame_2)

    thresh_1 = threshold_triangle(frame_1)
    thresh_2 = threshold_triangle(frame_2)
    thresh = (thresh_1 + thresh_2) / 2
    frame_1 = frame_1 > thresh
    frame_2 = frame_2 > thresh

    offset, error, _ = register_translation(frame_1, frame_2)
    if plot_compared_frames:
        tform = tf.SimilarityTransform(translation=[offset[1], offset[0]])
        pu.plot_frame(tf.warp(frame_1, tform), show_immediately=False)
        pu.plot_frame(frame_2, show_immediately=False)
        plt.show()

    return offset


def calculate_mm_per_pixel(dir_path, plot_compared_frames=False):
    """ Just used for testing, do not use for actual calculations. """
    index_file = open(dir_path + "index.csv")
    index_lines = index_file.readlines()

    readings = []  # x, y, index

    for i in range(1, len(index_lines)):  # Start at 1 to skip header.
        split = index_lines[i].strip().split(",")
        readings.append([float(split[0]), float(split[1]), split[2]])  # x, y, index number

    readings = np.array(readings)
    xs = np.array(readings[:, 0], dtype=float)
    ys = np.array(readings[:, 1], dtype=float)

    first_frames = []
    for r in range(len(readings)):
        path = dir_path + "movie_S" + str(readings[r][2]).rjust(4, '0') + "/a_C001H001S0001.cih"
        movie = mraw(path)
        first_frames.append(movie[0])
        movie.close()

    # For now just take the first two readings.
    # TODO: Make a method of detecting if geometry is in view and get average mm_per_pixel across a range.
    pixel_offset = vect.mag(calculate_offset(first_frames[10], first_frames[11], plot_compared_frames))
    mm_offset_x = float(readings[1][0]) - float(readings[0][0])
    mm_offset_y = float(readings[1][1]) - float(readings[0][1])
    mm_offset = math.sqrt(mm_offset_x ** 2 + mm_offset_y ** 2)

    return mm_offset / pixel_offset


if __name__ == '__main__':
    # cal_dir = "../../../Data/SidewaysSeries/w2.2h2.7/"
    # print(calculate_mm_per_pixel(cal_dir, plot_compared_frames=True))
    mraw_1 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0013/")
    mraw_2 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0001/")
    print(calculate_offset(mraw_1[0], mraw_2[0], True))
