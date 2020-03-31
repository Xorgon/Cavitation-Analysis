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


if __name__ == '__main__':
    # cal_dir = "../../../Data/SidewaysSeries/w2.2h2.7/"
    # print(calculate_mm_per_pixel(cal_dir, plot_compared_frames=True))
    mraw_1 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0013/")
    mraw_2 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0001/")
    print(calculate_offset(mraw_1[0], mraw_2[0], True))
