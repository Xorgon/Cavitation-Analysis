import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tf
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel, threshold_triangle
from skimage.restoration import denoise_bilateral

import util.plotting_utils as pu
import util.file_utils as fu


def calculate_offset(frame_1, frame_2, plot_compared_frames=False):
    """
    Calculates a required offset of frame_1 in direction dir such that it maps onto frame_2 as closely as possible.

    :param frame_1: Frame to be moved.
    :param frame_2: Frame to be mapped onto.
    :param plot_compared_frames: Whether to plot the final frame comparison.
    :return: Offset required in direction.
    """
    frame_1 = denoise_bilateral(frame_1, channel_axis=None, bins=2, mode='reflect', sigma_spatial=2.5)
    frame_2 = denoise_bilateral(frame_2, channel_axis=None, bins=2, mode='reflect', sigma_spatial=2.5)
    frame_1 = sobel(frame_1, mask=np.ones(frame_1.shape, dtype=bool))
    frame_2 = sobel(frame_2, mask=np.ones(frame_1.shape, dtype=bool))

    # thresh_1 = threshold_triangle(frame_1)
    # thresh_2 = threshold_triangle(frame_2)
    # thresh = (thresh_1 + thresh_2) / 2
    # frame_1 = frame_1 > thresh
    # frame_2 = frame_2 > thresh

    offset, error, _ = phase_cross_correlation(frame_1, frame_2, normalization=None)
    if plot_compared_frames:
        tform = tf.SimilarityTransform(translation=[offset[1], offset[0]])
        pu.plot_frame(tf.warp(frame_1, tform), show_immediately=False)
        pu.plot_frame(frame_2, show_immediately=False)
        plt.show()

    return offset


if __name__ == '__main__':
    # cal_dir = "../../../Data/SidewaysSeries/w2.2h2.7/"
    # print(calculate_mm_per_pixel(cal_dir, plot_compared_frames=True))
    # mraw_1 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0013/")
    # mraw_2 = fu.get_mraw_from_dir("../../../../Data/SlotSweeps/w1h3/movie_S0001/")
    mraw_1 = fu.get_mraw_from_dir("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 degassed again/movie_C001H001S0001/")
    mraw_2 = fu.get_mraw_from_dir("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/~25VF/Purer water between 3 degassed again/movie_C001H001S0002/")
    print(calculate_offset(mraw_1[0], mraw_2[0], True))
