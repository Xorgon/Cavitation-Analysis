# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:43:51 2017

@author: Ivo Peters
ivo.r.peters@gmail.com

version 1.5

Update version 1.2 (2017-12-07):
    Added a catch in case only one peak is found for the bubble size.
Update version 1.3 (2017-12-10):
    Reduced dpi of output images from 300 to 150.
Update version 1.4 (2018-03-29):
    Added a catch in case the peak consists of two points with exactly the same
    value.
Update version 1.5 (2018-04-08):
    Fixed use of non-integer as index in findPeaks function.
"""

import numpy as np
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from scipy import ndimage


def findPeaks(x, y, n=2, kernel=5):
    """
    Standard peak finding algorithm, finds the first n peaks using a kernel
    of size kernel.
    """
    peakIdx = []  # index of the peak
    yPeak = []  # peak value
    k = np.int((kernel - 1) / 2)
    for i in range(len(x) - (kernel - 1)):
        if np.max(y[i:i + kernel]) == y[i + k]:
            # If this is the first peak that is found, then the peak is added
            if len(peakIdx) < 1:
                peakIdx.append(i + k)
                yPeak.append(y[i + k])
            # Check if this is not a peak within the same range
            elif (i + k - peakIdx[-1]) > k:
                peakIdx.append(i + k)
                yPeak.append(y[i + k])
            if len(peakIdx) == n:
                break
    return peakIdx, yPeak


def makeBinary(I):
    """
    Turn a grayscale image into a binary image, trying to only keep the main
    bubble as a uniform white object on a fully black background.
    """
    # find a reasonable threshold
    thresh = threshold_otsu(I)
    # make a binary version of the image
    binary = I > thresh
    # try to separate small objects that are attached to the main bubble
    morph.binary_opening(binary, footprint=None, out=binary)
    # fill up holes to make the bright spot in the center of the bubble black
    ndimage.binary_fill_holes(binary, output=binary)
    # remove small objects (noise and object that we separated a few steps ago)
    binary = morph.remove_small_objects(binary, min_size=128)
    # return the binary image
    return binary


laserDelay = 151.0e-6  # delay (s) for Q-switch after lamp was triggered (setting in laser)
