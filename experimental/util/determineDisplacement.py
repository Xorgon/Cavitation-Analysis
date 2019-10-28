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
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from skimage.measure import label, regionprops
from scipy import ndimage

from experimental.util.mraw import mraw


def findPeaks(x, y, n=2, kernel=5):
    """
    Standard peak finding algorithm, finds the first n peaks using a kernel
    of size kernel.
    """
    xPeak = []  # x-position of the peak
    yPeak = []  # peak value
    k = np.int((kernel - 1) / 2)
    for i in range(len(x) - (kernel - 1)):
        if np.max(y[i:i + kernel]) == y[i + k]:
            # If this is the first peak that is found, then the peak is added
            if len(xPeak) < 1:
                xPeak.append(i + k)
                yPeak.append(y[i + k])
            # Check if this is not a peak within the same range
            elif (i + k - xPeak[-1]) > k:
                xPeak.append(i + k)
                yPeak.append(y[i + k])
            if len(xPeak) == n:
                break
    return xPeak, yPeak


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
    morph.binary_opening(binary, selem=None, out=binary)
    # fill up holes to make the bright spot in the center of the bubble black
    ndimage.binary_fill_holes(binary, output=binary)
    # remove small objects (noise and object that we separated a few steps ago)
    binary = morph.remove_small_objects(binary, min_size=256)
    # return the binary image
    return binary


fileName = 'movie0966/a.cih'
laserDelay = 151.0e-6  # delay (s) for Q-switch after lamp was triggered (setting in laser)

movieID = 966
baseDir = ''


def bubbleDisplacement(baseDir, movieID, laserDelay):
    # construct filename for movie to load
    fileName = '%smovie%04d/a.cih' % (baseDir, movieID)
    # load movie as mraw object
    movie = mraw(fileName)

    # Initiate lists
    xs = []
    ys = []
    ns = []
    areas = []

    # determine total delay in firing the laser
    totalLaserDelay = laserDelay
    # delay in frames
    laserFrame = int(round(movie.frame_rate * totalLaserDelay))
    # first frame we will analyse
    firstFrame = laserFrame + 1
    # read in a background image
    bgImage = np.int32(movie[0])

    for i in range(firstFrame, len(movie)):
        I = bgImage - np.int32(movie[i])
        binary = makeBinary(I)
        label_img = label(binary)
        props = regionprops(label_img, cache=False)
        if len(props):
            if len(props) > 1:
                maxArea = 0.0
                for j in range(len(props)):
                    if props[j].area > maxArea:
                        maxArea = props[j].area
                        maxIndex = j
                y, x = props[maxIndex].centroid
                areas.append(props[maxIndex].area)
            else:
                y, x = props[0].centroid
                areas.append(props[0].area)
            ns.append(i)
            xs.append(x)
            ys.append(y)

    # convert the lists to numpy arrrays:
    areas = np.array(areas)
    xs = np.array(xs)
    ys = np.array(ys)
    ns = np.array(ns)

    # calculate time (s)
    t = ns / movie.frame_rate

    # close the previous plot
    plt.close()
    # find and plot peaks
    plt.figure()
    plt.subplot(2, 3, 3)
    plt.plot(1000 * t, areas)
    plt.xlabel('t (ms)')
    plt.ylabel('area (sq. pix.)')
    peakFrames, yPeak = findPeaks(ns, areas, kernel=5)
    idx = ns[peakFrames]
    plt.plot(1000 * t[peakFrames], yPeak, 'o')

    # plot raw images
    if len(idx) > 0:
        plt.subplot(2, 3, 1)
        plt.imshow(movie[idx[0]], cmap=plt.cm.gray)
        plt.plot(xs[peakFrames[0]], ys[peakFrames[0]], 'x')
        plt.xticks([])
        plt.yticks([])
    if len(idx) > 1:
        plt.subplot(2, 3, 4)
        plt.imshow(movie[idx[1]], cmap=plt.cm.gray)
        plt.plot(xs[peakFrames[1]], ys[peakFrames[1]], 'x')
        plt.xticks([])
        plt.yticks([])

    # plot binary processed bubble images
    if len(idx) > 0:
        plt.subplot(2, 3, 2)
        binary1 = makeBinary(bgImage - np.int32(movie[idx[0]]))
        plt.imshow(binary1, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
    if len(idx) > 1:
        plt.subplot(2, 3, 5)
        binary2 = makeBinary(bgImage - np.int32(movie[idx[1]]))
        plt.imshow(binary2, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])

    # calculate and plot displacement on background image
    if len(idx) > 1:
        dx = xs[peakFrames[1]] - xs[peakFrames[0]]
        dy = ys[peakFrames[1]] - ys[peakFrames[0]]
        plt.subplot(2, 3, 6)
        plt.imshow(bgImage, cmap=plt.cm.gray)
        plt.plot(xs[peakFrames], ys[peakFrames], 'x')
        plt.quiver(xs[peakFrames[0]], ys[peakFrames[0]], dx, -dy,
                   scale_units='x',
                   scale=0.33,
                   width=0.02)

    plt.tight_layout()
    plt.show()
    plt.pause(0.001)

    # close the movie file
    movie.close()

    # save figure for checking proper working of algorithms
    plt.savefig('plotDump/movie%04d.png' % movieID, dpi=150)

    # return the bubble positions and bubble area
    if len(idx) > 1:
        return xs[peakFrames], ys[peakFrames], yPeak[0]
    elif len(idx) > 0:
        i = [peakFrames[0], peakFrames[0]]
        return xs[i], ys[i], yPeak[0]
    else:
        return [0, 0], [0, 0], 0
