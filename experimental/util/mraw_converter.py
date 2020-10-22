# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:36:00 2017

@author: Ivo Peters
ivo.r.peters@gmail.com

Modified by Elijah Andrews
"""

import numpy as np
# import matplotlib.pyplot as plt
import cv2

from experimental.util.mraw import mraw
import experimental.util.file_utils as file


def convert(mraw_obj, outputFile, codec='XVID', fps=24, frame_range=None, scale=1, contrast=1):
    movie = mraw_obj
    if frame_range is None:
        frame_range = (0, len(movie))
    out = cv2.VideoWriter(outputFile,
                          cv2.VideoWriter_fourcc(codec[0],
                                                 codec[1],
                                                 codec[2],
                                                 codec[3]),
                          fps,
                          (movie.width * scale, movie.height * scale),
                          0)
    for i in range(frame_range[0], frame_range[1]):
        frame = np.uint8(np.double(movie[i]) / (2 ** 12) * 255)  # Convert to 8 bit colour.
        if scale != 1:
            frame = cv2.resize(frame, (movie.width * scale, movie.height * scale))
        if contrast != 1:
            frame = cv2.convertScaleAbs(frame, alpha=contrast)
        out.write(frame)
    out.release()


def convert_mraw(mraw_obj, outputFile, codec='XVID', fps=24, separate_readings=False):
    movie = mraw_obj
    if separate_readings:
        repeats = movie.image_count // 100
        for i in range(repeats):
            filename = outputFile[:-4] + "_" + str(i) + outputFile[-4:]
            convert(movie, filename, frame_range=(i * 100, (i + 1) * 100), codec=codec, fps=fps, contrast=2)
    else:
        convert(mraw_obj, outputFile, codec=codec, fps=fps, contrast=2)


def convert_series(dir_path, codec="XVID", file_format="mp4"):
    index_file = open(dir_path + "index.csv")
    index_lines = index_file.readlines()
    index_file.close()

    input_data = []  # x, y, index

    for i in range(1, len(index_lines)):  # Start at 1 to skip header.
        split = index_lines[i].strip().split(",")
        input_data.append([float(split[0]), float(split[1]), split[2]])  # x, y, index number

    reading_prefix = file.get_prefix_from_idxs(dir_path, np.array(input_data)[:, 2])
    for i in range(len(input_data)):
        print("Converting reading {0}.".format(i))
        reading_path = dir_path + reading_prefix + str(input_data[i][2]).rjust(4, "0") + "/"
        mraw_obj = file.get_mraw_from_dir(reading_path)
        convert_mraw(mraw_obj, reading_path + "video." + file_format, codec=codec, separate_readings=True)


if __name__ == "__main__":
    convert_series(file.select_dir(), codec="mp4v")
