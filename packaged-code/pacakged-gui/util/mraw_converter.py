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
from PIL import Image
from subprocess import Popen, PIPE
import os
import multiprocessing

from util.mraw import mraw
from util.pixel_correction import safe_correct, load_norm_mat
import util.file_utils as file


def convert(mraw_obj, outputFile, codec='XVID', fps=24, frame_range=None, scale=1, contrast=1, crf=18,
            autoscale_brightness=False, norm_mat=None, writer='ffmpeg'):
    """
    Convert a given mraw object into a video file.
    :param mraw_obj: mraw object
    :param outputFile: output file path
    :param codec: four letter codec name - ignored if using writer='ffmpeg'
    :param fps: frames per second in the output file
    :param frame_range: range of frames to include, [a, b] inclusive of both a and b
    :param scale: image scale
    :param contrast: contrast ratio
    :param crf: Constant Rate Factor (quality), only used for writer='ffmpeg'
    :param autoscale_brightness: whether to scale brightness based on maximum and minimum brightnesses in the image
    :param writer: which video-writing method to use (cv2 or ffmpeg)
    """
    movie = mraw_obj
    if frame_range is None:
        frame_range = (0, len(movie))

    if writer == "cv2":
        if type(codec) is str and len(codec) == 4:
            codec = cv2.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3])
        out = cv2.VideoWriter(outputFile, codec, fps,
                              (movie.width * scale, movie.height * scale),
                              0)
    elif writer == "ffmpeg":
        p = Popen(['ffmpeg',
                   '-y',  # Overwrite files
                   '-f', 'image2pipe',  # Input format
                   '-r', '24',  # Framerate
                   '-i', '-',  # stdin
                   '-c:v', 'libx264',  # Codec
                   '-preset', 'slow',
                   '-crf', f'{crf}',  # H264 Constant Rate Factor (quality, lower is better)
                   '-loglevel', 'quiet',  # Stop yelling at me
                   '-flush_packets', '1',
                   outputFile], stdin=PIPE)

    movie_min = np.min([np.min(movie[i]) for i in range(frame_range[0], frame_range[1] + 1)])
    movie_max = np.max([np.max(movie[i]) for i in range(frame_range[0], frame_range[1] + 1)])
    movie_ptp = movie_max - movie_min
    for i in range(frame_range[0], frame_range[1] + 1):
        frame = movie[i]
        if norm_mat is not None:
            frame = safe_correct(frame, norm_mat)
        if autoscale_brightness:
            frame = ((frame - movie_min) / (movie_ptp / 255.0)).astype(np.uint8)
        else:
            frame = np.uint8(np.double(frame) / (2 ** 12) * 255)  # Convert to 8 bit colour.
        if scale != 1:
            frame = cv2.resize(frame, (movie.width * scale, movie.height * scale))
        if contrast != 1:
            frame = cv2.convertScaleAbs(frame, alpha=contrast)

        if writer == "cv2":
            out.write(frame)
        elif writer == "ffmpeg":
            im = Image.fromarray(frame)
            im.save(p.stdin, 'PNG')

    if writer == "cv2":
        out.release()
    elif writer == "ffmpeg":
        p.stdin.close()
        p.wait()


def convert_mraw(mraw_obj, outputFile, codec='XVID', fps=24, separate_readings=True, writer="ffmpeg", norm_mat=None):
    movie = mraw_obj
    if separate_readings:
        repeats = movie.image_count // 100
        for i in range(repeats):
            filename = outputFile[:-4] + "_" + str(i) + outputFile[-4:]
            convert(movie, filename, frame_range=(i * 100, (i + 1) * 100 - 1),
                    codec=codec, fps=fps, contrast=1, writer=writer, norm_mat=norm_mat)
    else:
        convert(mraw_obj, outputFile, codec=codec, fps=fps, contrast=1, writer=writer, norm_mat=norm_mat)


def convert_series(dir_path, codec="mp4v", file_format="mp4", writer="ffmpeg", px_correct=True):
    print(dir_path)
    index_file = open(dir_path + "index.csv")
    index_lines = index_file.readlines()
    index_file.close()

    input_data = []  # x, y, index

    for i in range(1, len(index_lines)):  # Start at 1 to skip header.
        split = index_lines[i].strip().split(",")
        input_data.append([float(split[0]), float(split[1]), split[2]])  # x, y, index number

    reading_prefix = file.get_prefix_from_idxs(dir_path, np.array(input_data)[:, 2])
    for i in range(len(input_data)):
        print("Converting reading {0}. - {1}".format(input_data[i][2], dir_path))
        reading_path = dir_path + reading_prefix + str(input_data[i][2]).rjust(4, "0") + "/"
        mraw_obj = file.get_mraw_from_dir(reading_path)
        if px_correct:
            norm_mat = load_norm_mat(dir_path)
        else:
            norm_mat = None
        convert_mraw(mraw_obj, reading_path + "video." + file_format,
                     codec=codec, separate_readings=True, writer=writer, norm_mat=norm_mat)


if __name__ == "__main__":
    # Anisotropy modelling paper
    # to_convert = ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
    #               "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/",
    #               "E:/Data/Lebo/Restructured Data/Square/",
    #               "E:/Data/Lebo/Restructured Data/Square 2/",
    #               "E:/Data/Lebo/Restructured Data/Square 3/",
    #               "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]

    to_convert = []

    # Porous plates paper
    root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

    for root, _, files in os.walk(root_dir):
        if "params.py" in files:
            to_convert.append(root + "/")

    for cdir in to_convert:
        print(cdir)
    print(f"Found {len(to_convert)} data sets")

    pool = multiprocessing.Pool(processes=os.cpu_count() - 1)  # Leave one core for the rest of us
    pool.map(convert_series, to_convert)
    pool.close()

    # this_path = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/" \
    #             "Steel plates/w48vf24circles/Between 3 holes/"
    # norm_mat = load_norm_mat(this_path)
    # this_mraw = file.get_mraw_from_dir(this_path + "movie_C001H001S0026/")
    # convert_mraw(this_mraw, this_path + "movie_C001H001S0026/" + "video.mp4", norm_mat=norm_mat)

    # TESTING OPTIONS
    # basic_norm_mat = load_norm_mat()

    # convert(file.get_mraw_from_dir(
    #     "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/2something Hole 1mm Acrylic 50mm Plate/The Lump/normal_angle_C001H001S0001/"),
    #     "converted.mp4", codec="mp4v", frame_range=[200, 300], scale=2, contrast=1, autoscale_brightness=True)

    # this_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Misc/OAP mirror testing/sono-100power-75att_C001H001S0001/"
    # convert(file.get_mraw_from_dir(this_dir), this_dir + "luminescence/luminescence1.png", codec=0,
    #         frame_range=[400, 499],
    #         scale=1, contrast=1, fps=0, autoscale_brightness=True, norm_mat=basic_norm_mat)
