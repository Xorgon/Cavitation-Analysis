import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QApplication


def lists_to_csv(file_dir, filename, lists, headers=None, overwrite=False):
    if os.path.exists(file_dir + filename) and not overwrite:
        print(f"Warning: file already exists. {file_dir + filename}")
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
    file = open(file_dir + filename, "w")
    if headers:
        line = ""
        for h in headers:
            line += h + ","
        line = line[:-1] + "\n"
        file.write(line)
    zipped_lists = np.transpose(lists)
    for entry in zipped_lists:
        line = ""
        for part in entry:
            line += str(part) + ","
        line = line[:-1] + "\n"
        file.write(line)
    file.close()


def csv_to_lists(file_dir, filename, has_headers=False):
    file = open(file_dir + filename, "r")
    lines = file.readlines()
    start_idx = 1 if has_headers else 0
    zipped_lists = []
    for line in lines[start_idx:]:
        line = line.strip(',\n')
        part_arr = []
        for part in line.split(","):
            part_arr.append(float(part))
        zipped_lists.append(part_arr)
    return np.transpose(zipped_lists)


def select_file(start_path, create_window=True):
    if create_window:
        window = QApplication([])
    file_path = str(QFileDialog.getOpenFileName(None, "Select File", start_path)[0])
    return file_path


def select_multiple_files(start_path, create_window=True):
    if create_window:
        window = QApplication([])
    file_paths = QFileDialog.getOpenFileNames(None, "Select File", start_path)[0]
    return file_paths


def select_dir(start_path, create_window=True):
    if create_window:
        window = QApplication([])
    dir_path = str(QFileDialog.getExistingDirectory(None, "Select Directory", start_path)) + "/"
    return dir_path
