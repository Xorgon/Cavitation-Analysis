import os
import re
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import importlib
import numpy as np

from util.mraw import mraw


def get_prefix_from_idxs(dir_path, idxs):
    """
    Finds the prefix used in a directory for a series of indexes. e.g. movie0001 vs movie_S0001.
    :param dir_path: Directory path.
    :param idxs: Array of indexes to check.
    :return: Prefix.
    """
    files = os.listdir(dir_path)

    prefix = None
    for idx in idxs:
        for file in files:
            if os.path.isdir(dir_path + file):
                suffix = str(idx).rjust(4, '0')
                match = re.match(r'(.*)' + suffix, file)
                if match:
                    file_prefix = match.group(1)
                    if prefix is None:
                        prefix = file_prefix
                    else:
                        if prefix != file_prefix:
                            print("Warning: Multiple directory prefixes found. Only using {0}".format(prefix))
    return prefix


def get_mraw_from_dir(dir_path):
    files = os.listdir(dir_path)
    headers = []
    for file in files:
        if re.match(r'(.*\.cih)', file):
            headers.append(file)

    if len(headers) > 1:
        print("Warning: Multiple headers found. Using {0}".format(headers[0]))
    if len(headers) == 0:
        print("Warning: No headers found in {0}".format(dir_path))
        return None

    return mraw(dir_path + headers[0])


def select_dir(start_path="../../../../../", create_window=True):
    if create_window:
        window = QApplication([])
    dir_path = str(QFileDialog.getExistingDirectory(None, "Select Directory", start_path)) + "/"
    return dir_path


def load_params(params_dir):
    sys.path.append(params_dir)
    import params

    importlib.reload(params)
    sys.path.remove(params_dir)
    return params

def lists_to_csv(file_dir, filename, lists, headers=None, overwrite=False):
    """
    Writes a list of lists to a CSV file where each list is a column
    :param file_dir: The directory in which to save the CSV file.
    :param filename: The name of the CSV file (including file extension).
    :param lists: The lists to save.
    :param headers: A list of header strings, one for each column, defaults to no headers.
    :param overwrite: Whether to overwrite an existing file, defaults to false.
    :return: Success of file write.
    """
    if os.path.exists(file_dir + filename) and not overwrite:
        print(f"Warning: file already exists. {file_dir + filename}")
        return False
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
    return True


def csv_to_lists(file_dir, filename, has_headers=False):
    """
    Reads a CSV file to a list of lists where each column becomes a list.
    :param file_dir: The directory from which to read the CSV file.
    :param filename: The name of the CSV file (including file extension).
    :param has_headers: Whether the CSV file has headers, defaults to False.
    :return: List of lists.
    """
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
