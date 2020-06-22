import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QApplication


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


def select_file(start_path, create_window=True):
    """
    Opens a file selection dialog.
    :param start_path: The path at which to open the dialog.
    :param create_window: Whether to create a new QApplication. May need to be set to false if QT is used elsewhere,
    such as in matplotlib.
    :return: String of the full file path.
    """
    if create_window:
        window = QApplication([])
    file_path = str(QFileDialog.getOpenFileName(None, "Select File", start_path)[0])
    return file_path


def select_multiple_files(start_path, create_window=True):
    """
    Opens a multiple file selection dialog.
    :param start_path: The path at which to open the dialog.
    :param create_window: Whether to create a new QApplication. May need to be set to false if QT is used elsewhere,
    such as in matplotlib.
    :return: A list of full file path strings.
    """
    if create_window:
        window = QApplication([])
    file_paths = QFileDialog.getOpenFileNames(None, "Select File", start_path)[0]
    return file_paths


def select_dir(start_path, create_window=True):
    """
    Opens a directory selection dialog.
    :param start_path: The path at which to open the dialog.
    :param create_window: Whether to create a new QApplication. May need to be set to false if QT is used elsewhere,
    such as in matplotlib.
    :return: A list of full file path strings.
    """
    if create_window:
        window = QApplication([])
    dir_path = str(QFileDialog.getExistingDirectory(None, "Select Directory", start_path)) + "/"
    return dir_path
