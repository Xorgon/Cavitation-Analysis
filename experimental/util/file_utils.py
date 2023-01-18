import os
import re
from experimental.util.mraw import mraw
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import importlib


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
