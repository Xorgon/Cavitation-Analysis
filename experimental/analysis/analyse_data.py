"""
Analyse all of the data in a selected directory.
"""

import os

import experimental.util.analysis_utils as au
import experimental.util.file_utils as file
from PyQt5.QtGui import QGuiApplication

reanalyse = True

w = QGuiApplication([])
dir_path = file.select_dir()
dirs_to_analyse = []
for root, dirs, files in os.walk(dir_path):
    if "index.csv" in files and ("readings_dump.csv" not in files or reanalyse):
        dirs_to_analyse.append(root)

for this_dir in dirs_to_analyse:
    print(f"Analysing {this_dir}")
    au.analyse_series(this_dir + "/")
    try:
        au.flag_invalid_readings(this_dir)
    except ValueError:
        print(f"Could not flag invalid readings in {this_dir}")
