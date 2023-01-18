import os
import shutil

from common.util.file_utils import csv_to_lists
from experimental.util.file_utils import get_prefix_from_idxs
from experimental.util.analysis_utils import load_readings

input_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"
output_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates packaged/"

include_invalid = True
overwrite_movies = False  # For recopying movies in case of change.
overwrite_plots = False

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    if "index.csv" not in files or "readings_dump.csv" not in files or "params.py" not in files:
        continue

    geom_dir = root.replace(input_dir, "")

    os.makedirs(output_dir + geom_dir, exist_ok=True)

    # Copy meta files.
    shutil.copyfile(input_dir + geom_dir + "/readings_dump.csv", output_dir + geom_dir + "/readings_dump.csv")
    shutil.copyfile(input_dir + geom_dir + "/params.py", output_dir + geom_dir + "/params.py")
    shutil.copyfile(input_dir + geom_dir + "/index.csv", output_dir + geom_dir + "/index.csv")
    shutil.copyfile(input_dir + geom_dir + "/description.txt", output_dir + geom_dir + "/description.txt")

    # Copy movies and plots.
    if include_invalid:
        _, _, idxs = csv_to_lists(input_dir + geom_dir, "/index.csv", has_headers=True)
        idxs = [int(idx) for idx in idxs]
    else:
        readings = load_readings(input_dir + geom_dir + "/readings_dump.csv")
        idxs = set([r.idx for r in readings])
    prefix = get_prefix_from_idxs(input_dir + geom_dir + "/", idxs)

    if include_invalid:
        for idx in idxs:
            movie_dir = geom_dir + "/" + prefix + str(idx).rjust(4, '0') + "/"
            os.makedirs(output_dir + movie_dir, exist_ok=True)
            for f in os.listdir(input_dir + movie_dir):
                if (f[:6] == "video_" and f[-4:] == '.mp4') and (
                        not os.path.exists(output_dir + movie_dir + f) or overwrite_movies):
                    shutil.copyfile(input_dir + movie_dir + f, output_dir + movie_dir + f)
                if (f[:15] == "analysis_plot_r" and f[-4:] == '.png') and (
                        not os.path.exists(output_dir + movie_dir + f) or overwrite_plots):
                    shutil.copyfile(input_dir + movie_dir + f, output_dir + movie_dir + f)
    else:
        for reading in readings:
            movie_dir = geom_dir + "/" + prefix + str(reading.idx).rjust(4, '0') + "/"
            os.makedirs(output_dir + movie_dir, exist_ok=True)

            video_name = f"video_{reading.repeat_number}.mp4"
            if not os.path.exists(output_dir + movie_dir + video_name) or overwrite_movies:
                shutil.copyfile(input_dir + movie_dir + video_name, output_dir + movie_dir + video_name)

            plot_name = f"analysis_plot_r{reading.repeat_number}.png"
            if not os.path.exists(output_dir + movie_dir + plot_name) or overwrite_plots:
                shutil.copyfile(input_dir + movie_dir + plot_name, output_dir + movie_dir + plot_name)
