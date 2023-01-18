import os
import shutil

from experimental.util.file_utils import get_prefix_from_idxs
from experimental.util.analysis_utils import load_readings

# Root name, copy movies?, sub-directories
input_dirs = [["Triangle", True, ["E:/Data/Lebo/Restructured Data/Equilateral triangle/",
                                  "E:/Data/Lebo/Restructured Data/Equilateral triangle 2/"]],

              ["Square", True, ["E:/Data/Lebo/Restructured Data/Square/",
                                "E:/Data/Lebo/Restructured Data/Square 2/",
                                "E:/Data/Lebo/Restructured Data/Square 3/"]],

              ["90 degree corner", False, ["E:/Data/Ivo/Restructured Data/90 degree corner/",
                                           "E:/Data/Ivo/Restructured Data/90 degree corner 2/",
                                           "E:/Data/Ivo/Restructured Data/90 degree corner 3/",
                                           "E:/Data/Ivo/Restructured Data/90 degree corner 4/"]],

              ["60 degree corner", False, ["E:/Data/Ivo/Restructured Data/60 degree corner/"]],

              ["Flat plate", True,
               ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Solid plate/"]],

              ["Slots", False,
               ["C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W1H3/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H12/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3a/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H6/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H9/",
                "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W4H12/"]
               ],
              ]

output_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Anisotropy Modelling/Data Archive/Data/"

include_invalid = False
overwrite_movies = True  # For recopying movies in case of change.
overwrite_plots = False

os.makedirs(output_dir, exist_ok=True)

for geom_dir, copy_movies, subdirs in input_dirs:
    print(geom_dir)
    os.makedirs(output_dir + geom_dir + "/", exist_ok=True)

    for sub_in_dir in subdirs:
        print("   " + sub_in_dir)
        if len(subdirs) > 1:
            sub_out_dir = output_dir + geom_dir + "/" + sub_in_dir.split('/')[-2] + "/"
        else:
            sub_out_dir = output_dir + geom_dir + "/"
        os.makedirs(sub_out_dir, exist_ok=True)

        # Copy meta files.
        shutil.copyfile(sub_in_dir + "/readings_dump.csv", sub_out_dir + "/readings_dump.csv")
        shutil.copyfile(sub_in_dir + "/params.py", sub_out_dir + "/params.py")
        shutil.copyfile(sub_in_dir + "/index.csv", sub_out_dir + "/index.csv")
        shutil.copyfile(sub_in_dir + "/description.txt", sub_out_dir + "/description.txt")

        # Copy movies and plots.
        readings = load_readings(sub_in_dir + "/readings_dump.csv")
        idxs = set([r.idx for r in readings])
        prefix = get_prefix_from_idxs(sub_in_dir + "/", idxs)

        if include_invalid:
            for idx in idxs:
                movie_dir = "/" + prefix + str(idx).rjust(4, '0') + "/"
                os.makedirs(sub_out_dir + movie_dir, exist_ok=True)
                for f in os.listdir(sub_in_dir + movie_dir):
                    if (f[:6] == "video_" and f[-4:] == '.mp4') \
                            or (f[:15] == "analysis_plot_r" and f[-4:] == '.png'):
                        shutil.copyfile(sub_in_dir + movie_dir + f, sub_out_dir + movie_dir + f)
        else:
            for reading in readings:
                movie_dir = "/" + prefix + str(reading.idx).rjust(4, '0') + "/"
                os.makedirs(sub_out_dir + movie_dir, exist_ok=True)

                if copy_movies:
                    video_name = f"video_{reading.repeat_number}.mp4"
                    if not os.path.exists(sub_out_dir + movie_dir + video_name) or overwrite_movies:
                        shutil.copyfile(sub_in_dir + movie_dir + video_name, sub_out_dir + movie_dir + video_name)

                plot_name = f"analysis_plot_r{reading.repeat_number}.png"
                if not os.path.exists(sub_out_dir + movie_dir + plot_name) or overwrite_plots:
                    shutil.copyfile(sub_in_dir + movie_dir + plot_name, sub_out_dir + movie_dir + plot_name)
