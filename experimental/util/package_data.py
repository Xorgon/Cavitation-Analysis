import os
import shutil

from experimental.util.file_utils import get_prefix_from_idxs
from experimental.util.analysis_utils import load_readings

input_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/"
output_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/Packaged/"

os.makedirs(output_dir)

for geom_dir in os.listdir(input_dir):
    os.makedirs(output_dir + geom_dir)

    # Copy meta files.
    shutil.copyfile(input_dir + geom_dir + "/readings_dump.csv", output_dir + geom_dir + "/readings_dump.csv")
    shutil.copyfile(input_dir + geom_dir + "/params.py", output_dir + geom_dir + "/params.py")
    shutil.copyfile(input_dir + geom_dir + "/index.csv", output_dir + geom_dir + "/index.csv")
    shutil.copyfile(input_dir + geom_dir + "/description.txt", output_dir + geom_dir + "/description.txt")

    # Copy all valid readings.
    readings = load_readings(input_dir + geom_dir + "/readings_dump.csv")
    idxs = set([r.idx for r in readings])
    prefix = get_prefix_from_idxs(input_dir + geom_dir + "/", idxs)
    for reading in readings:
        movie_dir = geom_dir + "/" + prefix + str(reading.idx).rjust(4, '0') + "/"
        os.makedirs(output_dir + movie_dir, exist_ok=True)
        video_name = f"video_{reading.repeat_number}.mp4"
        shutil.copyfile(input_dir + movie_dir + video_name, output_dir + movie_dir + video_name)

        plot_name = f"analysis_plot_r{reading.repeat_number}.png"
        shutil.copyfile(input_dir + movie_dir + plot_name, output_dir + movie_dir + plot_name)
