import numpy as np
import matplotlib.pyplot as plt
import os

from experimental.util.mraw import mraw
from experimental.util.file_utils import get_mraw_from_dir, get_prefix_from_idxs


def load_norm_mat(dataset_dir="C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/"
                              "Data/Steel plates/w20vf24squares/On a hole/"):
    """ Load a normalisation matrix for pixel errors. Default dataset dir is set for a clean data set. """
    subdirs = [d for d in os.listdir(dataset_dir) if "movie_C001H001S" in d or "movie_S" in d or "movie" in d]
    all_idxs = []
    max_idx = 0
    for sd in subdirs:
        if "movie_C001H001S" in sd:
            idx = int(sd.split("movie_C001H001S")[-1])
        elif "movie_S" in sd:
            idx = int(sd.split("movie_S")[-1])
        elif "movie" in sd:
            idx = int(sd.split("movie")[-1])
        else:
            raise RuntimeError("Could not find movies in " + dataset_dir)

        if idx > max_idx:
            max_idx = idx

        all_idxs.append(idx)

    prefix = get_prefix_from_idxs(dataset_dir, all_idxs)

    norm_mats = []
    if len(all_idxs) > 10:
        first_blank = max_idx - 2
    else:
        first_blank = max_idx
    for i in range(first_blank, max_idx + 1):
        max_mraw = get_mraw_from_dir(dataset_dir + f"{prefix}{i:04d}/")  # type: mraw
        med = np.median(max_mraw[0])
        norm_mats.append(med / max_mraw[0])
    return np.median(norm_mats, axis=0)


def safe_correct(frame, norm_mat, max_value=2 ** 12 - 1):
    """ Performs pixel correction but ignores saturated pixels. """
    out = np.copy(np.float64(frame))
    np.multiply(norm_mat, frame, out=out, where=frame != max_value)
    np.minimum(out, max_value, out=out)  # Cap values that become over-saturated (possible for almost-saturated pixels).
    return out


if __name__ == "__main__":
    mov_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/" \
              "Data/Steel plates/w12vf16triangles/On a hole/movie_C001H001S0001/"
    normalisation_mat = load_norm_mat(mov_dir + "../")

    mov = get_mraw_from_dir(mov_dir)  # type: mraw
    plt.figure()
    plt.imshow(mov[32], plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.figure()
    plt.imshow(normalisation_mat * mov[32], plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
