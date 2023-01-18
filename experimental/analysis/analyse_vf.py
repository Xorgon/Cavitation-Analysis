from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import os

root_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/Data/Steel plates/"

dirs = os.listdir(root_dir)

overwrite = True

mm_per_px = 0.0424

for this_dir in dirs:
    img_path = root_dir + this_dir + "/vf_measurement_C001H001S0001/vf_measurement_C001H001S0001000001.tif"
    if os.path.exists(img_path):
        img = io.imread(img_path, as_gray=True)
        threshs_8bit = np.array([30, 70, 95])  # Min, nominal, and max threshold values in 8-bit grayscale.
        threshs_10bit = 1023 * threshs_8bit / 255
        quantiles = [0.75, 0.5, 0.25]
        res_vfs = []
        res_areas = []
        res_sizes = []
        res_spacings = []
        for n, thresh in enumerate(threshs_10bit):
            cs = range(96, 256)
            vfs = []
            for c in cs:
                c_img = img[c:1024 - c, c:1024 - c]
                # thresh = threshold_otsu(c_img)
                # print(thresh)
                b_img = c_img > thresh
                vf = np.sum(b_img) / b_img.size
                vfs.append(vf)

            res_vf = np.quantile(vfs, quantiles[n])
            res_vfs.append(res_vf)

            b_img = img > thresh
            props = regionprops(label(b_img))
            med_area = np.median([p.area for p in props])
            if "circles" in this_dir:
                W = np.sqrt(4 * med_area / np.pi) * mm_per_px
                S = W * np.sqrt(np.pi * np.sqrt(3) / (6 * res_vf))
            elif "triangles" in this_dir:
                W = np.sqrt(4 * med_area / np.sqrt(3)) * mm_per_px
                S = W / np.sqrt(3 * res_vf)
            elif "squares" in this_dir:
                W = np.sqrt(med_area) * mm_per_px
                S = W / np.sqrt(res_vf)
            else:
                W = None
                S = None
                print("Shape not detected")

            res_areas.append(med_area * mm_per_px ** 2)
            res_sizes.append(W)
            res_spacings.append(S)

        vfs = [np.min(res_vfs), np.median(res_vfs), np.max(res_vfs)]
        Ws = [np.min(res_sizes), np.median(res_sizes), np.max(res_sizes)]
        As = [np.min(res_areas), np.median(res_areas), np.max(res_areas)]
        Ss = [np.min(res_spacings), np.median(res_spacings), np.max(res_spacings)]
        print(f"{this_dir:16s} - {100 * vfs[0]:4.1f} | {100 * vfs[1]:4.1f} | {100 * vfs[2]:4.1f} % "
              f"- {Ws[0]:5.3f} | {Ws[1]:5.3f} | {Ws[2]:5.3f} mm")

        names = ["min_vf", "vf", "max_vf", "min_W", "W", "max_W", "min_A", "A", "max_A", "min_S", "S", "max_S"]
        values = np.concatenate([vfs, Ws, As, Ss])
        for root, _, files in os.walk(root_dir + this_dir + "/"):
            if "params.py" in files:
                for j, (name, value) in enumerate(zip(names, values)):
                    f = open(root + "/params.py", "r+")
                    contents = f.read()
                    if f"\n{name} = " in contents and not overwrite:
                        continue
                    elif f"\n{name} = " in contents:
                        new_contents = contents.split("\n")
                        for i, line in enumerate(new_contents):
                            if "=" not in line:
                                new_contents[i] = ''  # Remove bugged lines
                            else:
                                if line.startswith(f"{name} = "):
                                    new_contents[i] = f"{name} = {value:.3f}  # Calculated by analyse_vf.py"
                                if line != '':
                                    new_contents[i] += "\n"
                        f.seek(0)  # Overwrite contents
                        f.truncate(0)
                        f.writelines(new_contents)
                    else:
                        f.seek(0)
                        if f.read()[-1] != "\n":
                            f.write("\n")
                        f.write(f"{name} = {value:.3f}  # Calculated by analyse_vf.py\n")
                    f.close()

    else:
        print(f"No vf measurement image found for {this_dir}")
