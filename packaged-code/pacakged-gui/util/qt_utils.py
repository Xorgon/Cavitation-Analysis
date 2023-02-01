import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from util.pixel_correction import safe_correct


def frame_to_pixmap(frame, autoscale=True, min=None, max=None, norm_mat=None):
    if norm_mat is not None:
        frame = safe_correct(frame, norm_mat)
    if autoscale:
        if min is not None and max is not None:
            img_8bit = ((frame - min) / ((max - min) / 255.0)).astype(np.uint8)  # map the data range to 0 - 255
        else:
            img_8bit = ((frame - frame.min()) / (frame.ptp() / 255.0)).astype(np.uint8)  # map the data range to 0 - 255
    else:
        img_8bit = (frame / (4095 / 255)).astype(np.uint8)
    img = QImage(img_8bit, img_8bit.shape[1], img_8bit.shape[0], QImage.Format_Grayscale8)
    pixmap = QPixmap(img)
    return pixmap
