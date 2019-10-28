import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def frame_to_pixmap(frame):
    img_8bit = ((frame - frame.min()) / (frame.ptp() / 255.0)).astype(np.uint8)  # map the data range to 0 - 255
    img = QImage(img_8bit, img_8bit.shape[1], img_8bit.shape[0], QImage.Format_Grayscale8)
    pixmap = QPixmap(img)
    return pixmap
