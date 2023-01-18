import cv2
import numpy as np


class MP4:
    # Declared to allow IDE to find these properties automatically.
    width, height, image_count = None, None, None

    # Hard code these for compatibility assuming that the movies are from MRAW data using the usual settings.
    frame_rate = 100000
    trigger_out_delay = 100 * 100e-9

    buf = None

    def __init__(self, fn):
        cap = cv2.VideoCapture(fn)
        self.image_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.buf = np.empty((self.image_count, self.height, self.width), np.dtype('uint8'))

        fc = 0
        ret = True

        while fc < self.image_count and ret:
            self.buf[fc] = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            fc += 1

        cap.release()

    def __len__(self):
        return self.image_count

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.get_frame, range(self.image_count)[key])

        return self.get_frame(key)

    def get_frame(self, idx):
        return self.buf[idx]

    def get_fps(self):
        return self.frame_rate
