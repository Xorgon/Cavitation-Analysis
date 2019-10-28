"""
Photron MRAW reader
Author: Ivo Peters (i.r.peters@soton.ac.uk)
Created on:    2016-10-26
Last revision: 2016-10-27
Version: 0.2

Chunks of code taken from:
CINE library: Dustin Kleckner (dkleckner@uchicago.edu)
pyMRAW.py: Jaka Javh (jaka.javh@fs.uni-lj.si)
"""

import numpy as np

# Processing the data in chunks keeps it in the L2 catch of the processor, increasing speed for large arrays by ~50%
CHUNK_SIZE = 6 * 10 ** 5  # Should be divisible by 3, 4 and 5!  This seems to be near-optimal.


def read_cih(filename):
    # create a dictionary which will contain all the properties defined in the cih file
    cih = dict()

    # read the cih (Camera Information Header)
    with open(filename, 'r') as f:
        # work thorugh the file line by line
        for line in f:
            if line == '\n':  # end of cif header
                break
            # split the line in property and value pairs (line_sp is a list with two entries)
            line_sp = line.replace('\n', '').split(' : ')
            # check if line_sp indeed contains a property name and a value
            if len(line_sp) == 2:
                key, value = line_sp
                # try to turn the value into a number (float or integer):
                try:
                    if '.' in value:
                        cih[key] = float(value)
                    else:
                        cih[key] = int(value)
                # otherwise the value is a string:
                except:
                    cih[key] = value
    return cih


def twelve2sixteen(a):
    b = np.zeros(a.size // 3 * 2, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2) = [a[j + i:j + CHUNK_SIZE:3].astype('u2') for i in range(3)]

        k = j // 3 * 2
        k2 = k + CHUNK_SIZE // 3 * 2

        b[k + 0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
        b[k + 1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)

    return b


class mraw(object):
    # Declared to allow IDE to find these properties automatically.
    width, height, imageSize, image_count = None, None, None, None

    def __init__(self, fn):
        cih = read_cih(fn)
        self.cih = cih

        self.width = cih['Image Width']
        self.height = cih['Image Height']
        self.image_count = cih['Total Frame']
        self.bit_depth = cih['EffectiveBit Depth']
        self.frame_rate = float(cih[
                                    'Record Rate(fps)'])  # turn the frame rate into a float to avoid problems with creating unwanted integers by accident when using this property
        self.trigger_out_delay = float(cih['Signal Delay Trigger Out Width(nsec)']) * 1e-9

        # calculate the size of an image in bytes:
        self.imageSize = self.width * self.height * self.bit_depth // 8

        self.fileName = fn[:-4] + '.mraw'
        self.f = open(self.fileName, 'rb')

    def __len__(self):
        return self.image_count

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.get_frame, range(self.image_count)[key])

        return self.get_frame(key)

    def get_fps(self):
        return self.frame_rate

    def get_frame(self, number):
        # calculate the position in bytes where the image is:
        bytePosition = number * self.imageSize
        # seek this position in the file:
        self.f.seek(bytePosition)
        # image = np.memmap(f, dtype=np.uint8, mode='r', shape=(self.width * self.height * 12 / 8))
        image = np.frombuffer(self.f.read(self.imageSize), np.uint8)
        image = twelve2sixteen(image)
        image = np.reshape(image, (self.height, self.width))
        return image

    def close(self):
        self.f.close()
