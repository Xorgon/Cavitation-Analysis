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

from os import path
import numpy as np
import warnings
import xmltodict

# Processing the data in chunks keeps it in the L2 catch of the processor, increasing speed for large arrays by ~50%
CHUNK_SIZE = 6 * 10 ** 5  # Should be divisible by 3, 4 and 5!  This seems to be near-optimal.

SUPPORTED_FILE_FORMATS = ['mraw', 'tiff']
SUPPORTED_EFFECTIVE_BIT_SIDE = ['lower', 'higher']


def read_cih(filename):
    name, ext = path.splitext(filename)
    if ext == '.cih':
        cih = dict()
        # read the cif header
        with open(filename, 'r') as f:
            for line in f:
                if line == '\n':  # end of cif header
                    break
                line_sp = line.replace('\n', '').split(' : ')
                if len(line_sp) == 2:
                    key, value = line_sp
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                        cih[key] = value
                    except:
                        cih[key] = value

    elif ext == '.cihx':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i]]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1] + 1])

        raw_cih_dict = xmltodict.parse(xml)
        cih = {
            'Date': raw_cih_dict['cih']['fileInfo']['date'],
            'Camera Type': raw_cih_dict['cih']['deviceInfo']['deviceName'],
            'Record Rate(fps)': float(raw_cih_dict['cih']['recordInfo']['recordRate']),
            'Shutter Speed(s)': float(raw_cih_dict['cih']['recordInfo']['shutterSpeed']),
            'Total Frame': int(raw_cih_dict['cih']['frameInfo']['totalFrame']),
            'Original Total Frame': int(raw_cih_dict['cih']['frameInfo']['recordedFrame']),
            'Image Width': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['width']),
            'Image Height': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['height']),
            'File Format': raw_cih_dict['cih']['imageFileInfo']['fileFormat'],
            'EffectiveBit Depth': int(raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['depth']),
            'EffectiveBit Side': raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['side'],
            'Color Bit': int(raw_cih_dict['cih']['imageDataInfo']['colorInfo']['bit']),
            'Comment Text': raw_cih_dict['cih']['basicInfo'].get('comment', ''),
            'Signal Delay Trigger Out Width(nsec)': int(raw_cih_dict['cih']['deviceInfo']['delayInfos']['delayInfo'][3]['value']) / 100  # 4th signal is Trigger Out Width in 100ns units
        }

    else:
        raise Exception('Unsupported configuration file ({:s})!'.format(ext))

    # check exceptions
    # ff = cih['File Format']
    # if ff.lower() not in SUPPORTED_FILE_FORMATS:
    #     raise Exception('Unexpected File Format: {:g}.'.format(ff))
    # bits = cih['Color Bit']
    # if bits < 12:
    #     warnings.warn('Not 12bit ({:g} bits)! clipped values?'.format(bits))
        # - may cause overflow')
        # 12-bit values are spaced over the 16bit resolution - in case of photron filming at 12bit
        # this can be meanded by dividing images with //16
    # if cih['EffectiveBit Depth'] != 12:
    #     warnings.warn('Not 12bit image!')
    # ebs = cih['EffectiveBit Side']
    # if ebs.lower() not in SUPPORTED_EFFECTIVE_BIT_SIDE:
    #     raise Exception('Unexpected EffectiveBit Side: {:g}'.format(ebs))
    # if (cih['File Format'].lower() == 'mraw') & (cih['Color Bit'] not in [8, 16]):
    #     raise Exception('pyMRAW only works for 8-bit and 16-bit files!')
    # if cih['Original Total Frame'] > cih['Total Frame']:
    #     warnings.warn('Clipped footage! (Total frame: {}, Original total frame: {})'.format(cih['Total Frame'], cih[
    #         'Original Total Frame']))

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

        if fn[-4:] == '.cih':
            self.fileName = fn[:-4] + '.mraw'
        elif fn[-5:] == '.cihx':
            self.fileName = fn[:-5] + '.mraw'
        else:
            raise ValueError("Input file is neither CIH or CIHX.")
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
