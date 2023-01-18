import numpy as np

def convert(mraw_obj, outputFile, codec='XVID', fps=24, frame_range=None, scale=1, contrast=1):
    movie = mraw_obj
    if frame_range is None:
        frame_range = (0, len(movie))
    out = cv2.VideoWriter(outputFile,
                          cv2.VideoWriter_fourcc(codec[0],
                                                 codec[1],
                                                 codec[2],
                                                 codec[3]),
                          fps,
                          (movie.width * scale, movie.height * scale),
                          0)
    for i in range(frame_range[0], frame_range[1]):
        frame = np.uint8(np.double(movie[i]) / (2 ** 12) * 255)  # Convert to 8 bit colour.
        if scale != 1:
            frame = cv2.resize(frame, (movie.width * scale, movie.height * scale))
        if contrast != 1:
            frame = cv2.convertScaleAbs(frame, alpha=contrast)
        out.write(frame)
    out.release()


input_dir = "C:/Users/eda1g15/OneDrive - University of Southampton/Research/Slot Geometries/Data/SlotSweeps/W2H3b"
load_readings