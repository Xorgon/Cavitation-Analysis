"""
A program that displays frames from positions and allows them to be measured.

TODO:
- Allow measurements to be saved for use in other scripts
    -> Ensure measurements allow for reproducibility and repeatability.
    -> Save to file and allow minor variations to assess error.
- Create a better position selection UI (grid of images?).
    -> Maybe make a composite image from all the positions after calibration.
    -> Snap to the nearest position centered on the cursor (allow clicking and dragging).
(- Implement zooming
    -> Zoom parameter exists but zoom center and actual zooming is not implemented.)?
"""

import ctypes
import importlib
import math
import os
import sys
import time

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QIcon, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextBrowser, QMainWindow, \
    QPushButton, QSlider, QSpacerItem, QSizePolicy, QStyle, QFileDialog, QLayout, QCheckBox

import util.vector_utils as vect
import util.analysis_utils as au
import util.calibration_utils as cu
import util.determineDisplacement as dd
import util.file_utils as file
import util.mraw_converter as mraw_converter
import util.qt_utils as qtu
from util.pixel_correction import load_norm_mat


class JumpSlider(QSlider):
    """ From https://stackoverflow.com/a/29639127/5270376 """

    def mousePressEvent(self, ev):
        """ Jump to click position """
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width()))

    def mouseMoveEvent(self, ev):
        """ Jump to pointer position while moving """
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width()))


class Player(QObject):
    frames = None
    fps = None
    skip = False
    frame_update = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, frames=100, fps=30) -> None:
        super().__init__()
        self.frames = frames
        self.fps = fps

    def play(self, i=0):
        last_time = None
        while i < self.frames and not self.skip:
            if last_time is None or time.time() - last_time > 1 / self.fps:
                self.frame_update.emit(i)
                i += 1
                last_time = time.time()
            app.processEvents()
        self.skip = False
        self.finished.emit()


class PositionSelection(QWidget):
    """
    TODO:
    -> Maybe make a composite image from all the positions after calibration.
    -> Snap to the nearest position centered on the cursor (allow clicking and dragging).
    """


class ImageDisplayWidget(QWidget):
    pixmap = None

    height = None
    width = None

    def set_size(self):
        self.setMinimumHeight(self.pixmap.height())
        self.setMinimumWidth(self.pixmap.width())

        self.setMaximumHeight(self.pixmap.height())
        self.setMaximumWidth(self.pixmap.width())

    def set_image(self, image):
        pixmap = qtu.frame_to_pixmap(image)
        pixmap = pixmap.scaledToHeight(pixmap.height())

        self.pixmap = pixmap

        self.set_size()
        self.update()

    def set_image_from_png(self, png_path, scale=0.5):
        self.pixmap = QPixmap(png_path)

        self.pixmap = self.pixmap.scaledToHeight(int(round(self.pixmap.height() * scale)))

        self.set_size()

        self.update()

    def clear_image(self):
        self.pixmap = None
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.gray)
        self.setPalette(p)

        self.update()

    def paintEvent(self, e):
        if self.pixmap is not None:
            # Draw frame.
            qp = QPainter()
            if qp.isActive():
                qp.setRenderHint(QPainter.Antialiasing)
            qp.begin(self)
            qp.drawPixmap(0, 0, self.pixmap)
            qp.end()


class DataWidget(QWidget):
    point_1 = None
    point_2 = None
    frame_idx = None
    repeat_idx = None
    position_idx = None
    total_frames = None
    movie = None
    norm_mat = None

    scale = None
    mm_per_pixel = None

    file_dir = None

    positions = None

    position_prefix = None

    draw_binary = False
    show_debug = False
    raw = False

    def __init__(self, text_area, position_status, file_dir, aux_box: ImageDisplayWidget, main_window: QMainWindow,
                 scale=1, parent=None, frame_idx=0, max_frames=100):
        if parent:
            super().__init__(parent)
        else:
            super().__init__()
        self.text_area = text_area
        self.position_status = position_status
        self.aux_box = aux_box
        self.main_window = main_window

        self.frame_idx = frame_idx
        self.max_frames = max_frames

        self.scale = scale

        self.change_dir(file_dir)

        self.width = self.scale * 384
        self.height = self.scale * 264
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))

        # self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.black)
        # self.setPalette(p)

        self.update_text()

    def sizeHint(self):
        return QSize(384, 264)

    def change_dir(self, file_dir):
        # Reset to zero to avoid jumping into a position that doesn't exist
        self.frame_idx = 0
        self.repeat_idx = 0

        self.file_dir = file_dir
        self.set_positions(file_dir)

        self.norm_mat = load_norm_mat(file_dir)
        if np.std(self.norm_mat) > 0.1:  # Something has probably gone wrong
            print("Warning: Pixel correction matrix (norm_mat) appears to have malfunctioned and so will be ignored.")
            self.norm_mat = None

        self.position_prefix = file.get_prefix_from_idxs(file_dir, np.array(self.positions)[:, 2][0:5])

        self.set_position(0)
        if hasattr(self.main_window, "calibration"):
            self.main_window.calibration.load_calibration()

    def set_positions(self, file_dir):
        index_file = open(file_dir + "index.csv")
        index_lines = index_file.readlines()

        positions = []  # x, y, index

        for i in range(1, len(index_lines)):  # Start at 1 to skip header.
            split = index_lines[i].strip().split(",")
            positions.append([float(split[0]), float(split[1]), split[2]])  # x, y, index number

        # Sort by x
        self.positions = sorted(positions, key=lambda r: float(r[2]))

    def update(self):
        super().update()
        self.update_text()
        self.aux_box.update()
        self.main_window.update()

    def resizeEvent(self, a0: QResizeEvent) -> None:
        last_scale = self.scale
        width_ratio = self.frameGeometry().width() / 384
        height_ratio = self.frameGeometry().height() / 264
        self.scale = min(width_ratio, height_ratio)

        if self.point_1 is not None:
            self.point_1[0] *= self.scale / last_scale
            self.point_1[1] *= self.scale / last_scale

        if self.point_2 is not None:
            self.point_2[0] *= self.scale / last_scale
            self.point_2[1] *= self.scale / last_scale

        # TODO: Fix this so that the dimension with extra space doesn't expand forever
        # self.frameGeometry().setWidth(round(self.scale * 384))
        # self.frameGeometry().setHeight(round(self.scale * 264))

    def set_position(self, position_idx):
        last_position_idx = self.position_idx
        self.position_idx = position_idx
        r_dir_path = self.file_dir + self.position_prefix + str(self.positions[self.position_idx][2]).rjust(4,
                                                                                                            '0') + "/"
        if self.movie is not None:
            self.movie.close()

        self.movie = file.get_mraw_from_dir(r_dir_path)
        self.move_points(last_position_idx)

    def paintEvent(self, e):
        r_dir_path = self.file_dir + self.position_prefix + str(self.positions[self.position_idx][2]).rjust(4,
                                                                                                            '0') + "/"
        image = self.movie[self.frame_idx + self.max_frames * self.repeat_idx]
        bg_img = self.movie[0]
        binary = dd.makeBinary(np.int32(bg_img) - np.int32(image))
        self.total_frames = self.movie.image_count

        movie_min, movie_max = np.min(image), np.max(image)

        png_path = r_dir_path + "analysis_plot_r{0}.png".format(self.repeat_idx)
        if os.path.exists(png_path):
            self.aux_box.set_image_from_png(png_path)
        else:
            self.aux_box.clear_image()
            self.main_window.update()

        # Convert to pixel map.
        if self.draw_binary:
            pixmap = qtu.frame_to_pixmap(np.int32(binary), autoscale=True)
        else:
            pixmap = qtu.frame_to_pixmap(image, autoscale=not self.raw, min=movie_min, max=movie_max,
                                         norm_mat=None if self.raw else self.norm_mat)
        pixmap = pixmap.scaledToHeight(pixmap.height() * self.scale)

        # Draw frame.
        qp = QPainter()
        if qp.isActive():
            qp.setRenderHint(QPainter.Antialiasing)
        qp.begin(self)
        qp.drawPixmap(0, 0, pixmap)

        # Draw bubble position
        if self.show_debug:
            b_pos_x, b_pos_y, area, ecc, sol, jet_tip = au.analyse_frame(image, bg_img, debug=True)

            if b_pos_x is not None and b_pos_y is not None and jet_tip is not None:
                self.draw_line(qp, [(b_pos_x + 1) * self.scale, (b_pos_y + 1) * self.scale],
                               [(jet_tip[0] + 1) * self.scale, (jet_tip[1] + 1) * self.scale], colour=Qt.green)

            if b_pos_x is not None and b_pos_y is not None:
                self.draw_point(qp, [(b_pos_x + 1) * self.scale, (b_pos_y + 1) * self.scale],
                                Qt.magenta)

            if area is not None:
                self.draw_circle(qp, [(b_pos_x + 1) * self.scale, (b_pos_y + 1) * self.scale],
                                 self.scale * np.sqrt(area / np.pi), Qt.magenta)

            if b_pos_x is not None and b_pos_y is not None \
                    and ecc is not None and sol is not None and jet_tip is not None:
                if area is None:
                    qp.drawText(b_pos_x * self.scale + 5,
                                b_pos_y * self.scale + 5, f"ecc = {ecc:.2f}, sol = {sol:.2f}")
                else:
                    tip_ratio = np.linalg.norm(np.subtract(jet_tip, [b_pos_x, b_pos_y])) / np.sqrt(area / np.pi)
                    qp.drawText((b_pos_x + np.sqrt(area / np.pi)) * self.scale,
                                (b_pos_y + np.sqrt(area / np.pi)) * self.scale,
                                f"ecc = {ecc:.2f}, sol = {sol:.2f}, R = {np.sqrt(area / np.pi):.2f}, tr = {tip_ratio:.2f}")

        # Draw ruler.
        if self.point_1 is not None and self.point_2 is None:
            self.draw_point(qp, self.point_1)
        if self.point_1 is not None and self.point_2 is not None:
            self.draw_line(qp)

        qp.end()

    def save_reading_movie(self):
        r_dir_path = self.file_dir + self.position_prefix + str(self.positions[self.position_idx][2]).rjust(4,
                                                                                                            '0') + "/"
        mov = file.get_mraw_from_dir(r_dir_path)

        filename = str(QFileDialog.getSaveFileName(self, "Select Save Directory",
                                                   r_dir_path + "video" + "_" + str(self.repeat_idx) + ".mp4")[0])
        if filename == "":
            return
        if filename[-4:] != ".mp4":
            filename += ".mp4"

        # These settings make OpenCV compain about stuff, but they still give much nicer quality.
        if self.raw:
            mraw_converter.convert(mov, filename, codec='avc1', fps=24,
                                   frame_range=(100 * self.repeat_idx, 100 * (self.repeat_idx + 1) - 1),
                                   autoscale_brightness=False, norm_mat=None, writer='cv2')
        else:
            mraw_converter.convert(mov, filename, codec='avc1', fps=24,
                                   frame_range=(100 * self.repeat_idx, 100 * (self.repeat_idx + 1) - 1),
                                   autoscale_brightness=True, norm_mat=self.norm_mat, writer='cv2')

    def draw_point(self, qp, point, colour=Qt.red):
        pen = QPen(colour, 4, Qt.SolidLine)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(pen)
        qp.drawPoint(point[0], point[1])

    def draw_line(self, qp, p1=None, p2=None, colour=Qt.red):
        pen = QPen(colour, 2, Qt.SolidLine)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(pen)
        if p1 is None or p2 is None:
            qp.drawLine(self.point_1[0], self.point_1[1], self.point_2[0], self.point_2[1])
        else:
            qp.drawLine(p1[0], p1[1], p2[0], p2[1])

    def draw_circle(self, qp, center, radius, colour=Qt.red):
        pen = QPen(colour, 2, Qt.SolidLine)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(pen)
        qp.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)

    def next_frame(self):
        self.frame_idx += 1
        if self.frame_idx >= self.max_frames:
            self.frame_idx = 0
        self.update()

    def previous_frame(self):
        self.frame_idx -= 1
        if self.frame_idx < 0:
            self.frame_idx = self.max_frames - 1
        self.update()

    def move_points(self, last_position_idx):
        if self.mm_per_pixel is not None:
            px_per_mm = 1 / self.mm_per_pixel
            last_position = self.positions[last_position_idx]
            this_position = self.positions[self.position_idx]
            if last_position[0] != this_position[0]:
                dx_mm = last_position[0] - this_position[0]
                if self.point_1 is not None:
                    self.point_1[0] += self.scale * dx_mm * px_per_mm
                if self.point_2 is not None:
                    self.point_2[0] += self.scale * dx_mm * px_per_mm
            if last_position[1] != this_position[1]:
                dy_mm = last_position[1] - this_position[1]
                if self.point_1 is not None:
                    self.point_1[1] -= self.scale * dy_mm * px_per_mm
                if self.point_2 is not None:
                    self.point_2[1] -= self.scale * dy_mm * px_per_mm

    def next_position(self):
        self.repeat_idx = 0
        if self.position_idx + 1 >= len(self.positions):
            self.set_position(0)
        else:
            self.set_position(self.position_idx + 1)
        self.update()

    def previous_position(self):
        self.repeat_idx = 0
        if self.position_idx - 1 < 0:
            self.set_position(len(self.positions) - 1)
        else:
            self.set_position(self.position_idx - 1)
        self.update()

    def next_repeat(self):
        self.repeat_idx += 1
        if self.frame_idx + self.repeat_idx * self.max_frames >= self.total_frames:
            self.repeat_idx = int(self.total_frames / self.max_frames) - 1
        self.update()

    def previous_repeat(self):
        self.repeat_idx -= 1
        if self.repeat_idx < 0:
            self.repeat_idx = 0
        self.update()

    def set_frame(self, frame_idx):
        self.frame_idx = frame_idx
        self.update()

    def set_draw_binary(self, draw_binary):
        self.draw_binary = draw_binary
        self.update()

    def set_show_debug(self, show_debug):
        self.show_debug = show_debug
        self.update()

    def set_raw(self, raw):
        self.raw = raw
        self.update()

    def update_text(self):
        # Line 1
        text = "Frame: {0}".format(self.frame_idx)
        text += "    Repeat: {0}".format(self.repeat_idx)
        if self.point_1 is not None:
            text += "    Point 1: {0:.1f}, {1:.1f} (px)".format(self.point_1[0] / self.scale,
                                                                self.point_1[1] / self.scale)
            if self.mm_per_pixel is not None:
                text += "  {0:.2f}, {1:.2f} (mm)".format(
                    self.positions[self.position_idx][0] + self.mm_per_pixel * self.point_1[0] / self.scale,
                    self.positions[self.position_idx][1] + self.mm_per_pixel * (264 - self.point_1[1] / self.scale))
        if self.point_2 is not None:
            text += "    Point 2: {0:.1f}, {1:.1f} (px)".format(self.point_2[0] / self.scale,
                                                                self.point_2[1] / self.scale)
            if self.mm_per_pixel is not None:
                text += "  {0:.2f}, {1:.2f} (mm)".format(
                    self.positions[self.position_idx][0] + self.mm_per_pixel * self.point_2[0] / self.scale,
                    self.positions[self.position_idx][1] + self.mm_per_pixel * (264 - self.point_2[1] / self.scale))
        # Line 2
        if self.point_1 is not None and self.point_2 is not None:
            dx = (self.point_2[0] - self.point_1[0]) / self.scale
            dy = (self.point_2[1] - self.point_1[1]) / self.scale
            length = math.sqrt(dx ** 2 + dy ** 2)
            text += "\nLength (px): {0:.2f}".format(length)
            text += "    dx (px) = {0:.2f}    dy (px) = {1:.2f}".format(dx, dy)

            # y measured backwards (point_2 to point_1) to correct for inverted axis.
            text += "    Angle (rad): {0:.4f}".format(math.atan2(-dy, dx))

            # Line 3
            if self.mm_per_pixel is not None:
                text += "\nLength (mm): {0:.2f}".format(length * self.mm_per_pixel)
                text += "    dx (mm) = {0:.2f}    dy (mm) = {1:.2f}".format(dx * self.mm_per_pixel,
                                                                            -dy * self.mm_per_pixel)

        self.text_area.setText(text)

        # Position status
        position_status_text = "x = {0:.2f}" \
                               "\ny = {1:.2f}" \
                               "\nidx = {2}" \
            .format(self.positions[self.position_idx][0], self.positions[self.position_idx][1],
                    self.positions[self.position_idx][2])
        self.position_status.setText(position_status_text)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.point_1 is None and self.point_2 is None:
                self.point_1 = [event.x(), event.y()]
            elif self.point_1 is not None and self.point_2 is None:
                self.point_2 = [event.x(), event.y()]
            else:
                self.point_1 = [event.x(), event.y()]
                self.point_2 = None
            self.update()
        if event.buttons() & Qt.RightButton:
            self.point_1 = None
            self.point_2 = None
            self.update()
        self.update_text()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.point_1 is not None:
                self.point_2 = [event.x(), event.y()]
                self.update()

    def get_current_frame_data(self):
        x = self.positions[self.position_idx][0]
        y = self.positions[self.position_idx][1]
        mov = file.get_mraw_from_dir(
            self.file_dir + self.position_prefix + str(self.positions[self.position_idx][2]).rjust(4, '0') + "/")
        image = mov[self.frame_idx + self.max_frames * self.repeat_idx]
        self.total_frames = mov.image_count
        mov.close()
        return x, y, image

    def incr_y(self):
        cur_x = self.positions[self.position_idx][0]
        cur_y = self.positions[self.position_idx][1]

        min_y_dif = None  # type: float
        min_x_dif = None  # type: float
        min_r = None
        for r in range(len(self.positions)):
            position_y = self.positions[r][1]
            position_x = self.positions[r][0]
            if position_y > cur_y:
                y_dif = abs(position_y - cur_y)
                x_dif = abs(position_x - cur_x)
                if min_y_dif is None or y_dif < min_y_dif:
                    min_x_dif = x_dif
                    min_y_dif = y_dif
                    min_r = r
                elif y_dif == min_y_dif and x_dif < min_x_dif:
                    min_x_dif = x_dif
                    min_r = r
        if min_r is not None:
            last_position_idx = self.position_idx
            self.set_position(min_r)
            self.repeat_idx = 0
            self.update()

    def decr_y(self):
        cur_x = self.positions[self.position_idx][0]
        cur_y = self.positions[self.position_idx][1]

        min_y_dif = None  # type: float
        min_x_dif = None  # type: float
        min_r = None
        for r in range(len(self.positions)):
            position_y = self.positions[r][1]
            position_x = self.positions[r][0]
            if position_y < cur_y:
                y_dif = abs(position_y - cur_y)
                x_dif = abs(position_x - cur_x)
                if min_y_dif is None or y_dif < min_y_dif:
                    min_x_dif = x_dif
                    min_y_dif = y_dif
                    min_r = r
                elif y_dif == min_y_dif and x_dif < min_x_dif:
                    min_x_dif = x_dif
                    min_r = r
        if min_r is not None:
            last_position_idx = self.position_idx
            self.set_position(min_r)
            self.repeat_idx = 0
            self.update()

    def incr_x(self):
        cur_x = self.positions[self.position_idx][0]
        cur_y = self.positions[self.position_idx][1]

        min_y_dif = None  # type: float
        min_x_dif = None  # type: float
        min_r = None
        for r in range(len(self.positions)):
            position_y = self.positions[r][1]
            position_x = self.positions[r][0]
            if position_x > cur_x:
                y_dif = abs(position_y - cur_y)
                x_dif = abs(position_x - cur_x)
                if min_x_dif is None or x_dif < min_x_dif:
                    min_x_dif = x_dif
                    min_y_dif = y_dif
                    min_r = r
                elif x_dif == min_x_dif and y_dif < min_y_dif:
                    min_y_dif = y_dif
                    min_r = r
        if min_r is not None:
            last_position_idx = self.position_idx
            self.set_position(min_r)
            self.repeat_idx = 0
            self.update()

    def decr_x(self):
        cur_x = self.positions[self.position_idx][0]
        cur_y = self.positions[self.position_idx][1]

        min_y_dif = None  # type: float
        min_x_dif = None  # type: float
        min_r = None
        for r in range(len(self.positions)):
            position_y = self.positions[r][1]
            position_x = self.positions[r][0]
            if position_x < cur_x:
                y_dif = abs(position_y - cur_y)
                x_dif = abs(position_x - cur_x)
                if min_x_dif is None or x_dif < min_x_dif:
                    min_x_dif = x_dif
                    min_y_dif = y_dif
                    min_r = r
                elif x_dif == min_x_dif and y_dif < min_y_dif:
                    min_y_dif = y_dif
                    min_r = r
        if min_r is not None:
            last_position_idx = self.position_idx
            self.set_position(min_r)
            self.repeat_idx = 0
            self.update()


class CalibrationWidget(QWidget):
    calibration_frames = None  # x, y, frame
    text_area = None
    mm_per_pixel = None

    def __init__(self, data_widget: DataWidget, parent=None):
        if parent:
            super().__init__(parent)
        else:
            super().__init__()

        self.data_widget = data_widget
        self.calibration_frames = []

        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignTop)
        vbox.setSizeConstraint(QLayout.SetFixedSize)

        label = QLabel("Calibration")
        label.setStyleSheet("QLabel { font-weight: bold; }")
        label.setAlignment(Qt.AlignHCenter)
        label.setMaximumHeight(16)

        text_area = QTextBrowser()
        text_area.setMaximumHeight(36)
        text_area.setMaximumWidth(100)
        self.text_area = text_area

        add_frame_button = QPushButton("Add Frame")
        add_frame_button.clicked.connect(self.take_frame)
        clear_frames_button = QPushButton("Clear Frames")
        clear_frames_button.clicked.connect(self.clear_frames)
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.calibrate)

        vbox.addWidget(label)
        vbox.addWidget(add_frame_button)
        vbox.addWidget(clear_frames_button)
        vbox.addWidget(calibrate_button)
        vbox.addWidget(text_area)
        self.setLayout(vbox)

        self.load_calibration()
        self.update_text()

    def take_frame(self):
        self.calibration_frames.append(self.data_widget.get_current_frame_data())
        self.update_text()

    def clear_frames(self):
        self.calibration_frames.clear()
        self.update_text()

    def load_calibration(self):
        file_dir = self.data_widget.file_dir
        if os.path.exists(file_dir + "params.py"):
            sys.path.append(file_dir)
            import params
            importlib.reload(params)
            sys.path.remove(file_dir)

            if params.mm_per_px:
                self.mm_per_pixel = params.mm_per_px
                self.data_widget.mm_per_pixel = params.mm_per_px
                self.update_text()

    def calibrate(self):
        mm_per_pixels = []
        for i in range(1, len(self.calibration_frames)):
            x_1 = self.calibration_frames[i - 1][0]
            y_1 = self.calibration_frames[i - 1][1]

            x_2 = self.calibration_frames[i][0]
            y_2 = self.calibration_frames[i][1]

            if x_2 == x_1 and y_2 == y_1:
                self.text_area.setText("Warning!\nDuplicate frame.")
                continue

            mm = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

            px_offset = cu.calculate_offset(self.calibration_frames[i - 1][2], self.calibration_frames[i][2])
            print(px_offset)
            px = vect.mag(px_offset)
            if px == 0:
                self.text_area.setText("Error!\nNo offset found.")
                continue
            mm_per_pixels.append(mm / px)

        if len(mm_per_pixels) == 0:
            return
        self.mm_per_pixel = np.mean(mm_per_pixels)
        self.data_widget.mm_per_pixel = self.mm_per_pixel
        self.update_text()

    def update_text(self):
        text = "Frames: {0}".format(len(self.calibration_frames))
        if self.mm_per_pixel is not None:
            text += "\nmm/px = {0:.4f}".format(self.mm_per_pixel)
        self.text_area.setText(text)


class BubbleAnalyser(QMainWindow, QObject):
    player_play_signal = pyqtSignal(int)

    def __init__(self, file_dir=None):
        QMainWindow.__init__(self)
        QObject.__init__(self)
        while file_dir is None or not os.path.exists(file_dir + "index.csv"):
            file_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", "../"))
            if file_dir == "":
                QTimer.singleShot(0, self.close)
                return
            else:
                file_dir += "/"

        self.title = 'Bubble Analysis - ' + file_dir

        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon('icon.png'))

        # Reset the App ID to interact more nicely with Windows (separate icon from generic Python etc.).
        myappid = u'bubble_analysis.py'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        # Build text area
        text_area = QTextBrowser()
        text_area.setMaximumHeight(52)

        # Build side bar area.
        side_bar = QVBoxLayout()
        side_bar.setAlignment(Qt.AlignTop)

        # Build position box
        position_box = QVBoxLayout()
        position_box.setContentsMargins(8, 0, 8, 0)
        position_box.setSizeConstraint(QLayout.SetFixedSize)
        position_label = QLabel("Position")
        position_label.setStyleSheet("QLabel { font-weight: bold; }")
        position_label.setAlignment(Qt.AlignHCenter)
        position_label.setMaximumHeight(16)
        position_box.setAlignment(Qt.AlignTop)

        position_back = QPushButton("Previous")
        position_back.setAutoRepeat(True)
        position_forward = QPushButton("Next")
        position_forward.setAutoRepeat(True)

        incr_x = QPushButton("x +")
        incr_x.setAutoRepeat(True)
        incr_x.setMaximumWidth(40)
        decr_x = QPushButton("x -")
        decr_x.setAutoRepeat(True)
        decr_x.setMaximumWidth(40)
        incr_y = QPushButton("y +")
        incr_y.setAutoRepeat(True)
        incr_y.setMaximumWidth(40)
        decr_y = QPushButton("y -")
        decr_y.setAutoRepeat(True)
        decr_y.setMaximumWidth(40)
        position_buttons = QVBoxLayout()
        position_buttons.setAlignment(Qt.AlignHCenter)
        x_buttons = QHBoxLayout()
        x_buttons.addWidget(decr_x)
        x_buttons.addWidget(incr_x)
        incr_y_wrapper = QHBoxLayout()
        incr_y_wrapper.addWidget(incr_y)
        decr_y_wrapper = QHBoxLayout()
        decr_y_wrapper.addWidget(decr_y)
        position_buttons.addLayout(incr_y_wrapper)
        position_buttons.addLayout(x_buttons)
        position_buttons.addLayout(decr_y_wrapper)

        flag_repeat = QPushButton("Flag Invalid Repeat")
        flag_repeat.clicked.connect(self.flag_invalid_repeat)

        position_status = QTextBrowser()
        position_status.setMaximumHeight(52)
        position_status.setMaximumWidth(100)
        position_box.addWidget(position_label)
        position_box.addWidget(position_forward)
        position_box.addWidget(position_back)
        position_box.addWidget(position_status)
        position_box.addLayout(position_buttons)
        position_box.addWidget(flag_repeat)

        # Build auxiliary box
        aux_box = ImageDisplayWidget()

        # Build image area
        self.data_widget = DataWidget(text_area, position_status, file_dir, aux_box, self)

        position_forward.clicked.connect(self.data_widget.next_position)
        position_back.clicked.connect(self.data_widget.previous_position)
        incr_x.clicked.connect(self.data_widget.incr_x)
        decr_x.clicked.connect(self.data_widget.decr_x)
        incr_y.clicked.connect(self.data_widget.incr_y)
        decr_y.clicked.connect(self.data_widget.decr_y)

        save_repeat = QPushButton("Save Repeat Movie")
        save_repeat.clicked.connect(self.data_widget.save_reading_movie)
        position_box.addWidget(save_repeat)

        dir_change = QPushButton("Change Directory")
        dir_change.clicked.connect(self.select_dir)
        position_box.addWidget(dir_change)

        # Build calibration widget
        self.calibration = CalibrationWidget(self.data_widget)
        side_bar.addWidget(self.calibration)
        side_bar.addLayout(position_box)

        # Build upper area (image and side buttons)
        upper_box = QHBoxLayout()
        # upper_box.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Preferred, QSizePolicy.Minimum))
        upper_box.addWidget(self.data_widget)
        # upper_box.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Preferred, QSizePolicy.Minimum))
        upper_box.addLayout(side_bar)

        # Build button area
        slider = JumpSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(1)
        slider.setValue(0)
        slider.setMinimum(0)
        slider.setMaximum(99)
        slider.valueChanged.connect(self.set_frame_from_slider)
        self.slider = slider

        self.player_thread = QThread()
        self.player = Player()
        self.player.moveToThread(self.player_thread)
        self.player.frame_update.connect(self.set_frame_from_player)
        self.player.finished.connect(self.on_player_finished)
        self.player_play_signal.connect(self.player.play)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)

        button_box = QHBoxLayout()
        frame_back = QPushButton("Previous Frame")
        frame_back.setAutoRepeat(True)
        frame_back.clicked.connect(self.previous_frame)
        button_box.addWidget(frame_back)
        frame_forward = QPushButton("Next Frame")
        frame_forward.setAutoRepeat(True)
        frame_forward.clicked.connect(self.next_frame)
        button_box.addWidget(frame_forward)

        button_box.addItem(spacer)

        rep_back = QPushButton("Previous Repeat")
        rep_back.setAutoRepeat(True)
        rep_back.clicked.connect(self.data_widget.previous_repeat)
        button_box.addWidget(rep_back)
        rep_forward = QPushButton("Next Repeat")
        rep_forward.setAutoRepeat(True)
        rep_forward.clicked.connect(self.data_widget.next_repeat)
        button_box.addWidget(rep_forward)

        button_box.addItem(spacer)

        self.loop_toggle = QCheckBox("Loop")
        self.loop_toggle.stateChanged.connect(lambda _: self.toggle_loop(self.loop_toggle.isChecked()))
        button_box.addWidget(self.loop_toggle)

        binary_toggle = QCheckBox("Binary")
        binary_toggle.stateChanged.connect(lambda _: self.data_widget.set_draw_binary(binary_toggle.isChecked()))
        button_box.addWidget(binary_toggle)

        debug_toggle = QCheckBox("Debug")
        debug_toggle.stateChanged.connect(lambda _: self.data_widget.set_show_debug(debug_toggle.isChecked()))
        button_box.addWidget(debug_toggle)

        auto_bright_toggle = QCheckBox("Raw Output")
        auto_bright_toggle.setChecked(False)
        auto_bright_toggle.stateChanged.connect(
            lambda _: self.data_widget.set_raw(auto_bright_toggle.isChecked()))
        button_box.addWidget(auto_bright_toggle)

        self.rand_toggle = QCheckBox("Randomise")
        button_box.addWidget(self.rand_toggle)

        # Build window contents
        vbox = QVBoxLayout()  # Main box.
        hbox = QHBoxLayout()  # Box containing the main box and auxiliary box.
        central_widget = QWidget()
        central_widget.setLayout(hbox)

        # Put together the main box
        vbox.addLayout(upper_box)
        vbox.addWidget(slider)
        vbox.addLayout(button_box)
        vbox.addWidget(text_area)

        # Add the auxiliary box
        hbox.addLayout(vbox)
        hbox.addWidget(aux_box)

        self.setCentralWidget(central_widget)
        self.show()
        self.update()
        self.player_thread.start()

    def set_frame_from_slider(self):
        self.data_widget.set_frame(self.slider.value())

    def set_frame_from_player(self, i):
        self.data_widget.set_frame(i)
        self.slider.setValue(i)

    def toggle_loop(self, loop):
        if loop:
            self.slider.setEnabled(False)
            self.player_play_signal.emit(self.slider.value())
        else:
            self.player.skip = True  # This is probably pretty unhealthy but it'll do

    def on_player_finished(self):
        if self.loop_toggle.isChecked():
            if self.rand_toggle.isChecked():
                rand_pos = np.random.randint(len(self.data_widget.positions))
                self.data_widget.set_position(rand_pos)
                rand_rep = np.random.randint(int(self.data_widget.total_frames / self.data_widget.max_frames))
                self.data_widget.repeat_idx = rand_rep
                self.data_widget.update()

            self.player_play_signal.emit(0)
        else:
            self.slider.setEnabled(True)

    def next_frame(self):
        self.data_widget.next_frame()
        self.slider.setValue(self.data_widget.frame_idx)

    def previous_frame(self):
        self.data_widget.previous_frame()
        self.slider.setValue(self.data_widget.frame_idx)

    def update(self):
        # self.resize(self.minimumSizeHint())
        super().update()

    def flag_invalid_repeat(self):
        f = open(self.data_widget.file_dir + "invalid_readings.txt", "a")
        f.write("{0}:{1}\n".format(self.data_widget.positions[self.data_widget.position_idx][2],
                                   self.data_widget.repeat_idx))
        f.close()

    def select_dir(self):
        file_dir = None
        while file_dir is None or not os.path.exists(file_dir + "index.csv"):
            file_dir = str(
                QFileDialog.getExistingDirectory(self, "Select Directory", self.data_widget.file_dir + "../"))
            if file_dir == "":
                return
            else:
                file_dir += "/"

        self.calibration.clear_frames()
        self.calibration.load_calibration()
        self.data_widget.change_dir(file_dir)
        self.slider.setSliderPosition(0)
        self.title = 'Bubble Analysis - ' + file_dir
        self.setWindowTitle(self.title)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Slash:
            self.loop_toggle.setChecked(not self.loop_toggle.isChecked())
        if e.key() == Qt.Key_Comma:
            self.data_widget.previous_frame()
            self.slider.setValue(self.data_widget.frame_idx)
        if e.key() == Qt.Key_Period:
            self.data_widget.next_frame()
            self.slider.setValue(self.data_widget.frame_idx)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BubbleAnalyser()
    # ex.show()
    # ex = BubbleAnalyser("../../Data/Test1/HSweep4/")
    sys.exit(app.exec_())
