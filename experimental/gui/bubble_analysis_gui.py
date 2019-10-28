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
import math
import os
import sys
import importlib

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextBrowser, QMainWindow, \
    QPushButton, QSlider, QSpacerItem, QSizePolicy, QStyle, QFileDialog, QLayout

import experimental.util.calibration_utils as cu
import experimental.util.file_utils as file
import experimental.util.qt_utils as qtu
import experimental.util.mraw_converter as mraw_converter
import common.util.vector_utils as vect


class JumpSlider(QSlider):
    """ From https://stackoverflow.com/a/29639127/5270376 """

    def mousePressEvent(self, ev):
        """ Jump to click position """
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width()))

    def mouseMoveEvent(self, ev):
        """ Jump to pointer position while moving """
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width()))


class PositionSelection(QWidget):
    """
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

        # self.setMinimumHeight(0)
        # self.setMinimumWidth(0)
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

    zoom = None
    mm_per_pixel = None

    positions = None

    position_prefix = None

    def __init__(self, text_area, position_status, file_dir, aux_box: ImageDisplayWidget, main_window: QMainWindow,
                 zoom=2, parent=None, frame_idx=0, max_frames=100):
        if parent:
            super().__init__(parent)
        else:
            super().__init__()
        self.text_area = text_area
        self.position_status = position_status
        self.aux_box = aux_box
        self.main_window = main_window

        self.frame_idx = frame_idx
        self.repeat_idx = 0
        self.position_idx = 0
        self.max_frames = max_frames

        self.zoom = zoom
        self.file_dir = file_dir
        self.set_positions(file_dir)

        self.position_prefix = file.get_prefix_from_idxs(file_dir, np.array(self.positions)[:, 2][0:5])

        self.width = 2 * 384
        self.height = 2 * 264
        self.setMinimumHeight(self.height)
        self.setMinimumWidth(self.width)

        self.update_text()

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

    def paintEvent(self, e):
        # Open movie.
        r_dir_path = self.file_dir + self.position_prefix + str(self.positions[self.position_idx][2]).rjust(4,
                                                                                                            '0') + "/"
        mov = file.get_mraw_from_dir(r_dir_path)
        image = mov[self.frame_idx + self.max_frames * self.repeat_idx]
        self.total_frames = mov.image_count
        mov.close()

        png_path = r_dir_path + "analysis_plot_r{0}.png".format(self.repeat_idx)
        if os.path.exists(png_path):
            self.aux_box.set_image_from_png(png_path)
        else:
            self.aux_box.clear_image()
            self.main_window.update()

        # Convert to pixel map.
        pixmap = qtu.frame_to_pixmap(image)
        pixmap = pixmap.scaledToHeight(pixmap.height() * 2)

        # Draw frame.
        qp = QPainter()
        if qp.isActive():
            qp.setRenderHint(QPainter.Antialiasing)
        qp.begin(self)
        qp.drawPixmap(0, 0, pixmap)

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
        mraw_converter.convert(mov, filename, codec='X264', fps=10,
                               frame_range=(100 * self.repeat_idx, 100 * (self.repeat_idx + 1)), contrast=2)

    def draw_point(self, qp, point):
        pen = QPen(Qt.red, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawPoint(point[0], point[1])

    def draw_line(self, qp):
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(self.point_1[0], self.point_1[1], self.point_2[0], self.point_2[1])

    def next_frame(self):
        self.frame_idx += 1
        if self.frame_idx >= self.max_frames:
            self.frame_idx = self.max_frames - 1
        self.update()

    def previous_frame(self):
        self.frame_idx -= 1
        if self.frame_idx < 0:
            self.frame_idx = 0
        self.update()

    def move_points(self, last_position_idx):
        if self.mm_per_pixel is not None:
            px_per_mm = 1 / self.mm_per_pixel
            last_position = self.positions[last_position_idx]
            this_position = self.positions[self.position_idx]
            if last_position[0] != this_position[0]:
                dx_mm = last_position[0] - this_position[0]
                if self.point_1 is not None:
                    self.point_1[0] += self.zoom * dx_mm * px_per_mm
                if self.point_2 is not None:
                    self.point_2[0] += self.zoom * dx_mm * px_per_mm
            if last_position[1] != this_position[1]:
                dy_mm = last_position[1] - this_position[1]
                if self.point_1 is not None:
                    self.point_1[1] -= self.zoom * dy_mm * px_per_mm
                if self.point_2 is not None:
                    self.point_2[1] -= self.zoom * dy_mm * px_per_mm

    def next_position(self):
        last_position_idx = self.position_idx
        self.position_idx += 1
        self.repeat_idx = 0
        if self.position_idx >= len(self.positions):
            self.position_idx = len(self.positions) - 1

        self.move_points(last_position_idx)
        self.update()

    def previous_position(self):
        last_position_idx = self.position_idx
        self.repeat_idx = 0
        self.position_idx -= 1
        if self.position_idx < 0:
            self.position_idx = 0

        self.move_points(last_position_idx)
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

    def update_text(self):
        # Line 1
        text = "Frame: {0}".format(self.frame_idx)
        text += "    Repeat: {0}".format(self.repeat_idx)
        if self.point_1 is not None:
            text += "    Point 1: {0:.1f}, {1:.1f} (px)".format(self.point_1[0] / self.zoom,
                                                                self.point_1[1] / self.zoom)
            if self.mm_per_pixel is not None:
                text += "  {0:.2f}, {1:.2f} (mm)".format(
                    self.positions[self.position_idx][0] + self.mm_per_pixel * self.point_1[0] / self.zoom,
                    self.positions[self.position_idx][1] + self.mm_per_pixel * (264 - self.point_1[1] / self.zoom))
        if self.point_2 is not None:
            text += "    Point 2: {0:.1f}, {1:.1f} (px)".format(self.point_2[0] / self.zoom,
                                                                self.point_2[1] / self.zoom)
            if self.mm_per_pixel is not None:
                text += "  {0:.2f}, {1:.2f} (mm)".format(
                    self.positions[self.position_idx][0] + self.mm_per_pixel * self.point_2[0] / self.zoom,
                    self.positions[self.position_idx][1] + self.mm_per_pixel * (264 - self.point_2[1] / self.zoom))
        # Line 2
        if self.point_1 is not None and self.point_2 is not None:
            dx = (self.point_2[0] - self.point_1[0]) / self.zoom
            dy = (self.point_2[1] - self.point_1[1]) / self.zoom
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
            self.position_idx = min_r
            self.move_points(last_position_idx)
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
            self.position_idx = min_r
            self.move_points(last_position_idx)
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
            self.position_idx = min_r
            self.move_points(last_position_idx)
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
            self.position_idx = min_r
            self.move_points(last_position_idx)
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


class BubbleAnalyser(QMainWindow):
    def __init__(self, file_dir=None):
        super().__init__()
        while file_dir is None or not os.path.exists(file_dir + "index.csv"):
            file_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", "../../../../Data/"))
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
        label = QLabel()
        self.data_widget = DataWidget(text_area, position_status, file_dir, aux_box, self, parent=label)
        label.setMinimumWidth(self.data_widget.width)
        label.setMinimumHeight(self.data_widget.height)

        position_forward.clicked.connect(self.data_widget.next_position)
        position_back.clicked.connect(self.data_widget.previous_position)
        incr_x.clicked.connect(self.data_widget.incr_x)
        decr_x.clicked.connect(self.data_widget.decr_x)
        incr_y.clicked.connect(self.data_widget.incr_y)
        decr_y.clicked.connect(self.data_widget.decr_y)

        save_repeat = QPushButton("Save Repeat Movie")
        save_repeat.clicked.connect(self.data_widget.save_reading_movie)
        position_box.addWidget(save_repeat)

        # Build calibration widget
        self.calibration = CalibrationWidget(self.data_widget)
        side_bar.addWidget(self.calibration)
        side_bar.addLayout(position_box)

        # Build upper area (image and side buttons)
        upper_box = QHBoxLayout()
        upper_box.addWidget(label)
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

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

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

        # Build window contents
        vbox = QVBoxLayout()  # Main box.
        hbox = QHBoxLayout()  # Box containing the main box and auxiliary box.
        central_widget = QWidget()
        central_widget.setLayout(hbox)
        vbox.addLayout(upper_box)
        vbox.addWidget(slider)
        vbox.addLayout(button_box)
        vbox.addWidget(text_area)
        hbox.addLayout(vbox)
        hbox.addWidget(aux_box)
        self.setCentralWidget(central_widget)
        self.show()
        self.update()

    def set_frame_from_slider(self):
        self.data_widget.set_frame(self.slider.value())

    def next_frame(self):
        self.data_widget.next_frame()
        self.slider.setValue(self.data_widget.frame_idx)

    def previous_frame(self):
        self.data_widget.previous_frame()
        self.slider.setValue(self.data_widget.frame_idx)

    def update(self):
        self.resize(self.minimumSizeHint())
        super().update()

    def flag_invalid_repeat(self):
        f = open(self.data_widget.file_dir + "invalid_readings.txt", "a")
        f.write("{0}:{1}\n".format(self.data_widget.position_idx + 1, self.data_widget.repeat_idx))
        f.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BubbleAnalyser()
    # ex = BubbleAnalyser("../../Data/Test1/HSweep4/")
    sys.exit(app.exec_())
