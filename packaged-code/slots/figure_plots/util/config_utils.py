import sys
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QCheckBox, QPushButton


class ConfigDialog(QDialog):
    def __init__(self, config_dict: dict, *args, **kwargs):
        super(ConfigDialog, self).__init__(*args, *kwargs)
        self.config_dict = config_dict
        self.check_boxes = []

        layout = QVBoxLayout()

        for key in config_dict.keys():
            if type(key) is not str:
                raise ValueError("Configuration key must be of type 'str'.")
            if type(config_dict[key]) == bool:
                check_box = QCheckBox(key)
                check_box.setChecked(config_dict[key])
                layout.addWidget(check_box)
                self.check_boxes.append(check_box)

        button = QPushButton("Plot")
        button.clicked.connect(self.finish_config)
        layout.addWidget(button)

        self.setLayout(layout)

    def finish_config(self):
        for check_box in self.check_boxes:  # type: QCheckBox
            self.config_dict[check_box.text()] = check_box.isChecked()
        self.accept()


def get_config(config_dict, create_window=True):
    if create_window:
        window = QApplication([])
    config_dialog = ConfigDialog(config_dict)
    config_dialog.exec_()
    return config_dialog.config_dict


if __name__ == "__main__":
    get_config({"test a": False, "test b": True})
