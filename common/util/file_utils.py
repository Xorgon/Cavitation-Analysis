from PyQt5.QtWidgets import QFileDialog, QApplication


def array_to_csv(path, filename, array):
    file = open(path + filename, "w")
    for entry in array:
        line = ""
        for part in entry:
            line += str(part) + ","
        line += "\n"
        file.write(line)
    file.close()


def select_file(start_path, create_window=True):
    if create_window:
        window = QApplication([])
    file_path = str(QFileDialog.getOpenFileName(None, "Select File", start_path)[0])
    return file_path


def select_dir(start_path, create_window=True):
    if create_window:
        window = QApplication([])
    dir_path = str(QFileDialog.getExistingDirectory(None, "Select Directory", start_path)) + "/"
    return dir_path
