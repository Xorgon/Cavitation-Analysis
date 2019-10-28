# Bubble Analysis GUI
`bubble_analysis_gui.py` is a GUI program written with PyQt5 that allows a series of readings to be inspected and analysed easily. A demonstration can be found [here](https://xorg.us/2018-07-18_12-07-23.mp4).

#### Instructions
1) On running the script a file selection prompt will appear. Select the folder that contains the index.csv file.
2) Find as many readings in your data that contain the same geometry and have a common coordinate (for example a constant y). Navigate to an empty frame (no bubble) and click 'Add Frame' under 'Calibration'. If you select incorrect frames you can restart by clicking 'Clear Frames'. When you have selected a series of frames click 'Calibrate'. This will take a moment to run and will result in a 'mm/px' value being shown in the 'Calibration' status box.
3) You can navigate through your data with the navigation buttons provided or slide the slider to advance throug the frames.
4) You can click on a frame to take measurements, left click once at the start of your measurement and again at the end. A red line will appear and the status box at the bottom of the window will show details about your measurement. The angle noted is measured from the positive horizontal (x) direction anti-clockwise (Note: the order in which you choose your points is important in angle measurement). Right clicking will clear your measurement.

#### Recording Data
To use this analysis tool data must be recorded in the correct format. <Insert explanation>
- csv with header etc.
- Individual folders, filenames etc.