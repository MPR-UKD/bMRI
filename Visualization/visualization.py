import sys

import PIL.Image
from scipy import ndimage
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QWidget,
    QSlider,
    QHBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def calc_scaling_factor(dicom_shape):
    return 1000 // max(dicom_shape[1], dicom_shape[2])


class ImageViewer(QMainWindow):
    def __init__(self, dicom, fit_maps, fit_function, time_points):
        super(ImageViewer, self).__init__()

        self.echo_time = 0
        self.time_points = time_points
        self.dicom = dicom
        self.fit_maps = fit_maps
        self.fit_function = fit_function
        self.scaling_factor = calc_scaling_factor(dicom.shape)
        self.current_slice = 0
        self.current_params = self.fit_maps[:, :, :, self.current_slice]

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.image_label.mousePressEvent = self.update_fit_function
        #self.image_label.mouseMoveEvent = self.update_fit_function
        self.scaling_factor = calc_scaling_factor(dicom.shape)
        width, height = dicom.shape[1] * self.scaling_factor, dicom.shape[2] * self.scaling_factor
        self.image_label.setFixedSize(width, height)


        # Display the first slice
        self.display_slice()

        # Create a horizontal slider to change the slice
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.dicom.shape[-1] - 1)
        self.slider.valueChanged.connect(self.change_slice)

        # Create a container widget to hold the plot of the fit function
        self.plot_container = QWidget(self)

        # Set up the axes, but don't plot any data initially
        self.x_fit = np.linspace(0, self.time_points[-1], 1000)

        # Set the layout of the FitFunctionWidget
        layout = QHBoxLayout()
        self.plot_container.setLayout(layout)

        self.init_fit_function()

        # Set the layout of the ImageViewer
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.plot_container)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def change_slice(self, slice_num):
        self.current_slice = slice_num
        self.current_params = self.fit_maps[:, :, :, self.current_slice]
        self.display_slice()

    from scipy import ndimage

    def display_slice(self):
        # Get the size of the widget
        size = self.image_label.size()

        # Get the image slice and normalize it to the range [0, 255]
        image = self.dicom[self.echo_time, :, :, self.current_slice]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        # Zoom the image by a factor of 5
        image_zoomed = ndimage.zoom(image, (self.scaling_factor, self.scaling_factor), order=0, mode='nearest')

        # Convert the zoomed image to a QImage and create a QPixmap from it
        qimage = QImage(image_zoomed, image_zoomed.shape[1], image_zoomed.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Scale the pixmap to fit the size of the image label widget
        pixmap = pixmap.scaled(size, Qt.KeepAspectRatio)

        # Set the pixmap on the image label and display it
        self.image_label.setPixmap(pixmap)
        self.image_label.show()

    def init_fit_function(self):
        self.fit_function_widget = FitFunctionWidget(
            [np.NAN] * len(self.time_points), self.fit_function,
            [np.NAN] * len(self.current_params[:, 0, 0]),
            self.time_points, self
        )
        self.plot_container.layout().addWidget(self.fit_function_widget)

    def update_fit_function(self, event):
        x = event.pos().x() // self.scaling_factor
        y = event.pos().y() // self.scaling_factor
        try:
            pixel_params = self.current_params[:, y, x]
        except IndexError:
            return None
        raw_data = self.dicom[:, y, x, self.current_slice]
        self.fit_function_widget.update_plot(pixel_params, raw_data)

class FitFunctionWidget(QWidget):
    def __init__(self, raw_data, fit_function, params, time_points, parent=None):
        super().__init__(parent)
        self.y_raw = raw_data
        self.fit_function = fit_function
        self.params = params
        self.time_points = time_points

        # Add a plot area to the widget
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        # Plot the raw data and fit function
        self.x_fit = np.linspace(0, self.time_points[-1], 1000)
        self.y_fit = self.fit_function(self.x_fit, *self.params)
        self.axes.plot(
            self.time_points, self.y_raw, "o", markersize=4, label="Raw data"
        )
        self.axes.plot(self.x_fit, self.y_fit, "-", label="Fit function")

        # Set the layout of the widget
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self, params=None, raw_data=None):
        if params is not None:
            self.params = params
        if raw_data is not None:
            self.y_raw = raw_data
        self.y_fit = self.fit_function(self.x_fit, *self.params)
        self.axes.clear()
        self.axes.plot(
            self.time_points, self.y_raw, "o", markersize=4, label="Raw data"
        )
        self.axes.plot(self.x_fit, self.y_fit, "-", label="Fit function")
        self.canvas.draw()


if __name__ == "__main__":
    import pydicom
    from pydicom.data import get_testdata_files

    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)
    default = ds.pixel_array

    # Create a 3D DICOM array
    dicom = np.zeros((10, 128, 128, 42))
    for i in range(10):
        for ii in range(42):
            dicom[i, :, :, ii] = default * (11 - i) + np.random.random((128, 128)) * 500

    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Create fitted_map
    fit_maps = np.random.rand(2, 128, 128, 42)

    def fit(x, a, b):
        return a * x + b

    # Create an instance of the ImageViewer class

    app = QApplication(sys.argv)

    viewer = ImageViewer(dicom, fit_maps, fit, time_points)
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())
