import sys
from numba import njit
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
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from scipy import ndimage
from pathlib import Path
from src.Utilitis import load_nii


def calc_scaling_factor(dicom_shape: tuple[int, int, int]) -> int:
    """
    Calculate scaling factor based on DICOM shape.

    :param dicom_shape: Shape of the DICOM data.
    :return: Scaling factor.
    """
    return 1000 // max(dicom_shape[1], dicom_shape[2])


class ImageViewer(QMainWindow):
    """
    ImageViewer class to visualize DICOM data and fitted maps.
    """

    def __init__(self):
        super().__init__()

    def start(
        self,
        dicom: np.ndarray | Path,
        fit_maps: np.ndarray | list | Path,
        fit_function: callable,
        time_points: list[int],
        c_int: int | None = None,
        alpha: float = 0.3,
        normalize: bool = True,
    ):
        """
        Initialize the ImageViewer.

        :param dicom: DICOM data array.
        :param fit_maps: Array of fitted maps.
        :param fit_function: Fitting function.
        :param time_points: List of time points.
        :param c_int: Color intensity index, optional.
        :param alpha: Alpha value for overlay, optional.
        :param normalize: Flag to normalize data, optional.
        """
        if isinstance(dicom, Path):
            dicom = load_nii(dicom).array
        if isinstance(fit_maps, Path):
            fit_maps = load_nii(fit_maps).array
            fit_maps[fit_maps == -1] = np.NAN
        self.echo_time = 0
        self.time_points = time_points
        self.dicom = dicom
        self.alpha = alpha
        self.norm = normalize
        self.fit_maps = np.array(fit_maps)
        self.color_map = fit_maps[c_int] if c_int is not None else None
        self.fit_function = fit_function
        self.scaling_factor = calc_scaling_factor(dicom.shape)
        self.current_slice = 0
        self.current_params = self.fit_maps[:, :, :, self.current_slice]

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.image_label.mousePressEvent = self.update_fit_function
        # self.image_label.mouseMoveEvent = self.update_fit_function
        self.scaling_factor = calc_scaling_factor(dicom.shape)
        width, height = (
            dicom.shape[1] * self.scaling_factor,
            dicom.shape[2] * self.scaling_factor,
        )
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

    def change_slice(self, slice_num: int):
        """
        Change displayed slice.

        :param slice_num: Slice number to display.
        """
        self.current_slice = slice_num
        self.current_params = self.fit_maps[:, :, :, self.current_slice]
        self.display_slice()

    def display_slice(self):
        """
        Display the current slice.
        """
        # Get the size of the widget
        size = self.image_label.size()

        # Get the image slice and normalize it to the range [0, 1]
        image = self.dicom[self.echo_time, :, :, self.current_slice]
        image = (image - image.min()) / (image.max() - image.min())

        # Zoom the image by a factor of 5
        image_zoomed = ndimage.zoom(
            image, (self.scaling_factor, self.scaling_factor), order=0, mode="nearest"
        )

        # Convert the zoomed image to an RGB image
        image_zoomed_rgb = np.dstack((image_zoomed, image_zoomed, image_zoomed))

        # Normalize the color map to the range [0, 1]
        if self.color_map is not None:
            color_map = self.color_map[:, :, self.current_slice]
            color_map_norm = (color_map - color_map.min()) / (
                color_map.max() - color_map.min()
            )

            # Zoom the color map by a factor of 5
            color_map_zoomed = ndimage.zoom(
                color_map_norm,
                (self.scaling_factor, self.scaling_factor),
                order=0,
                mode="nearest",
            )
            jet_cmap = get_cmap("jet")
            color_map_zoomed_rgb = jet_cmap(color_map_zoomed)

            # Overlay the color map on the DICOM image pixel by pixel
            alpha = 0.3
            for i in range(image_zoomed.shape[0]):
                for j in range(image_zoomed.shape[1]):
                    if color_map_zoomed[i][j] > 0:
                        image_zoomed_rgb[i][j] = (1 - alpha) * image_zoomed_rgb[i][
                            j
                        ] + alpha * color_map_zoomed_rgb[i][j][:3]

        image_zoomed_rgb = (image_zoomed_rgb * 255).round().astype("int8")
        # Convert the zoomed RGB image to a QImage and create a QPixmap from it
        qimage = QImage(
            image_zoomed_rgb,
            image_zoomed_rgb.shape[1],
            image_zoomed_rgb.shape[0],
            image_zoomed_rgb.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)

        # Set the pixmap as the background image of the label
        self.image_label.setPixmap(pixmap)

    def init_fit_function(self):
        """
        Initialize the fit function plot.
        """
        self.fit_function_widget = FitFunctionWidget(
            [np.NAN] * len(self.time_points),
            self.fit_function,
            [np.NAN] * len(self.current_params[:, 0, 0]),
            self.time_points,
            self,
        )
        self.plot_container.layout().addWidget(self.fit_function_widget)

    def update_fit_function(self, event):
        """
        Update the fit function plot based on an event.

        :param event: The triggered event.
        """
        x = event.pos().x() // self.scaling_factor
        y = event.pos().y() // self.scaling_factor
        try:
            pixel_params = self.current_params[:, y, x]
        except IndexError:
            return None
        raw_data = self.dicom[:, y, x, self.current_slice].astype("float64")
        if self.norm:
            raw_data /= raw_data.max()
        self.fit_function_widget.update_plot(pixel_params, raw_data)


class FitFunctionWidget(QWidget):
    """
    Widget for displaying the fitting function.
    """

    def __init__(
        self,
        raw_data: list[float],
        fit_function: callable,
        params: list[float],
        time_points: list[int],
        parent: QWidget = None,
    ):
        """
        Initialize the FitFunctionWidget.

        :param raw_data: List of raw data points.
        :param fit_function: Fitting function.
        :param params: List of parameters for the fitting function.
        :param time_points: List of time points.
        :param parent: Parent widget, optional.
        """
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
        """
        Update the plot with new parameters or raw data.

        :param params: List of new parameters, optional.
        :param raw_data: List of new raw data, optional.
        """
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


def example_1():
    import pydicom
    from pydicom.data import get_testdata_files

    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)
    default = ds.pixel_array

    # Create a 3D DICOM array
    dicom = np.zeros((10, 128, 128, 42))
    for i in range(10):
        for ii in range(42):
            dicom[i, :, :, ii] = default * (i + 1)

    dicom = dicom / dicom.max() * 4016
    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Create fitted_map
    fit_maps = np.array(
        [
            (dicom[-1] - dicom[0]) / (time_points[-1] - time_points[0]),
            np.zeros_like(dicom[0]),
        ]
    )

    @njit
    def fit(x, a, b):
        return a * x + b

    # Create an instance of the ImageViewer class

    app = QApplication(sys.argv)

    viewer = ImageViewer(dicom, fit_maps, fit, time_points, 0, normalize=False)
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())


def example_2():
    from src.Fitting import T1rho_T2prep

    t1rho_folder = (
        Path(__file__).parent.parent.parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "T1rho"
    )
    t1rho = T1rho_T2prep(dim=3)
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.start(
        dicom=t1rho_folder / "dicom.nii.gz",
        fit_maps=t1rho_folder / "params.nii.gz",
        fit_function=t1rho.fit_function,
        time_points=[0, 20, 80, 140],
        c_int=1
    )
    viewer.show()
    sys.exit(app.exec_())


def example_t2star():
    from src.Fitting import T2_T2star
    t2_star_folder = (
        Path(__file__).parent.parent.parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "7_T2-star_map_3D_cor_18818"
    )
    t2star = T2_T2star(dim=3)
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.start(
        dicom=t2_star_folder / "dicom.nii.gz",
        fit_maps=t2_star_folder / "params.nii.gz",
        fit_function=t2star.fit_function,
        time_points=[0, 20, 80, 140],
        c_int=1
    )
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    example_2()
