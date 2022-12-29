from Visualization.image_viewer import *

def show(dicom, fit_maps, fit, time_points):
    # Create an instance of the ImageViewer class

    app = QApplication(sys.argv)

    viewer = ImageViewer(dicom, fit_maps, fit, time_points, 1)
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())
