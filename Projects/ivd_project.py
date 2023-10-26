import numpy as np

from src import *
from pathlib import Path
from multiprocessing import freeze_support

# T2*
def t2star_fitting_example(t2star_folder: Path):
    t2star = T2_T2star(
        dim=3, boundary=([0.9, 0, -np.Inf], [2, 50, np.inf]), normalize=True
    )
    t2star.run(
        dicom_folder=t2star_folder,
        mask_file=t2star_folder / "mask.nii.gz",
        min_r2=0.75,
    )


def view_t2(t2_folder):
    from src.Utilitis import load_nii
    from src.Visualization.image_viewer import ImageViewer
    from src.Fitting import T2_T2star
    from PyQt5.QtWidgets import QApplication
    import sys
    from pathlib import Path

    # Replace the paths below with the paths to your files
    path_to_dicom = t2_folder / "dicom.nii.gz"
    path_to_fit_maps = t2_folder / "params.nii.gz"

    # Define the fit function
    t2 = T2_T2star(dim=3)
    fit_function = t2.fit_function

    # List of time points / Alternative you can read the saved timepoints with the fitting clas
    time_points = [9.400000000000000355e+00,
1.880000000000000071e+01,
2.819999999999999929e+01,
3.760000000000000142e+01,
4.700000000000000000e+01,
5.639999999999999858e+01,
6.579999999999999716e+01,
7.520000000000000284e+01,
8.459999999999999432e+01,
9.400000000000000000e+01,
]

    # Create a QApplication instance
    app = QApplication(sys.argv)

    # Create an instance of the ImageViewer class
    viewer = ImageViewer()

    # Start the viewer
    viewer.start(
        dicom=path_to_dicom,
        fit_maps=path_to_fit_maps,
        fit_function=fit_function,
        time_points=time_points,
        c_int=1,
        alpha=0.3,
        normalize=True
    )

    # Show the viewer
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())
# T2
def t2_fitting_example(t2_folder: Path):
    t2 = T2_T2star(dim=3, boundary=([0.9, 20, -0.5], [3, 90, 0.5]), normalize=True)
    t2.run(dicom_folder=t2_folder, mask_file=t2_folder / "mask.nii.gz", min_r2=0.7)


# T1rho
def t1rho_fitting_example(t1rho_folder):
    t1rho = T1rho_T2prep(
        dim=3, boundary=([1, 40, -0.4], [3, 150, 0.4]), normalize=True, config=None
    )
    tsl = t1rho.get_TSL(10, 30)
    t1rho.run(
        dicom_folder=t1rho_folder,
        mask_file=t1rho_folder / "mask.nii.gz",
        tsl=tsl,
    )


if __name__ == "__main__":
    freeze_support()
    root = Path("/Users/ludgerradke/Documents/bMRI_Test_data_translated/Test-MRTC_BMRI01_GAPH38923/20231026_0950")


    t2_folder = root / "6_T2_map_00673"
    #view_t2(t2_folder)
    t2_fitting_example(t2_folder)

    #t2star_folder = root / "8_T2-star_map_3D_sag_01631"
    #t2_fitting_example(t2star_folder)

    #t2_folder = root / "6_T2_map_00673"
    #t2star_fitting_example(t2star_folder)

    #t1rho_folder = root / "T1rho"
    #t1rho_fitting_example(t1rho_folder)
