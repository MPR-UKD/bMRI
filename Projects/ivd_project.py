from multiprocessing import freeze_support
from pathlib import Path

import numpy as np

from src import *


def view_t2_t2star(folder, dim=2):
    from src.Visualization.image_viewer import ImageViewer
    from src.Fitting import T2_T2star
    from PyQt5.QtWidgets import QApplication
    import sys

    # Replace the paths below with the paths to your files
    path_to_dicom = folder / "dicom.nii.gz"
    path_to_fit_maps = folder / "params.nii.gz"

    # Define the fit function
    t2 = T2_T2star(dim=dim)
    fit_function = t2.fit_function

    # List of time points / Alternative you can read the saved timepoints with the fitting clas
    time_points = np.loadtxt(folder / "acquisition_times.txt")

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
        normalize=True,
    )

    # Show the viewer
    viewer.show()

    # Run the PyQt5 application
    sys.exit(app.exec_())


########################################################################################################################
def t2_fitting_example(t2_folder: Path):
    t2 = T2_T2star(dim=3, boundary=([0.9, 20, -0.5], [3, 90, 0.5]), normalize=True)
    t2.run(dicom_folder=t2_folder, mask_file=t2_folder / "mask.nii.gz", min_r2=0.7)


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


def t2star_fitting_example(t2star_folder: Path):
    t2star = T2_T2star(
        dim=2, boundary=([0.9, 0, -np.Inf], [2, 50, np.inf]), normalize=True
    )
    t2star.run(
        dicom_folder=t2star_folder,
        mask_file=t2star_folder / "mask.nii.gz",
        min_r2=0.75,
    )


def t1_evaluation(t1_map_folder, t1_mask_path):
    fitted_map = FittedMap(low_percentile=1, up_percentile=99)
    fitted_map(dcm_folder=t1_map_folder, mask_file=t1_mask_path)


if __name__ == "__main__":
    freeze_support()
    root = Path(
        "/Users/ludgerradke/Documents/DICOM_translated/MRTC-Test_HWS_BWS_02_GAPH41824/20231109_1407"
    )

    # T2star
    t2_star_folder = root / "10_T2-star_map_3D_sag_01214"
    assert t2_star_folder.exists()
    t2star_fitting_example(t2_star_folder)
    # view_t2_t2star(t2_star_folder, 2)

    # T2
    t2_folder = root / "29_T2_map_fatsat_hfnormal_02141"
    assert t2_folder.exists()
    t2_fitting_example(t2_folder)
    view_t2_t2star(t2_folder, 2)

    # T1
    t1_folder = root / "18_T1_Images_01418"
    t1_mak_path = root / "16_T1_map_sag_FLIP1_01417" / "mask.nii.gz"
    t1_evaluation(t1_folder, t1_mak_path)
