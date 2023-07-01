import numpy as np

from src import *
from pathlib import Path
from multiprocessing import freeze_support


error_text = (
    "Note: Sample code - path to medical file is missing.\n\n"
    "Please note that this sample code references a path to a file that is not in the GitHub repository. "
    "This is because the file is sensitive medical data that cannot be shared publicly.\n\n"
    "To run the script successfully, you will need to make a few adjustments:\n"
    "1. place the relevant medical data in a directory on your local system.\n"
    "2. adjust the file path in the script so that it points to the directory with your medical data."
)


# T2*
def t2star_fitting_example():
    t2_star_folder = (
        Path(__file__).parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "7_T2-star_map_3D_cor_18818"
    )
    if not t2_star_folder.exists():
        raise EnvironmentError(error_text)
    t2star = T2_T2star(
        dim=3, boundary=([0.9, 0, -np.Inf], [2, 50, np.inf]), normalize=True
    )
    t2star.run(
        dicom_folder=t2_star_folder,
        mask_file=t2_star_folder / "mask.nii.gz",
        min_r2=0.75,
    )


# T2
def t2_fitting_example():
    t2_folder = (
        Path(__file__).parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "10_T2_map_cor_25681"
    )
    if not t2_folder.exists():
        raise FileNotFoundError(error_text)
    t2 = T2_T2star(dim=3, boundary=([0.9, 5, -0.5], [3, 40, 0.5]), normalize=True)
    t2.run(dicom_folder=t2_folder, mask_file=t2_folder / "mask.nii.gz", min_r2=0.7)


# T1rho
def t1rho_fitting_example():
    t1rho_folder = (
        Path(__file__).parent / "test" / "resources" / "20211206_1038" / "T1rho"
    )
    if not t1rho_folder.exists():
        raise EnvironmentError(error_text)
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
    # t2_fitting_example()
    t2star_fitting_example()
    # t1rho_fitting_example()
