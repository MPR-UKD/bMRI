from src import *
from pathlib import Path
from multiprocessing import freeze_support


# T2* Auswertung
def t2star_fitting_example():
    t2_star_folder = (
        Path(__file__).parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "7_T2-star_map_3D_cor_18818"
    )
    t2star = T2_T2star(dim=3, boundary=([0.9, 0, -0.2], [1.5, 45, 0.2]), normalize=True)
    t2star.run(
        dicom_folder=t2_star_folder,
        mask_file=t2_star_folder / "mask.nii.gz",
        min_r2=0.75
    )


# T2 Auswertung
def t2_fitting_example():
    t2_folder = (
        Path(__file__).parent
        / "test"
        / "resources"
        / "20211206_1038"
        / "10_T2_map_cor_25681"
    )
    t2 = T2_T2star(dim=3, boundary=([0.9, 20, -0.2], [1.5, 80, 0.2]), normalize=True)
    t2.run(
        dicom_folder=t2_folder,
        mask_file=t2_folder / "mask.nii.gz",
        min_r2=0.7
    )


# T1rho Auswertung
def t1rho_fitting_example():
    t1rho_folder = (
        Path(__file__).parent / "test" / "resources" / "20211206_1038" / "T1rho"
    )
    t1rho = T1rho_T2prep(
        dim=3, boundary=([0.9, 20, -0.2], [1.5, 85, 0.2]), normalize=True, config=None
    )
    t1rho.run(
        dicom_folder=t1rho_folder,
        mask_file=t1rho_folder / "mask.nii.gz",
        tsl=[0, 20, 80, 140],
    )


if __name__ == "__main__":
    freeze_support()
    t2_fitting_example()
    t2star_fitting_example()
    t1rho_fitting_example()
