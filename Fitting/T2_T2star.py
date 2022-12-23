from typing import Union

import pydicom
from pathlib import Path
from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from .AbstractFitting import *


def mono_exp(x: np.ndarray, S0: float, t2_t2star: float, offset: float) -> np.ndarray:
    """
    Fit function for T2* relaxation time.

    Parameters:
    - x: 1D array of echo times
    - S0: initial signal intensity
    - t2_t2star: T2* relaxation time
    - offset: constant offset to add to the fit curve

    Returns:
    - fit: 1D array of fitted signal intensities at the given echo times
    """
    return S0 * np.exp(-x / t2_t2star) + offset


class T2_T2Star(AbstractFitting):
    def __init__(
        self, dicom_path: Union[str, Path], dim: int, boundary: Tuple, fit_config: dict
    ):
        """
        T2Star fitting class. Inherits from AbstractFitting class.

        Parameters:
        - dicom_path: path to the DICOM image or directory containing DICOM images
        - dim: number of dimensions in the DICOM image (2 or 3)
        - boundary: tuple of lower and upper bounds for the fit parameters
        - fit_config: dictionary of additional fitting configuration options
        """
        self.dicom_path = dicom_path
        self.dim = dim

        # Set the fit function to the provided t2star function
        super().__init__(mono_exp, boundary=boundary, fit_config=fit_config)

    def fit(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        pools: int = cpu_count(),
        min_r2: float = -np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the T2* relaxation time for the given DICOM image data.

        Parameters:
        - dicom: 3D or 4D array of DICOM image data
        - mask: 2D or 3D array of mask indicating which pixels to include in the fit
        - min_r2: minimum R^2 value for a fit to be considered valid

        Returns:
        - fit_maps: 3D or 4D array of fitted T2* values
        - r2_map: 2D or 3D array of R^2 values for each fit
        """

        # Call the fit method from the parent class using the provided dicom, mask, and x data
        fit_maps, r2_map = super().fit(dicom, mask, x, pools=pools, min_r2=min_r2)

        return fit_maps, r2_map

    def read_data(self, folder: str | Path):

        if self.dim == 2:
            dcm_files = get_dcm_list(folder)
            dcm_files = [[dcm] for dcm in dcm_files]
            TEs, order = get_tes(dcm_files)
        elif self.dim == 3:
            dcm_files = get_dcm_list(folder)
            if len(dcm_files) == 0:
                echos = folder.glob("*/")
                dcm_files = [get_dcm_list(echo) for echo in echos]
                dcm_files = [item for sublist in dcm_files for item in sublist]
            dcm_files = split_dcm_list(dcm_files)
            TEs, order = get_tes(dcm_files)
        else:
            raise NotImplementedError
        # echos, z, x, y --> echos, x, y, z
        dicom = np.array([get_dcm_array(dcm_files[o]) for o in order]).transpose(
            0, 3, 2, 1
        )
        return dicom, TEs


def get_tes(dcm_files: list):
    TEs = []
    for dcm in dcm_files:
        info = pydicom.dcmread(dcm[0])
        if info.EchoTime not in TEs:
            TEs.append(info.EchoTime)
    tes = np.array([float(te) for te in TEs])
    order = np.argsort(tes)
    return tes[order], order
