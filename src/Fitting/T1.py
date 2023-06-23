from pathlib import Path
from typing import Union, List, Tuple

import pydicom
import numpy as np
from numba import njit

from src.Utilitis.read import get_dcm_list, split_dcm_list, get_dcm_array
from .AbstractFitting import AbstractFitting, ABC, cpu_count


class InversionRecoveryT1(AbstractFitting, ABC):
    def __init__(self, boundary: Union[tuple, None] = None, normalize: bool = False) -> None:
        """
        Initializes the InversionRecoveryT1 object.

        Parameters:
        - boundary (tuple | None, optional): Boundary for the curve fit parameters
        - normalize (bool, optional): Whether to normalize the data (default is False)
        """
        super(InversionRecoveryT1, self).__init__(
            inversion_recovery_t1, boundary=boundary, normalize=normalize
        )

    def fit(
            self,
            dicom: np.ndarray,
            mask: np.ndarray,
            x: np.ndarray,
            pools: int = cpu_count(),
            min_r2: float = -np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the T1 relaxation time for the given DICOM image data.

        Parameters:
        - dicom: 3D or 4D array of DICOM image data
        - mask: 2D or 3D array of mask indicating which pixels to include in the fit
        - x: 1D array of independent variable data
        - pools: Number of pools to use for multiprocessing
        - min_r2: minimum R^2 value for a fit to be considered valid

        Returns:
        - fit_maps: 3D or 4D array of fitted T1 values
        - r2_map: 2D or 3D array of R^2 values for each fit
        """
        # Call the fit method from the parent class using the provided dicom, mask, and x data
        fit_maps, r2_map = super().fit(dicom, mask, x, pools=pools, min_r2=min_r2)

        return fit_maps, r2_map

    def read_data(self, folder: Union[str, Path, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads DICOM data from the given folder or list of folders.

        Parameters:
        - folder: Path to the folder or list of folders containing DICOM files

        Returns:
        - dicom: 3D or 4D array of DICOM image data
        - x: 1D array of independent variable data
        """
        if type(folder) is not list:
            folder = Path(folder)
            echos = folder.glob("*/")
        else:
            echos = [Path(_) for _ in folder]
        dcm_files = [get_dcm_list(echo) for echo in echos]
        dcm_files_flatted = [item for sublist in dcm_files for item in sublist]
        dcm_files = split_dcm_list(dcm_files_flatted)
        order, x = get_ti(dcm_files)
        dicom = np.array([get_dcm_array(dcm_files[o]) for o in order]).transpose(
            0, 3, 2, 1
        )
        return dicom, x


def get_ti(dcm_files: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves inversion time information from the DICOM files.

    Parameters:
    - dcm_files: List of DICOM files

    Returns:
    - order: Order of inversion times
    - x: Array of inversion times
    """
    x = []
    for dcm in dcm_files:
        info = pydicom.dcmread(dcm[0])
        x.append(info.InversionTime)
    x = np.array([float(ti) for ti in x])
    order = np.argsort(x)
    return order, x[order]


@njit
def inversion_recovery_t1(x: np.ndarray, S0: float, t1: float, offset: float) -> np.ndarray:
    """
    Calculates the inversion recovery curve for T1 relaxation.

    Parameters:
    - x: Array of independent variable data
    - S0: Amplitude parameter
    - t1: T1 relaxation time parameter
    - offset: Offset parameter

    Returns:
    - The calculated inversion recovery curve
    """
    return S0 * (1 - 2 * np.exp(-x / t1)) + offset
