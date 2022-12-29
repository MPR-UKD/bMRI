from pathlib import Path

import pydicom

from Utilitis.read import get_dcm_list, split_dcm_list, get_dcm_array
from .AbstractFitting import *


class InversionRecoveryT1(AbstractFitting, ABC):
    def __init__(self, boundary: tuple):
        super(InversionRecoveryT1, self).__init__(
            inversion_recovery_t1, boundary=boundary
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


    def read_data(self, folder: str | Path | list):
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


def get_ti(dcm_files: list):
    x = []
    for dcm in dcm_files:
        info = pydicom.dcmread(dcm[0])
        x.append(info.InversionTime)
    x = np.array([float(ti) for ti in x])
    order = np.argsort(x)
    return order, x[order]


def inversion_recovery_t1(x: np.ndarray, S0: float, t1: float, offset: float):
    return S0 * (1 - np.exp(-x / t1)) + offset
