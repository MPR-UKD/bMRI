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


class T2_T2star(AbstractFitting):
    def __init__(self, dim: int, boundary: tuple | None = None, fit_config: dict | None = None, normalize: bool = False):

        super(T2_T2star, self).__init__(mono_exp, boundary=boundary, fit_config=fit_config, normalize=normalize)
        self.dim = dim

    def fit(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        pools: int = cpu_count(),
        min_r2: float = -np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:

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
