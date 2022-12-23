from abc import ABC

from .AbstractFitting import AbstractFitting
import numpy as np
from pathlib import Path
from Utilitis.read import get_dcm_list, split_dcm_list, get_dcm_array
import pydicom


class InversionRecoveryT1(AbstractFitting, ABC):
    def __init__(self, boundary: tuple):
        super(InversionRecoveryT1, self).__init__(
            inversion_recovery_t1, boundary=boundary
        )

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
