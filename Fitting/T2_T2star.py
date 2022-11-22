import numpy as np
from .AbstractFitting import AbstractFitting
from pathlib import Path
from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
import pydicom

def fit(x, S0, t2_t2star, offset):
    return S0 * np.exp(-x / t2_t2star) + offset


class T2_T2star(AbstractFitting):

    def __init__(self, dim, boundary=None):
        super(T2_T2star, self).__init__(fit, boundary=boundary)
        self.dim = dim

    def read_data(self, folder: str | Path):

        if self.dim == 2:
            dcm_files = get_dcm_list(folder)
            dcm_files = [[dcm] for dcm in dcm_files]
            TEs = get_tes(dcm_files)
        if self.dim == 3:
            dcm_files = get_dcm_list(folder)
            if len(dcm_files) == 0:
                echos = folder.glob('*/')
                dcm_files = [get_dcm_list(echo) for echo in echos]
                dcm_files = [item for sublist in dcm_files for item in sublist]
            dcm_files = split_dcm_list(dcm_files)
            TEs = get_tes(dcm_files)
        else:
            raise NotImplementedError
        # echos, z, x, y --> echos, x, y, z
        dicom = np.array([get_dcm_array(dcm) for dcm in dcm_files]).transpose(0, 3, 2, 1)
        return dicom, TEs


def get_tes(dcm_files):
    TEs = []
    for dcm in dcm_files:
        info = pydicom.dcmread(dcm[0])
        if info.EchoTime not in TEs:
            TEs.append(info.EchoTime)
    return [float(te) for te in TEs]
