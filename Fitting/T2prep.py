from pathlib import Path

import numpy as np

from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from .AbstractFitting import AbstractFitting


def fit_prep_wrapper(TR, T1, alpha, TE):
    def fit(x, S0, t2prep, offset):
        counter = np.exp(-t2prep / TE) * (1 - np.exp(-(TR - t2prep) / T1))
        denominator = 1 - np.cos(alpha) * np.exp(-t2prep / TE) * np.exp(
            -(TR - t2prep) / T1
        )
        return S0 * np.sin(alpha) * counter / denominator * np.exp(-x / t2prep) + offset

    return fit


class T2prep(AbstractFitting):
    def __init__(self, dim, config, boundary=None):
        fit = fit_prep_wrapper(
            config["TR"], config["T1"], config["alpha"], config["TE"]
        )
        super(T2prep, self).__init__(fit, boundary=boundary)
        self.dim = dim

    def read_data(self, folder: str | Path):

        if self.dim == 2:
            dcm_files = get_dcm_list(folder)
            dcm_files = [[dcm] for dcm in dcm_files]
        elif self.dim == 3:
            dcm_files = get_dcm_list(folder)
            if len(dcm_files) == 0:
                echos = folder.glob("*/")
                dcm_files = [get_dcm_list(echo) for echo in echos]
                dcm_files = [item for sublist in dcm_files for item in sublist]
            dcm_files = split_dcm_list(dcm_files)
        else:
            raise NotImplementedError
        # echos, z, x, y --> echos, x, y, z
        dicom = np.array([get_dcm_array(dcm) for dcm in dcm_files]).transpose(
            0, 3, 2, 1
        )
        return dicom, None
