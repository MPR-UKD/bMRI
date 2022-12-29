from pathlib import Path

from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from .AbstractFitting import *


def fit_prep_wrapper(TR: float, T1: float, alpha: float, TE: float):
    def fit(x: np.ndarray, S0: float, t2prep: float, offset: float):
        counter = np.exp(-t2prep / TE) * (1 - np.exp(-(TR - t2prep) / T1))
        denominator = 1 - np.cos(alpha) * np.exp(-t2prep / TE) * np.exp(
            -(TR - t2prep) / T1
        )
        return S0 * np.sin(alpha) * counter / denominator * np.exp(-x / t2prep) + offset

    return fit


class T2prep(AbstractFitting):
    def __init__(self, dim: int, config: dict, boundary: tuple | None = None):
        fit = fit_prep_wrapper(
            config["TR"], config["T1"], config["alpha"], config["TE"]
        )
        super(T2prep, self).__init__(fit, boundary=boundary)
        self.dim = dim

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
