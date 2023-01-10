from pathlib import Path

from Fitting.AbstractFitting import *
from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list


# Assessment of T1, T1ρ, and T2 values of the ulnocarpal disc in healthy subjects at 3 tesla
# DOI: 10.1016/j.mri.2014.05.010
# Eq. 4


def fit_T1rho_wrapper_raush(TR: float, T1: float, alpha: float):
    def fit(x: np.ndarray, S0: float, t1rho: float, offset: float) -> np.ndarray:
        counter = (1 - np.exp(-(TR - x) / T1)) * np.exp(-x / t1rho)
        denominator = 1 - np.cos(alpha) * np.exp(-x / t1rho) * np.exp(-(TR - x) / T1)
        return S0 * np.sin(alpha) * counter / denominator + offset

    return fit


# 3D SPIN-LOCK IMAGING OF HUMAN GLIOMAS
# https://doi.org/10.1016/S0730-725X(99)00041-7
# Appendix
def fit_T1rho_wrapper_aronen(
        TR: float, T1: float, alpha: float, TE: float, T2star: float
):
    @njit
    def fit(x: np.ndarray, S0: float, t1rho: float, offset: float) -> np.ndarray:
        tau = TR - x
        counter = (
                S0
                * np.exp(-x / t1rho)
                * (1 - np.exp(-tau / T1))
                * np.sin(alpha)
                * np.exp(-TE / T2star)
        )
        denominator = 1 - np.cos(alpha) * np.exp(-tau / T1) * np.exp(-x / t1rho)
        return counter / denominator + offset

    return fit


class T1rho_T2prep(AbstractFitting):
    def __init__(self, dim: int, config: dict, boundary: tuple | None = None, normalize: bool = False):
        # fit = fit_T1rho_wrapper_raush(config["TR"], config["T1"], config["alpha"])
        fit = fit_T1rho_wrapper_aronen(
            config["TR"], config["T1"], config["alpha"], config["TE"], config["T2star"]
        )
        super(T1rho_T2prep, self).__init__(fit, boundary=boundary, normalize=normalize)
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

    def get_TSL(self, first_SL: int = 10, inc_SL: int = 30, n: int = 4) -> np.ndarray:
        x = [0, 2 * first_SL]
        for _ in range(1, n - 1):
            x.append(x[-1] + 2 * inc_SL)
        return x

    def read_data(self, folder: str | Path):

        folder = Path(folder)
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
