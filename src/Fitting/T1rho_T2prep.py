from pathlib import Path
from typing import Union, Tuple, Dict, List
import numpy as np
from numba import njit
from src.Fitting.AbstractFitting import AbstractFitting, cpu_count
from src.Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from src.Utilitis import save_results, load_nii, save_nii


def fit_T1rho_wrapper_raush(TR: float, T1: float, alpha: float):
    """
    Creates a T1rho fitting function based on Rausch et al.

    Parameters:
    - TR: Repetition time in MRI sequence
    - T1: Longitudinal relaxation time
    - alpha: Flip angle

    Returns:
    - fit function
    """

    def fit(x: np.ndarray, S0: float, t1rho: float, offset: float) -> np.ndarray:
        counter = (1 - np.exp(-(TR - x) / T1)) * np.exp(-x / t1rho)
        denominator = 1 - np.cos(alpha) * np.exp(-x / t1rho) * np.exp(-(TR - x) / T1)
        return S0 * np.sin(alpha) * counter / denominator + offset

    return fit


def fit_T1rho_wrapper_aronen(
    TR: float, T1: float, alpha: float, TE: float, T2star: float
):
    """
    Creates a T1rho fitting function based on Aronen et al.

    Parameters:
    - TR: Repetition time in MRI sequence
    - T1: Longitudinal relaxation time
    - alpha: Flip angle
    - TE: Echo time
    - T2star: Transverse relaxation time

    Returns:
    - fit function
    """

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


def fit_mono_exp_wrapper():
    """
    Creates a mono-exponential fitting function.

    Returns:
    - fit function
    """

    @njit
    def mono_exp(x: np.ndarray, S0: float, t1rho: float, offset: float) -> np.ndarray:
        return S0 * np.exp(-x / t1rho) + offset

    return mono_exp


def fit_T2_wrapper_aronen(TR: float, T1: float, alpha: float, TE: float, T2star: float):
    @njit
    def fit(x: np.ndarray, S0: float, t2: float, offset: float) -> np.ndarray:
        tau = TR - x
        counter = (
            S0
            * np.exp(-x / t2)
            * (1 - np.exp(-tau / T1))
            * np.sin(alpha)
            * np.exp(-TE / T2star)
        )
        denominator = 1 - np.cos(alpha) * np.exp(-tau / T1) * np.exp(-x / t2)
        return counter / denominator + offset

    return fit


class T1rho_T2prep(AbstractFitting):
    def __init__(
        self,
        dim: int,
        config: Union[Dict, None] = None,
        boundary: Union[Tuple, None] = None,
        normalize: bool = False,
        mode_T2: bool = False,
    ):
        """
        Initializes T1rho_T2prep object for fitting T1rho and T2 in MRI data.

        Parameters:
        - dim: Dimension of the MRI data (2 or 3)
        - config: Configuration dictionary with MRI sequence parameters
        - boundary: Tuple representing the boundary conditions
        - normalize: Whether to normalize the data
        - mode_T2: If True, fitting is done for T2 instead of T1rho
        """

        if config is not None:
            if not mode_T2:
                fit = fit_T1rho_wrapper_aronen(
                    config["TR"],
                    config["T1"],
                    config["alpha"],
                    config["TE"],
                    config["T2star"],
                )
            else:
                fit = fit_T2_wrapper_aronen(
                    config["TR"],
                    config["T1"],
                    config["alpha"],
                    config["TE"],
                    config["T2star"],
                )
        else:
            fit = fit_mono_exp_wrapper()

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
        Fit the T1rho or T2 relaxation time for the given DICOM image data.

        Parameters:
        - dicom: 3D or 4D array of DICOM image data
        - mask: 2D or 3D array of mask indicating which pixels to include in the fit
        - x: Array of independent variable data
        - pools: Number of parallel pools for computation
        - min_r2: minimum R^2 value for a fit to be considered valid

        Returns:
        - fit_maps: 3D or 4D array of fitted T1rho or T2 values
        - r2_map: 2D or 3D array of R^2 values for each fit
        """

        # Call the fit method from the parent class using the provided dicom, mask, and x data
        fit_maps, r2_map = super().fit(dicom, mask, x, pools=pools, min_r2=min_r2)

        return fit_maps, r2_map

    def get_TSL(self, first_SL: int = 10, inc_SL: int = 30, n: int = 4) -> np.ndarray:
        """
        Generate an array of spin-lock times (TSL).

        Parameters:
        - first_SL: First spin-lock time
        - inc_SL: Increment in spin-lock time
        - n: Number of spin-lock times

        Returns:
        - Array of spin-lock times
        """
        x = [0, first_SL]
        for _ in range(1, n - 1):
            x.append(x[-1] + inc_SL)
        return np.array(x)

    def run(
        self,
        dicom_folder: Path,
        mask_file: Path,
        tsl: np.ndarray | List,
        pools: int = 0,
        min_r2: float = -np.inf,
        save_dicom_as_nii: bool = True,
    ):
        """
        Run full evaluation pipline.

        Args:
            - dicom_folder: Path to the folder containing DICOM files
            - mask_file: Path to the nifti mask file
            - tsl: Array of spin-lock times
            - pools: Number of parallel pools for computation (optional)
            - min_r2: minimum R^2 value for a fit to be considered valid (optional)
            - save_dicom_as_nii: Save dicom array as nifti for image_viewer

        Returns:
            results
        """
        tsl = np.array(tsl)
        data, _ = self.read_data(dicom_folder)
        mask = load_nii(mask_file)
        if save_dicom_as_nii:
            save_nii(
                data[:, :, :, ::-1],
                mask.affine,
                mask.header,
                dicom_folder / "dicom.nii.gz",
            )
            self.save_times(tsl, dicom_folder / "acquisition_times.txt")
        fit_map, r2 = self.fit(dicom=data, mask=mask.array, x=tsl, pools=pools)
        results = save_results(
            fit_map=fit_map,
            r2=r2,
            affine=mask.affine,
            function=self.fit_function,
            nii_folder=dicom_folder,
            results_path=dicom_folder,
            mask=mask.array,
            header=mask.header,
        )
        return results

    def read_data(self, folder: Union[str, Path]) -> Tuple[np.ndarray, None]:
        """
        Reads DICOM data from the specified folder.

        Parameters:
        - folder: Path to the folder containing DICOM files

        Returns:
        - dicom: 3D or 4D array of DICOM image data
        - None
        """
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

        order = self.check_order(dcm_files)
        # echos, z, x, y --> echos, x, y, z
        dicom = (
            np.array([get_dcm_array(dcm_files[o]) for o in order])
            .transpose(0, 3, 2, 1)
            .astype("int16")
        )
        return dicom, None

    def check_order(self, dcm_files: list[list[Path]]) -> np.ndarray:
        """
        Check dcm
        Args:
            dcm_files:

        Returns:

        """
        signal = [get_dcm_array(dcm).mean() for dcm in dcm_files]
        return np.argsort(signal)[::-1]
