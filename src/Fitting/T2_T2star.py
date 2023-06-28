from typing import Union, List, Tuple
import numpy as np
import pydicom
from pathlib import Path
from src.Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from src.Utilitis import save_results, load_nii, save_nii
from .AbstractFitting import AbstractFitting, cpu_count


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
    def __init__(
        self,
        dim: int,
        boundary: Union[tuple, None] = None,
        fit_config: Union[dict, None] = None,
        normalize: bool = False,
    ):
        """
        Initialize T2_T2star object.

        Parameters:
        - dim: Dimensionality of the data (2 or 3)
        - boundary: Tuple representing boundary conditions (optional)
        - fit_config: Configuration for the fitting (optional)
        - normalize: Boolean indicating whether to normalize the data
        """
        super(T2_T2star, self).__init__(
            mono_exp, boundary=boundary, fit_config=fit_config, normalize=normalize
        )
        self.dim = dim

    def run(
        self,
        dicom_folder: Path,
        mask_file: Path,
        pools: int = 0,
        min_r2: float = -np.inf,
        save_dicom_as_nii: bool = True,
    ):
        """
        Run full evaluation pipline.

        Args:
            - dicom_folder:  Path to the folder containing DICOM files
            - mask_file:  Path to the nifti mask file
            - pools: Number of parallel pools for computation (optional)
            - min_r2: minimum R^2 value for a fit to be considered valid (optional)
            - save_dicom_as_nii: Save dicom array as nifti for image_viewer (optional)

        Returns:
            results
        """
        data, te = self.read_data(dicom_folder)
        mask = load_nii(mask_file)
        if save_dicom_as_nii:
            save_nii(data[:, :, :, ::-1], mask.affine, mask.header, dicom_folder / "dicom.nii.gz")
            self.save_times(te, dicom_folder / "acquisition_times.txt")
        fit_map, r2 = self.fit(dicom=data, mask=mask.array, x=te, pools=pools)
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
        - x: Array of independent variable data
        - pools: Number of parallel pools for computation (optional)
        - min_r2: minimum R^2 value for a fit to be considered valid (optional)

        Returns:
        - fit_maps: 3D or 4D array of fitted T2* values
        - r2_map: 2D or 3D array of R^2 values for each fit
        """
        fit_maps, r2_map = super().fit(dicom, mask, x, pools=pools, min_r2=min_r2)
        return fit_maps, r2_map

    def read_data(self, folder: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads DICOM data from the specified folder.

        Parameters:
        - folder: Path to the folder containing DICOM files

        Returns:
        - dicom: 3D or 4D array of DICOM image data
        - TEs: Array of echo times
        """
        folder = Path(folder)
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


def get_tes(dcm_files: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and sorts the echo times from DICOM files.

    Parameters:
    - dcm_files: List of lists of DICOM file paths

    Returns:
    - tes: Sorted array of echo times
    - order: Indices that would sort the echo times
    """
    TEs = []
    for dcm in dcm_files:
        info = pydicom.dcmread(dcm[0])
        if info.EchoTime not in TEs:
            TEs.append(info.EchoTime)
    tes = np.array([float(te) for te in TEs])
    order = np.argsort(tes)
    return tes[order], order
