from src.Utilitis.read import get_dcm_array, load_nii
from pathlib import Path
import numpy as np
from typing import Union, Tuple, Any


class FittedMap:
    def __init__(self, low_percentile: int = 5, up_percentile: int = 95) -> None:
        """
        Initializes the FittedMap object.

        Parameters:
        - low_percentile (int, optional): The lower percentile to consider in the map (default is 5)
        - up_percentile (int, optional): The upper percentile to consider in the map (default is 95)
        """
        assert low_percentile < up_percentile
        assert 0 <= low_percentile <= 100
        assert 0 <= up_percentile <= 100

        self.low_percentile = low_percentile
        self.up_percentile = up_percentile

    def __call__(self, dcm_folder: Union[str, Path], mask_file: Union[str, Path]) -> Tuple[np.ndarray, Any]:
        """
        Process the fitted map.

        Parameters:
        - dcm_folder (str | Path): The path to the directory containing DICOM files
        - mask_file (str | Path): The path to the mask file

        Returns:
        - Tuple[np.ndarray, Any]: The processed fitted map and the mask
        """
        dcm_folder = Path(dcm_folder)
        mask_file = Path(mask_file)

        # Load the DICOM files and transpose the dimensions
        fitted_map = get_dcm_array([_ for _ in dcm_folder.glob("*.dcm")]).transpose(
            (2, 1, 0)
        )

        # Load the mask
        mask = load_nii(mask_file)

        # Process the fitted map
        for i in range(1, int(mask.array.max()) + 1):
            fit_map = fitted_map.copy().astype("float64")
            fit_map[mask != i] = 0
            try:
                low = np.percentile(fit_map[mask.array == i], self.low_percentile)
                up = np.percentile(fit_map[mask.array == i], self.up_percentile)
                fit_map[fit_map < low] = np.nan
                fit_map[fit_map > up] = np.nan
            except IndexError:
                pass

        return fitted_map, mask
