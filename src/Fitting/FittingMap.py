from src.Utilitis import save_results, load_nii, get_dcm_array
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

    def __call__(
        self, dcm_folder: Union[str, Path], mask_file: Union[str, Path]
    ) -> Tuple[np.ndarray, Any]:
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
        fitted_map = (
            get_dcm_array([_ for _ in dcm_folder.glob("*.dcm")])
            .transpose((2, 1, 0))
            .astype("float32")
        )

        # Load the mask
        mask = load_nii(mask_file)

        # Process the fitted map
        fitted_map[(mask.array == 0)] = np.NAN
        for i in range(1, int(mask.array.max()) + 1):
            try:
                low = np.percentile(fitted_map[mask.array == i], self.low_percentile)
                up = np.percentile(fitted_map[mask.array == i], self.up_percentile)
                fitted_map[(mask.array == i) & (fitted_map < low)] = np.NAN
                fitted_map[(mask.array == i) & (fitted_map > up)] = np.NAN
            except IndexError:
                pass

        save_results(
            fit_map=fitted_map,
            affine=mask.affine,
            nii_folder=dcm_folder,
            results_path=dcm_folder,
            function=None,
            r2=None,
            mask=mask.array,
            header=mask.header,
            return_params=None,
        )
        return fitted_map, mask
