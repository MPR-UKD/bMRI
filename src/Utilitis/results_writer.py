import nibabel as nib
import numpy as np
from pathlib import Path
import csv
from typing import Callable, List, Union, Any, Optional
from src.Utilitis.utils import get_function_parameter


def save_results(
    fit_map: np.ndarray,
    affine: np.ndarray,
    nii_folder: Union[str, Path],
    results_path: Union[str, Path],
    function: Optional[Callable] = None,
    r2: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    header: Optional[np.ndarray] = None,
    return_params: Optional[List[str]] = None,
    decimal: str = ",",
) -> Optional[List[dict]]:
    """
    Save results of the fitting procedure to Nifti and CSV files.

    :param function: Fitting function used.
    :param fit_map: Array of fitting maps.
    :param r2: Array of R squared values.
    :param mask: Array of the masks.
    :param affine: Affine transformation to be used in the Nifti files.
    :param header: Header information to be used in the Nifti files.
    :param nii_folder: Path to the folder to store the Nifti files.
    :param results_path: Path to the folder to store the CSV results.
    :param return_params: List of parameters to return, if None returns all.
    :param decimal: Decimal separator for the numbers in the CSV files.
    :return: List of result dictionaries.
    """
    return_list = []
    nii_folder = Path(nii_folder)
    results_path = Path(results_path)
    nii_folder.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if r2 is not None:
        save_nii(r2, affine, header, nii_folder / "r2.nii.gz")

    save_nii(fit_map, affine, header, nii_folder / "params.nii.gz")

    parameters = ["value"] if function is None else get_function_parameter(function)

    for ii, parameter in enumerate(parameters):
        save_nii(fit_map[ii], affine, header, nii_folder / f"{parameter}_map.nii.gz")
        results = {}
        for i in range(1, int(mask.max()) + 1):
            m = np.where(mask == i, 1, 0)

            times = fit_map[ii][m == 1]
            if times.size == 0:
                continue

            results[str(i)] = [
                f"{np.nanmean(times):.2f}",
                f"{np.nanstd(times):.2f}",
                f"{np.nanmin(times):.2f}",
                f"{np.nanmax(times):.2f}",
                f"{len(times[~np.isnan(times)]):.2f}/{np.sum(m):.2f}",
                f"{np.nanmean(r2[m == 1]):.2f}" if r2 is not None else "NaN",
            ]
        with open(
            results_path.as_posix() + f"_{parameter}.csv", mode="w", newline=""
        ) as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(
                ["mask_index", "mean", "std", "min", "max", "Pixels", "Mean R^2"]
            )
            for key, value in results.items():
                value = [v.replace(".", decimal) for v in value]
                writer.writerow([key] + value)
        if return_params is None or parameter in return_params:
            return_list.append(results)

    return return_list if return_list else None


def save_nii(nii: np.ndarray, affine: np.ndarray, header: Any, file: Path) -> None:
    """
    Save a numpy array as a Nifti file.

    :param nii: The array to be saved.
    :param affine: Affine transformation for the Nifti file.
    :param header: Header information for the Nifti file.
    :param file: Path to save the Nifti file.
    """
    nii = nii.astype("uint16")
    nib.save(nib.Nifti1Image(nii, affine=affine, header=header), file)
