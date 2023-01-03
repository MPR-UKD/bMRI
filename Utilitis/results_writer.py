import nibabel as nib
import numpy as np
from pathlib import Path
import csv
from typing import Callable
from Utilitis.utils import get_function_parameter

def save_results(
    function: Callable | None,
    fit_map: np.ndarray | None,
    r2: np.ndarray | None,
    mask: np.ndarray | None,
    affine: np.ndarray,
    header: np.ndarray | None,
    nii_folder: str | Path,
    results_path: str | Path,
    decimal: str = ",",
):
    nii_folder = Path(nii_folder)
    results_path = Path(results_path)
    nii_folder.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if r2 is not None:
        save_nii(r2, affine, header, nii_folder / "r2.nii.gz")

    save_nii(fit_map, affine, header, nii_folder / "params.nii.gz")

    if function is not None:
        parameters = get_function_parameter(function)
    else:
        parameters = ''
    for ii, parameter in enumerate(parameters):
        save_nii(fit_map[ii], affine, header, nii_folder / f"{parameter}_map.nii.gz")
        results = {}
        for i in range(1, int(mask.max()) + 1):
            m = mask.copy()
            m = np.where(m == i, 1, 0)

            times = fit_map[ii][m == 1]
            if len(times) == 0:
                continue

            results[str(i)] = [
                "%.2f" % np.nanmean(times),
                "%.2f" % np.nanstd(times),
                "%.2f" % np.nanmin(times),
                "%.2f" % np.nanmax(times),
                "%.2f" % len(times[~np.isnan(times)]) + "/" + "%.2f" % np.sum(m),
                "%.2f" % np.nanmean(r2[m == 1]) if r2 is not None else "NaN",
            ]
        with open(results_path.as_posix() + f'_{parameter}.csv', mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(
                ["mask_index", "mean", "std", "min", "max", "Pixels", "Mean R^2"]
            )
            for key, value in results.items():
                value = [v.replace(".", decimal) for v in value]
                writer.writerow([key] + value)



def save_nii(nii: np.ndarray, affine, header, file: Path):
    nii = nii.astype('uint16')
    nib.save(nib.Nifti1Image(nii, affine=affine, header=header), file)
