import nibabel as nib
import numpy as np
from pathlib import Path
import csv


def save_results(fit_map: np.ndarray | None,
                 r2: np.ndarray | None,
                 mask: np.ndarray | None,
                 affine: np.ndarray,
                 header: np.ndarray,
                 nii_folder: str | Path,
                 result_file: str | Path,
                 decimal: str = ','):
    nii_folder = Path(nii_folder)
    nii_folder.mkdir(parents=True, exist_ok=True)

    save_nii(fit_map, affine, header, nii_folder / 'map.nii.gz')
    save_nii(r2, affine, header, nii_folder / 'r2.nii.gz')

    results = {}
    for i in range(1, int(mask.max()) + 1):
        m = mask.copy()
        m = np.where(m == i, 1, 0)

        fit_map = np.multiply(fit_map, m)
        k = fit_map.copy()
        k[k > 0] = 1
        fit_map = np.where(fit_map != 0.0, fit_map, np.nan)

        r_squares = np.multiply(self.r_squares, m)
        r_squares = np.where(r_squares != 0, r_squares, np.nan)

        results[str(i)] = ['%.2f' % np.nanmean(fit_map), '%.2f' % np.nanstd(fit_map),
                           '%.2f' % np.nanmin(fit_map), '%.2f' % np.nanmax(fit_map),
                           '%.2f' % np.nansum(k) + '/' + '%.2f' % np.sum(m),
                           '%.2f' % np.nanmean(r_squares)]
        with open(result_file, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['mask_index', 'mean', 'std', 'min', 'max', 'Pixels', 'Mean R^2'])
            for key, value in results.items():
                value = [v.replace('.', decimal) for v in value]
                writer.writerow([key] + value)
    return results


def save_nii(nii, affine, header, file):
    nii = nii.astype(np.uint16)
    nib.save(nib.Nifti1Image(nii, affine=affine, header=header), file)