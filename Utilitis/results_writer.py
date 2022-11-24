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
    if r2 is not None:
        save_nii(r2, affine, header, nii_folder / 'r2.nii.gz')

    results = {}
    for i in range(1, int(mask.max()) + 1):
        m = mask.copy()
        m = np.where(m == i, 1, 0)

        times = fit_map[m == 1]
        if len(times) == 0:
            continue

        results[str(i)] = ['%.2f' % np.nanmean(times), '%.2f' % np.nanstd(times),
                           '%.2f' % np.nanmin(times), '%.2f' % np.nanmax(times),
                           '%.2f' % len(times[~np.isnan(times)]) + '/' + '%.2f' % np.sum(m),
                           '%.2f' % np.nanmean(r2[m == 1]) if r2 is not None else 'NaN']
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
