from typing import Callable
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from pathlib import Path

class AbstractFitting(ABC):

    def __init__(self,
                 fit_function: Callable,
                 boundary: tuple):
        self.fit_function = fit_function
        self.bounds = boundary
        self.fit_config = None

    @abstractmethod
    def set_fit_config(self):
        pass

    @abstractmethod
    def read_data(self, folder: str | Path | list):
        raise NotImplementedError

    def fit(self,
            dicom: np.ndarray,
            mask: np.ndarray,
            x: np.ndarray,
            multiprocessing: bool = False,
            min_r2: float = -np.inf):
        fit_map = np.zeros(mask.shape)
        r2_map = np.zeros(mask.shape)

        if multiprocessing:
            with Pool(cpu_count()) as pool:
                idxs, map_slices, r2_slices = zip(*pool.map(
                    fit_slice,
                    [(dicom[:, :, :, i], mask[:, :, i], x, self.fit_function, self.bounds, min_r2, self.fit_config)
                     for i in range(dicom.shape[-1])]
                ))
                for i in idxs:
                    fit_map[:, :, i], r2_map[:, :, i] = map_slices[i], r2_slices[i]
        else:
            for i in range(dicom.shape[-1]):
                fit_map[:, :, i], r2_map[:, :, i] = fit_slice(dicom[:, :, :, i],
                                                              mask[:, :, i], x,
                                                              self.fit_function,
                                                              self.bounds,
                                                              min_r2,
                                                              self.fit_config)
        return fit_map, r2_map


def fit_slice(dicom: np.ndarray,
              mask: np.ndarray,
              x: np.ndarray,
              fit_function: Callable,
              bounds: tuple | None = None,
              min_r2: float = 0.9,
              config: dict | None = None,
              slice_idx: int | None = None):
    dicom = dicom.astype('float64')
    fit_map = np.full((dicom.shape[1], dicom.shape[2]), np.nan)
    r2_map = np.full((dicom.shape[1], dicom.shape[2]), np.nan)
    if mask.max() != 0:
        non_zero_mask_pixel = np.argwhere(mask != 0)
        for row, column in non_zero_mask_pixel:
            y = dicom[:, row, column]
            y /= y.max()
            try:
                if bounds is None and config is None:
                    param, param_cov = curve_fit(fit_function, x, y)
                elif config is None:
                    param, param_cov = curve_fit(fit_function, x, y, bounds=bounds)
                elif bounds is None:
                    param, param_cov = curve_fit(fit_function, x, y, xtol=config['xtol'], maxfev=config['maxfev'])
                else:
                    raise IndexError
            except RuntimeError:
                continue
            except ValueError:
                continue
            residuals = y - fit_function(x, param[0], param[1], param[2])
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            if r2 < min_r2:
                continue
            fit_map[row, column] = param[1]
            r2_map[row, column] = r2

    if slice_idx is None:
        return fit_map, r2_map
    else:
        return slice_idx, fit_map, r2_map
