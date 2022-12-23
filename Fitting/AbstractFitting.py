from abc import ABC
from multiprocessing import Pool, cpu_count
from typing import Callable

import numpy as np
from numba import njit
from scipy.optimize import curve_fit


class AbstractFitting(ABC):
    def __init__(self, fit_function: Callable, boundary: tuple | None = None):
        self.fit_function = fit_function
        self.bounds = boundary
        self.fit_config = None

    def set_fit_config(self, fit_config):
        self.fit_config = fit_config

    def fit(
            self,
            dicom: np.ndarray,
            mask: np.ndarray,
            x: np.ndarray,
            multiprocessing: bool = False,
            min_r2: float = -np.inf,
    ):
        assert len(mask.shape) == len(dicom.shape) - 1
        if len(mask.shape) == 2:
            # Change size of dicom and mask to 3D
            mask = np.expand_dims(mask, axis=2)
            dicom = np.expand_dims(dicom, axis=3)
        assert dicom.shape[0] == len(x)
        # Get the number of parameters in the fit function
        num_params = len(curve_fit(self.fit_function, x, dicom[:, 0, 0, 0])[0])

        # Init fit_map list - Each fit_map is a 3D array and storage the result of the fitting parameter (order
        # similar to the fit function)
        fit_maps = [np.zeros(mask.shape) for _ in range(num_params)]
        r2_map = np.zeros(mask.shape)

        if multiprocessing:
            with Pool(cpu_count()) as pool:
                idxs, map_slices, r2_slices = zip(
                    *pool.starmap(
                        fit_slice,
                        [
                            (
                                dicom[:, :, :, i],
                                mask[:, :, i],
                                x,
                                self.fit_function,
                                self.bounds,
                                min_r2,
                                self.fit_config,
                                i,
                            )
                            for i in range(dicom.shape[-1])
                        ],
                    )
                )
                for i in idxs:
                    r2_map[:, :, i] = r2_slices[i]
                    for j, slice_param_map in enumerate(map_slices[i]):
                        fit_maps[j][:, :, i] = slice_param_map
        else:
            for i in range(dicom.shape[-1]):
                _, slice_param_maps, slice_r2_map = fit_slice(
                    dicom[:, :, :, i],
                    mask[:, :, i],
                    x,
                    self.fit_function,
                    self.bounds,
                    min_r2,
                    self.fit_config,
                )
                r2_map[:, :, i] = slice_r2_map
                for j, slice_param_map in enumerate(slice_param_maps):
                    fit_maps[j][:, :, i] = slice_param_map
        return fit_maps, r2_map


def fit_slice(
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        fit_function,
        bounds: np.ndarray = None,
        min_r2: float = 0.9,
        config: dict = None,
        slice_idx: int = None,
        normalize: bool = False,
) -> tuple:
    """Fit the curve defined by `fit_function` to the data in `dicom` and `mask` using `fit_pixel`.

    Parameters:
    dicom (numpy.ndarray): A 3D array of DICOM data.
    mask (numpy.ndarray): A 2D array of masks.
    x (numpy.ndarray): An array of independent variables.
    fit_function (function): A fit function.
    bounds (numpy.ndarray, optional): Bounds for the curve fitting.
    min_r2 (float, optional): The minimum R-squared value for a successful fit.
    config (dict, optional): A dictionary of optimization options.
    slice_idx (int, optional): An index for the slice being fitted.

    Returns:
    tuple: A tuple containing the slice index (if provided), the fitting parameter maps, and the R-squared map.
    """
    # Convert dicom to float64 type
    dicom = dicom.astype("float64")

    # Get the number of parameters in the fit function
    num_params = len(curve_fit(fit_function, x, dicom[:, 0, 0])[0])

    # Initialize empty parameter maps and r2 map
    param_maps = [
        np.full((dicom.shape[1], dicom.shape[2]), np.nan) for _ in range(num_params)
    ]
    r2_map = np.full((dicom.shape[1], dicom.shape[2]), np.nan)

    # If mask has no non-zero values, return the empty maps
    if mask.max() == 0:
        return slice_idx, *param_maps, r2_map

    # Get the coordinates of non-zero mask pixels
    non_zero_mask_pixel = np.argwhere(mask != 0)

    # Init p0
    if bounds is None:
        p0 = num_params * [1]
    else:
        lower, upper = bounds
        p0 = [np.mean([l, u]) for l, u in zip(lower, upper)]
    # Iterate over the non-zero mask pixels
    for row, column in non_zero_mask_pixel:
        # Get the y-values for the pixel
        y = dicom[:, row, column]

        if normalize:
            # Normalize the y-values
            y /= y.max()

        # Fit the curve to the pixel data
        try:
            param, param_cov = fit_pixel(fit_function, x, y, bounds, config, p0)
        except:
            # If the fit failed, skip this pixel
            continue

        # Update initial fitting parameter
        p0 = param
        # Calculate the residuals and the R-squared value for the fit
        residuals = y - fit_function(x, *param)
        r2 = get_r2(residuals, y)

        # If the R-squared value is below the minimum skip the pixel
        if r2 < min_r2:
            continue
        for i, p in enumerate(param):
            param_maps[i][row, column] = p
        r2_map[row, column] = r2
    return slice_idx, param_maps, r2_map


def fit_pixel(fit_function, x, y, bounds=None, config=None, p0=None):
    kwargs = {}
    if bounds is not None:
        kwargs['bounds'] = bounds
    if config is not None:
        for key in config:
            kwargs[key] = config[key]
    if p0 is not None:
        kwargs['p0'] = p0
    param, param_cov = curve_fit(fit_function, x, y, **kwargs)
    return param, param_cov


@njit
def get_r2(residuals: np.ndarray, y: np.ndarray) -> float:
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)
