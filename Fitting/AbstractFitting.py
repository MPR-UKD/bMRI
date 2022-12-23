from abc import ABC
from multiprocessing import Pool, cpu_count
from typing import Callable

import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from functools import partial
from itertools import repeat

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
            pools: int = cpu_count(),
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
        fit_maps = np.empty((num_params, *mask.shape))
        r2_map = np.zeros(mask.shape)

        # Create a partial function with the fixed arguments for fit_pixel
        fit_pixel_fixed = partial(fit_pixel, fit_function=self.fit_function, bounds=self.bounds, config=self.fit_config)

        # Create an iterator of arguments to pass to fit_pixel
        pixel_args = zip(
            (dicom[:, i, j, k] for i, j, k in zip(*np.nonzero(mask))),
            repeat(x),
        )

        # Use a Pool to fit the pixels in parallel
        with Pool(cpu_count()) as pool:
            # Call fit_pixel for each pixel and store the result in a list
            pixel_results = pool.starmap(fit_pixel_fixed, pixel_args)

            # Iterate through the list of results and store the fitting parameters and r2 values in the appropriate
            # positions in the fit_maps and r2_map arrays
            for i, j, k, param in zip(*np.nonzero(mask), pixel_results):
                if param is None:
                    continue
                r2_map[i, j, k] = calculate_r2(dicom[:, i, j, k], self.fit_function, param, x)
                if r2_map[i, j, k] > min_r2:
                    for p_num, p in enumerate(param):
                        fit_maps[p_num][i, j, k] = p

        return fit_maps, r2_map


def fit_pixel(y, x, fit_function, bounds=None, config=None):
    kwargs = {}
    if bounds is not None:
        kwargs['bounds'] = bounds
    if config is not None:
        for key in config:
            kwargs[key] = config[key]
    try:
        param, param_cov = curve_fit(fit_function, x, y, **kwargs)
    except (RuntimeError, ValueError):
        param = None
    return param


def calculate_r2(y, fit_function, param, x):
    residuals = y - fit_function(x, *param)
    return get_r2(residuals, y)

@njit
def get_r2(residuals: np.ndarray, y: np.ndarray) -> float:
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)
