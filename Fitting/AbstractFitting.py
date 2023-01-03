from abc import ABC
from multiprocessing import Pool, cpu_count
from typing import Callable, Tuple

import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from functools import partial
from itertools import repeat


class AbstractFitting(ABC):
    def __init__(
            self,
            fit_function: Callable,
            boundary: Tuple[float, float] = None,
            fit_config: dict | None = None,
            normalize: bool = False
    ):
        self.fit_function = fit_function
        self.bounds = boundary
        self.fit_config = fit_config
        self.normalize = normalize

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
        dicom = dicom.astype('float64')
        assert len(mask.shape) == len(dicom.shape) - 1
        if len(mask.shape) == 2:
            # Change size of dicom and mask to 3D
            mask = np.expand_dims(mask, axis=2)
            dicom = np.expand_dims(dicom, axis=3)
        assert dicom.shape[0] == len(x)
        # Get the number of parameters in the fit function
        num_params = len(curve_fit(self.fit_function, x, dicom[:, 0, 0, 0])[0])

        # Init fit_maps list - Each fit_map is a 3D array and stores the result of the fitting parameter (order
        # similar to the fit function)
        fit_maps = np.empty((num_params, *mask.shape))
        r2_map = np.zeros(mask.shape)

        # Create a partial function with the fixed arguments for fit_pixel
        fit_pixel_fixed = partial(
            fit_pixel,
            fit_function=self.fit_function,
            bounds=self.bounds,
            config=self.fit_config,
            normalize=self.normalize
        )

        # Create an iterator of arguments to pass to fit_pixel
        pixel_args = zip(
            (dicom[:, i, j, k] for i, j, k in zip(*np.nonzero(mask))),
            repeat(x),
        )

        # Use a Pool to fit the pixels in parallel
        if pools != 0:
            with Pool(pools) as pool:
                # Call fit_pixel for each pixel and store the result in a list
                pixel_results = pool.starmap(fit_pixel_fixed, pixel_args)
        else:
            pixel_results = [fit_pixel_fixed(*p) for p in pixel_args]

        # Iterate through the list of results and store the fitting parameters and r2 values in the appropriate
        # positions in the fit_maps and r2_map arrays
        for i, j, k, param in zip(*np.nonzero(mask), pixel_results):
            if param is None:
                continue
            r2_map[i, j, k] = calculate_r2(
                dicom[:, i, j, k], self.fit_function, param, x, self.normalize
            )
            if r2_map[i, j, k] > min_r2:
                for p_num, p in enumerate(param):
                    fit_maps[p_num][i, j, k] = p

        return np.array(fit_maps), r2_map


def fit_pixel(
        y: np.ndarray,
        x: np.ndarray,
        fit_function: Callable,
        bounds: Tuple[float, float] = None,
        config: dict = None,
        normalize: bool = False
) -> np.ndarray:
    """
    Fits a curve to the given data using the provided fit function.

    Parameters:
    - y: 1D array of dependent variable data
    - x: 1D array of independent variable data
    - fit_function: function to use for fitting the curve
    - bounds: optional bounds for the curve fit parameters
    - config: optional config dictionary for curve_fit function

    Returns:
    - param: array of curve fit parameters
    """
    if normalize:
        y /= y.max()
    kwargs = {'xtol': 0.0000000001}
    if bounds is not None:
        kwargs["bounds"] = bounds
    if config is not None:
        for key in config:
            kwargs[key] = config[key]
    try:
        param, param_cov = curve_fit(fit_function, x, y, **kwargs)
    except (RuntimeError, ValueError):
        param = None
    return param


def calculate_r2(
        y: np.ndarray, fit_function: Callable, param: np.ndarray, x: np.ndarray, normalize: bool = False
) -> float:
    if normalize:
        y /= y.max()
    residuals = y - fit_function(x, *param)
    return get_r2(residuals, y)


@njit
def get_r2(residuals: np.ndarray, y: np.ndarray) -> float:
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)
