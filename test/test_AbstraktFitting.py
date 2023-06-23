import numpy as np
import pytest
from numba import njit

from src.Fitting import AbstractFitting


def test_init():
    # Test that the fit_function and boundary are correctly assigned
    fit_function = lambda x, a, b: a * x + b
    boundary = (0, 1)
    fitting = AbstractFitting(fit_function, boundary)
    assert fitting.fit_function == fit_function
    assert fitting.bounds == boundary


def test_set_fit_config():
    # Test that the fit_config attribute is correctly set
    fit_function = lambda x, a, b: a * x + b
    boundary = (0, 1)
    fitting = AbstractFitting(fit_function, boundary)
    fit_config = {"maxfev": 1000}
    fitting.set_fit_config(fit_config)
    assert fitting.fit_config == fit_config


@pytest.mark.parametrize("pools", [0, 1, 2, 3])
def test_fit(pools):
    # Test that the fit function returns the expected output

    # Define a fit function
    @njit
    def fit_function(x, a, b):
        return a * x + b

    # Create an instance of the AbstractFitting class
    fitting = AbstractFitting(fit_function)

    # Generate some data using the fit function
    a, b = 1, 2
    x = np.array([1, 2, 3, 4, 5, 6])
    y = fit_function(x, a, b)
    dicom = np.array([np.ones((2, 2)) * y[i] for i in range(len(x))])
    dicom = dicom.reshape((6, 2, 2, 1))

    # Create a mask
    mask = np.ones((2, 2, 1))

    # Fit the data
    fit_maps, r2_map = fitting.fit(dicom, mask, x, pools=pools)

    # Assert that the output is as expected
    assert len(fit_maps) == 2
    assert abs(fit_maps[0][0, 0] - a) < 0.001
    assert abs(fit_maps[1][0, 0] - b) < 0.001


def test_fit_reshaped_2D_to_3D():
    # Define a fit function
    @njit
    def fit_function(x, a, b):
        return a * x + b

    # Create an instance of the AbstractFitting class
    fitting = AbstractFitting(fit_function)

    # Generate some data using the fit function
    a, b = 1, 2
    x = np.array([1, 2, 3, 4, 5, 6])
    y = fit_function(x, a, b)
    dicom = np.array([np.ones((2, 2)) * y[i] for i in range(len(x))])
    dicom = dicom.reshape((6, 2, 2))

    # Create a mask
    mask = np.ones((2, 2))

    # Fit the data
    fit_maps, r2_map = fitting.fit(dicom, mask, x)

    # Assert that the output is as expected
    assert len(fit_maps) == 2
    assert len(r2_map.shape) == 3


if __name__ == "__main__":
    pytest.main()
