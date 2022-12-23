import time

import numpy as np
import pytest
from numba import njit

from Fitting.AbstractFitting import AbstractFitting


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


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_fit(multiprocessing):
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
    fit_maps, r2_map = fitting.fit(dicom, mask, x, multiprocessing=multiprocessing)

    # Assert that the output is as expected
    assert len(fit_maps) == 2
    assert abs(fit_maps[0][0, 0] - a) < 0.001
    assert abs(fit_maps[1][0, 0] - b) < 0.001


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_fit_reshaped_2D_to_3D(multiprocessing):
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
    fit_maps, r2_map = fitting.fit(dicom, mask, x, multiprocessing=multiprocessing)

    # Assert that the output is as expected
    assert len(fit_maps) == 2
    assert len(r2_map.shape) == 3


def test_multiprocessing_performance():
    size_x, size_y = 200, 200

    @njit
    def fit_function(x, a, b, c):
        return a * np.exp(-x / b) + c

    # Create an instance of the AbstractFitting class
    fitting = AbstractFitting(fit_function)

    # Generate some data using the fit function
    a, b, c = 1, 2, 3
    x = np.array([1, 2, 3, 4, 5, 6])
    y = fit_function(x, a, b, c)
    dicom = np.array([np.ones((size_x, size_y)) * y[i] for i in range(len(x))])
    dicom = dicom.reshape((6, size_x, size_y, 1))

    # Create a mask
    mask = np.ones((size_x, size_y, 1))

    # Fit the data
    start = time.time()
    fit_maps, r2_map = fitting.fit(dicom, mask, x, multiprocessing=False)
    end = time.time()
    print(end - start)
    # Assert that the output is as expected
    assert len(fit_maps) == 3
    assert len(r2_map.shape) == 3


# 86.4
# 68.1

if __name__ == "__main__":
    pytest.main()
