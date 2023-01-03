import numpy as np


def get_function_parameter(f):
    return f.__code__.co_varnames[:f.__code__.co_argcount][1:]


def binary_grow_3d(array: np.ndarray, dist: int = 1, threshold: float = 0.25) -> np.ndarray:
    """
    This function grows a binary 3D array by setting all surrounding elements of 1s to 1s.
    The distance of the surrounding elements to consider can be specified through the dist parameter.
    The default value for dist is 1, meaning that only the direct neighbors of 1s will be set to 1s.
    """
    # create a copy of the input array
    grow_array = array.copy()

    # find the indices of all the 1s in the array
    x, y, z = np.where(array == 1)
    if len(x) == 0:
        return array
    # find the minimum and maximum indices for each dimension
    x_min, x_max = x.min() - dist, x.max() + dist
    y_min, y_max = y.min() - dist, y.max() + dist
    z_min, z_max = z.min() - dist, z.max() + dist

    # iterate through each element in the array
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            for k in range(z_min, z_max):
                # if the element is 1, skip it
                if array[i, j, k] == 1:
                    continue
                # if the mean value of the surrounding elements is greater than threshold, set the element to 1
                if array[i - dist:i + dist, j - dist:j + dist, k - dist:k + dist].mean() > threshold:
                    grow_array[i, j, k] = 1

    # return the grown array
    return grow_array
