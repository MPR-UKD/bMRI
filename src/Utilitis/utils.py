import numpy as np
from typing import Callable


def get_function_parameter(f: Callable) -> tuple:
    """
    Get the names of the parameters of a given function.

    Parameters:
    - f: The function for which to get parameter names.

    Returns:
    - Tuple containing the names of the parameters.
    """
    return f.__code__.co_varnames[: f.__code__.co_argcount][1:]


def binary_grow_3d(
    array: np.ndarray, dist: int = 1, threshold: float = 0.25
) -> np.ndarray:
    """
    Grow a binary 3D array by setting surrounding elements of 1s to 1s.
    The distance of the surrounding elements to consider can be specified through the dist parameter.
    Only the elements whose average value within the specified distance is above the threshold will be set to 1s.

    Parameters:
    - array: A 3D ndarray with binary values (0s and 1s).
    - dist: Distance of surrounding elements to consider (default is 1).
    - threshold: Threshold for average value to set an element to 1 (default is 0.25).

    Returns:
    - A 3D ndarray with the same shape as the input array, but with potentially more 1s.
    """
    # Create a copy of the input array
    grow_array = array.copy()

    # Find the indices of all the 1s in the array
    x, y, z = np.where(array == 1)

    # If no elements are 1, return the original array
    if len(x) == 0:
        return array

    # Find the minimum and maximum indices for each dimension, ensuring they are within bounds
    x_min = max(x.min() - dist, 0)
    x_max = min(x.max() + dist, array.shape[0])

    y_min = max(y.min() - dist, 0)
    y_max = min(y.max() + dist, array.shape[1])

    z_min = max(z.min() - dist, 0)
    z_max = min(z.max() + dist, array.shape[2])

    # Iterate through each element in the array
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            for k in range(z_min, z_max):
                # If the element is 1, skip it
                if array[i, j, k] == 1:
                    continue

                # Calculate local indices for neighborhood
                local_i_min = max(i - dist, 0)
                local_i_max = min(i + dist, array.shape[0])

                local_j_min = max(j - dist, 0)
                local_j_max = min(j + dist, array.shape[1])

                local_k_min = max(k - dist, 0)
                local_k_max = min(k + dist, array.shape[2])

                # If the mean value of the surrounding elements is greater than threshold, set the element to 1
                if (
                    array[
                        local_i_min:local_i_max,
                        local_j_min:local_j_max,
                        local_k_min:local_k_max,
                    ].mean()
                    > threshold
                ):
                    grow_array[i, j, k] = 1

    # Return the grown array
    return grow_array
