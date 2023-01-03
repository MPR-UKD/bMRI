import numpy as np

def get_function_parameter(f):
    return f.__code__.co_varnames[:f.__code__.co_argcount][1:]

def binary_grow_3d(array: np.ndarray) -> np.ndarray:
    # create a copy of the input array
    grow_array = array.copy()

    # get the shape of the array
    rows, cols, depth = array.shape

    # iterate through each element in the array
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                # if the element is 1, set all its surrounding elements to 1
                if array[i,j,k] == 1:
                    if i > 0:
                        grow_array[i-1,j,k] = 1
                    if i < rows-1:
                        grow_array[i+1,j,k] = 1
                    if j > 0:
                        grow_array[i,j-1,k] = 1
                    if j < cols-1:
                        grow_array[i,j+1,k] = 1
                    if k > 0:
                        grow_array[i,j,k-1] = 1
                    if k < depth-1:
                        grow_array[i,j,k+1] = 1

    # return the grown array
    return grow_array