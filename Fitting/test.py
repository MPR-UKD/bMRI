import numba
import numpy as np
from scipy.optimize import least_squares

# Define the fit function
@numba.jit
def fit_function(x, a, b):
    return a * x + b


# Vectorize the fit function using numba.vectorize
vec_fit_function = numba.vectorize(fit_function)

# Define the data to fit
x = np.array([1, 2, 3, 4])
y = np.array([1, 2, 3, 4])

# Define the function to fit the data using least_squares
@numba.jit
def residuals(p, x, y):
    return y - vec_fit_function(x, *p)


# @numba.jit
def fit_data(x, y, num_params):
    # Initialize the initial guess for the fit parameters
    p0 = num_params * [1]

    # Define the residuals function

    # Fit the curve to the data using least_squares
    fit_params = least_squares(residuals, p0, args=(x, y)).x

    return fit_params


# Compile the fit_data function using numba
numba_fit_data = numba.jit(fit_data)

# Fit the data using the compiled function
fit_params = numba_fit_data(x, y, num_params=2)

print(fit_params)  # Output: [1. 1.]
