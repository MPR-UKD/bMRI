
# AbstractFitting

The **'AbstractFitting'** class is an abstract base class (ABC) that defines a template for fitting curves to DICOM data. The class has the following attributes and methods:

- **'fit_function'**: a fit function that will be used to fit the curves to the data
- **'bounds'**: bounds for the curve fitting (optional)
- **'fit_config'**: a dictionary of optimization options (optional)
- **'set_fit_config'**: a method that sets the value of the fit_config attribute
- **'read_data'**: an abstract method that must be implemented in a subclass. It should read in the DICOM data from a given folder or list of files.
- **'fit'**: a method that fits curves to each slice of a 4D DICOM array using either a single process or multiple processes in parallel, depending on the value of the iprocessing parameter. The method returns a list of fitting parameter maps and an R-squared map for each slice.

The **'fit_slice'** function is a helper function that fits the curve defined by **'fit_function'** to the data in a single slice of a 3D DICOM array and returns a tuple containing the slice index (if provided), the fitting parameter maps, and the R-squared map. The function takes the following parameters:

- **'dicom'**: a 3D array of DICOM data
- **'mask'**: a 2D array of masks
- **'x'**: an array of independent variables
- **'fit_function'**: a fit function
- **'bounds'**: bounds for the curve fitting (optional)
- **'min_r2'**: the minimum R-squared value for a successful fit (optional)
- **'config'**: a dictionary of optimization options (optional)
- **'slice_idx'**: an index for the slice being fitted (optional)