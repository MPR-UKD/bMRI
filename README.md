**Still in progress**

# bMRI [![Actions Status](https://github.com/ludgerradke/bMRI/blob/main/.github/workflows/test.yml/badge.svg)](https://github.com/ludgerradke/bMRI/blob/main/.github/workflows/test.yml)


bMRI is an open-source framework designed for the analysis of bio-sensitive MRI data. The goal of this project is to provide a comprehensive, flexible and easy-to-use tool for researchers and professionals working with MRI data. **The project is currently still in progress** and is completely written in Python. 

The framework is focused on leveraging the power of Python's scientific computing stack to process and analyze MRI data. It uses widely adopted libraries such as NumPy and SciPy to ensure efficiency and interoperability with other Python tools. 

## License

This project is licensed under the GPL-3.0 license.

## Code Structure

The codebase is structured around a few key components:

1. The `Fitting` package: This package provides tools and utilities for curve fitting and the analysis of DICOM medical imaging data, specifically focusing on `T2*`, `T2`, `T1rho` and `T1` relaxation times.
2. The `Utilitis` package: This package provides some support function for reading and writing.
3. The `Visualisation` package: THhs package provides the ImageViewer Class, which enables the visualisation of data and results (see below).

## Installation and Requirements

You can install all requirements with the command:`pip install -r requirements.txt`. 

## Usage

The `Fitting` objects can process 2D or 3D data and can be adjusted by passing configurations and optional boundary values. All classes have a 'read_data' function for reading the DICOM images, a 'Fit' function for determining the relaxation times, and a 'Run' function, which automatically reads the images and saves the results as .csv as well as .nii for later statistical analysis and visualizations. Optionally, the number of parallel pools for computation and a minimum R^2 value for a valid fit can be specified.

### T2 and T2*-Evaluation

The T2 and T2* evaluation is based on a mono-exponential adjustment.

Here is an example code that shows how to use the `T2_T2star` class to evaluate T2 and T2* relaxation times:

```python
from pathlib import Path
from src.Fitting import T2_T2star

# Define the DICOM directory and the mask file path
dicom_dir = Path("/path/to/dicom/directory")
mask_file = Path("/path/to/mask/file")

# Initialize the T2_T2star object
t2_t2star = T2_T2star(dim=3, boundary=(0, 1000), normalize=True)

# Run the full evaluation pipeline
results = t2_t2star.run(dicom_folder=dicom_dir, mask_file=mask_file)
```

### T1rho and T2prep Evaluation

The T1rho and T2prep evaluation can be accomplished through the specialized `T1rho_T2prep` class. A key feature of this class is its flexibility to choose between different fitting models according to various research studies. By default, if no specific configuration is provided, the fitting falls back to a mono-exponential fitting function. However, it also supports more complex models such as those developed by Rausch et al. and Aronen et al. These models take into account additional MRI sequence parameters such as repetition time (TR), longitudinal relaxation time (T1), flip angle (alpha), echo time (TE), and transverse relaxation time (T2star).

- The **Rausch model** (`fit_T1rho_wrapper_raush`) is based on a T1rho fitting function that models the signal with the given parameters and considers the counter and denominator terms in the signal equation.
- The **Aronen model** (`fit_T1rho_wrapper_aronen`) is another T1rho fitting function that takes additional parameters into account and utilizes a more complex model to fit the data.
- The **Mono-Exponential model** (`fit_mono_exp_wrapper`) is a simple model where the signal is an exponential decay.

Example usage:

```python
from pathlib import Path
from src.Fitting import T1rho_T2prep

# Define the DICOM directory and the mask file path
dicom_dir = Path("/path/to/dicom/directory")
mask_file = Path("/path/to/mask/file")

# Define the fitting configuration
config = {
    "TR": 1000,
    "T1": 1500,
    "alpha": 90,
    "TE": 20,
    "T2star": 50
}

# Initialize the T1rho_T2prep object
t1rho_t2prep = T1rho_T2prep(dim=3, config=config, boundary=(0, 1000), normalize=True)

# Run the full evaluation pipeline
results = t1rho_t2prep.run(dicom_folder=dicom_dir, mask_file=mask_file, tsl=[10, 20, 30, 40])
```

### T1 Evaluation

The T1 evaluation is performed using the `T1` class. Like the other classes, the `T1` class can be used to evaluate the T1 relaxation time based on DICOM images. Typically, T1 relaxation time is calculated using various methods, such as the Inversion Recovery (IR) method or the Look-Locker method. In this example, I will provide a general structure for using the `T1` class, assuming it is implemented similarly to the T2/T2* and T1rho/T2prep classes.

```python
from pathlib import Path
from src.Fitting import T1

# Define the DICOM directory and the mask file path
dicom_dir = Path("/path/to/dicom/directory")
mask_file = Path("/path/to/mask/file")

# Initialize the T1 object
t1_evaluator = T1(dim=3, boundary=(0, 5000), normalize=True)

# Run the full evaluation pipeline
results = t1_evaluator.run(dicom_folder=dicom_dir, mask_file=mask_file)
```

### Evaluation of FitMaps (e.g., dGEMRIC)

In medical imaging, evaluating Fitted Maps such as dGEMRIC (delayed Gadolinium-Enhanced MRI of Cartilage) is crucial for analyzing certain aspects of the tissues, like the distribution of contrast agents within the cartilage. The FittedMap class in this example is designed for processing and analyzing such data.


### Image Viewer

The Image Viewer is a PyQt5 based graphical user interface (GUI) that enables visualization of DICOM data and fitted maps. The viewer is contained in the `ImageViewer` class, which provides functionalities to display DICOM images in slices along with an overlaid color map, and a plot of the fitting function for the selected voxel.

![](assets/image_viewer.gif)
#### Features:
- Display DICOM images in slices with a slider to navigate through the slices.
- Display fitted maps as an overlay on the DICOM images.
- Display the fit function in a separate plot.
- Normalize the DICOM data to a range of [0,1].
- Use customizable color maps.
- Zoom the image for better visualization.

#### Usage:

The `ImageViewer` class is used by creating an instance of the class and calling the `start()` method with the necessary parameters, as demonstrated in the example code below.

#### Example Code

```python
from src.Utilitis import load_nii
from src.Visualization.image_viewer import ImageViewer
from src.Fitting import T2_T2star
from PyQt5.QtWidgets import QApplication
import sys
from pathlib import Path

# Replace the paths below with the paths to your files
path_to_dicom = Path('path_to_dicom_file')
path_to_fit_maps = Path('path_to_fit_maps')

# Define the fit function
t2 = T2_T2star(dim=3)
fit_function = t2.fit_function

# List of time points / Alternative you can read the saved timepoints with the fitting clas
time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a QApplication instance
app = QApplication(sys.argv)

# Create an instance of the ImageViewer class
viewer = ImageViewer()

# Start the viewer
viewer.start(
    dicom=path_to_dicom,
    fit_maps=path_to_fit_maps,
    fit_function=fit_function,
    time_points=time_points,
    c_int=1,
    alpha=0.3,
    normalize=True
)

# Show the viewer
viewer.show()

# Run the PyQt5 application
sys.exit(app.exec_())
```

## Contributing

We welcome contributions to bMRI! If you have a feature request, bug report, or want to contribute code, please feel free to open an issue or submit a pull request. When submitting code, please make sure to follow the existing code style and conventions.
