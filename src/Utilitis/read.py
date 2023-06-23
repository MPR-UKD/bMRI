from pathlib import Path
import pydicom
import numpy as np
import nibabel as nib
from natsort import natsorted
from typing import List, Dict, Any, Union
from collections import defaultdict


def get_dcm_list(folder: Path) -> List[Path]:
    """
    Get a naturally sorted list of DICOM files from a directory.

    :param folder: The directory containing the DICOM files.
    :return: A list of paths to the DICOM files.
    """
    return natsorted(folder.glob("*.dcm"))


def split_dcm_list(dcm_list: List[Path]) -> List[List[Path]]:
    """
    Split the list of DICOM files based on Slice Location.

    :param dcm_list: List of paths to DICOM files.
    :return: List of lists of DICOM file paths, split by slice location.
    """
    locations = {}
    for f in dcm_list:
        try:
            d = pydicom.dcmread(f)
        except BaseException:
            continue
        if d["SliceLocation"].value in locations.keys():
            locations[d["SliceLocation"].value].append(f)
        else:
            locations[d["SliceLocation"].value] = [f]
    locations = check_locations(locations)
    split_dcmList = [locations[key] for key in natsorted(list(locations.keys()))]
    echo_list = [[] for _ in range(len(split_dcmList[0]))]
    keys = list(locations.keys())
    keys.sort()
    for key in keys:
        echos = locations[key]
        for idx in range(len(echo_list)):
            echo_list[idx].append(echos[idx])
    return echo_list


def check_locations(locations: Dict[Any, List[Path]]) -> Dict[Any, List[Path]]:
    """
    Check if the number of echo images is consistent in the given locations dict.

    :param locations: Dictionary of locations with lists of DICOM file paths.
    :return: A modified locations dictionary.
    """
    keys = list(locations.keys())
    ls = [len(locations[key]) for key in keys]
    echos = np.median(ls)
    idx = [i for i, l in enumerate(ls) if (l - echos) != 0.0]

    if len(idx) == 2:
        key_a, key_b = keys[idx[0]], keys[idx[1]]
        locations[key_a].extend(locations[key_b])
        del locations[key_b]
    return locations


def get_dcm_array(data: List[Path]) -> np.ndarray:
    """
    Load DICOM images from file paths and convert them into a numpy array.

    :param data: List of paths to DICOM files.
    :return: 3D numpy array representing the DICOM image stack.
    """
    array = []
    for d in data:
        try:
            with pydicom.dcmread(d) as img:
                info = pydicom.dcmread(d)
                try:
                    img = img.pixel_array * info.RescaleSlope + info.RescaleIntercept
                except AttributeError:
                    img = img.pixel_array
                array.append(img)
        except Exception as e:
            print(f"Error reading DICOM file {d}: {e}")
            continue
    return np.array(array)


class Mask:
    """
    Mask class representing a 3D mask with affine transformation and header data.
    """

    def __init__(self, array: np.ndarray, affine: np.ndarray, header: Any) -> None:
        self.array = array
        self.affine = affine
        self.header = header


def load_nii(file: Path) -> Mask:
    """
    Load a NIfTI file and create a Mask object from it.

    :param file: Path to the NIfTI file.
    :return: Mask object representing the NIfTI data.
    """
    nimg = nib.load(file)
    mask = Mask(nimg.get_fdata()[:, :, ::-1], nimg.affine, nimg.header)
    return mask
