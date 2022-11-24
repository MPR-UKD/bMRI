from pathlib import Path
import pydicom
import numpy as np
import nibabel as nib
from natsort import natsorted


def get_dcm_list(folder: Path):
    return natsorted(folder.glob("*.dcm"))


def split_dcm_list(dcm_list: list):
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


def check_locations(locations: dict):
    keys = [key for key in locations.keys()]
    ls = [len(locations[key]) for key in locations.keys()]
    echos = np.median(ls)
    idx = []
    for i, l in enumerate(ls):
        if (l - echos) != 0.0:
            idx.append(i)
    if len(idx) == 2:
        locations[keys[idx[0]]] += locations[keys[idx[1]]]
        locations.pop(keys[idx[1]])
    return locations


def get_dcm_array(data: list):
    array = []
    for d in data:
        img = pydicom.dcmread(d).pixel_array
        info = pydicom.dcmread(d)
        try:
            img = img * info.RescaleSlope + info.RescaleIntercept
        except AttributeError:
            pass
        array.append(img)
    # array = array[::-1]
    return np.array(array)


class Mask:
    def __init__(self, array, affine, header):
        self.array = array
        self.affine = affine
        self.header = header


def load_nii(file: Path):
    nimg = nib.load(file)
    mask = Mask(nimg.get_fdata()[:, :, ::-1], nimg.affine, nimg.header)
    return mask
