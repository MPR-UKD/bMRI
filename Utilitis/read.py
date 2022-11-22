from pathlib import Path
import pydicom
import numpy as np
import nibabel as nib


def get_dcm_list(folder: Path):
    return sorted(folder.glob('*.dcm'))

def split_dcm_list(dcm_list: list):
    locations = {}
    for f in dcm_list:
        try:
            d = pydicom.dcmread(f)
        except BaseException:
            continue
        if d['SliceLocation'].value in locations.keys():
            locations[d['SliceLocation'].value].append(f)
        else:
            locations[d['SliceLocation'].value] = [f]
    split_dcmList = [locations[key] for key in locations.keys()]
    echo_list = [[] for _ in range(len(split_dcmList[0]))]
    keys = list(locations.keys())
    keys.sort()
    for key in keys:
        echos = locations[key]
        echos = sort_echo_list(echos)
        for idx in range(len(echo_list)):
            echo_list[idx].append(echos[idx])
    return echo_list


def sort_echo_list(echos: list):
    instance_numbers = [int(pydicom.dcmread(file).InstanceNumber) for file in echos]
    series_numbers = [int(pydicom.dcmread(file).SeriesNumber) for file in echos]

    # remove duplicates
    instance_numbers = list(set(instance_numbers))
    series_numbers = list(set(series_numbers))
    if len(instance_numbers) > 1:
        order = np.argsort(instance_numbers)
    elif len(series_numbers) > 1:
        order = np.argsort(series_numbers)
    else:
        print("Warning: Sorting not possible!")
        return echos
    return [echos[o] for o in order]


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
    array = array[::-1]
    return np.array(array)


class Mask:
    def __init__(self, array, affine, header):
        self.array = array
        self.affine = affine
        self.header = header

def load_nii(file):
    nimg = nib.load(file)
    mask = Mask(nimg.get_fdata()[:, :, ::-1], nimg.affine, nimg.header)
    return mask
