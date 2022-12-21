from Utilitis import load_nii, get_dcm_array, get_dcm_list
from Fitting import *
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


class Overlay:
    def __init__(self, dcm_array, map_file, mask_file):
        self.mask = load_nii(mask_file).array[:, :, ::-1]
        self.map = load_nii(map_file).array
        self.dcm = dcm_array

    def get_img(self, slice_nr):
        dcm = self.dcm[:, :, slice_nr]
        mask = self.mask[:, :, slice_nr]
        f_map = self.map[:, :, slice_nr]
        f_map[mask == 0] = np.nan
        plt.imshow(np.rot90(dcm[160:350, 180:370], 3), cmap='gray')
        plt.imshow(np.rot90(f_map[160:350, 180:370], 3), cmap='jet', alpha=0.75, vmax=25, vmin=0)
        plt.axis('off')
        c_bar = plt.colorbar()
        c_bar.set_label('T$_{2}^{*}$ [ms]', fontsize=22)
        plt.savefig(
         r'F:\Projekt_Schweineknie\img.png', dpi=1200
        )
    def get_most_pixel(self):
        count = []
        for slice in range(self.mask.shape[-1]):
            temp = self.mask[:, :, slice]
            count.append(len(temp[temp != 0]))
        return np.argmax(count)

    def get_mid_slice(self):
        count = []
        for slice in range(self.mask.shape[-1]):
            temp = self.mask[:, :, slice]
            count.append(len(temp[temp != 0]))
        return round(np.median(np.argwhere(np.array(count) != 0)))

    def get_most_pixel_near_mid(self, c):
        mid = self.get_mid_slice()
        count = []
        for slice in range(self.mask.shape[-1]):
            if abs(mid - slice) > c:
                count.append(0)
                continue
            temp = self.mask[:, :, slice]
            count.append(len(temp[temp != 0]))
        return np.argmax(count)


if __name__ == '__main__':
    knee_nr = 22 #4
    folder = [_ for _ in Path(r'F:\Projekt_Schweineknie\Daten').glob(f'*{knee_nr}*\*\*T2-star_map_*')][0]
    t2 = T2_T2star(dim=3, boundary=None)
    dicom, x = t2.read_data(folder)
    overlay = Overlay(
        dcm_array = dicom[0],
        map_file= folder / 'map.nii.gz',
        mask_file = folder / 'mask.nii.gz')
    overlay.get_img(overlay.get_most_pixel_near_mid(5))