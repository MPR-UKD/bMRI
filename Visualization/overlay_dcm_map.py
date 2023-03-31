from Utilitis import load_nii, get_dcm_array, get_dcm_list
from Fitting import *
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


class Overlay:
    def __init__(self,
                 dcm_array: np.ndarray | Path,
                 map_array: np.ndarray | Path,
                 mask_arry: np.ndarray | Path,
                 min = 0,
                 max = 200):
        self.dcm = dcm_array if type(dcm_array) == np.ndarray else load_nii(dcm_array).array
        self.map = map_array if type(map_array) == np.ndarray else load_nii(map_array).array
        self.mask = mask_arry if type(mask_arry) == np.ndarray else load_nii(mask_arry).array
        self.dcm = self.dcm[0]
        self.map = self.map[1]
        self.map[self.mask == 0] = np.nan
        self.min = min
        self.max = max
        self.label = "T$_{2}$ [ms]"

    def overlay_img(self, slice_idx, file_name):
        dcm = np.rot90(self.dcm[:, :, slice_idx], 3)
        f_map = np.rot90(self.map[:, :, slice_idx], 3)
        plt.imshow(dcm, cmap="gray")
        plt.imshow(
            f_map,
            cmap="jet",
            alpha=0.75,
            vmax=self.max,
            vmin=self.min,
        )
        plt.axis("off")
        c_bar = plt.colorbar()
        c_bar.set_label(self.label, fontsize=22)

        plt.savefig(file_name, dpi=1200)
        plt.close()
