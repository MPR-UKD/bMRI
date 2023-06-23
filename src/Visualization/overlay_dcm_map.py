from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from src.Utilitis import load_nii


class Overlay:
    def __init__(self, dcm_array: np.ndarray | Path, map_array: np.ndarray | Path, mask_array: np.ndarray | Path,
                 min_val: float = 0, max_val: float = 200, label: str = ""):
        """
        Initialize the Overlay.

        :param dcm_array: DICOM data array or Path.
        :param map_array: Map data array or Path.
        :param mask_array: Mask data array or Path.
        :param min_val: Minimum value for the overlay, optional.
        :param max_val: Maximum value for the overlay, optional.
        :param label: Label for the color bar, optional.
        """
        self.dcm = dcm_array if isinstance(dcm_array, np.ndarray) else load_nii(dcm_array).array
        self.map = map_array if isinstance(map_array, np.ndarray) else load_nii(map_array).array
        self.mask = mask_array if isinstance(mask_array, np.ndarray) else load_nii(mask_array).array
        self.map[self.mask == 0] = np.nan
        self.min = min_val
        self.max = max_val
        self.label = label

    def overlay_img(self, slice_idx: int, file_name: str):
        """
        Overlay the image with the selected map and save the result.

        :param slice_idx: Index of the slice to be overlayed.
        :param file_name: Name of the file to save the overlayed image.
        """
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
