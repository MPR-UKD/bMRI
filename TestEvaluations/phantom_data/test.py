from pathlib import Path
from Fitting import *
from Utilitis import load_nii, save_results
import numpy as np


if __name__ == '__main__':
    dGEMRIC_folder = Path('./dGEMRIC')
    T1rho_folder = Path('./T1rho')
    T2_folder = Path('./T2')
    T2star_folder = Path('./T2star')

    ###################################################################################################################
    #                                             ----- dGEMRIC -----                                                 #
    ###################################################################################################################
    dGEMRIC = FittedMap()
    dGEMRIC_map, mask = dGEMRIC(dGEMRIC_folder, dGEMRIC_folder / 'mask.nii.gz')
    results = save_results(dGEMRIC_map, None, mask.array, mask.affine, mask.header,
                           dGEMRIC_folder, dGEMRIC_folder / 'results.csv')
    print("dGEMRIC results" + "\n" + "_" * 20)
    print(results)


    ###################################################################################################################
    #                                               ----- T1rho -----                                                 #
    ###################################################################################################################
    # Sequence parameters
    # TR = ... ms                                       T2star =
    # TE = ... ms                                       T1 =
    # alpha = ... ms
    # first TSL =
    # Delta TSL = ... ms
    ###################################################################################################################
    t1rho = T1rho(dim=3,
                  boundary=([0.2, 25, -0.1], [4000, 140, 0.1]),
                  config={"T1": 800,
                          "TR": 3500,
                          "alpha": 15,
                          "TE": 5.78,
                          "T2star": 10})
    dicom, _ = t1rho.read_data(T1rho_folder)
    x = t1rho.get_TSL(first_SL=10, inc_SL=30, n=10)
    mask = load_nii(Path(T1rho_folder) / 'mask.nii.gz')
    t1rho_map, r2 = t1rho.fit(dicom=dicom, mask=mask.array, x=np.array(x))
    results = save_results(t1rho_map, r2, mask.array, mask.affine, mask.header,
                           T1rho_folder, T1rho_folder / 'results.csv')
    print("T1rho results" + "\n" + "_" * 20)
    print(results)

    ###################################################################################################################
    #                                                ----- T2 -----                                                   #
    # TEs = ................. ms
    ###################################################################################################################
    t2 = T2_T2star(dim=3,
                   boundary=([0.2, 5, -0.1], [1, 80, 0.1]))
    dicom, x = t2.read_data(T2_folder)
    assert x == []
    mask = load_nii(T2_folder / 'mask.nii.gz')
    T2_map, r2 = t2.fit(dicom=dicom, mask=mask.array, x=np.array(x))
    results = save_results(T2_map, r2, mask.array, mask.affine, mask.header,
                           T2_folder, T2_folder / 'results.csv')
    print("T2 results" + "\n" + "_" * 20)
    print(results)


