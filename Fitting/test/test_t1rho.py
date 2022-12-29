import numpy as np
import pytest

from Fitting.T1rho import T1rho, fit_T1rho_wrapper_aronen
from Visualization.show import show

def test():
    config = {"TR": 3000, "T1": 1000, "alpha": 20, "TE": 12, "T2star": 10}

    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70]) * 2
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    f_t1rho = fit_T1rho_wrapper_aronen(TR=3000, T1=1000, alpha=20, TE=12, T2star=10)
    t1rhos = [30, 35, 40, 70]
    dicom[:, 0, 0, 0] = f_t1rho(x, 1000, t1rhos[0], 200)
    dicom[:, 0, 1, 0] = f_t1rho(x, 1000, t1rhos[1], 200)
    dicom[:, 1, 0, 0] = f_t1rho(x, 1000, t1rhos[2], 200)
    dicom[:, 1, 1, 0] = f_t1rho(x, 1000, t1rhos[3], 200)

    t1rho = T1rho(2, config, normalize=True, boundary=((1, 0, 0), (2, 100, 1)))
    fit_map, __ = t1rho.fit(dicom=dicom, mask=mask, x=x)
    show(dicom, fit_map, f_t1rho, x)


if __name__ == '__main__':
    pytest.main()
