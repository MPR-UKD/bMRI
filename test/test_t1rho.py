import numpy as np
import pytest

from src.Fitting.T1rho_T2prep import T1rho_T2prep, fit_T1rho_wrapper_aronen


def test():
    config = {"TR": 3000, "T1": 1000, "alpha": 20, "TE": 12, "T2star": 10}

    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    f_t1rho = fit_T1rho_wrapper_aronen(TR=3000, T1=1000, alpha=20, TE=12, T2star=10)
    t1rhos = [30, 35, 40, 70]
    dicom[:, 0, 0, 0] = f_t1rho(x, 1000, t1rhos[0], 200)
    dicom[:, 0, 1, 0] = f_t1rho(x, 1000, t1rhos[1], 200)
    dicom[:, 1, 0, 0] = f_t1rho(x, 1000, t1rhos[2], 200)
    dicom[:, 1, 1, 0] = f_t1rho(x, 1000, t1rhos[3], 200)

    t1rho = T1rho_T2prep(2, config, normalize=True, boundary=((1, 0, 0), (40, 140, 1)))
    fit_map, __ = t1rho.fit(dicom=dicom, mask=mask, x=x)
    assert abs(fit_map[1][0, 0, 0] - t1rhos[0]) < 1
    assert abs(fit_map[1][0, 1, 0] - t1rhos[1]) < 1
    assert abs(fit_map[1][1, 0, 0] - t1rhos[2]) < 1
    assert abs(fit_map[1][1, 1, 0] - t1rhos[3]) < 1


if __name__ == "__main__":
    pytest.main()
