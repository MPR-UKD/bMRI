import numpy as np
import pytest

from src.Fitting.T2_T2star import T2_T2star, mono_exp


def test_T2():

    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80,90, 100, 110, 120, 130, 140])
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    t2 = [30, 35, 40, 70]
    dicom[:, 0, 0, 0] = mono_exp(x, S0=1000, t2_t2star=t2[0], offset=100)
    dicom[:, 0, 1, 0] = mono_exp(x, S0=1000, t2_t2star=t2[1], offset=100)
    dicom[:, 1, 0, 0] = mono_exp(x, S0=1000, t2_t2star=t2[2], offset=100)
    dicom[:, 1, 1, 0] = mono_exp(x, S0=1000, t2_t2star=t2[3], offset=100)

    t1rho = T2_T2star(3, boundary=((0,0,0),(np.inf, 200, np.inf)))
    fit_map, __ = t1rho.fit(dicom=dicom, mask=mask, x=x)
    assert abs(fit_map[1][0, 0, 0] - t2[0]) < 1
    assert abs(fit_map[1][0, 1, 0] - t2[1]) < 1
    assert abs(fit_map[1][1, 0, 0] - t2[2]) < 1
    assert abs(fit_map[1][1, 1, 0] - t2[3]) < 1

def test_T2_norm():

    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80,90, 100, 110, 120, 130, 140])
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    t2 = [30, 35, 40, 70]
    dicom[:, 0, 0, 0] = mono_exp(x, S0=1000, t2_t2star=t2[0], offset=100)
    dicom[:, 0, 1, 0] = mono_exp(x, S0=1000, t2_t2star=t2[1], offset=100)
    dicom[:, 1, 0, 0] = mono_exp(x, S0=1000, t2_t2star=t2[2], offset=100)
    dicom[:, 1, 1, 0] = mono_exp(x, S0=1000, t2_t2star=t2[3], offset=100)

    t1rho = T2_T2star(3, boundary=((0,0,0),(np.inf, 200, np.inf)), normalize=True)
    fit_map, __ = t1rho.fit(dicom=dicom, mask=mask, x=x)
    assert abs(fit_map[1][0, 0, 0] - t2[0]) < 1
    assert abs(fit_map[1][0, 1, 0] - t2[1]) < 1
    assert abs(fit_map[1][1, 0, 0] - t2[2]) < 1
    assert abs(fit_map[1][1, 1, 0] - t2[3]) < 1


if __name__ == '__main__':
    pytest.main()
