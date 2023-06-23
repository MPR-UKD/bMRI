import numpy as np
import pytest

from src.Fitting.T1 import InversionRecoveryT1, inversion_recovery_t1


def test_T1():
    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 200, 400, 600, 800, 1000, 1200, 1500, 2800])
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    t1 = [300, 500, 800, 1200]
    dicom[:, 0, 0, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[0], offset=-0)
    dicom[:, 0, 1, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[1], offset=-0)
    dicom[:, 1, 0, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[2], offset=-0)
    dicom[:, 1, 1, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[3], offset=-0)

    t1_fit = InversionRecoveryT1(boundary=((0, 200, 0), (np.inf, 1600, np.inf)))
    fit_map, __ = t1_fit.fit(dicom=dicom, mask=mask, x=x)
    assert abs(fit_map[1][0, 0, 0] - t1[0]) < 1
    assert abs(fit_map[1][0, 1, 0] - t1[1]) < 1
    assert abs(fit_map[1][1, 0, 0] - t1[2]) < 1
    assert abs(fit_map[1][1, 1, 0] - t1[3]) < 1


def test_T1_norm():
    mask = np.ones(shape=(2, 2, 1))
    x = np.array([0, 200, 400, 600, 800, 1000, 1200, 1500, 280])
    dicom = np.zeros(shape=(len(x), 2, 2, 1))
    t1 = [300, 500, 800, 1200]
    dicom[:, 0, 0, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[0], offset=0)
    dicom[:, 0, 1, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[1], offset=0)
    dicom[:, 1, 0, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[2], offset=0)
    dicom[:, 1, 1, 0] = inversion_recovery_t1(x, S0=1000, t1=t1[3], offset=0)

    t1_fit = InversionRecoveryT1(boundary=((0, 200, 0), (np.inf, 1600, np.inf)), normalize=True)
    fit_map, __ = t1_fit.fit(dicom=dicom, mask=mask, x=x)
    assert abs(fit_map[1][0, 0, 0] - t1[0]) < 1
    assert abs(fit_map[1][0, 1, 0] - t1[1]) < 1
    assert abs(fit_map[1][1, 0, 0] - t1[2]) < 1
    assert abs(fit_map[1][1, 1, 0] - t1[3]) < 1


if __name__ == '__main__':
    pytest.main()
