import unittest

import numpy as np

from ..T1rho import T1rho, fit_T1rho_wrapper


class TestT1rho(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"TR": 3000, "T1": 1000, "alpha": 20, "TE": 12, "T2star": 10}

        self.mask = np.ones(shape=(2, 2, 1))
        self.x = np.array([0, 10, 20, 30, 40, 50, 60, 70])
        dicom = np.zeros(shape=(len(self.x), 2, 2, 1))
        f_t1rho = fit_T1rho_wrapper(TR=3000, T1=1000, alpha=20, TE=12, T2star=10)
        t1rhos = [30, 35, 40, 70]
        dicom[:, 0, 0, 0] = f_t1rho(self.x, 1000, t1rhos[0], 200)
        dicom[:, 0, 1, 0] = f_t1rho(self.x, 1000, t1rhos[1], 200)
        dicom[:, 1, 0, 0] = f_t1rho(self.x, 1000, t1rhos[2], 200)
        dicom[:, 1, 1, 0] = f_t1rho(self.x, 1000, t1rhos[3], 200)
        self.dicom = dicom

    def test(self):
        t1rho = T1rho(2, self.config)
        fit_map, __ = t1rho.fit(dicom=self.dicom, mask=self.mask, x=self.x)
        self.assertTrue(abs(fit_map[0, 0] - 30) < 1)
        self.assertTrue(abs(fit_map[0, 1] - 35) < 1)
        self.assertTrue(abs(fit_map[1, 0] - 40) < 1)
        self.assertTrue(abs(fit_map[1, 1] - 70) < 1)
