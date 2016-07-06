from __future__ import division
import unittest
import numpy as np
from mock import Mock
from os.path import dirname, join, abspath, exists
import sys
# ROOTDIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(dirname(dirname(__file__)))
from ktr_shuttle_ode import ParamHolder
from kinase_estimation_dynamics import kinase_dynamics_ode, fit_params_kinase_dynamics, trapezoid_err

class Test_dynamics(unittest.TestCase):
    def setUp(self):
        pass

    def test_kinase_dynamic_ode_outputs_ndarray(self):
        ps_dict = dict(k_v=1, k_cat=20, Km=0.1, k_dc=0.1, k_dn=0.1, Kmd=0.1, r_total=1)
        pset = ParamHolder(ps_dict)
        pset.k_iu, pset.k_eu, pset.k_ip, pset.k_ep = 0.5, 0.5, 0.5, 0.5
        time = np.arange(15)
        trapezoid_params = [0, 1, 2, 3, 0, 0, 0]
        kins = [0, 2, 1]
        ret = kinase_dynamics_ode(kins, time, pset, trapezoid_params)
        self.assertTrue(isinstance(ret, np.ndarray))
        self.assertEqual(ret.shape, (15, 4))

    def test_fit_params_kinase_dynamics_outputs_ndarray(self):
        ps_dict = dict(k_v=1, k_cat=20, Km=0.1, k_dc=0.1, k_dn=0.1, Kmd=0.1, r_total=1,
                       k_iu=0.5, k_eu=0.5, k_ip=0.5, k_ep=0.5)
        time = np.arange(15)
        trapezoid_params = [0, 1, 2, 3, 0, 0, 0]
        ret = fit_params_kinase_dynamics(trapezoid_params, ps_dict, time)
        self.assertTrue(isinstance(ret, np.ndarray))
        self.assertEqual(ret.shape, (3, ))

    def test_trapezoid_func(self):
        p0 = [0, 1, 2, 3] + np.random.random(3).tolist()
        t = np.arange(0, 12, 2)
        y = np.array([0, 0, 10, 10, 5, 5])
        print trapezoid_err(p0, t, y)

if __name__ == '__main__':
    unittest.main()
