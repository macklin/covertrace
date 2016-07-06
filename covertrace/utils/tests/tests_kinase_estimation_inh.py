from __future__ import division
import unittest
import numpy as np
from mock import Mock
from os.path import dirname, join, abspath, exists
import sys
# ROOTDIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(dirname(dirname(__file__)))
from kinase_estimation_inh import fit_params_inhibitor
from kinase_estimation_inh import calc_active_kinase_at_steady_state
from kinase_estimation_inh import calc_rep_profile_at_steady_state
from ktr_shuttle_ode import ParamHolder


class Test_fit_params_inhibitor(unittest.TestCase):
    def setUp(self):
        pass

    # def atest_normalize_data(self):
    #     ps_dict = dict(k_v=4, k_iu=0.44, k_eu=0.11, k_ip=0.16, k_ep=0.2,
    #                    k_cat=20, Km=3, k_dc=0.03, k_dn=0.03, Kmd=0.1, r_total=0.4)
    #     ts_time = np.arange(15)
    #     ts_cn = np.exp(-0.3*np.arange(15))
    #     x0 = [0.5, 0.5, 0.5]
    #     ret = fit_params_inhibitor(x0, ts_time, ts_cn, ps_dict)
    #     self.assertTrue(isinstance(ret, np.ndarray))

    def test_calc_rep_profile_at_steady_state(self):
        # Assume nuclear/cytoplasmic volume are the same.
        ps_dict = dict(k_v=1, k_cat=20, Km=0.1, k_dc=0.1, k_dn=0.1, Kmd=0.1, r_total=1)
        pset = ParamHolder(ps_dict)
        # Phosphorylation status does not affect localization.
        pset.k_iu, pset.k_eu, pset.k_ip, pset.k_ep = 0.5, 0.5, 0.5, 0.5
        rep_pro1 = calc_rep_profile_at_steady_state(0, pset)  # Active kinsase is 0.
        self.assertTrue(rep_pro1[2] < 0.01)  # cp does not exist
        self.assertTrue(rep_pro1[3] < 0.01)  # np does not exist
        rep_pro2 = calc_rep_profile_at_steady_state(1, pset)  # Active kinsase is 1.
        self.assertTrue(rep_pro2[0] < 0.01)  # cu does not exist
        self.assertTrue(rep_pro2[1] < 0.01)  # nu does not exist

if __name__ == '__main__':
    unittest.main()
