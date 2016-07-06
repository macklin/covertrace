from __future__ import division
import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import odeint
# from scipy.interpolate import interp1d
from scipy.optimize import minimize
from ktr_shuttle_ode import main_ode, ParamHolder

# Parameters
'''r_total: Ths can be inferred experimentally or from the literature '''

'''
We would like to know importing and exporting rate constants for both unphosphorylated and phosphorylated proteins.
(k_iu, k_eu, k_ip, k_ep). These parameters can be inferred by measuring dynamics of AA mutants
and EE mutants followed by LMB treatment.

k_cat, Km from the literature.

k_dc=k_dn: dephosphorylation Vmax. Assume it's sam, given the fact that phosphatase is believed to be
more promisucuos.
Kmd

d_cyto = -k_i * cyto + k_e * nuc
d_nuc =


parameters:
r_total, k_cat, Km, Kmd, (k_iu, k_eu, k_ip, k_ep), k_dc, k_dn

'''

r_total = 0.4  # uM, total reporter concentration
k_v = 4  # ratio of cytosolic volume to nuclear volume
k_iu = 0.44  # 1/min, nuclear import of unphosphorylated reporter
k_eu = 0.11  # 1/min, nuclear export of unphosphorylated reporter
k_ip = 0.16  # 1/min, nuclear import of phosphorylated reporter
k_ep = 0.2  # 1/min, nuclear export of phosphorylated reporter
k_cat = 20  # 1/min, catalytic rate constant of kinase and reporter
Km = 3  # uM, Michaelis constant for kinase and reporter
k_dc = 0.03  # uM/min, dephosphorylation Vmax of reporter in cytosol
k_dn = 0.03  # uM/min, dephosphorylation Vmax of reporter in nucleus
Kmd = 0.1  # uM, Michaelis constant for dephosphorylation of reporter
kin_c = None
kin_t = None




def calc_active_kinase_at_steady_state(rcn, pset, x0=[0.01]):
    """At given parameters, calculate the kinase concentration
    such that it matches with the observed cytoplasmic-to-nuclear ratio.
    """
    func = lambda kin: calc_cn_ratio_steady_state(kin, pset) - rcn
    ret = minimize(func, x0=x0, bounds=((0, None), ))
    kinase = ret.x[0]
    return kinase


def calc_rep_profile_at_steady_state(kin, pset):
    """At given kinase concentration and parameters,
    calculate the reporter profile (c_u, n_u, c_p, c_p)
    such that it minimizes sum of squared dy, meaning at pseudo-steady state."""
    bnds = ((0, None), (0, None), (0, None), (0, None))
    kin = float(kin)
    pset.time_points = [0, 1]
    pset.kin_c_with_time, pset.kin_n_with_time = [kin, kin], [kin, kin]
    x0 = [pset.r_total, 0, 0, 0]

    func = lambda y: (np.array(main_ode(y, 0, pset))**2).sum()
    ret = minimize(func, x0=x0, bounds=bnds)
    return ret.x


def calc_cn_ratio_steady_state(kin, pset):
    """At given kinase concentration and parameters,
    calculate cytoplasmic-to-nuclear ratio at the pseudo-steady state.
    """
    x = calc_rep_profile_at_steady_state(kin, pset)
    return (x[0] + x[2])/(x[1] + x[3])

def inhibitor_ode(x, time, rcn_init, pset):
    pset.k_dn, pset.k_dc = x[0], x[0]
    pset.Kmd = x[1]
    rcn_final = x[2]
    pset.time_points = [0, 1]

    # Calculate kinase concentration at steady state such that it matches with rcn
    kin0 = calc_active_kinase_at_steady_state(rcn=rcn_init, pset=pset)
    # At given kinase concentration, calculate reporter profile.
    y0 = calc_rep_profile_at_steady_state(kin0, pset)
    # At given rcn_final, calculate kinase concentration at steady state.
    kin_after_inh = calc_active_kinase_at_steady_state(rcn=rcn_final, pset=pset)

    pset.kin_c_with_time = [kin_after_inh, kin_after_inh]
    pset.kin_n_with_time = [kin_after_inh, kin_after_inh]

    pset.time_points = [time[0], time[-1]]  # actual_data
    # At time 0, inihibition started. kinase is inactive so it follows kinase (rcn_final)
    # but reporter profile at time 0 follows y0. Calculate cytoplasmic to nuclear times series.
    ts = odeint(main_ode, y0, time, (pset, ), rtol=1e-4)
    return (ts[:, 0] + ts[:, 2])/(ts[:, 1] + ts[:, 3])


def fit_params_inhibitor(x0, ts_time, ts_cn, pset_dict):
    """
    Output: k_d*, Kmd, rcn_final
    """
    pset = ParamHolder(pset_dict)
    func = lambda x: ((inhibitor_ode(x, ts_time, ts_cn[0], pset) - ts_cn)**2).sum()
    bnds = ((0, 1), (0, None), (0, None))
    ret = minimize(func, x0=x0, bounds=bnds)
    return ret.x


if __name__ == "__main__":

    ps = dict(k_v=4, k_iu=0.44, k_eu=0.11, k_ip=0.16, k_ep=0.2,
              k_cat=20, Km=3, k_dc=0.03, k_dn=0.03, Kmd=0.1, r_total=0.4,
              time_points=[0, 1], kin_c_with_time=[1,1], kin_n_with_time=[1,1])
    # ps = ParamHolder(ps)
    ts_time = np.arange(15)
    ts_cn = np.exp(-0.3*np.arange(15))
    x0 = np.random.random(3)
    ret_store = []
    for i in range(3):
        x0 = np.random.random(3)
        ret = fit_params_inhibitor(x0, ts_time, ts_cn, ps)
        ret_store.append(ret)
    print ret_store
