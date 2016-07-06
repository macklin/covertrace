import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute
from functools import partial
from ktr_shuttle_ode import main_ode, ParamHolder
from kinase_estimation_inh import calc_rep_profile_at_steady_state
from scipy.integrate import odeint


def trapezoid_func(t, t1, t2, t3, t4, c1, c2, c3):
    if t <= t1:
        return c1
    elif (t > t1) and (t <= t2):
        return c1 + (c2 - c1) * (t-t1)/(t2-t1)
    elif (t > t2) and (t <= t3):
        return c2
    elif (t > t3) and (t <= t4):
        return c2 - (c2 - c3) * (t-t3)/(t4-t3)
    elif t > t4:
        return c3


def trapezoid_err(params, t, y):
    y_p = np.zeros(y.shape)
    for num, ti in enumerate(t):
        y_p[num] = trapezoid_func(ti, *params)
    return ((y - y_p)**2).sum()


def fit_trapezoid(t, y, p0=None):
    if p0 is None:
        p0 = [0, 1, 2, 3] + np.random.random(3).tolist()
    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[0]},  # t1 < t2
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]},  # t2 < t3
            {'type': 'ineq', 'fun': lambda x:  x[3] - x[2]},  # t3 < t4
            {'type': 'ineq', 'fun': lambda x:  x[5] - x[4]},  # c1 < c2
            {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},  # c3 < c2
            {'type': 'ineq', 'fun': lambda x:  x})  # non-negative parameters
    fun = partial(trapezoid_err, t=t, y=y)
    res = minimize(fun, p0, constraints=cons)
    return res.x


def fit_params_kinase_dynamics(trapezoid_params, pset_dict, time, kin_max=1, x0=np.random.random(3)):
    pset = ParamHolder(pset_dict)
    rcn = construct_ts_from_trap_params(time, *trapezoid_params)
    func = lambda x: ((kinase_dynamics_ode_rcn(x, time, pset, trapezoid_params) - rcn)**2).sum()
    bnds = ((0, kin_max),) * 3
    ret = minimize(func, x0=x0, bounds=bnds)
    return ret.x


def construct_ts_from_trap_params(time, t1, t2, t3, t4, c1, c2, c3):
    return np.interp(time, [t1, t2, t3, t4], [c1, c2, c2, c3])


def kinase_dynamics_ode_rcn(kins, time, pset, trapezoid_params):
    ts = kinase_dynamics_ode(kins, time, pset, trapezoid_params)
    return (ts[:, 0] + ts[:, 2])/(ts[:, 1] + ts[:, 3])


def kinase_dynamics_ode(kins, time, pset, trapezoid_params):
    if isinstance(pset, dict):
        pset = ParamHolder(pset)
    k1, k2, k3 = kins  # active kinase at each time in trapezoidal form
    t1, t2, t3, t4 = trapezoid_params[:4]
    # get model to steady state
    rep0 = calc_rep_profile_at_steady_state(k1, pset)

    pset.time_points = [t1, t2, t3, t4, time[-1]]
    pset.kin_c_with_time = [k1, k2, k2, k3, k3]
    pset.kin_n_with_time = [k1, k2, k2, k3, k3]
    ts = odeint(main_ode, rep0, time, (pset, ), rtol=1e-4)
    return ts




if __name__ == "__main__":

    t = np.arange(0, 5, 0.5)
    y = np.array([0.5, 0.5, 2, 6, 7, 6, 3, 0.6, 0.8, 0.2])
    y0 = [3, 5, 7, 9, 0, 6, 2.5]
    y0 = [i*0.5 for i in y0]
    trap_params = fit_trapezoid(t, y)

    ps = dict(k_v=4, k_iu=0.44, k_eu=0.11, k_ip=0.16, k_ep=0.2,
              k_cat=20, Km=3, k_dc=0.03, k_dn=0.03, Kmd=0.1, r_total=0.4,
              time_points=[0, 1], kin_c_with_time=[1, 1], kin_n_with_time=[1, 1])
    # ps = ParamHolder(ps)
    # kinase_dynamic_ode(ps)

    # trapezoid_params = [0, 1, 2, 3, 0, 0, 0]
    k1, k2, k3 = fit_params_kinase_dynamics(trap_params, ps, t)
    print kinase_dynamics_ode((k1, k2, k3), t, ps, trap_params)
