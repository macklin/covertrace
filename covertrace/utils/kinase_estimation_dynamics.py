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


# def trapezoid_func2(t, c1, c2, c3, t1, t2, t3, t4):
#     if t <= t1:
#         return c1
#     elif (t > t1) and (t <= t2):
#         return c1 + (c2 - c1) * (t-t1)/(t2-t1)
#     elif (t > t2) and (t <= t3):
#         return c2
#     elif (t > t3) and (t <= t4):
#         return c2 - (c2 - c3) * (t-t3)/(t4-t3)
#     elif t > t4:
#         return c3


def trapezoid_err(params, t, y):
    y_p = np.zeros(y.shape)
    for num, ti in enumerate(t):
        y_p[num] = trapezoid_func(np.float(ti)/t.max(), *params)
    return ((y - y_p)**2).sum()


# def trapezoid_err2(params, t, y, c1, c2, c3):
#     y_p = np.zeros(y.shape)
#     for num, ti in enumerate(t):
#         y_p[num] = trapezoid_func2(np.float(ti)/t.max(), c1, c2, c3, *params)
#     return ((y - y_p)**2).sum()


def fit_trapezoid(t, y, p0=None, tbuf=[0.05, 0.05, 0.05]):
    if p0 is None:
        p0 = [0.2, 0.4, 0.6, 0.8, y[0], y.max(), y[-1]]
    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[0] - tbuf[0]},  # t1 < t2
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1] - tbuf[1]},  # t2 < t3
            {'type': 'ineq', 'fun': lambda x:  x[3] - x[2] - tbuf[2]},  # t3 < t4
            {'type': 'ineq', 'fun': lambda x:  x[5] - x[4]},  # c1 < c2
            {'type': 'ineq', 'fun': lambda x:  x[5] - x[6]},  # c3 < c2
            {'type': 'ineq', 'fun': lambda x:  x})  # non-negative parameters
    bnds = ((0, 1), ) * 4 + ((y.min(), y.max()), ) * 3
    fun = partial(trapezoid_err, t=t, y=y)
    res = minimize(fun, p0, constraints=cons, bounds=bnds, tol=1e-12, options=dict(max_iter=10000))
    # convert Ts from relative to absolute time.
    Ts = np.interp(res.x[:4], np.linspace(0, 1, len(t)), t)
    return np.concatenate((Ts, res.x[4:]))


# def fit_trapezoid2(t, y, p0=None, c_max=5):
#     if p0 is None:
#         # p0 = [0.2, 0.4, 0.5, 0.75] + np.random.random(3).tolist()
#         p0 = [0.1, 0.6, 0.7, 0.8]
#     cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[0]},  # t1 < t2
#             {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]},  # t2 < t3
#             {'type': 'ineq', 'fun': lambda x:  x[3] - x[2]})  # t3 < t4
#     bnds = ((0, 1.0), ) * 4
#     cs =  [y[0], y.max(), y[-1]]
#     fun = partial(trapezoid_err2, t=t, y=y, c1=cs[0], c2=cs[1], c3=cs[2])
#     res = minimize(fun, p0, constraints=cons, bounds=bnds, tol=1e-8, options=dict(disp=True))
#     # convert Ts from relative to absolute time.
#     Ts = np.interp(res.x[:4], np.linspace(0, 1, len(t)), t)
#     print res
#     print Ts
#     return np.concatenate((Ts, res.x[4:]))



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

    # t = np.arange(0, 5, 0.5)
    # y = np.array([0.5, 0.5, 2, 6, 7, 6, 3, 0.6, 0.8, 0.2])
    # y0 = [3, 5, 7, 9, 0, 6, 2.5]
    # y0 = [i*0.5 for i in y0]
    # trap_params = fit_trapezoid(t, y)
    # ps = dict(k_v=4, k_iu=0.44, k_eu=0.11, k_ip=0.16, k_ep=0.2,
    #           k_cat=20, Km=3, k_dc=0.03, k_dn=0.03, Kmd=0.1, r_total=0.4,
    #           time_points=[0, 1], kin_c_with_time=[1, 1], kin_n_with_time=[1, 1])
    # k1, k2, k3 = fit_params_kinase_dynamics(trap_params, ps, t)
    # print kinase_dynamics_ode((k1, k2, k3), t, ps, trap_params)
    from numpy import array
    tes=array([ 0.24771574,  0.25555009,  0.24858223,  0.23608617,  0.23356704,
                0.23404256,  0.21572052,  0.23339012,  0.33829364,  0.4148061 ,
                0.38658535,  0.4122093 ,  0.36057141,  0.36304349,  0.299591  ,
                0.27535546,  0.25878733,  0.22735043,  0.22575387,  0.21827412,
                0.24344087,  0.23410526,  0.21227622,  0.22704507,  0.22143453,
                0.21454842,  0.22906198,  0.22011222,  0.20725389,  0.21312949,
                0.21758437,  0.21351351,  0.20987654,  0.21842697,  0.22017545,
                0.21915285,  0.21466431,  0.20662598,  0.21990439,  0.2206655 ,
                0.22759227,  0.23342294,  0.23226951,  0.2221231 ,  0.23108666,
                0.25371179,  0.2326858 ,  0.21723869,  0.23443815,  0.23870173,
                0.23701566,  0.22178683,  0.23070803,  0.21497585,  0.21636952,
                0.22148696,  0.22091195,  0.20910816,  0.21022524,  0.21433851])
    time = np.arange(0, 15000, 250)  # in minute
    trap_params = fit_trapezoid(time, tes)
