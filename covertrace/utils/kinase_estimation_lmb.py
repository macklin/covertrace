import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from export_figures import save_subplots_in_pdf

bnds = ((0, 1), (1e-6, None), (1e-6, None))


def call_estimate_h_ki_kv(time, ts_ratio_all, inh_timing, k_v, x0=None):
    if not x0:
        x0 = np.random.random(3)
    time_pre, time_post = time[time <= inh_timing], time[time > inh_timing]
    param_store, sim_store = [], []
    for cell_ts in ts_ratio_all:
        ((params), sim, exp) = estimate_imex_constants(time_pre, time_post, cell_ts, k_v, x0)
        param_store.append(params)
        sim_store.append(sim)
    est_h, est_k_e, est_k_i = np.array(param_store).T
    return est_h.tolist(), est_k_e.tolist(), est_k_i.tolist(), np.array(sim_store)


class Estimator_h_ki_kv_orig(object):
    def __init__(self, time, ts_ratio, inh_timing, k_v, x0=None):
        self.time = time
        self.ts_ratio = ts_ratio
        self.inh_timing = inh_timing
        self.k_v = k_v
        if not x0:
            x0 = np.random.random(3)
        self.x0 = x0
        self.set_time_pre_post()

    def set_time_pre_post(self):
        self.time_pre = self.time[self.time <= self.inh_timing]
        self.time_post = self.time[self.time > self.inh_timing]

    def estimate(self):
        store = []
        for ts_cell in self.ts_ratio:
            ret = self.estimate_imex_constants(ts_cell)
            store.append(ret)
        return store

    def estimate_imex_constants(self, ts_cell):
        ret = minimize(self.optim_err_lmb, x0=self.x0, args=(self.time_pre, self.time_post, ts_cell, self.k_v), bounds=bnds)
        sim, dat = self.calc_ts_sim_ratio([ret.x[0], ret.x[1], ret.x[2]], self.time_pre, self.time_post, ts_cell, self.k_v)
        return ret.x, sim, dat

    @classmethod
    def optim_err_lmb(cls, x, *args):
        ts_sim_ratio, ts_exp_ratio = cls.calc_ts_sim_ratio(x, *args)
        return ((ts_sim_ratio - ts_exp_ratio) ** 2).sum()

    @classmethod
    def calc_ts_sim_ratio(cls, x, *args):
        '''x = [h, k_e, k_i]
        '''
        h, k_e, k_i = x[0], x[1], x[2]
        time_pre, time_post, ts_exp_ratio, k_v = args[0], args[1], args[2], args[3]

        ini_r_c = 1.0
        ini_r_n = k_i/k_e

        pre_inh = odeint(cls.ode_mutant_model, [ini_r_c, ini_r_n], time_pre, (k_v, k_i, k_e), rtol=1e-4)

        post_inh = odeint(cls.ode_mutant_model, [pre_inh[-1][0], pre_inh[-1][1]], time_post,
                          (k_v, k_i, k_e * h), rtol=1e-4)
        pre_inh_ratio = [i[0]/i[1] for i in pre_inh]
        post_inh_ratio = [i[0]/i[1] for i in post_inh]
        ts_sim_ratio = np.array(pre_inh_ratio + post_inh_ratio)
        return ts_sim_ratio, ts_exp_ratio

    @staticmethod
    def ode_mutant_model(y, t, *args):
        k_v, k_i, k_e = args[0], args[1], args[2]
        r_c, r_n = y[0], y[1]
        d_r_c = -k_i * r_c + k_e * r_n
        d_r_n = k_v * k_i * r_c - k_v * k_e * r_n
        return [d_r_c, d_r_n]



class Estimator_h_ki_kv(object):
    def __init__(self, time, ts_ratio, t_inh, k_v, x0=None, _save=True):
        self.time = time
        self.ts_ratio = ts_ratio
        self.t_inh = t_inh
        self.k_v = k_v
        if not x0:
            x0 = np.random.random(3)
        self.x0 = x0
        self._save = _save

    def estimate(self):
        param_store, sim_store = [], []
        for ts_cell in self.ts_ratio:
            ((params), sim, exp) = self.estimate_imex_constants(ts_cell)
            param_store.append(params)
            sim_store.append(sim)
        est_h, est_k_e, est_k_i = np.array(param_store).T
        if self._save:
            save_subplots_in_pdf(zip(sim_store, list(self.ts_ratio)))
        return est_h.tolist(), est_k_e.tolist(), est_k_i.tolist(), np.array(sim_store)

    @classmethod
    def optim_err_lmb(cls, x, *args):
        ts_sim_ratio, ts_exp_ratio = cls.calc_ts_sim_ratio(x, *args)
        return ((ts_sim_ratio - ts_exp_ratio) ** 2).sum()

    def estimate_imex_constants(self, ts_cell):
        ext_args = (self.time, ts_cell, self.k_v, self.t_inh)
        ret = minimize(self.optim_err_lmb, x0=self.x0, args=ext_args, bounds=bnds)
        sim, dat = self.calc_ts_sim_ratio([ret.x[0], ret.x[1], ret.x[2]], *ext_args)
        return ret.x, sim, dat

    @classmethod
    def calc_ts_sim_ratio(cls, x, *args):
        '''x = [h, k_e, k_i]
        '''
        h, k_e, k_i = x[0], x[1], x[2]
        time, ts_exp_ratio, k_v, inh_t = args[0], args[1], args[2], args[3]

        ini_r_c = 1.0
        ini_r_n = k_i/k_e

        params = (k_v, k_i, k_e, h, inh_t)
        pre_inh = odeint(cls.ode_mutant_model, [ini_r_c, ini_r_n], time, params, rtol=1e-4)

        pre_inh_ratio = [i[0]/i[1] for i in pre_inh]
        ts_sim_ratio = np.array(pre_inh_ratio)
        return ts_sim_ratio, ts_exp_ratio

    @staticmethod
    def ode_mutant_model(y, t, *args):
        k_v, k_i, p_k_e, h, t_inh = args[0], args[1], args[2], args[3], args[4]
        if t > t_inh:
            k_e = p_k_e * h
        else:
            k_e = p_k_e
        r_c, r_n = y[0], y[1]
        d_r_c = -k_i * r_c + k_e * r_n
        d_r_n = k_v * k_i * r_c - k_v * k_e * r_n
        return [d_r_c, d_r_n]


def estimate_imex_constants(time_pre, time_post, cell_ts, k_v, x0):
    ret = minimize(optim_err_lmb, x0=x0, args=(time_pre, time_post, cell_ts, k_v), bounds=bnds)
    sim, dat = calc_ts_sim_ratio([ret.x[0], ret.x[1], ret.x[2]], time_pre, time_post, cell_ts, k_v)
    return ret.x, sim, dat


def optim_err_lmb(x, *args):
    ts_sim_ratio, ts_exp_ratio = calc_ts_sim_ratio(x, *args)
    return ((ts_sim_ratio - ts_exp_ratio) ** 2).sum()


def calc_ts_sim_ratio(x, *args):
    '''x = [h, k_e, k_i]
    '''
    h, k_e, k_i = x[0], x[1], x[2]
    time_pre, time_post, ts_exp_ratio, k_v = args[0], args[1], args[2], args[3]

    ini_r_c = 1.0
    ini_r_n = k_i/k_e

    pre_inh = odeint(ode_mutant_model, [ini_r_c, ini_r_n], time_pre, (k_v, k_i, k_e), rtol=1e-4)
    post_inh = odeint(ode_mutant_model, [pre_inh[-1][0], pre_inh[-1][1]], time_post,
                      (k_v, k_i, k_e * h), rtol=1e-4)
    pre_inh_ratio = [i[0]/i[1] for i in pre_inh]
    post_inh_ratio = [i[0]/i[1] for i in post_inh]
    ts_sim_ratio = np.array(pre_inh_ratio + post_inh_ratio)
    return ts_sim_ratio, ts_exp_ratio


def ode_mutant_model(y, t, *args):
    k_v, k_i, k_e = args[0], args[1], args[2]
    r_c, r_n = y[0], y[1]
    d_r_c = -k_i * r_c + k_e * r_n
    d_r_n = k_v * k_i * r_c - k_v * k_e * r_n
    return [d_r_c, d_r_n]

if __name__ == "__main__":

    test = np.array([[   0.85539716,  0.98306596,  0.79492003,  0.83921164,  0.8596788 ,
                         0.66810036,  0.60740292,  0.4666273 ,  0.38555443,  0.3813481 ,
                         0.37428337,  0.36253563,  0.34255821,  0.33464825,  0.30366647,
                         0.29342857,  0.28485838,  0.29465523,  0.28893906,  0.27903092,
                         0.28078684,  0.26884019,  0.26093832,  0.26764244,  0.25823894,
                         0.25763062,  0.2346514 ,  0.22468513,  0.21562658,  0.20786516,
                         0.20599613,  0.21841024,  0.22473785,  0.23508772,  0.21124828,
                         0.21973307,  0.20993394,  0.19277674,  0.20206282,  0.18052366],
                     [   0.88014102,  0.88673466,  0.98207885,  0.8096624 ,  0.65587044,
                         0.53011203,  0.50520831,  0.42707631,  0.40064308,  0.38502824,
                         0.35037595,  0.35449263,  0.34892872,  0.33788979,  0.2829091 ,
                         0.25933239,  0.25641027,  0.25866473,  0.22512527,  0.22380787,
                         0.20256151,  0.19854972,  0.18939136,  0.18602329,  0.16192706,
                         0.18336813,  0.21337028,  0.22133236,  0.23764969,  0.25885612,
                         0.25004506,  0.25288683,  0.26721051,  0.27056709,  0.25865498,
                         0.29063731,  0.28248113,  0.27161238,  0.3125    ,  0.30643514],
                     [   0.32481202,  0.40428886,  0.3537088 ,  0.37818182,  0.28504184,
                         0.33829251,  0.261529  ,  0.24716599,  0.15758526,  0.18435754,
                         0.15366744,  0.11923877,  0.07976012,  0.07291362,  0.0425949 ,
                         0.03199863,  0.03518605,  0.03445754,  0.03598423,  0.03773277,
                         0.03104213,  0.02217226,  0.03096978,  0.0303684 ,  0.03001605,
                         0.02642868,  0.02429218,  0.02414929,  0.02583201,  0.02265638,
                         0.0282838 ,  0.02720313,  0.02786743,  0.02894322,  0.02314563,
                         0.02903063,  0.0253649 ,  0.02753036,  0.03261049,  0.03555983]])
    test_ts = test
    time = np.linspace(0, 20, 40) # Imaged every 30 sec
    inh_timing = 2.5
    # est_h, est_k_e, est_k_i, sim = calling(time, test_ts, inh_timing, 4)

    eh = Estimator_h_ki_kv(time, test_ts, inh_timing, k_v=4.0, x0=None)
    hh, k_e,k_i, sim = eh.estimate()
