import numpy as np


class ParamHolder(object):
    __slots__ = ['k_v', 'k_iu', 'k_eu', 'k_ip', 'k_ep',
                 'k_cat', 'Km', 'k_dc', 'k_dn', 'Kmd', 'r_total',
                 'time_points', 'kin_c_with_time', 'kin_n_with_time']

    def __init__(self, dict_params):
        for i, ii in dict_params.iteritems():
            setattr(self, i, ii)

def main_ode(y, t, p):
    c_u, n_u, c_p, n_p = y[0], y[1], y[2], y[3]

    kin_c = np.interp(t, p.time_points, p.kin_c_with_time)
    kin_n = np.interp(t, p.time_points, p.kin_n_with_time)

    d_c_u = -kin_c * p.k_cat * c_u/(c_u+p.Km)\
        + p.k_dc * c_p/(c_p + p.Kmd) - p.k_iu * c_u + p.k_eu * n_u
    d_n_u = -kin_n * p.k_cat * n_u/(n_u + p.Km) + p.k_dn * n_p/(n_p + p.Kmd)\
        + p.k_v * p.k_iu * c_u - p.k_v * p.k_eu * n_u
    d_c_p = kin_c * p.k_cat * c_u/(c_u + p.Km) - p.k_dc * c_p/(c_p + p.Kmd)\
        - p.k_ip * c_p + p.k_ep * n_p
    d_n_p = p.r_total - c_u - n_u/p.k_v - c_p - n_p/p.k_v
    return [d_c_u, d_n_u, d_c_p, d_n_p]
