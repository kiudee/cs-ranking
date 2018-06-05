import numpy as np
from theano import tensor as tt

from csrank.util import normalize


def replace_inf_theano(x):
    if tt.any(tt.isinf(x)):
        x = tt.switch(tt.isinf(x), 2e+300, x)
        x = tt.switch(tt.isnan(x), 2e+300, x)
    return x


def replace_inf_np(x):
    if np.any(np.isinf(x)):
        x[np.isinf(x)] = 2e+300
        x[np.isnan(x)] = 2e+300
    return x


def replace_nan_theano(p):
    if tt.any(tt.isnan(p)):
        p = tt.switch(tt.isnan(p), 1.0, p)
        p = tt.switch(tt.isinf(p), 1.0, p)
        p = normalize(p)
    return p


def replace_nan_np(p):
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 1.0
        p[np.isinf(p)] = 1.0
        p = normalize(p)
    return p