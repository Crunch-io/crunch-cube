# encoding: utf-8

"""The CDF of the largest eigenvalue of a Wishart distribution.

Compute the CDF of the Wishart distribution using the algorithm of
Chiani (2014). The Wishart CDF is not defined in scipy.stats.wishart, nor in
tensorflow_probability.distributions.wishart. At the time of writing, the only
implementation of this CDF is the R package rootWishart. This largely
follows that, but using scipy implementations rather than Boost, and omitting
multi-precision support.
"""

from __future__ import division

import numpy as np
from scipy.special import gamma, gammainc
from scipy.linalg import det
from scipy.constants import pi

try:
    xrange
except NameError:
    xrange = range


def wishartCDF(X, n_min, n_max):
    w_p = np.zeros(X.shape)
    for i, x in np.ndenumerate(X):
        w_p[i] = _wishart_pfaffian(x, n_min, n_max)
    return _normalizing_const(n_min, n_max) * w_p


def _wishart_pfaffian(x, n_min, n_max):
    """Value providing the CDF of the largest eigenvalue of a Wishart(n, p)
    evaluated at x.
    """

    size = n_min + (n_min % 2)
    alpha = 0.5 * (n_max - n_min - 1)
    A = np.zeros([size, size])
    alpha_ind = np.arange(n_min, dtype=np.float)
    alpha_vec = alpha_ind + alpha + 1
    gammainc_a = gammainc(alpha_vec, 0.5 * x)
    if n_min != size:
        alpha_ind = alpha_ind.astype(np.int)
        other_ind = np.full(n_min, size - 1, dtype=np.int)
        alpha_last = (n_max + n_min + 1) / 2
        A[other_ind, alpha_ind] = (
            np.float_power(2, -alpha_last) * gammainc_a / gamma(alpha_last)
        )
        A[alpha_ind, other_ind] = -A[other_ind, alpha_ind]

    p = gammainc_a
    g = gamma(alpha_vec)
    q_ind = np.arange(2 * n_min - 2)
    q_vec = 2 * alpha + q_ind + 2
    q = np.float_power(0.5, q_vec) * gamma(q_vec) * gammainc(q_vec, x)

    # TODO consider index tricks instead of iteration here
    for i in xrange(n_min):
        b = 0.5 * p[i] * p[i]
        for j in xrange(i, n_min - 1):
            b -= q[i + j] / (g[i] * g[j + 1])
            A[j + 1, i] = p[i] * p[j + 1] - 2 * b
            A[i, j + 1] = -A[j + 1, i]
    return np.sqrt(det(A))


def _normalizing_const(n_min, n_max):
    size = n_min + (n_min % 2)
    alpha = 0.5 * (n_max - n_min - 1)
    K1 = np.float_power(pi, 0.5 * n_min * n_min)
    K1 /= (
        np.float_power(2, 0.5 * n_min * n_max)
        * _mgamma(0.5 * n_max, n_min)
        * _mgamma(0.5 * n_min, n_min)
    )

    K2 = np.float_power(2, alpha * size + 0.5 * size * (size + 1))
    for i in xrange(size):
        K2 *= gamma(alpha + i + 1)
    return K1 * K2


def _mgamma(x, m):
    res = np.float_power(pi, 0.25 * m * (m - 1))
    for i in xrange(m):
        res *= gamma(x - 0.5 * i)
    return res
