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


def wishartCDF(X, n_min, n_max):
    w_p = np.zeros(X.shape)
    for i, x in np.ndenumerate(X):
        w_p[i] = _wishart_pfaffian(x, n_min, n_max)
    return _normalizing_const(n_min, n_max) * w_p


def _wishart_pfaffian(x, n_min, n_max):
    """Value providing the CDF of the largest eigenvalue of a Wishart(n, p)
    evaluated at x.

    if (n_min != n_mat) {
        // Fill in extra column
        VectorXmp alpha_vec(n_min);
        T alpha_last = 0.5*(n_max + n_min + 1);
        for (int i = 0; i < n_min; i++) {
            alpha_vec(i) = alpha + i + 1;
            A(i, n_min) = pow(2, -alpha_last)*boost::math::gamma_p(alpha_vec(i), 0.5*xx)/boost::math::tgamma(alpha_last);
            A(n_min, i) = - A(i, n_min);
        }
    }
    """
    size = n_min + (n_min % 2)
    alpha = 0.5 * (n_max - n_min - 1)
    A = np.zeros([size, size])
    alpha_ind = np.arange(n_min, dtype=float)
    alpha_vec = alpha_ind + alpha + 1
    gammainc_a = gammainc(alpha_vec, x / 2)

    if n_min != size:
        alpha_ind = alpha_ind.astype(np.int)
        other_ind = np.full(n_min, size - 1, dtype=np.int)
        alpha_last = (n_max + n_min + 1) / 2
        A[other_ind, alpha_ind] = (
            np.power(2, -alpha_last) * gammainc_a / gamma(alpha_last)
        )
        # TODO check that this is not transposed
        A[alpha_ind, other_ind] = -A[other_ind, alpha_ind]
        # print(A)
    """

    for (int i = 0; i < n_min; i++) {
        p(i) = boost::math::gamma_p(alpha + i + 1, 0.5*xx);
        g(i) = boost::math::tgamma(alpha + i + 1);
    }
    for (int i = 0; i < (2*n_min - 2); i++) {
        q(i) = pow(0.5, 2*alpha + i + 2) * boost::math::tgamma(2*alpha+i+2)*boost::math::gamma_p(2*alpha + i + 2, xx);
    }
    """
    p = gammainc_a
    g = gamma(alpha_vec)
    q_ind = np.arange(2 * n_min - 2)
    q = (
        np.power(0.5, 2 * alpha + q_ind + 2)
        * gamma(2 * alpha + q_ind + 2)
        * gammainc(2 * alpha + q_ind + 2, x)
    )

    """


    for (int i = 0; i < n_min; i++) {
        b = 0.5*p(i)*p(i);
        for(int j = i; j < (n_min - 1); j++) {
            b -= q(i+j)/(g(i)*g(j+1));
            A(i, j+1) = p(i)*p(j+1) - 2*b;
            A(j+1, i) = -A(i, j+1);
        }
        Rcpp::checkUserInterrupt();
    }
    """

    # Lets do the inefficient iteration first and figure out index tricks later
    for i in range(n_min):
        b = 0.5 * p[i] * p[i]
        for j in range(i, n_min - 1):
            b -= q[i + j] / g[i] * g[j + 1]
            A[j + 1, i] = p[i] * p[j + 1] - 2 * b
            A[i, j + 1] = -A[j + 1, i]

    return np.sqrt(det(A))


def _normalizing_const(n_min, n_max):
    """int n_mat = n_min + (n_min % 2);
    double alpha = 0.5*(n_max - n_min - 1);
    // Compute constant
    double K1 = pow(M_PI, 0.5*n_min*n_min);
    K1 /= pow(2, 0.5*n_min*n_max)*mgamma_C(0.5*n_max, n_min, false)*mgamma_C(0.5*n_min, n_min, false);
    double K2 = pow(2, alpha*n_mat+0.5*n_mat*(n_mat+1));
    for (int k = 0; k < n_mat; k++) {
        K2 *= boost::math::tgamma(alpha + k + 1);
    }
    return K1*K2;"""
    size = n_min + (n_min % 2)
    alpha = 0.5 * (n_max - n_min - 1)
    K1 = (
        np.power(pi, 0.5 * n_min * n_min)
        / np.power(2, 0.5 * n_min * n_max)
        * _mgamma(0.5 * n_max, n_min)
        * _mgamma(0.5 * n_min, n_min)
    )
    K2 = np.power(2, alpha * size + 0.5 * size * (size + 1))
    for i in range(size):
        K2 *= gamma(alpha + i + 1)
    print K1*K2
    return K1 * K2


def _mgamma(x, m):
    res = pow(pi, 0.25 * m * (m - 1))
    for i in range(m):
        res *= gamma(x - 0.5 * i)
    return res
