# encoding: utf-8
# pylint: disable=too-few-public-methods, invalid-name

"""The CDF of the largest eigenvalue of a Wishart distribution.

Compute the CDF of the Wishart distribution using the algorithm of
Chiani (2014). The Wishart CDF is not defined in scipy.stats.wishart, nor in
tensorflow_probability.distributions.wishart. At the time of writing, the only
implementation of this CDF is the R package rootWishart. This largely
follows that, but using scipy implementations rather than Boost, and omitting
multi-precision support.

For more info about the formulae, and the terse variable naming,
please refer to the paper:

"Distribution of the largest eigenvalue for real Wishart and Gaussian random
matrices and a simple approximation for the Tracy-Widom distribution"

found at: https://arxiv.org/pdf/1209.3394.pdf
"""

from __future__ import division

import numpy as np
from scipy.special import gamma, gammainc
from scipy.linalg import det
from scipy.constants import pi
from cr.cube.util import lazyproperty

try:
    xrange
except NameError:  # pragma: no cover
    xrange = range


class WishartCDF:
    """Implementation of Cumulative Distribution Function (CDF)."""

    def __init__(self, chisq, n_min, n_max):
        self._chisq = chisq
        self.n_min = n_min
        self._n_max = n_max

    @lazyproperty
    def values(self):
        """ndarray of wishart CDF values"""
        return self.K * self.wishart_pfaffian

    @lazyproperty
    def wishart_pfaffian(self):
        """ndarray of wishart pfaffian CDF, before normalization"""
        return np.array(
            [Pfaffian(self, val).value for i, val in np.ndenumerate(self._chisq)]
        ).reshape(self._chisq.shape)

    @lazyproperty
    def alpha(self):
        """int representing common basis of off-diagonal elements of A"""
        return 0.5 * (self._n_max - self.n_min - 1)

    @lazyproperty
    def alpha_ind(self):
        """indices of off-diagonal elements of A"""
        return np.arange(self.n_min)

    @lazyproperty
    def alpha_last(self):
        """elements of row and column to append to A when its order/size is odd"""
        return (self._n_max + self.n_min + 1) / 2

    @lazyproperty
    def alpha_vec(self):
        """elements of row and column to append to A when its order/size is odd"""
        return self.alpha_ind + self.alpha + 1

    @lazyproperty
    def other_ind(self):
        """last row or column of square A"""
        return np.full(self.n_min, self.size - 1, dtype=np.int)

    @staticmethod
    def _mgamma(x, m):
        mgamma = np.float_power(pi, 0.25 * m * (m - 1))
        for i in xrange(m):
            mgamma *= gamma(x - 0.5 * i)
        return mgamma

    @lazyproperty
    def K(self):
        """Normalizing constant for wishart CDF."""
        K1 = np.float_power(pi, 0.5 * self.n_min * self.n_min)
        K1 /= (
            np.float_power(2, 0.5 * self.n_min * self._n_max)
            * self._mgamma(0.5 * self._n_max, self.n_min)
            * self._mgamma(0.5 * self.n_min, self.n_min)
        )

        K2 = np.float_power(
            2, self.alpha * self.size + 0.5 * self.size * (self.size + 1)
        )
        for i in xrange(self.size):
            K2 *= gamma(self.alpha + i + 1)
        return K1 * K2

    @lazyproperty
    def size(self):
        """int representing the size of the wishart matrix.

        If the original size is even, that this is the size that's returned. If the
        size is odd, then the next even number is returned, because of the row/col
        insertion, that happens as the part of the algorithm
        """
        return self.n_min + (self.n_min % 2)


class Pfaffian:
    """Implementation of the Pfaffian value object class."""

    def __init__(self, wishart_cdf, chisq_ordinal):
        self._wishart_cdf = wishart_cdf
        self._chisq_val = chisq_ordinal

    @lazyproperty
    def value(self):
        """return float Cumulative Distribution Function.

        The return value represents a floating point number of the CDF of the
        largest eigenvalue of a Wishart(n, p) evaluated at chisq_val.
        """
        wishart = self._wishart_cdf

        # Prepare variables for integration algorithm
        A = self.A
        p = self._gammainc_a
        g = gamma(wishart.alpha_vec)
        q_ind = np.arange(2 * wishart.n_min - 2)
        q_vec = 2 * wishart.alpha + q_ind + 2
        q = np.float_power(0.5, q_vec) * gamma(q_vec) * gammainc(q_vec, self._chisq_val)

        # Perform integration (i.e. calculate Pfaffian CDF)
        for i in xrange(wishart.n_min):
            # TODO consider index tricks instead of iteration here
            b = 0.5 * p[i] * p[i]
            for j in xrange(i, wishart.n_min - 1):
                b -= q[i + j] / (g[i] * g[j + 1])
                A[j + 1, i] = p[i] * p[j + 1] - 2 * b
                A[i, j + 1] = -A[j + 1, i]

        if np.any(np.isnan(A)):
            return 0
        return np.sqrt(det(A))

    @lazyproperty
    def A(self):
        """ndarray - a skew-symmetric matrix for integrating the target distribution"""
        wishart = self._wishart_cdf

        base = np.zeros([wishart.size, wishart.size])
        if wishart.n_min % 2:
            # If matrix has odd number of elements, we need to append a
            # row and a col, in order for the pfaffian algorithm to work
            base = self._make_size_even(base)
        return base

    @lazyproperty
    def _gammainc_a(self):
        return gammainc(self._wishart_cdf.alpha_vec, 0.5 * self._chisq_val)

    def _make_size_even(self, base):
        wishart = self._wishart_cdf
        alpha_ind = wishart.alpha_ind
        other_ind = wishart.other_ind
        alpha_last = wishart.alpha_last

        base[other_ind, alpha_ind] = (
            np.float_power(2, -alpha_last) * self._gammainc_a / gamma(alpha_last)
        )
        base[alpha_ind, other_ind] = -base[other_ind, alpha_ind]

        return base
