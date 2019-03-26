# encoding: utf-8

"""T-score based P-values of pairwise comparison or columns of a contingency table."""

from __future__ import division

import numpy as np
from scipy.stats import t

from cr.cube.util import lazyproperty

try:
    xrange
except NameError:  # pragma: no cover
    # pylint: disable=invalid-name
    xrange = range


class PairwiseSignificance:
    """Implementation of p-vals and t-tests for each column proportions comparison."""

    def __init__(self, slice_, axis=0, weighted=True, alpha=0.05, only_larger=True):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger

    @lazyproperty
    def values(self):
        """list of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are coliumns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        slice_, axis, weighted = self._slice, self._axis, self._weighted
        return [
            _ColumnPairwiseSignificance(
                slice_, col_idx, axis, weighted, self._alpha, self._only_larger
            )
            for col_idx in range(self._slice.shape[1])
        ]

    @lazyproperty
    def pairwise_indices(self):
        return np.array([sig.pairwise_indices for sig in self.values]).T


# pylint: disable=too-few-public-methods
class _ColumnPairwiseSignificance:
    """Value object providing matrix of T-score based pairwise-comparison P-values"""

    def __init__(
        self, slice_, col_idx, axis=0, weighted=True, alpha=0.05, only_larger=True
    ):
        self._slice = slice_
        self._col_idx = col_idx
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger

    @lazyproperty
    def t_stats(self):
        props = self._slice.proportions(axis=0)
        diff = props - props[:, [self._col_idx]]
        margin = self._slice.margin(axis=0, weighted=self._weighted)
        var_props = props * (1.0 - props) / margin
        se_diff = np.sqrt(var_props + var_props[:, [self._col_idx]])
        return diff / se_diff

    @lazyproperty
    def p_vals(self):
        unweighted_n = self._slice.margin(axis=0, weighted=False)
        df = unweighted_n + unweighted_n[self._col_idx] - 2
        return 2 * (1 - t.cdf(abs(self.t_stats), df=df))

    @lazyproperty
    def pairwise_indices(self):
        significance = self.p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats < 0, significance)
        return [tuple(np.where(sig_row)[0]) for sig_row in significance]
