# encoding: utf-8

"""T-score based P-values of pairwise comparison or columns of a contingency table."""

from __future__ import division

import numpy as np
from scipy.stats import t

from cr.cube.util import lazyproperty, intersperse_hs_in_std_res

try:
    xrange
except NameError:  # pragma: no cover
    # pylint: disable=invalid-name
    xrange = range


class PairwiseSignificance:
    """Implementation of p-vals and t-tests for each column proportions comparison."""

    def __init__(
        self, slice_, axis=0, weighted=True, alpha=0.05, only_larger=True, hs_dims=None
    ):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger
        self._hs_dims = hs_dims

    @lazyproperty
    def values(self):
        """list of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are coliumns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        # TODO: Figure out how to intersperse pairwise objects for columns
        # that represent H&S
        return [
            _ColumnPairwiseSignificance(
                self._slice,
                col_idx,
                self._axis,
                self._weighted,
                self._alpha,
                self._only_larger,
                self._hs_dims,
            )
            for col_idx in range(self._slice.shape[1])
        ]

    @lazyproperty
    def pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        pwi = np.array([sig.pairwise_indices for sig in self.values]).T

        if self._hs_dims and 1 in self._hs_dims:
            # If we need to account for the dimension 1 in pairwise indices, we need
            # to intersperse with NaNs. The dimension 0 is already tackled
            # when determining the indices.
            pwi = intersperse_hs_in_std_res(self._slice, (1,), pwi)

        return pwi


# pylint: disable=too-few-public-methods
class _ColumnPairwiseSignificance:
    """Value object providing matrix of T-score based pairwise-comparison P-values"""

    def __init__(
        self,
        slice_,
        col_idx,
        axis=0,
        weighted=True,
        alpha=0.05,
        only_larger=True,
        hs_dims=None,
    ):
        self._slice = slice_
        self._col_idx = col_idx
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger
        self._hs_dims = hs_dims

    @lazyproperty
    def _t_stats(self):
        props = self._slice.proportions(axis=0)
        diff = props - props[:, [self._col_idx]]
        margin = self._slice.margin(axis=0, weighted=self._weighted)
        var_props = props * (1.0 - props) / margin
        se_diff = np.sqrt(var_props + var_props[:, [self._col_idx]])
        return diff / se_diff

    @lazyproperty
    def t_stats(self):
        return intersperse_hs_in_std_res(self._slice, self._hs_dims, self._t_stats)

    @lazyproperty
    def p_vals(self):
        unweighted_n = self._slice.margin(axis=0, weighted=False)
        df = unweighted_n + unweighted_n[self._col_idx] - 2
        p_vals = 2 * (1 - t.cdf(abs(self._t_stats), df=df))
        return intersperse_hs_in_std_res(self._slice, self._hs_dims, p_vals)

    @lazyproperty
    def pairwise_indices(self):
        significance = self.p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats < 0, significance)
        return [tuple(np.where(sig_row)[0]) for sig_row in significance]
