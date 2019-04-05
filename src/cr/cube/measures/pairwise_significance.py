# encoding: utf-8

"""T-score based P-values of pairwise comparison or columns of a contingency table."""

from __future__ import division

import numpy as np
from scipy.stats import t

from cr.cube.util import (
    lazyproperty,
    intersperse_hs_in_std_res,
    apply_pruning_mask,
    compress_pruned,
)

try:
    xrange
except NameError:  # pragma: no cover
    # pylint: disable=invalid-name
    xrange = range


class PairwiseSignificance:
    """Implementation of p-vals and t-tests for each column proportions comparison."""

    def __init__(
        self,
        slice_,
        axis=0,
        weighted=True,
        alpha=0.05,
        only_larger=True,
        hs_dims=None,
        prune=False,
    ):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger
        self._hs_dims = hs_dims
        self._prune = prune

    @lazyproperty
    def values(self):
        """list of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are coliumns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        return [
            _ColumnPairwiseSignificance(
                self._slice,
                col_idx,
                self._axis,
                self._weighted,
                self._alpha,
                self._only_larger,
                self._hs_dims,
                self._prune,
            )
            for col_idx in range(self._slice.shape[1 - self._axis])
            if not self._is_pruned(col_idx)
        ]

    @lazyproperty
    def pairwise_indices(self):
        return np.array([sig.pairwise_indices for sig in self.values]).T

    def _is_pruned(self, col_idx):
        if not self._prune:
            return False
        return not np.any(self._slice.as_array().T[col_idx])


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
        prune=False,
    ):
        self._slice = slice_
        self._col_idx = col_idx
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha
        self._only_larger = only_larger
        self._hs_dims = hs_dims
        self._prune = prune

    @lazyproperty
    def _props(self):
        return self._slice.proportions(axis=self._axis)

    @lazyproperty
    def _t_stats(self):
        diff = self._props - self._props[:, [self._col_idx]]
        margin = self._slice.margin(axis=self._axis, weighted=self._weighted)
        var_props = self._props * (1.0 - self._props) / margin
        se_diff = np.sqrt(var_props + var_props[:, [self._col_idx]])
        return diff / se_diff

    @lazyproperty
    def t_stats(self):
        t_stats = intersperse_hs_in_std_res(self._slice, self._hs_dims, self._t_stats)
        if self._prune:
            t_stats = apply_pruning_mask(self._slice, t_stats, self._hs_dims)
            t_stats = compress_pruned(t_stats)
        return t_stats

    @lazyproperty
    def p_vals(self):
        unweighted_n = self._slice.margin(axis=self._axis, weighted=False)
        df = unweighted_n + unweighted_n[self._col_idx] - 2
        p_vals = 2 * (1 - t.cdf(abs(self._t_stats), df=df))
        p_vals = intersperse_hs_in_std_res(self._slice, self._hs_dims, p_vals)
        if self._prune:
            p_vals = apply_pruning_mask(self._slice, p_vals, self._hs_dims)
            p_vals = compress_pruned(p_vals)
        return p_vals

    @lazyproperty
    def pairwise_indices(self):
        significance = self.p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats < 0, significance)
        pwi = [tuple(np.where(sig_row)[0]) for sig_row in significance]
        return pwi
