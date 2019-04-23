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
            for col_idx in range(self._slice.get_shape(hs_dims=self._hs_dims)[1])
        ]

    @lazyproperty
    def pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        return np.array([sig.pairwise_indices for sig in self.values]).T

    @lazyproperty
    def summary_pairwise_indices(self):
        """ndarray containing tuples of pairwise indices for the column summary."""
        summary_pairwise_indices = np.empty(
            self.values[0].t_stats.shape[1], dtype=object
        )
        summary_pairwise_indices[:] = [
            sig.summary_pairwise_indices for sig in self.values
        ]
        return summary_pairwise_indices


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
    def _unweighted_col_margin(self):
        return self._slice.margin(
            axis=0, weighted=False, include_transforms_for_dims=self._hs_dims
        )

    @lazyproperty
    def t_stats(self):
        props = self._slice.proportions(
            axis=0, include_transforms_for_dims=self._hs_dims
        )
        diff = props - props[:, [self._col_idx]]
        var_props = props * (1.0 - props) / self._unweighted_col_margin
        se_diff = np.sqrt(var_props + var_props[:, [self._col_idx]])
        return diff / se_diff

    @lazyproperty
    def p_vals(self):
        return 2 * (1 - t.cdf(abs(self.t_stats), df=self._df))

    @lazyproperty
    def pairwise_indices(self):
        significance = self.p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats < 0, significance)
        return [tuple(np.where(sig_row)[0]) for sig_row in significance]

    @lazyproperty
    def summary_pairwise_indices(self):
        significance = self.summary_p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.summary_t_stats < 0, significance)
        return tuple(np.where(significance)[0])

    @lazyproperty
    def summary_t_stats(self):
        total_margin = self._slice.margin(weighted=self._weighted)
        col_margin_props = self._unweighted_col_margin / total_margin
        diff = col_margin_props - col_margin_props[self._col_idx]
        var_props = col_margin_props * (1.0 - col_margin_props) / total_margin
        se_diff = np.sqrt(var_props + var_props[self._col_idx])
        return diff / se_diff

    @lazyproperty
    def summary_p_vals(self):
        return 2 * (1 - t.cdf(abs(self.summary_t_stats), df=self._df))

    @lazyproperty
    def _df(self):
        selected_unweighted_n = (
            self._unweighted_n[self._col_idx]
            if self._unweighted_n.ndim < 2
            else self._unweighted_n[:, self._col_idx][:, None]
        )
        return self._unweighted_n + selected_unweighted_n - 2

    @lazyproperty
    def _unweighted_n(self):
        return self._slice.margin(
            axis=0, weighted=False, include_transforms_for_dims=self._hs_dims
        )
