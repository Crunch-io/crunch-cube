# encoding: utf-8

"""T-score based P-values of pairwise comparison or columns of a contingency table."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import t

from cr.cube.util import lazyproperty

try:
    xrange
except NameError:  # pragma: no cover
    xrange = range


class PairwiseSignificance(object):
    """Implementation of p-vals and t-tests for each column proportions comparison."""

    def __init__(self, slice_, alpha=0.05, only_larger=True):
        self._slice = slice_
        self._alpha = alpha
        self._only_larger = only_larger

    @classmethod
    def pairwise_indices(cls, slice_, alpha, only_larger):
        """ -> 2D ndarray of tuples of pairwise indices."""
        return cls(slice_, alpha, only_larger)._pairwise_indices

    @classmethod
    def scale_mean_pairwise_indices(cls, slice_, alpha, only_larger):
        """ -> 1D ndarray of tuples of column-indices meeting pairwise-t threshold.

        Indicates "whole columns" that are significantly different, based on the mean of
        the scale (category numeric-values) for each column. The length of the array is
        that of the columns-dimension.
        """
        return cls(slice_, alpha, only_larger)._scale_mean_pairwise_indices

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

    @lazyproperty
    def values(self):
        """list of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are coliumns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        return [
            _ColumnPairwiseSignificance(
                self._slice, col_idx, self._alpha, self._only_larger
            )
            for col_idx in xrange(self._slice.shape[1])
        ]

    @lazyproperty
    def _pairwise_indices(self):
        """2D ndarray containing tuples of pairwise indices."""
        return np.array([sig.pairwise_indices for sig in self.values]).T

    @lazyproperty
    def _scale_mean_pairwise_indices(self):
        """ndarray containing tuples of pairwise indices."""
        return np.array([sig.scale_mean_pairwise_indices for sig in self.values]).T


class _ColumnPairwiseSignificance(object):
    """Value object providing matrix of T-score based pairwise-comparison P-values"""

    def __init__(self, slice_, col_idx, alpha=0.05, only_larger=True):
        self._slice = slice_
        self._col_idx = col_idx
        self._alpha = alpha
        self._only_larger = only_larger

    @lazyproperty
    def p_vals(self):
        return 2 * (1 - t.cdf(abs(self.t_stats), df=self._df))

    @lazyproperty
    def p_vals_scale_means(self):
        return 2 * (1 - t.cdf(abs(self.t_stats_scale_means), df=self._two_sample_df))

    @lazyproperty
    def pairwise_indices(self):
        significance = self.p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats < 0, significance)
        return [tuple(np.where(sig_row)[0]) for sig_row in significance]

    @lazyproperty
    def scale_mean_pairwise_indices(self):
        """
        List of tuples indicating the significance

        Considreing this output: [(3,), (2, 3), (3,), ()] and scale_means values
        [26, 30, 21, 11], each element contains a tuple of other element
        indices that are significantly different from the present element's
        index in a two-tailed test with alpha=.05 by default. The element at
        index 0 (26) indicates that it differs significantly only from the
        element at index 3 (11)
        """
        significance = self.p_vals_scale_means < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.t_stats_scale_means > 0, significance)
        return tuple(np.where(significance)[0])

    @lazyproperty
    def summary_p_vals(self):
        return 2 * (1 - t.cdf(abs(self.summary_t_stats), df=self._df))

    @lazyproperty
    def summary_pairwise_indices(self):
        significance = self.summary_p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(self.summary_t_stats < 0, significance)
        return tuple(np.where(significance)[0])

    @lazyproperty
    def summary_t_stats(self):
        col_margin_props = self._slice.column_base / self._slice.table_margin
        diff = col_margin_props - col_margin_props[self._col_idx]
        var_props = (
            col_margin_props * (1.0 - col_margin_props) / self._slice.table_margin
        )
        se_diff = np.sqrt(var_props + var_props[self._col_idx])
        return diff / se_diff

    @lazyproperty
    def t_stats(self):
        if self._slice.cube_is_mr_by_itself:
            return self.t_stats_correct
        props = self._slice.column_proportions
        diff = props - props[:, [self._col_idx]]
        var_props = props * (1.0 - props) / self._slice.column_base
        se_diff = np.sqrt(var_props + var_props[:, [self._col_idx]])
        return diff / se_diff

    @lazyproperty
    def t_stats_correct(self):
        """It returns the t_statistic for MR variables considering the overlaps"""
        diff, se_diff = self._slice.overlaps_tstats
        t_stats = diff[:, self._col_idx, :] / se_diff[:, self._col_idx, :]
        t_stats[:, self._col_idx] = 0
        return t_stats

    @lazyproperty
    def t_stats_scale_means(self):
        """
        This property calculates the Two-tailed t-test using the formula:
        t = X1 - X2 / Sx1x2 * sqrt(1/n1 + 1/n2)
        where X1 and X2 are the scale mean value for the 2 sample we're
        comparing, n1 and n2 are the number of people from the 1st ans 2nd
        sample who provided a response to the survey, Sx1x2 is the standard
        deviation. In this case the standard deviation is:
        Sx1x2 = sqrt(((n1-1)*s2x1 + (n2-2)*s2x2)/(n1+n2+2)), where s2x1 and
        s2x2 are the is the standard deviation for sample 1 and 2.

        :return: numpy 1D array of tstats
        """

        variance = self._slice.var_scale_means_row
        # Sum for each column of the counts that have not a nan index in the
        # related numeric counts
        not_a_nan_index = ~np.isnan(self._slice.rows_dimension_numeric)
        counts = np.sum(self._slice.counts[not_a_nan_index, :], axis=0)

        standard_deviation = np.sqrt(
            np.divide(
                ((counts[self._col_idx] - 1) * variance[self._col_idx])
                + ((counts - 1) * np.array(variance)),
                (counts[self._col_idx] + counts - 2),
            )
        )

        tstats_scale_means = (
            self._slice.scale_means_row[self._col_idx] - self._slice.scale_means_row
        ) / (standard_deviation * np.sqrt((1 / counts[self._col_idx]) + (1 / counts)))

        return tstats_scale_means

    @lazyproperty
    def _df(self):
        # if the cube to which the slice belongs is a CATxMRxITSELF
        # returns the n1 + n2 as degrees of freedom, n1 + n2 -2 otherwise
        selected_unweighted_n = (
            self._slice.column_base[self._col_idx]
            if self._slice.column_base.ndim < 2
            else self._slice.column_base[:, self._col_idx][:, None]
        )
        if self._slice.cube_is_mr_by_itself:
            return self._slice.column_base + selected_unweighted_n
        return self._slice.column_base + selected_unweighted_n - 2

    @lazyproperty
    def _two_sample_df(self):
        not_a_nan_index = ~np.isnan(self._slice.rows_dimension_numeric)
        counts = np.sum(self._slice.counts[not_a_nan_index, :], axis=0)
        return counts[self._col_idx] + counts - 2
