# encoding: utf-8

"""P-values of pairwise comparison or columns of a contingency table."""

from __future__ import division

import numpy as np

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.distributions.wishart import WishartCDF
from cr.cube.util import lazyproperty

try:
    xrange
except NameError:  # pragma: no cover
    # pylint: disable=invalid-name
    xrange = range


# pylint: disable=too-few-public-methods
class PairwisePvalues:
    """Value object providing matrix of pairwise-comparison P-values"""

    def __init__(self, slice_, axis=0, weighted=True):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted

    @lazyproperty
    def values(self):
        """Square matrix of pairwise Chi-square along axis, as numpy.ndarray."""
        chisq = self._pairwise_chisq

        def _prepare_result(props):
            return self._intersperse_insertions_rows_and_columns(
                1.0 - WishartCDF(props, self._n_min, self._n_max).values
            )

        if isinstance(chisq, list):
            return [_prepare_result(table) for table in chisq]
        return _prepare_result(chisq)

    @lazyproperty
    def _categorical_pairwise_chisq(self):
        """Pairwise comparisons (Chi-Square) along axis, as numpy.ndarray.

        Returns a square, symmetric matrix of test statistics for the null
        hypothesis that each vector along *axis* is equal to each other.
        """
        return self._calculate_chi_squared(
            self._numel, self._proportions, self._margin, self._observed
        )

    @staticmethod
    def _calculate_chi_squared(numel, proportions, margin, observed):
        chisq = np.zeros([numel, numel])
        for i in xrange(1, numel):
            for j in xrange(0, numel - 1):
                chisq[i, j] = chisq[j, i] = np.sum(
                    np.square(proportions[:, i] - proportions[:, j]) / observed
                ) / (1 / margin[i] + 1 / margin[j])
        return chisq

    @lazyproperty
    def _mr_x_cat_pairwise_chisq(self):
        """TODO: Add docstring"""
        res = []
        mr_proportions = self._slice.proportions(axis=self._axis, include_mr_cat=True)

        for (margin_idx, table) in enumerate(mr_proportions):
            margin = self._margin[margin_idx]
            off_margin = self._slice.margin(axis=1, include_mr_cat=True)[margin_idx]
            observed = off_margin / np.sum(off_margin)
            res.append(
                self._calculate_chi_squared(self._numel, table, margin, observed)
            )
        return res

    @lazyproperty
    def _margin(self):
        """Margin for the axis as numpy.ndarray."""
        return self._slice.margin(axis=self._axis)

    @lazyproperty
    def _off_margin(self):
        """Margin for the opposite axis as numpy.ndarray."""
        return self._slice.margin(axis=(1 - self._axis))

    @lazyproperty
    def _pairwise_chisq(self):
        """Pairwise Chi-squared statistics along axis, as numpy.ndarray.

        Zscore is a measure of statistical significance of observed vs.
        expected counts. It's only applicable to a 2D contingency tables.
        """
        if self._slice.dim_types[0] == DT.MR_SUBVAR:
            return self._mr_x_cat_pairwise_chisq
        return self._categorical_pairwise_chisq

    @lazyproperty
    def _proportions(self):
        """Slice proportions for *axis* as numpy.ndarray."""
        return self._slice.proportions(axis=self._axis)

    @lazyproperty
    def _n_max(self):
        """Size (zero based) of the bigger of the two slice's dimension, as int."""
        return max(self._slice.get_shape()) - 1

    @lazyproperty
    def _n_min(self):
        """Size (zero based) of the smaller of the two slice's dimension, as int."""
        return min(self._slice.get_shape()) - 1

    @lazyproperty
    def _numel(self):
        """Number of elements of the dimension opposite to axis, as int."""
        return self._slice.get_shape()[1 - self._axis]

    @lazyproperty
    def _observed(self):
        """Observed marginal proportions, as float."""
        total = self._slice.margin()
        return self._off_margin / total

    @lazyproperty
    def _insertions_indices(self):
        """Return H&S indices of the pairwise comparison dimension."""
        return self._slice.inserted_hs_indices()[1 - self._axis]

    def _intersperse_insertions_rows_and_columns(self, pairwise_pvals):
        """Return pvals matrix with inserted NaN rows and columns, as numpy.ndarray.

        Insertions (Headers and Subtotals) create offset in calculated pvals, and
        these need to be taken into account, when converting them to columnar letters
        representation. For this reason, we need to insert an all-NaN row and a
        column in the right indices (the inserted indices of the H&S, in the
        respective dimension).
        """
        for i in self._insertions_indices:
            pairwise_pvals = np.insert(pairwise_pvals, i, np.nan, axis=0)
            pairwise_pvals = np.insert(pairwise_pvals, i, np.nan, axis=1)
        return pairwise_pvals
