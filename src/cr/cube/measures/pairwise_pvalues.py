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
class PairwiseSignificance:
    """Value object providing matrix of pairwise-comparison P-values"""

    def __init__(self, slice_, axis=0, weighted=True):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted

    @classmethod
    def pvals(cls, slice_, axis=0, weighted=True):
        return cls._factory(slice_, axis, weighted)._pvals

    @staticmethod
    def _calculate_chi_squared(numel, proportions, margin, observed):
        chisq = np.zeros([numel, numel])
        for i in xrange(1, numel):
            for j in xrange(0, numel - 1):
                chisq[i, j] = chisq[j, i] = np.sum(
                    np.square(proportions[:, i] - proportions[:, j]) / observed
                ) / (1 / margin[i] + 1 / margin[j])
        return chisq

    def _calculate_pvals_from_chi_squared(self, props):
        return self._intersperse_insertions_rows_and_columns(
            1.0 - WishartCDF(props, self._n_min, self._n_max).values
        )

    @staticmethod
    def _factory(slice_, axis, weighted):
        if slice_.dim_types[0] == DT.MR_SUBVAR:
            return _MrXCatPairwiseSignificance(slice_, axis, weighted)
        return _CatXCatPairwiseSignificance(slice_, axis, weighted)

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

    @lazyproperty
    def _margin(self):
        """Margin for the axis as numpy.ndarray."""
        return self._slice.margin(axis=self._axis)

    @lazyproperty
    def _numel(self):
        """Number of elements of the dimension opposite to axis, as int."""
        return self._slice.get_shape()[1 - self._axis]

    @lazyproperty
    def _n_max(self):
        """Size (zero based) of the bigger of the two slice's dimension, as int."""
        return max(self._slice.get_shape()) - 1

    @lazyproperty
    def _n_min(self):
        """Size (zero based) of the smaller of the two slice's dimension, as int."""
        return min(self._slice.get_shape()) - 1

    @lazyproperty
    def _off_margin(self):
        return self._slice.margin(axis=1, include_mr_cat=self._include_mr_cat)

    @lazyproperty
    def _proportions(self):
        return self._slice.proportions(
            axis=self._axis, include_mr_cat=self._include_mr_cat
        )


class _CatXCatPairwiseSignificance(PairwiseSignificance):
    """Pairwise significance for CAT x CAT type slices."""

    _include_mr_cat = False

    @lazyproperty
    def _observed(self):
        """Observed marginal proportions, as float."""
        total = self._slice.margin()
        return self._off_margin / total

    @lazyproperty
    def _pairwise_chisq(self):
        """Pairwise comparisons (Chi-Square) along axis, as numpy.ndarray.

        Returns a square, symmetric matrix of test statistics for the null
        hypothesis that each vector along *axis* is equal to each other.
        """
        return self._calculate_chi_squared(
            self._numel, self._proportions, self._margin, self._observed
        )

    @lazyproperty
    def _pvals(self):
        """Square matrix of pairwise Chi-square along axis, as numpy.ndarray."""
        return self._calculate_pvals_from_chi_squared(self._pairwise_chisq)


class _MrXCatPairwiseSignificance(PairwiseSignificance):
    """Pairwise significance for MR x CAT type slices."""

    _include_mr_cat = True

    @lazyproperty
    def _pairwise_chisq(self):
        """Pairwise comparisons (Chi-Square) along axis, as numpy.ndarray.

        Returns a list of square and symmetric matrices of test statistics for the null
        hypothesis that each vector along *axis* is equal to each other.
        """
        return [
            self._calculate_chi_squared(
                self._numel,
                mr_subvar_proportions,
                self._margin[idx],
                self._off_margin[idx] / np.sum(self._off_margin[idx]),
            )
            for (idx, mr_subvar_proportions) in enumerate(self._proportions)
        ]

    @lazyproperty
    def _pvals(self):
        """Square matrix of pairwise Chi-square along axis, as numpy.ndarray."""
        return [
            self._calculate_pvals_from_chi_squared(mr_subvar_chisq)
            for mr_subvar_chisq in self._pairwise_chisq
        ]
