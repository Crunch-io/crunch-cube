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
class WishartPairwiseSignificance:
    """Value object providing matrix of pairwise-comparison P-values"""

    def __init__(self, slice_, axis=0, weighted=True, alpha=0.05):
        self._slice = slice_
        self._axis = axis
        self._weighted = weighted
        self._alpha = alpha

    @classmethod
    def pvals(cls, slice_, axis=0, weighted=True):
        """Wishart CDF values for slice columns as square ndarray.

        Wishart CDF (Cumulative Distribution Function) is calculated to determine
        statistical significance of slice columns, in relation to all other columns.
        These values represent the answer to the question "How much is a particular
        column different from each other column in the slice".
        """
        return cls._factory(slice_, axis, weighted).pvals

    @classmethod
    def pairwise_indices(cls, slice_, axis=0, weighted=True):
        return cls._factory(slice_, axis, weighted).pairwise_indices

    def _chi_squared(self, proportions, margin, observed):
        """return ndarray of chi-squared measures for proportions' columns.

        *proportions* (ndarray): The basis of chi-squared calcualations
        *margin* (ndarray): Column margin for proportions (See `def _margin`)
        *observed* (ndarray): Row margin proportions (See `def _observed`)
        """
        n = self._element_count
        chi_squared = np.zeros([n, n])
        for i in xrange(1, n):
            for j in xrange(0, n - 1):
                denominator = 1 / margin[i] + 1 / margin[j]
                chi_squared[i, j] = chi_squared[j, i] = (
                    np.sum(np.square(proportions[:, i] - proportions[:, j]) / observed)
                    / denominator
                )
        return chi_squared

    @staticmethod
    def _pairwise_sig_indices(significance_by_col):
        """Return tuple like (1, 3), (6,), or () identifying pairwise sig cols.

        The *significance_by_col* parameter is a 1D boolean array with one value for
        each column in this crosstab, a True value signifying pairwise significance.
        """
        return tuple(
            i
            for i, is_pairwise_significant in enumerate(significance_by_col)
            if is_pairwise_significant
        )

    def _pvals_from_chi_squared(self, pairwise_chisq):
        """return statistical significance for props' columns.

        *pairwise_chisq* (ndarray) Matrix of chi-squared values (bases for Wishart CDF)
        """
        return self._intersperse_insertion_rows_and_columns(
            1.0 - WishartCDF(pairwise_chisq, self._n_min, self._n_max).values
        )

    @staticmethod
    def _factory(slice_, axis, weighted):
        """return subclass for PairwiseSignificance, based on slice dimension types."""
        if slice_.dim_types[0] == DT.MR_SUBVAR:
            return _MrXCatPairwiseSignificance(slice_, axis, weighted)
        return _CatXCatPairwiseSignificance(slice_, axis, weighted)

    @lazyproperty
    def _insertion_indices(self):
        """Return H&S indices of the pairwise comparison dimension."""
        return self._slice.inserted_hs_indices()[1 - self._axis]

    def _intersperse_insertion_rows_and_columns(self, pairwise_pvals):
        """Return pvals matrix with inserted NaN rows and columns, as numpy.ndarray.

        Each insertion (a header or a subtotal) creates an offset in the calculated
        pvals. These need to be taken into account when converting each pval to a
        corresponding column letter. For this reason, we need to insert an all-NaN
        row and a column at the right indices. These are the inserted indices of each
        insertion, along respective dimensions.
        """
        for i in self._insertion_indices:
            pairwise_pvals = np.insert(pairwise_pvals, i, np.nan, axis=0)
            pairwise_pvals = np.insert(pairwise_pvals, i, np.nan, axis=1)
        return pairwise_pvals

    @lazyproperty
    def _margin(self):
        """Margin for the axis as numpy.ndarray."""
        return self._slice.margin(axis=self._axis)

    @lazyproperty
    def _element_count(self):
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
    def _opposite_axis_margin(self):
        """ndarray representing margin along the axis opposite of self._axis

        In the process of calculating p-values for the column significance testing we
        need both the margin along the primary axis and the percentage margin along
        the opposite axis.
        """
        off_axis = 1 - self._axis
        return self._slice.margin(axis=off_axis, include_mr_cat=self._include_mr_cat)

    @lazyproperty
    def _proportions(self):
        """ndarray representing slice proportions along correct axis."""
        return self._slice.proportions(
            axis=self._axis, include_mr_cat=self._include_mr_cat
        )


class _CatXCatPairwiseSignificance(WishartPairwiseSignificance):
    """Pairwise significance for CAT x CAT type slices."""

    _include_mr_cat = False

    @lazyproperty
    def pairwise_indices(self):
        return np.array(
            [
                self._pairwise_sig_indices(pvals_row < self._alpha)
                for pvals_row in self.pvals
            ]
        )

    @lazyproperty
    def pvals(self):
        """Square matrix of pairwise Chi-square along axis, as numpy.ndarray."""
        return self._pvals_from_chi_squared(self._pairwise_chisq)

    @lazyproperty
    def _observed(self):
        """Observed marginal proportions, as float."""
        total = self._slice.margin()
        return self._opposite_axis_margin / total

    @lazyproperty
    def _pairwise_chisq(self):
        """Pairwise comparisons (Chi-Square) along axis, as numpy.ndarray.

        Returns a square, symmetric matrix of test statistics for the null
        hypothesis that each vector along *axis* is equal to each other.
        """
        return self._chi_squared(self._proportions, self._margin, self._observed)


class _MrXCatPairwiseSignificance(WishartPairwiseSignificance):
    """Pairwise significance for MR x CAT type slices."""

    _include_mr_cat = True

    @lazyproperty
    def pairwise_indices(self):
        return np.array(
            [
                [
                    self._pairwise_sig_indices(pvals_row < self._alpha)
                    for pvals_row in pvals_matrix
                ]
                for pvals_matrix in self.pvals
            ]
        )

    @lazyproperty
    def pvals(self):
        """Square matrix of pairwise Chi-square along axis, as numpy.ndarray."""
        return [
            self._pvals_from_chi_squared(mr_subvar_chisq)
            for mr_subvar_chisq in self._pairwise_chisq
        ]

    @lazyproperty
    def _pairwise_chisq(self):
        """Pairwise comparisons (Chi-Square) along axis, as numpy.ndarray.

        Returns a list of square and symmetric matrices of test statistics for the null
        hypothesis that each vector along *axis* is equal to each other.
        """
        return [
            self._chi_squared(
                mr_subvar_proportions,
                self._margin[idx],
                self._opposite_axis_margin[idx]
                / np.sum(self._opposite_axis_margin[idx]),
            )
            for (idx, mr_subvar_proportions) in enumerate(self._proportions)
        ]
