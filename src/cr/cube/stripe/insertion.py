# encoding: utf-8

"""Provides insertion values according to a variety of strategies.

A stripe can have inserted rows (subtotals so far) that summarize two or more other
vectors by ostensibly "adding" them. Simple addition works for counts, but more
sophisticated methods are required for higher-order measures.

This module provides the various strategies required for computing subtotals and is
primarily used by measure objects as a collaborator to handle this aspect.
"""

import numpy as np

from cr.cube.util import lazyproperty


class _BaseSubtotals:
    """Base class for Subtotals objects."""

    def __init__(self, base_values, rows_dimension):
        self._base_values = base_values
        self._rows_dimension = rows_dimension

    @classmethod
    def subtotal_values(cls, base_values, rows_dimension):
        """Return (n_row_subtotals,) ndarray of subtotal values."""
        return cls(base_values, rows_dimension)._subtotal_values

    @lazyproperty
    def _subtotal_values(self):
        """(n_row_subtotals,) ndarray of subtotal values for stripe."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.array([])

        return np.array([self._subtotal_value(subtotal) for subtotal in subtotals])

    @lazyproperty
    def _subtotal_value(self):
        """Return scalar value of `subtotal` row."""
        raise NotImplementedError(
            f"`{type(self).__name__}` must implement `._subtotal_value`"
        )  # pragma: no cover

    @lazyproperty
    def _row_subtotals(self):
        """Sequence of _Subtotal object for each subtotal in rows-dimension."""
        return self._rows_dimension.subtotals


class NanSubtotals(_BaseSubtotals):
    """Subtotal blocks for measures that cannot meaningfully be subtotaled.

    Each subtotal value is `np.nan`.
    """

    @lazyproperty
    def _subtotal_values(self):
        """Return (n_row_subtotals,) ndarray of np.nan values."""
        return np.full(len(self._row_subtotals), np.nan)


class NegativeTermSubtotals(_BaseSubtotals):
    """Subtotal blocks that are only the negative terms from differences

    These values are the sum of the "negative" terms (0 for regular categories and
    regular subtotals, and sum of just the negative terms in a subtotal difference).
    """

    def _subtotal_value(self, subtotal):
        """Return scalar value of `subtotal` row."""
        return np.sum(self._base_values[subtotal.subtrahend_idxs])


class PositiveTermSubtotals(_BaseSubtotals):
    """Subtotal blocks that are only the positive terms from differences

    These values are the sum of the "negative" terms (0 for regular categories and
    regular subtotals, and sum of just the negative terms in a subtotal difference).
    """

    def _subtotal_value(self, subtotal):
        """Return scalar value of `subtotal` row."""
        return np.sum(self._base_values[subtotal.addend_idxs])


class SumSubtotals(_BaseSubtotals):
    """Subtotals created by np.sum() on addends, primarily bases.

    This sums together both addends AND subtrahends because some measures such as
    bases are additive even across subtrahends of subtotals.
    """

    def __init__(self, base_values, rows_dimension):
        super(SumSubtotals, self).__init__(base_values, rows_dimension)

    def _subtotal_value(self, subtotal):
        """Return scalar value of `subtotal` row."""
        base_values = self._base_values

        addend_sum = np.sum(base_values[subtotal.addend_idxs])
        subtrahend_sum = np.sum(base_values[subtotal.subtrahend_idxs])

        return addend_sum - subtrahend_sum
