# encoding: utf-8

"""Provides insertion values according to a variety of strategies.

A stripe can have inserted rows (subtotals so far) that summarize two or more other
vectors by ostensibly "adding" them. Simple addition works for counts, but more
sophisticated methods are required for higher-order measures.

This module provides the various strategies required for computing subtotals and is
primarily used by measure objects as a collaborator to handle this aspect.
"""

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
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


class WaveDiffSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by adding and subtracting terms for wave differences.

    This class handles a special case for wave differences when a CAT_DATE variable is
    involved in the calculation.

    A wave difference for a CAT_DATE variable is calculate subtracting at the
    percentages level: (count1/base1) - (count2/base2).
    """

    def __init__(self, base_values, counts, default_values, rows_dimension):
        super(WaveDiffSubtotals, self).__init__(base_values, rows_dimension)
        self._counts = counts
        self._default_values = default_values

    @classmethod
    def subtotal_values(cls, base_values, counts, default_values, rows_dimension):
        """Return (n_row_subtotals,) ndarray of subtotal values."""
        return cls(base_values, counts, default_values, rows_dimension)._subtotal_values

    def _multiple_subtrahends_or_addends(self, subtotal):
        """Returns true if the subtotal has multiple addend or subtrahend terms."""
        return any(subtotal.subtrahend_idxs) and (
            len(subtotal.subtrahend_idxs) > 1 or len(subtotal.addend_idxs) > 1
        )

    @lazyproperty
    def _subtotal_values(self):
        """(n_row_subtotals,) ndarray of subtotal values for stripe."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.array([])

        if self._rows_dimension.dimension_type != DT.CAT_DATE:
            return self._default_values

        return np.array(
            [
                self._subtotal_value(subtotal, default)
                for subtotal, default in zip(subtotals, self._default_values)
            ]
        )

    def _subtotal_value(self, subtotal, default):
        """Return scalar value of wafe diff `subtotal` row."""
        if len(subtotal.subtrahend_idxs) > 0 and len(subtotal.addend_idxs) > 0:
            if self._multiple_subtrahends_or_addends(subtotal):
                return np.nan
            base_values = self._base_values
            counts = self._counts
            base_addend_sum = np.sum(base_values[subtotal.addend_idxs])
            base_subtrahend_sum = np.sum(base_values[subtotal.subtrahend_idxs])
            counts_addend_sum = np.sum(counts[subtotal.addend_idxs])
            counts_subtrahend_sum = np.sum(counts[subtotal.subtrahend_idxs])
            return (counts_addend_sum / base_addend_sum) - (
                counts_subtrahend_sum / base_subtrahend_sum
            )
        return default
