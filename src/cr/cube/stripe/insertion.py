# encoding: utf-8

"""Provides insertion values according to a variety of strategies.

A stripe can have inserted rows (subtotals so far) that summarize two or more other
vectors by ostensibly "adding" them. Simple addition works for counts, but more
sophisticated methods are required for higher-order measures.

This module provides the various strategies required for computing subtotals and is
primarily used by measure objects as a collaborator to handle this aspect.
"""

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty


class _BaseSubtotals(object):
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
        raise NotImplementedError(
            "`%s` must implement `._subtotal_values`" % type(self).__name__
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


class SumSubtotals(_BaseSubtotals):
    """Subtotals created by np.sum() on addends, primarily bases.

    This sums together both addends AND subtrahends because some measures such as
    bases are additive even across subtrahends of subtotals.
    """

    @lazyproperty
    def _subtotal_values(self):
        """(n_row_subtotals,) ndarray of subtotal values for stripe."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.array([])

        return np.array([self._subtotal_value(subtotal) for subtotal in subtotals])

    def _subtotal_value(self, subtotal):
        """Return scalar value of `subtotal` row."""
        base_values = self._base_values

        addend_sum = np.sum(base_values[subtotal.addend_idxs])
        subtrahend_sum = np.sum(base_values[subtotal.subtrahend_idxs])

        return addend_sum - subtrahend_sum
