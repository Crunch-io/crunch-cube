# encoding: utf-8

"""Provides insertion values according to a variety of strategies.

A stripe can have inserted rows (subtotals so far) that summarize two or more other
vectors by ostensibly "adding" them. Simple addition works for counts, but more
sophisticated methods are required for higher-order measures.

This module provides the various strategies required for computing subtotals and is
primarily used by measure objects as a collaborator to handle this aspect.
"""

from __future__ import division

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
        )


class SumSubtotals(_BaseSubtotals):
    """Subtotals created by np.sum() on addends, primarily counts."""
