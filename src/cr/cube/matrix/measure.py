# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

from cr.cube.util import lazyproperty


class SecondOrderMeasures(object):
    """Intended to be a singleton for a given cube-result.

    It will give the same values if duplicated, just sacrificing some time and memory
    performance. Provides access to the variety of possible second-order measure objects
    for its cube-result. All construction and computation are lazy so only actually
    requested measures consume resources.
    """

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this cube-result."""
        raise NotImplementedError


class _BaseSecondOrderMeasure(object):
    """Base class for all second-order measure objects."""


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        raise NotImplementedError
