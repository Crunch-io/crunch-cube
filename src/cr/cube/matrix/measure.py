# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

from cr.cube.matrix.cubemeasure import CubeMeasures
from cr.cube.matrix.subtotals import SumSubtotals
from cr.cube.util import lazyproperty


class SecondOrderMeasures(object):
    """Intended to be a singleton for a given cube-result.

    It will give the same values if duplicated, just sacrificing some time and memory
    performance. Provides access to the variety of possible second-order measure objects
    for its cube-result. All construction and computation are lazy so only actually
    requested measures consume resources.
    """

    def __init__(self, cube, dimensions, slice_idx):
        self._cube = cube
        self._dimensions = dimensions
        self._slice_idx = slice_idx

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this cube-result."""
        return _UnweightedCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def weighted_counts(self):
        """_WeightedCounts measure object for this cube-result."""
        return _WeightedCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def _cube_measures(self):
        """CubeMeasures collection object for this cube-result.

        This collection provides access to all cube-measure objects for the cube-result.
        The collection is provided to each measure object so it can access the cube
        measures it is based on.
        """
        return CubeMeasures(self._cube, self._dimensions, self._slice_idx)


class _BaseSecondOrderMeasure(object):
    """Base class for all second-order measure objects."""

    def __init__(self, dimensions, second_order_measures, cube_measures):
        self._dimensions = dimensions
        self._second_order_measures = second_order_measures
        self._cube_measures = cube_measures

    @lazyproperty
    def _unweighted_cube_counts(self):
        """_BaseUnweightedCubeCounts subclass instance for this measure.

        Provides cube measures associated with unweighted counts, including
        unweighted-counts and cell, vector, and table bases.
        """
        return self._cube_measures.unweighted_cube_counts


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        return SumSubtotals.blocks(
            self._unweighted_cube_counts.unweighted_counts, self._dimensions
        )


class _WeightedCounts(_BaseSecondOrderMeasure):
    """Provides the weighted-counts measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        raise NotImplementedError
