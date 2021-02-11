# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

from cr.cube.util import lazyproperty


# === MEASURE COLLECTION ===


class StripeMeasures(object):
    """Intended to be a singleton for a given cube-result.

    It will give the same values if duplicated, just sacrificing some time and memory
    performance. Provides access to the variety of possible second-order measure objects
    for its cube-result. All construction and computation are lazy so only actually
    requested measures consume resources.
    """

    def __init__(self, cube, rows_dimension, ca_as_0th, slice_idx):
        self._cube = cube
        self._rows_dimension = rows_dimension
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this stripe."""
        return _UnweightedCounts(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def _cube_measures(self):
        """CubeMeasures collection object for this cube-result.

        This collection provides access to all cube-measure objects for the cube-result.
        The collection is provided to each measure object so it can access the cube
        measures it is based on.
        """
        raise NotImplementedError


# === INDIVIDUAL MEASURES ===


class _BaseSecondOrderMeasure(object):
    """Base class for all second-order measure objects."""

    def __init__(self, rows_dimension, measures, cube_measures):
        self._rows_dimension = rows_dimension
        self._measures = measures
        self._cube_measures = cube_measures

    @lazyproperty
    def blocks(self):
        """(base_values, subtotal_values) pair comprising the "blocks" of this measure.

        Use of this default implementation assumes implementation of a `._base_values`
        and `._subtotal_values` property in the subclass. A measure which is computed
        differently can override this `.blocks` property instead of implementing those
        two components.
        """
        raise NotImplementedError


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a stripe."""
