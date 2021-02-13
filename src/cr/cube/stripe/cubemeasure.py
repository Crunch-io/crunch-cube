# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

from __future__ import division

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class CubeMeasures(object):
    """Provides access to all cube-measure objects for this stripe."""

    def __init__(self, cube, rows_dimension, ca_as_0th, slice_idx):
        self._cube = cube
        self._rows_dimension = rows_dimension
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx

    @lazyproperty
    def unweighted_cube_counts(self):
        """_BaseUnweightedCubeCounts subclass object for this stripe."""
        return _BaseUnweightedCubeCounts.factory(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )


class _BaseCubeMeasure(object):
    """Base class for all cube-measure objects."""

    def __init__(self, rows_dimension):
        self._rows_dimension = rows_dimension


# === UNWEIGHTED COUNTS ===


class _BaseUnweightedCubeCounts(_BaseCubeMeasure):
    """Base class for unweighted-count cube-measure variants."""

    def __init__(self, rows_dimension, unweighted_counts):
        super(_BaseUnweightedCubeCounts, self).__init__(rows_dimension)
        self._unweighted_counts = unweighted_counts

    @classmethod
    def factory(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return _BaseUnweightedCubeCounts subclass instance appropriate to `cube`."""
        if ca_as_0th:
            return _CatUnweightedCubeCounts(
                rows_dimension, cube.unweighted_counts[slice_idx]
            )

        if rows_dimension.dimension_type == DT.MR:
            return _MrUnweightedCubeCounts(rows_dimension, cube.unweighted_counts)

        return _CatUnweightedCubeCounts(rows_dimension, cube.unweighted_counts)

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(
            "`%s` must implement `.pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        raise NotImplementedError(
            "`%s` must implement `.unweighted_counts`" % type(self).__name__
        )


class _CatUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a non-MR stripe."""

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts


class _MrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for an MR slice.

    Its `._unweighted_counts` is a 2D ndarray with axes (rows, sel/not).
    """
