# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

from __future__ import division

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
        raise NotImplementedError


class _BaseCubeMeasure(object):
    """Base class for all cube-measure objects."""


# === UNWEIGHTED COUNTS ===


class _BaseUnweightedCubeCounts(_BaseCubeMeasure):
    """Base class for unweighted-count cube-measure variants."""

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        raise NotImplementedError(
            "`%s` must implement `.unweighted_counts`" % type(self).__name__
        )
