# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

from __future__ import division

import numpy as np

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

    @lazyproperty
    def weighted_cube_counts(self):
        """_BaseWeightedCubeCounts subclass object for this stripe."""
        return _BaseWeightedCubeCounts.factory(
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
    def bases(self):
        """1D np.int64 ndarray of unweighted table-proportion denonimator per row."""
        raise NotImplementedError("`%s` must implement `.bases`" % type(self).__name__)

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
    def bases(self):
        """1D np.int64 ndarray of table-proportion denonimator (base) for each row.

        Each row in a CAT stripe has the same base (the table-base).
        """
        return np.broadcast_to(self.table_base, self._unweighted_counts.shape)

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the unweighted count for
        each row.
        """
        return self._unweighted_counts

    @lazyproperty
    def table_base(self):
        """Scalar np.int64 unweighted-N for overall stripe.

        This is the unweighted count of respondents who provided a valid response to
        the question.
        """
        return np.sum(self._unweighted_counts)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts


class _MrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for an MR slice.

    Its `._unweighted_counts` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts[:, 0]


# === WEIGHTED COUNTS ===


class _BaseWeightedCubeCounts(_BaseCubeMeasure):
    """Base class for weighted-count cube-measure variants."""

    def __init__(self, rows_dimension, weighted_counts):
        super(_BaseWeightedCubeCounts, self).__init__(rows_dimension)
        self._weighted_counts = weighted_counts

    @classmethod
    def factory(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return _BaseWeightedCubeCounts subclass instance appropriate to `cube`."""
        if ca_as_0th:
            return _CatWeightedCubeCounts(rows_dimension, cube.counts[slice_idx])

        if rows_dimension.dimension_type == DT.MR:
            return _MrWeightedCubeCounts(rows_dimension, cube.counts)

        return _CatWeightedCubeCounts(rows_dimension, cube.counts)

    @lazyproperty
    def weighted_counts(self):
        """1D np.float/int64 ndarray of weighted-count for each row of stripe.

        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        raise NotImplementedError(
            "`%s` must implement `.weighted_counts`" % type(self).__name__
        )


class _CatWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for a non-MR stripe.

    Its `._weighted_counts` is a 1D ndarray with axes (rows,).
    """

    @lazyproperty
    def weighted_counts(self):
        """1D np.float/int64 ndarray of weighted-count for each row of stripe.

        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        return self._weighted_counts


class _MrWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for an MR slice.

    Its `._weighted_counts` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def weighted_counts(self):
        """1D np.float/int64 ndarray of weighted-count for each row of stripe."""
        return self._weighted_counts[:, 0]
