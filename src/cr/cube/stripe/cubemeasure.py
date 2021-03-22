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
    def cube_means(self):
        """_BaseCubeMeans subclass object for this stripe."""
        return _BaseCubeMeans.factory(self._cube, self._rows_dimension)

    @lazyproperty
    def cube_stddev(self):
        """_BaseCubeStdDev subclass object for this stripe."""
        return _BaseCubeStdDev.factory(self._cube, self._rows_dimension)

    @lazyproperty
    def cube_sum(self):
        """_BaseCubeSum subclass object for this stripe."""
        return _BaseCubeSums.factory(self._cube, self._rows_dimension)

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


# === MEANS ===


class _BaseCubeMeans(_BaseCubeMeasure):
    """Base class for means cube-measure variants."""

    def __init__(self, rows_dimension, means):
        super(_BaseCubeMeans, self).__init__(rows_dimension)
        self._means = means

    @classmethod
    def factory(cls, cube, rows_dimension):
        """Return _BaseCubeMeans subclass instance appropriate to `cube`."""
        if cube.means is None:
            raise ValueError("cube-result does not contain cube-means measure")
        MeansCls = (
            _MrCubeMeans if rows_dimension.dimension_type == DT.MR else _CatCubeMeans
        )
        return MeansCls(rows_dimension, cube.means)

    @lazyproperty
    def means(self):
        """1D np.float64 ndarray of mean for each stripe row."""
        raise NotImplementedError(
            "`%s` must implement `.means`" % type(self).__name__
        )  # pragma: no cover


class _CatCubeMeans(_BaseCubeMeans):
    """Means cube-measure for a non-MR stripe."""

    @lazyproperty
    def means(self):
        """1D np.float64 ndarray of mean for each stripe row."""
        return self._means


class _MrCubeMeans(_BaseCubeMeans):
    """Means cube-measure for an MR stripe.

    Its `.means` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def means(self):
        """1D np.float64 ndarray of mean for each stripe row."""
        return self._means[:, 0]


# === SUM ===


class _BaseCubeSums(_BaseCubeMeasure):
    """Base class for sum cube-measure variants."""

    def __init__(self, rows_dimension, sums):
        super(_BaseCubeSums, self).__init__(rows_dimension)
        self._sums = sums

    @classmethod
    def factory(cls, cube, rows_dimension):
        """Return _BaseCubeSum subclass instance appropriate to `cube`."""
        if cube.sums is None:
            raise ValueError("cube-result does not contain cube-sum measure")
        SumCls = _MrCubeSums if rows_dimension.dimension_type == DT.MR else _CatCubeSums
        return SumCls(rows_dimension, cube.sums)

    @lazyproperty
    def sums(self):
        """1D np.float64 ndarray of sum for each stripe row."""
        raise NotImplementedError(
            "`%s` must implement `.sum`" % type(self).__name__
        )  # pragma: no cover


class _CatCubeSums(_BaseCubeSums):
    """Sums cube-measure for a non-MR stripe."""

    @lazyproperty
    def sums(self):
        """1D np.float64 ndarray of sum for each stripe row."""
        return self._sums


class _MrCubeSums(_BaseCubeSums):
    """Sums cube-measure for an MR stripe.
    Its `.sums` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def sums(self):
        """1D np.float64 ndarray of sum for each stripe row."""
        return self._sums[:, 0]


# === STD DEV ===


class _BaseCubeStdDev(_BaseCubeMeasure):
    """Base class for stddev cube-measure variants."""

    def __init__(self, rows_dimension, stddev):
        super(_BaseCubeStdDev, self).__init__(rows_dimension)
        self._stddev = stddev

    @classmethod
    def factory(cls, cube, rows_dimension):
        """Return _BaseCubeStdDev subclass instance appropriate to `cube`."""
        if cube.stddev is None:
            raise ValueError("cube-result does not contain cube-stddev measure")
        StdDevCls = (
            _MrCubeStdDev if rows_dimension.dimension_type == DT.MR else _CatCubeStdDev
        )
        return StdDevCls(rows_dimension, cube.stddev)

    @lazyproperty
    def stddev(self):
        """1D np.float64 ndarray of stddev for each stripe row."""
        raise NotImplementedError(
            "`%s` must implement `.stddev`" % type(self).__name__
        )  # pragma: no cover


class _CatCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for a non-MR stripe."""

    @lazyproperty
    def stddev(self):
        """1D np.float64 ndarray of stddev for each stripe row."""
        return self._stddev


class _MrCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for an MR stripe.
    Its `.stddev` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def stddev(self):
        """1D np.float64 ndarray of stddev for each stripe row."""
        return self._stddev[:, 0]


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

        if rows_dimension.dimension_type == DT.NUM_ARRAY:
            return _NumArrUnweightedCubeCounts(rows_dimension, cube.unweighted_counts)

        if rows_dimension.dimension_type == DT.MR:
            return _MrUnweightedCubeCounts(rows_dimension, cube.unweighted_counts)

        return _CatUnweightedCubeCounts(rows_dimension, cube.unweighted_counts)

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of unweighted table-proportion denonimator per row."""
        raise NotImplementedError(
            "`%s` must implement `.bases`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(
            "`%s` must implement `.pruning_base`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def unweighted_counts(self):
        """1D np.float64 ndarray of unweighted-count for each row of stripe."""
        raise NotImplementedError(
            "`%s` must implement `.unweighted_counts`" % type(self).__name__
        )  # pragma: no cover


class _CatUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a non-MR stripe."""

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denonimator (base) for each row.

        Each row in a CAT stripe has the same base (the table-base).
        """
        return np.broadcast_to(self.table_base, self._unweighted_counts.shape)

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the unweighted count for
        each row.
        """
        return self._unweighted_counts

    @lazyproperty
    def table_base(self):
        """Scalar np.float64 unweighted-N for overall stripe.

        This is the unweighted count of respondents who provided a valid response to
        the question.
        """
        return np.sum(self._unweighted_counts)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.float64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts


class _MrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for an MR slice.

    Its `._unweighted_counts` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denonimator (base) for each row.

        Each row in an MR stripe has a distinct base. These values include both the
        selected and unselected counts.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.float64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts[:, 0]


class _NumArrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a numeric array stripe."""

    @lazyproperty
    def bases(self):
        """1D np.int64 ndarray of table-proportion denonimator for each cell."""
        return self._unweighted_counts

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row."""
        return self._unweighted_counts

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        return self._unweighted_counts


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
    def bases(self):
        """1D np.float64 ndarray of table-proportion denominator for each cell."""
        raise NotImplementedError(
            "`%s` must implement `.bases`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def table_margin(self):
        """Scalar or 1D np.float64 array of weighted-N for overall stripe."""
        raise NotImplementedError(
            "`%s` must implement `.table_margin`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def weighted_counts(self):
        """1D np.float64 ndarray of weighted-count for each row of stripe.

        When the cube-result has no weight these values are the same as the
        unweighted-counts.
        """
        raise NotImplementedError(
            "`%s` must implement `.weighted_counts`" % type(self).__name__
        )  # pragma: no cover


class _CatWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for a non-MR stripe.

    Its `._weighted_counts` is a 1D ndarray with axes (rows,).
    """

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denominator for each cell."""
        return np.broadcast_to(self.table_margin, self._weighted_counts.shape)

    @lazyproperty
    def table_margin(self):
        """Scalar np.float64 weighted-N for overall stripe.

        This is the weighted count of respondents who provided a valid response to
        the question.
        """
        return np.sum(self._weighted_counts)

    @lazyproperty
    def weighted_counts(self):
        """1D np.float64 ndarray of weighted-count for each row of stripe.

        When the cube-result has no weight these values are the same as the
        unweighted-counts.
        """
        return self._weighted_counts


class _MrWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for an MR slice.

    Its `._weighted_counts` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denominator for each cell."""
        # --- (weighted) bases for an MR slice is the already 1D table-margin ---
        return self.table_margin

    @lazyproperty
    def table_margin(self):
        """1D np.float64 weighted-N for each row of stripe.

        This is the weighted count of respondents who provided a valid response to the
        question. Both selecting and not-selecting the subvar/option are valid
        responses, so this value includes both the selected and unselected counts.
        """
        return np.sum(self._weighted_counts, axis=1)

    @lazyproperty
    def weighted_counts(self):
        """1D np.float64 ndarray of weighted-count for each row of stripe."""
        return self._weighted_counts[:, 0]
