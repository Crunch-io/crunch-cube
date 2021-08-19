# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class CubeMeasures:
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
        """_BaseCubeCounts for unweighted counts subclass object for this stripe."""
        valid_counts = self._cube.unweighted_valid_counts
        counts = (
            valid_counts if valid_counts is not None else self._cube.unweighted_counts
        )

        return _BaseCubeCounts.factory(
            counts, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )

    @lazyproperty
    def weighted_cube_counts(self):
        """_BaseCubeCounts for weighted subclass object for this stripe."""
        valid_counts = self._cube.weighted_valid_counts
        counts = valid_counts if valid_counts is not None else self._cube.counts

        return _BaseCubeCounts.factory(
            counts, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )


class _BaseCubeMeasure:
    """Base class for all cube-measure objects."""

    def __init__(self, rows_dimension):
        self._rows_dimension = rows_dimension


# === COUNTS (WEIGHTED & UNWEIGHTED) ===


class _BaseCubeCounts(_BaseCubeMeasure):
    """Base class for count cube-measure variants."""

    def __init__(self, rows_dimension, counts):
        super(_BaseCubeCounts, self).__init__(rows_dimension)
        self._counts = counts

    @classmethod
    def factory(cls, counts, rows_dimension, ca_as_0th, slice_idx):
        """Return _BaseCubeCounts subclass instance appropriate to `cube`."""
        if ca_as_0th:
            return _CatCubeCounts(rows_dimension, counts[slice_idx])

        # --- Cat arrays require 2 dimensions, so we only have to worry about num arrays
        if rows_dimension.dimension_type == DT.NUM_ARRAY:
            return _NumArrCubeCounts(rows_dimension, counts)

        if rows_dimension.dimension_type == DT.MR:
            return _MrCubeCounts(rows_dimension, counts)

        return _CatCubeCounts(rows_dimension, counts)

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of unweighted table-proportion denonimator per row."""
        raise NotImplementedError(
            f"`{type(self).__name__}` must implement `.bases`"
        )  # pragma: no cover

    @lazyproperty
    def counts(self):
        """1D np.float64 ndarray of count for each row of stripe."""
        raise NotImplementedError(
            f"`{type(self).__name__}` must implement `.unweighted_counts`"
        )  # pragma: no cover

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of N for each matrix row."""
        raise NotImplementedError(
            f"`{type(self).__name__}` must implement `.pruning_base`"
        )  # pragma: no cover

    @lazyproperty
    def table_base(self):
        """Optional scalar value of the base for the whole stripe.

        Only defined on CAT cubes because the array types do not have a single
        base for the whole stripe.
        """
        return None


class _CatCubeCounts(_BaseCubeCounts):
    """Unweighted-counts cube-measure for a non-MR stripe."""

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denonimator (base) for each row.

        Each row in a CAT stripe has the same base (the table-base).
        """
        return np.broadcast_to(self.table_base, self._counts.shape)

    @lazyproperty
    def counts(self):
        """1D np.float64 ndarray of unweighted-count for each row of stripe."""
        return self._counts

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of N for each matrix row.

        Because this matrix has no MR dimension, this is simply the count for each row.
        """
        return self._counts

    @lazyproperty
    def table_base(self):
        """Scalar np.float64 N for overall stripe.

        This is the count of respondents who provided a valid response to the question.
        """
        return np.sum(self._counts)


class _MrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for an MR slice.

    Its `._counts` is a 2D ndarray with axes (rows, sel/not).
    """

    @lazyproperty
    def bases(self):
        """1D np.float64 ndarray of table-proportion denonimator (base) for each row.

        Each row in an MR stripe has a distinct base. These values include both the
        selected and unselected counts.
        """
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def counts(self):
        """1D np.float64 ndarray of unweighted-count for each row of stripe."""
        return self._counts[:, 0]

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._counts, axis=1)


class _NumArrCubeCounts(_BaseCubeCounts):
    """Unweighted-counts cube-measure for a numeric array stripe."""

    @lazyproperty
    def bases(self):
        """1D np.int64 ndarray of table-proportion denonimator for each cell."""
        return self._counts

    @lazyproperty
    def counts(self):
        """1D np.int64 ndarray of unweighted-count for each row of stripe."""
        return self._counts

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row."""
        return self._counts


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
            f"`{type(self).__name__}` must implement `.means`"
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
            f"`{type(self).__name__}` must implement `.stddev`"
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
            f"`{type(self).__name__}` must implement `.sum`"
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
