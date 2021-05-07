# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

from __future__ import division

import numpy as np
from scipy.stats.contingency import expected_freq

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class CubeMeasures(object):
    """Provides access to all cube-measure objects for this cube-result."""

    def __init__(self, cube, dimensions, slice_idx):
        self._cube = cube
        self._dimensions = dimensions
        self._slice_idx = slice_idx

    @lazyproperty
    def cube_means(self):
        """_BaseCubeMeans subclass object for this cube-result."""
        return _BaseCubeMeans.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_overlaps(self):
        """_BaseCubeOverlaps subclass object for this cube-result."""
        return _BaseCubeOverlaps.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_sum(self):
        """_BaseCubeSums subclass object for this cube-result."""
        return _BaseCubeSums.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_stddev(self):
        """_BaseCubeStdDev subclass object for this cube-result."""
        return _BaseCubeStdDev.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def unweighted_cube_counts(self):
        """_BaseUnweightedCubeCounts subclass object for this cube-result."""
        return _BaseUnweightedCubeCounts.factory(
            self._cube, self._dimensions, self._slice_idx
        )

    @lazyproperty
    def weighted_cube_counts(self):
        """_BaseWeightedCounts subclass object for this cube-result."""
        return _BaseWeightedCubeCounts.factory(
            self._cube, self._dimensions, self._slice_idx
        )


class _BaseCubeMeasure(object):
    """Base class for all cube-measure objects."""

    def __init__(self, dimensions):
        self._dimensions = dimensions

    @classmethod
    def _slice_idx_expr(cls, cube, slice_idx):
        """Return np.s_ advanced-indexing slice object to extract values for slice_idx.

        `cube` can contain data for multiple `_Slice` objects. The returned `numpy`
        advanced-indexing expression selects only those values of `cube` that pertain
        to the slice indicated by `slice_idx`.
        """
        # --- for a 2D cube we take the whole thing (1D is not expected here) ---
        if cube.ndim < 3:
            return np.s_[:]

        # --- if 0th dimension of a >2D cube is MR, we only take the "Selected" portion
        # --- of the indicated initial-MR subvar, because the slice is to represent the
        # --- values for "respondents who selected" that MR response (and not those who
        # --- didn't select it or did not respond).
        if cube.dimension_types[0] == DT.MR:
            return np.s_[slice_idx, 0]

        # --- for other 3D cubes we just select the 2D "table" portion associated with
        # --- the `slice_idx`-th table dimension element.
        return np.s_[slice_idx]


# === MEANS ===


class _BaseCubeMeans(_BaseCubeMeasure):
    """Base class for mean cube-measure variants."""

    def __init__(self, dimensions, means):
        super(_BaseCubeMeans, self).__init__(dimensions)
        self._means = means

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeMeans subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-means measure.
        """
        if cube.means is None:
            raise ValueError("cube-result does not contain cube-means measure")
        dimension_types = cube.dimension_types[-2:]
        CubeMeansCls = (
            _MrXMrCubeMeans
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeMeans
            if dimension_types[0] == DT.MR
            else _CatXMrCubeMeans
            if dimension_types[1] == DT.MR
            else _CatXCatCubeMeans
        )
        return CubeMeansCls(
            dimensions, cube.means[cls._slice_idx_expr(cube, slice_idx)]
        )

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of cube means."""
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.means`" % type(self).__name__
        )


class _CatXCatCubeMeans(_BaseCubeMeans):
    """Means cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means


class _CatXMrCubeMeans(_BaseCubeMeans):
    """Means cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means[:, :, 0]


class _MrXCatCubeMeans(_BaseCubeMeans):
    """Means cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means[:, 0, :]


class _MrXMrCubeMeans(_BaseCubeMeans):
    """Means cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._means[:, 0, :, 0]


# === OVERLAPS ===


class _BaseCubeOverlaps(_BaseCubeMeasure):
    """Base class for overlap cube-measure variants."""

    def __init__(self, dimensions, overlaps, valid_overlaps):
        super(_BaseCubeOverlaps, self).__init__(dimensions)
        self._overlaps = overlaps
        self._valid_overlaps = valid_overlaps

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeOverlaps subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-overlaps measure
        or if it doesn't include valid-cube-overlaps measure.
        """
        if cube.overlaps is None:
            raise ValueError(
                "cube-result does not contain cube-overlaps measure"
            )  # pragma: no cover
        if cube.valid_overlaps is None:
            raise ValueError(
                "cube-result does not contain cube-valid-overlaps measure"
            )  # pragma: no cover

        dimension_types = tuple(d.dimension_type for d in dimensions)
        idx_expr = cls._slice_idx_expr(cube, slice_idx)
        args = (dimensions, cube.overlaps[idx_expr], cube.valid_overlaps[idx_expr])
        return (
            _MrXMrOverlaps(*args)
            if dimension_types == (DT.MR, DT.MR)
            else _CatXMrOverlaps(*args)
        )

    @lazyproperty
    def overlaps(self):
        """3D np.float64 ndarray of cube overlaps."""
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.overlaps`" % type(self).__name__
        )


class _CatXMrOverlaps(_BaseCubeOverlaps):
    """Overlaps cube-measure for a NOT_MR_X_MR slice."""

    @lazyproperty
    def tile_repetitions(self):
        """tuple of number of repetitions of selected base matrix."""
        return (self._overlaps.shape[0], 1, 1)

    @lazyproperty
    def selected_bases(self):
        """3D np.float64 ndarray of selected overlaps between MR subvariables, per cat.

        For a CAT x MR matrix, the overlaps are calculated for each category, and then
        for each subvariables pair for that category (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is CAT x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the result
        of the `cube_overlap` measure, and representing the "pairing" with each subvar
        of the previous MR_SUBVAR dimension.

        From this shape, we only need the "Selected" part of the MR_SEL dimension, so
        we need to select the 0th element along the 2nd axis [:, :, 0]. But we also need
        to add all of the categories together, since that's what's used for the base
        value in all of the significance calculations. We then tile these bases, since
        they're the same for each row (all the categories added together). The tiling is
        done so that the API of this class can be uniform for all users.
        """
        return np.tile(np.sum(self._overlaps[:, :, 0], axis=0), self.tile_repetitions)

    @lazyproperty
    def valid_bases(self):
        """3D np.float64 ndarray of valid overlaps between MR subvariables, per cat.

        For a CAT x MR matrix, the overlaps are calculated for each category, and then
        for each subvariables pair for that category (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is CAT x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the result
        of the `cube_overlap` measure, and representing the "pairing" with each subvar
        of the previous MR_SUBVAR dimension.

        From this shape, we only need the "Selected" + "Other" part of the MR_SEL
        dimension (i.e. all except missing), so we need to add the 0th and the 1st
        element along the 2nd axis sum([:, :, 0:2]).
        """
        valids = np.sum(self._valid_overlaps[:, :, 0:2], axis=2)
        return np.tile(np.sum(valids, axis=0), self.tile_repetitions)


class _MrXMrOverlaps(_BaseCubeOverlaps):
    """Overlaps cube-measure for a MR_X_MR slice."""

    @lazyproperty
    def selected_bases(self):
        """3D np.float64 ndarray of selected overlaps bases between MR subvariables.

        For a MR x MR slice, the overlaps are calculated for each MR subvar row, and
        then for each subvariables pair for that row (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is MR_SUBVAR x MR_SEL x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one
        being the result of the `cube_overlap` measure, and representing the "pairing"
        with each subvar of the last MR_SUBVAR dimension.

        From this shape, we only need the "Selected" part of the last MR_SEL dimension,
        but we need both "Selected" and "Other" from the first MR_SEL dimension. We
        therefore need to select the 0th element along the 3rd axis, but we need to sum
        the 0th and 1st elements along the 1st axis (first MR_SEL dimension). The sum
        of the selected and other counts of the first MR_SEL is what's always used to
        represent bases of such crosstabs (we cannot simply ignore the other counts).
        """
        return np.sum(self._overlaps[:, 0:2, :, 0], axis=1)

    @lazyproperty
    def valid_bases(self):
        """3D np.float64 ndarray of valid overlaps between MR subvariables, per row.

        For a MR x MR slice, the overlaps are calculated for each row, and then
        for each subvariables pair for that row (which will produce a square matrix for
        each row). So the output shape that we get back from the database is
        MR_SUBVAR x MR_SEL x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the
        result of the `cube_overlap` measure, and representing the "pairing" with each
        subvar of the last MR_SUBVAR dimension.

        From this shape, we only need the "Selected" + "Other" part of both MR_SEL
        dimensions (i.e. all except missing), so we need to add the 0th and the 1st
        element along the 1st and 3rd axes sum([:, 0:2, :, 0:2], axis=(1, 3)).
        """
        return np.sum(self._valid_overlaps[:, 0:2, :, 0:2], axis=(1, 3))


# === STD DEV ===


class _BaseCubeStdDev(_BaseCubeMeasure):
    """Base class for stddev cube-measure variants."""

    def __init__(self, dimensions, stddev):
        super(_BaseCubeStdDev, self).__init__(dimensions)
        self._stddev = stddev

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeStdDev subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-stddev measure.
        """
        if cube.stddev is None:
            raise ValueError("cube-result does not contain cube-stddev measure")
        dimension_types = cube.dimension_types[-2:]
        CubeSumsCls = (
            _MrXMrCubeStdDev
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeStdDev
            if dimension_types[0] == DT.MR
            else _CatXMrCubeStdDev
            if dimension_types[1] == DT.MR
            else _CatXCatCubeStdDev
        )
        return CubeSumsCls(
            dimensions, cube.stddev[cls._slice_idx_expr(cube, slice_idx)]
        )

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of cube stddev."""
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.stddev`" % type(self).__name__
        )


class _CatXCatCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev


class _CatXMrCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev[:, :, 0]


class _MrXCatCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev[:, 0, :]


class _MrXMrCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._stddev[:, 0, :, 0]


# === SUMS ===


class _BaseCubeSums(_BaseCubeMeasure):
    """Base class for sum cube-measure variants."""

    def __init__(self, dimensions, sums):
        super(_BaseCubeSums, self).__init__(dimensions)
        self._sums = sums

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeSums subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-sum measure.
        """
        if cube.sums is None:
            raise ValueError("cube-result does not contain cube-sum measure")
        dimension_types = cube.dimension_types[-2:]
        CubeSumsCls = (
            _MrXMrCubeSums
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeSums
            if dimension_types[0] == DT.MR
            else _CatXMrCubeSums
            if dimension_types[1] == DT.MR
            else _CatXCatCubeSums
        )
        return CubeSumsCls(dimensions, cube.sums[cls._slice_idx_expr(cube, slice_idx)])

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of cube sum."""
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.sum`" % type(self).__name__
        )


class _CatXCatCubeSums(_BaseCubeSums):
    """Sums cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums


class _CatXMrCubeSums(_BaseCubeSums):
    """Sums cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums[:, :, 0]


class _MrXCatCubeSums(_BaseCubeSums):
    """Sums cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums[:, 0, :]


class _MrXMrCubeSums(_BaseCubeSums):
    """Sums cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._sums[:, 0, :, 0]


# === UNWEIGHTED COUNTS ===


class _BaseUnweightedCubeCounts(_BaseCubeMeasure):
    """Base class for unweighted-count cube-measure variants."""

    def __init__(self, dimensions, unweighted_counts, diff_nans):
        super(_BaseUnweightedCubeCounts, self).__init__(dimensions)
        self._unweighted_counts = unweighted_counts
        self._diff_nans = diff_nans

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseUnweightedCubeCounts subclass instance appropriate to `cube`."""
        dimension_types = cube.dimension_types[-2:]
        valid_counts = cube.unweighted_valid_counts
        counts = valid_counts if valid_counts is not None else cube.unweighted_counts
        diff_nans = True if valid_counts is not None else False
        UnweightedCubeCountsCls = (
            _NumArrayXMrUnweightedCubeCounts
            if dimension_types == (DT.NUM_ARRAY, DT.MR)
            else _NumArrayXCatUnweightedCubeCounts
            if dimension_types[0] == DT.NUM_ARRAY
            else _MrXMrUnweightedCubeCounts
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatUnweightedCubeCounts
            if dimension_types[0] == DT.MR
            else _CatXMrUnweightedCubeCounts
            if dimension_types[1] == DT.MR
            else _CatXCatUnweightedCubeCounts
        )

        return UnweightedCubeCountsCls(
            dimensions, counts[cls._slice_idx_expr(cube, slice_idx)], diff_nans
        )

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise unweighted-N for each matrix cell."""
        return np.broadcast_to(self.columns_base, self.unweighted_counts.shape)

    @lazyproperty
    def columns_base(self):
        """1D or 2D np.float64 ndarray of unweighted column-proportion denominator."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.columns_base`" % type(self).__name__
        )

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.columns_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def diff_nans(self):
        """Boolean, indicated if subtotal differences (rows and cols) has to be NaN."""
        return self._diff_nans

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.row_bases`" % type(self).__name__
        )

    @lazyproperty
    def rows_base(self):
        """1D or 2D np.float64 ndarray of unweighted row-proportion denominator."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.rows_base`" % type(self).__name__
        )

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.rows_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def table_base(self):
        """Scalar, 1D, or 2D np.float64 ndarray of unweighted table proportion denom."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.table_base`" % type(self).__name__
        )

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of unweighted table-proportion denominator per cell."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.table_bases`" % type(self).__name__
        )

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.unweighted_counts`" % type(self).__name__
        )


class _CatXCatUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def columns_base(self):
        """1D ndarray of np.float64 unweighted-N for each matrix column."""
        return np.sum(self.unweighted_counts, axis=0)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each column.
        """
        return np.sum(self._unweighted_counts, axis=0)

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        return np.broadcast_to(self.rows_base[:, None], self._unweighted_counts.shape)

    @lazyproperty
    def rows_base(self):
        """1D ndarray of np.float64 unweighted-N for each matrix row."""
        return np.sum(self.unweighted_counts, axis=1)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each row.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def table_base(self):
        """np.float64 count of actual respondents who answered both questions.

        Each dimension of a CAT_X_CAT matrix represents a categorical question. Only
        responses that include answers to both those questions appear as entries in the
        valid elements of those dimensions. The sum total of all valid answers is the
        sample size, aka "N". The term "base" derives from its use as the denominator
        for the table-proportions measure.
        """
        return np.sum(self.unweighted_counts)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each matrix cell."""
        return np.broadcast_to(self.table_base, self._unweighted_counts.shape)

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts


class _CatXMrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimension need not actually be CAT, as long as it's not MR.
    Its `._unweighted_counts` is a 3D ndarray with axes (rows, cols, selected/not).
    """

    @lazyproperty
    def columns_base(self):
        """2D ndarray of np.float64 unweighted-N for each matrix column."""
        return np.sum(self.unweighted_counts, axis=0)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        # --- in the CAT_X_MR case, rows_base is already the right (2D) value ---
        return self.rows_base

    @lazyproperty
    def rows_base(self):
        """2D np.float64 ndarray of row-wise unweighted-N for this matrix.

        An X_MR matrix has a distinct row-base for each cell. This is because not all
        responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each X_MR cell is the sum of its selected and unselected
        unweighted counts.
        """
        # --- sel/not axis (2) is summed, rows and columns are preserved ---
        return np.sum(self._unweighted_counts, axis=2)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """1D np.float64 unweighted-N for each column of table.

        Because the matrix is X_MR, each column (MR-subvar) has a distinct base.
        """
        # --- unweighted-counts is (nrows, ncols, selected/not) so axis 1 is preserved
        # --- to provide a distinct value for each MR subvar. Both selected and
        # --- not-selected counts contribute to base.
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of unweighted table-proportion denominator per cell."""
        return np.broadcast_to(self.table_base, self.unweighted_counts.shape)

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell."""
        return self._unweighted_counts[:, :, 0]


class _MrXCatUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT, as long as it's not MR.
    Its `._unweighted_counts` is a 3D ndarray with axes (rows, sel/not, cols).
    """

    @lazyproperty
    def columns_base(self):
        """2D np.float64 ndarray of unweighted-N for this matrix.

        An MR_X matrix has a distinct column-base for each cell. This is because not all
        responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each MR_X cell is the sum of its selected and unselected
        unweighted counts.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 1))

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        return np.broadcast_to(self.rows_base[:, None], self.unweighted_counts.shape)

    @lazyproperty
    def rows_base(self):
        """1D ndarray of np.float64 unweighted-N for each matrix row."""
        # --- only row-selecteds contribute ([:, 0, :]), sum is across columns (axis=1
        # --- after rows sel/not axis is collapsed), rows are retained.
        return np.sum(self._unweighted_counts[:, 0, :], axis=1)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """1D np.float64 ndarray (column) of unweighted-N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct base, since not all of
        the multiple responses were necessarily offered to all respondents. The base for
        each row indicates the number of respondents who were offered that option.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each matrix cell."""
        return np.broadcast_to(self.table_base[:, None], self.unweighted_counts.shape)

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell."""
        return self._unweighted_counts[:, 0, :]


class _MrXMrUnweightedCubeCounts(_BaseUnweightedCubeCounts):
    """Unweighted-counts cube-measure for an MR_X_MR slice.

    Its `._unweighted_counts` is a 4D ndarray with axes (rows, sel/not, cols, sel/not).
    """

    @lazyproperty
    def columns_base(self):
        """2D np.float64 ndarray of unweighted-N for this matrix.

        An MR_X_MR matrix has a distinct column-base for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each MR_X cell is the sum of the selected column counts for
        both the selected and unselected row values.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        This includes both selected and unselected counts for the row MR but only
        column-selecteds contribute.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=(0, 1))

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        # --- in the MR_X_MR case, rows-base is already the 2D row-unweighted-bases ---
        return self.rows_base

    @lazyproperty
    def rows_base(self):
        """2D np.float64 ndarray of unweighted-N for this matrix.

        An MR_X matrix has a distinct row-base for each cell, the sum of sel-sel and
        sel-not for each cell
        """
        # --- only selecteds in rows contribute ([:, 0, :, :]), selected and not from
        # --- columns both contribute (axis=2 after rows sel/not axis is collapsed),
        # --- both rows and columns are retained, producing a 2D result.
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=2)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        This includes both selected and unselected counts for the column MR, but only
        selecteds are considered for the rows dimension.
        """
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """2D np.float64 ndarray of distinct unweighted N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-base.
        """
        # --- unweighted_counts is 4D of shape (nrows, 2, ncols, 2):
        # --- (MR_SUBVAR (nrows), MR_CAT (sel/not), MR_SUBVAR (ncols), MR_CAT (sel/not))
        # --- Reduce the second and fourth axes with sum() producing 2D (nrows, ncols).
        # --- This sums (selected, selected), (selected, not), (not, selected) and
        # --- (not, not) cells of the subtable for each matrix cell.
        return np.sum(self._unweighted_counts, axis=(1, 3))

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each matrix cell."""
        # --- in the MR_X_MR case, table-base is already the 2D table-bases.
        return self.table_base

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._unweighted_counts[:, 0, :, 0]


class _NumArrayXCatUnweightedCubeCounts(_CatXCatUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a slice with no NUM ARRAY row dimension."""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise unweighted-N for each matrix cell."""
        return np.broadcast_to(
            np.sum(self.columns_base, axis=0), self.unweighted_counts.shape
        )

    @lazyproperty
    def columns_base(self):
        """2D ndarray of np.float64 unweighted-N for each matrix column."""
        return self._unweighted_counts


class _NumArrayXMrUnweightedCubeCounts(_CatXMrUnweightedCubeCounts):
    """Unweighted-counts cube-measure for a slice with NUM_ARRAY x MR dimensions."""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise unweighted-N for each matrix cell."""
        return np.broadcast_to(
            np.sum(self.columns_base, axis=0), self.unweighted_counts.shape
        )

    @lazyproperty
    def columns_base(self):
        """2D ndarray of np.float64 unweighted-N for each matrix column."""
        return self._unweighted_counts[:, :, 0]


# === WEIGHTED COUNTS ===


class _BaseWeightedCubeCounts(_BaseCubeMeasure):
    """Base class for weighted-count cube-measure variants."""

    def __init__(self, dimensions, weighted_counts, diff_nans):
        super(_BaseWeightedCubeCounts, self).__init__(dimensions)
        self._weighted_counts = weighted_counts
        self._diff_nans = diff_nans

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseWeightedCounts subclass instance appropriate to `cube`."""
        dimension_types = cube.dimension_types[-2:]
        valid_counts = cube.weighted_valid_counts
        counts = valid_counts if valid_counts is not None else cube.counts
        diff_nans = True if valid_counts is not None else False
        WeightedCubeCountsCls = (
            _MrXMrWeightedCubeCounts
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatWeightedCubeCounts
            if dimension_types[0] == DT.MR
            else _CatXMrWeightedCubeCounts
            if dimension_types[1] == DT.MR
            else _CatXCatWeightedCubeCounts
        )
        return WeightedCubeCountsCls(
            dimensions, counts[cls._slice_idx_expr(cube, slice_idx)], diff_nans
        )

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-proportion denominator for each cell."""
        return np.broadcast_to(self.columns_margin, self.weighted_counts.shape)

    @lazyproperty
    def columns_margin(self):
        """1D ndarray of np.float64 weighted-N for each matrix column."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.columns_margin`" % type(self).__name__
        )

    @lazyproperty
    def diff_nans(self):
        """Boolean, indicated if subtotal differences (rows and cols) has to be NaN."""
        return self._diff_nans

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of (weighted) row-proportion denominator per cell."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.row_bases`" % type(self).__name__
        )

    @lazyproperty
    def rows_margin(self):
        """1D ndarray of np.float64 weighted-N for each matrix row."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.rows_margin`" % type(self).__name__
        )

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of weighted table-proportion denominator per cell."""
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.table_bases`" % type(self).__name__
        )

    @lazyproperty
    def table_margin(self):
        """Scalar, 1D, or 2D np.float64 ndarray of weighted-N for matrix.

        There are four cases, one for each of CAT_X_CAT, CAT_X_MR, MR_X_CAT, and
        MR_X_MR. Both CAT_X_MR and MR_X_CAT produce a 1D array, but the former is a
        "row" and the latter is a "column".
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.table_margin`" % type(self).__name__
        )

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.weighted_counts`" % type(self).__name__
        )

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of zscores for each valid matrix cell."""
        raise NotImplementedError(  # pragma: no cover
            "`%s` must implement `.zscores`" % type(self).__name__
        )

    def _array_type_std_res(self, counts, total, rowsum, colsum):
        """Return 2D np.float64 ndarray of std-res value for each cell of MR matrix.

        This is a utility method used by a matrix with one or more MR dimensions. The
        caller forms the input arrays based on which of its dimensions are MR.
        """
        # --- if the matrix is "defective", in the sense that it doesn't have at least
        # --- two rows and two columns that are "full" of data, don't calculate zscores.
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            return np.full(counts.shape, np.nan)

        expected_counts = rowsum * colsum / total
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)


class _CatXCatWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def columns_margin(self):
        """1D ndarray of np.float64 weighted N for each matrix column."""
        return np.sum(self._weighted_counts, axis=0)

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each matrix cell."""
        return np.broadcast_to(self.rows_margin[:, None], self._weighted_counts.shape)

    @lazyproperty
    def rows_margin(self):
        """1D np.float64 ndarray of weighted-N for each matrix row."""
        return np.sum(self._weighted_counts, axis=1)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        return np.broadcast_to(self.table_margin, self._weighted_counts.shape)

    @lazyproperty
    def table_margin(self):
        """Scalar np.float64 weighted-N for overall table.

        This is the weighted count of respondents who provided a valid response to
        both questions. Because both dimensions are CAT, the table-margin value is the
        same for all cells of the matrix.
        """
        return np.sum(self._weighted_counts)

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        return self._weighted_counts

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        counts = self.weighted_counts

        # --- If the matrix is "defective", in the sense that it doesn't have at least
        # --- two rows and two columns that are "full" of data, don't calculate zscores.
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            return np.full(counts.shape, np.nan)

        residuals = counts - expected_freq(counts)

        # --- variance of the residuals ---
        rows_margin = self.rows_margin
        columns_margin = self.columns_margin
        table_margin = self.table_margin
        variance_of_residuals = (
            np.outer(rows_margin, columns_margin)
            * np.outer(table_margin - rows_margin, table_margin - columns_margin)
            / table_margin ** 3
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            return residuals / np.sqrt(variance_of_residuals)


class _CatXMrWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimension need not actually be CAT, as long as it's not MR.
    Its `._weighted_counts` is a 3D ndarray with axes (rows, cols, selected/not).
    """

    @lazyproperty
    def columns_margin(self):
        """1D ndarray of np.float64 weighted N for each matrix column."""
        # --- only selected counts contribute to the columns margin, which is summed
        # --- across the rows (axis 0).
        return np.sum(self._weighted_counts[:, :, 0], axis=0)

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each matrix cell."""
        # --- in the X_MR case, row-weighted-bases is the already-2D rows-margin ---
        return self.rows_margin

    @lazyproperty
    def rows_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of this matrix.

        A matrix with an MR columns dimension has a distinct rows-margin for each cell.
        This is because not all column responses (subvars) are necessarily offered to
        each respondent. The weighted-count for each X_MR cell is the sum of its
        selected and unselected weighted counts.
        """
        # --- selected and not-selected both contribute to margin (axis=2), both rows
        # --- and columns are retained.
        return np.sum(self._weighted_counts, axis=2)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        # --- table-margin is a row-vector (distinct margin for each column), so
        # --- broadcast is vertical
        return np.broadcast_to(self.table_margin, self.weighted_counts.shape)

    @lazyproperty
    def table_margin(self):
        """1D np.float64 ndarray of weighted-N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table margin.
        """
        # --- weighted-counts is (rows, cols, selected/not) so axis 1 is preserved to
        # --- provide a distinct value for each MR subvar.
        return np.sum(self._weighted_counts, axis=(0, 2))

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        return self._weighted_counts[:, :, 0]

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of z-score for each matrix cell."""
        return self._array_type_std_res(
            self.weighted_counts,
            self.table_margin,
            np.sum(self._weighted_counts, axis=2),
            np.sum(self.weighted_counts, axis=0),
        )


class _MrXCatWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for an MR_X_NOT_MR slice.

    Its `._weighted_counts` is a 3D ndarray with axes (rows, sel/not, cols).
    """

    @lazyproperty
    def columns_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of this matrix.

        An MR_X matrix has a distinct column-margin for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. The
        weighted-count for each MR_X cell is the sum of its selected and unselected
        weighted counts.
        """
        # --- sum sel and not-sel (axis 1), rows and columns are retained ---
        return np.sum(self._weighted_counts, axis=1)

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each matrix cell."""
        return np.broadcast_to(self.rows_margin[:, None], self.weighted_counts.shape)

    @lazyproperty
    def rows_margin(self):
        """1D np.float64 ndarray of weighted-N for each matrix row."""
        return np.sum(self.weighted_counts, axis=1)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        return np.broadcast_to(self.table_margin[:, None], self.weighted_counts.shape)

    @lazyproperty
    def table_margin(self):
        """1D np.float64 ndarray of weighted-N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct table margin, since not
        all of the multiple responses were necessarily offered to all respondents. The
        table-margin for each row indicates the weighted number of respondents who were
        offered that option.
        """
        return np.sum(self._weighted_counts, axis=(1, 2))

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        return self._weighted_counts[:, 0, :]

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of z-score for each matrix cell."""
        return self._array_type_std_res(
            self.weighted_counts,
            self.table_margin[:, np.newaxis],
            np.sum(self.weighted_counts, axis=1)[:, None],
            np.sum(self._weighted_counts, axis=1),
        )


class _MrXMrWeightedCubeCounts(_BaseWeightedCubeCounts):
    """Weighted-counts cube-measure for an MR_X_MR slice.

    Its `._weighted_counts` is a 4D ndarray with axes (rows, sel/not, cols, sel/not).
    """

    @lazyproperty
    def columns_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of matrix.

        An MR_X matrix has a distinct columns-margin for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. Each
        MR_X_MR cell has four counts: sel-sel, sel-not, not-sel, and not-not. Only
        sel-sel and not-sel contribute to the columns-margin.
        """
        # --- only column-selected counts contribute ([:, :, :, 0]), row-selected and
        # --- not-selected are summed (axis=1), rows and columns are retained.
        return np.sum(self._weighted_counts[:, :, :, 0], axis=1)

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each matrix cell."""
        # --- in the X_MR case, row-margins is the already-2D rows-margin ---
        return self.rows_margin

    @lazyproperty
    def rows_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of matrix.

        An X_MR matrix has a distinct rows-margin for each cell. Each MR_X_MR cell has
        four counts: sel-sel, sel-not, not-sel, and not-not. Only sel-sel and sel-not
        contribute to the rows-margin.
        """
        # --- only selecteds in rows contribute ([:, 0, :, :]), selected and not from
        # --- columns both contribute (axis=2 after rows sel/not axis is collapsed),
        # --- both rows and columns are retained, producing a 2D result.
        return np.sum(self._weighted_counts[:, 0, :, :], axis=2)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        # --- in MR_X_MR case, table-bases is the already-2D table-margin ---
        return self.table_margin

    @lazyproperty
    def table_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-margin.
        """
        # --- Reduce second and fourth axes (the two MR_CAT dimensions) with sum()
        # --- producing 2D (nrows, ncols). This sums the (selected, selected),
        # --- (selected, not), (not, selected) and (not, not) cells of the subtable for
        # --- each matrix cell. Rows and columns are retained.
        return np.sum(self._weighted_counts, axis=(1, 3))

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts. Only *selected*
        counts contribute to these values.
        """
        return self._weighted_counts[:, 0, :, 0]

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean each cell's
        value is.
        """
        return self._array_type_std_res(
            self.weighted_counts,
            self.table_margin,
            np.sum(self._weighted_counts, axis=3)[:, 0, :],
            np.sum(self._weighted_counts, axis=1)[:, :, 0],
        )


# === LEGACY MATRIX OBJECTS ===


class BaseCubeResultMatrix(object):
    """Base class for all cube-result matrix (2D second-order analyzer) objects."""

    def __init__(
        self, dimensions, weighted_counts, unweighted_counts, counts_with_missings=None
    ):
        self._dimensions = dimensions
        self._weighted_counts = weighted_counts
        self._unweighted_counts = unweighted_counts
        self._counts_with_missings = counts_with_missings

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return a base-matrix object of appropriate type for `cube`."""
        dimension_types = cube.dimension_types[-2:]
        MatrixCls = (
            _MrXMrMatrix
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatMatrix
            if dimension_types[0] == DT.MR
            else _CatXMrMatrix
            if dimension_types[1] == DT.MR
            else _CatXCatMatrix
        )
        return MatrixCls(dimensions, *cls._sliced_counts(cube, slice_idx))

    @lazyproperty
    def column_index(self):
        """2D np.float64/np.nan ndarray of column-index value for each matrix cell.

        Column-index answers the question "are respondents in this row-category more or
        less likely than the overall table population to choose the answer represented
        by this column?". For example, if the row is "Hispanic" and the column is
        home-ownership, a value of 100 indicates hispanics are no less and no more
        likely to own their home than the overall population. If that value was 150, it
        would indicate hispanics are 50% more likely to own their home than the general
        population (or the population surveyed anyway).
        """
        return self.column_proportions / self._baseline * 100

    @lazyproperty
    def column_proportions(self):
        """2D ndarray of np.float64 between 0.0 and 1.0.

        The value represents the ratio of each cell count to the total count (margin)
        for its column.
        """
        return self.weighted_counts / self.columns_margin

    @lazyproperty
    def columns_dimension(self):
        """The `Dimension` object representing column elements of this matrix."""
        return self._dimensions[1]

    @lazyproperty
    def columns_margin(self):
        """1D/2D np.float64 ndarray of weighted-N for each column of matrix."""
        raise NotImplementedError(
            "`%s` must implement `.columns_margin`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        raise NotImplementedError(
            "`%s` must implement `.columns_pruning_base`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def rows_base(self):
        """1D/2D np.float64 ndarray of unweighted-N for each matrix row/cell."""
        raise NotImplementedError(
            "`%s` must implement `.rows_base`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def rows_dimension(self):
        """The `Dimension` object representing row elements of this matrix."""
        return self._dimensions[0]

    @lazyproperty
    def rows_margin(self):
        """1D/2D np.float64 ndarray of weighted-N for each matrix row/cell."""
        raise NotImplementedError(
            "`%s` must implement `.rows_margin`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(
            "`%s` must implement `.rows_pruning_base`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def table_base(self):
        """Scalar, 1D, or 2D ndarray of np.float64 unweighted-N for this slice.

        This value has four distinct forms, depending on the subclass.
        """
        raise NotImplementedError(
            "`%s` must implement `.table_base`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def table_margin(self):
        """np.float64 scalar or a 1D or 2D np.float64 ndarray table margin.

        The table margin is the overall sample size of the matrix. This is the weighted
        count of respondents who were asked both questions and provided a valid response
        for both (including not-selecting an MR option/subvar).

        A matrix with a multiple-response (MR) dimension produces a 1D ndarray value.
        When both dimensions are MR, the return value is a 2D ndarray and there is
        a distinct table-base value for each "cell" of the matrix. A CAT_X_CAT matrix
        produces a scalar value for this property.
        """
        raise NotImplementedError(
            "`%s` must implement `.table_margin" % self.__class__.__name__
        )  # pragma: no cover

    @lazyproperty
    def table_stderrs(self):
        """2D np.float64 ndarray of table-percent std-error for each matrix cell."""
        raise NotImplementedError(
            "`%s` must implement `.table_stderrs" % self.__class__.__name__
        )  # pragma: no cover

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell."""
        raise NotImplementedError(
            "`%s` must implement `.unweighted_counts" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        raise NotImplementedError(
            "`%s` must implement `.weighted_counts`" % type(self).__name__
        )  # pragma: no cover

    @staticmethod
    def _cube_slice_expression(cube, slice_idx):
        """return `np.s_` object with correct slicing for the cube type."""
        if cube.ndim <= 2:
            return np.s_[:]

        # --- If 0th dimension of a >2D cube is MR, we only need the "Selected"
        # --- counts, because it's "just" used to tabulate.
        if cube.dimension_types[0] == DT.MR:
            return np.s_[slice_idx, 0]

        # --- If we have a cube with more than 2 dimensions we need to extract the
        # --- appropriate slice (element of the 0th dimension).
        return np.s_[slice_idx]

    @classmethod
    def _sliced_counts(cls, cube, slice_idx):
        """Return tuple of cube counts, prepared for regular matrix construction.

        Depending on the type of the cube, we need to extract the proper counts for the
        counstruction of a particular slice (matrix). In case of cubes that have more
        then 2 dimensions, we only need a particular slice (a particular selected
        element of the 0th dimension).

        If, in addition to being >2D cube, the 0th dimension is multiple response, we
        need to extract only the selected counts, since we're "just" dealing with the
        tabulation.
        """
        slice_expr = cls._cube_slice_expression(cube, slice_idx)
        return (
            cube.counts[slice_expr],
            cube.unweighted_counts[slice_expr],
            cube.counts_with_missings[slice_expr],
        )

    @lazyproperty
    def _valid_row_idxs(self):
        """ndarray-style index for only valid (non-missing) rows.

        Suitable for indexing a raw measure array to include only valid rows.
        """
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)


class _CatXCatMatrix(BaseCubeResultMatrix):
    """Matrix for CAT_X_CAT cubes and base class for most other matrix classes.

    Despite the name, this matrix is used for CA_SUBVAR and CA_CAT dimension too, since
    these behave the same from a base-matrix perspective.

    `counts_with_missings` is the raw weighted counts array, needed to compute the
    column-index.
    """

    @lazyproperty
    def column_index(self):
        """2D np.float64/np.nan ndarray of column-index value for each matrix cell.

        Column-index answers the question "are respondents in this row-category more or
        less likely than the overall table population to choose the answer represented
        by this column?". For example, if the row is "Hispanic" and the column is
        home-ownership, a value of 100 indicates hispanics are no less and no more
        likely to own their home than the overall population. If that value was 150, it
        would indicate hispanics are 50% more likely to own their home than the general
        population (or the population surveyed anyway).
        """
        return self.column_proportions / self._baseline * 100

    @lazyproperty
    def columns_margin(self):
        """1D ndarray of np.float64 weighted N for each matrix column."""
        return np.sum(self.weighted_counts, axis=0)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each column.
        """
        return np.sum(self._unweighted_counts, axis=0)

    @lazyproperty
    def rows_base(self):
        """1D ndarray of np.float64 unweighted-N for each matrix row."""
        return np.sum(self.unweighted_counts, axis=1)

    @lazyproperty
    def rows_margin(self):
        """1D np.float64 ndarray of weighted-N for each matrix row."""
        return np.sum(self._weighted_counts, axis=1)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each row.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def table_base(self):
        """np.float64 count of actual respondents who answered both questions.

        Each dimension of a CAT_X_CAT matrix represents a categorical question. Only
        responses that include answers to both those questions appear as entries in the
        valid elements of those dimensions. The sum total of all valid answers is the
        sample size, aka "N" or "base".
        """
        return np.sum(self.unweighted_counts)

    @lazyproperty
    def table_margin(self):
        """Scalar np.float64 weighted-N for overall table.

        This is the weighted count of respondents who provided a valid response to
        both questions. Because both dimensions are CAT, the table-margin value is the
        same for all cells of the matrix.
        """
        return np.sum(self._weighted_counts)

    @lazyproperty
    def table_stderrs(self):
        """2D np.float64 ndarray of table-percent std-error for each matrix cell.

        Standard error is sqrt(variance/N).
        """
        return np.sqrt(self._table_proportion_variances / self.table_margin)

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        If cube is unweighted, this is the same as unweighted_counts.
        """
        return self._weighted_counts

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each row in matrix.

        The shape of the return value is (nrows, 1). The baseline value for a row is the
        proportion of all values that appear in that row. A baseline for a 4 x 3 matrix
        looks like this:

            [[0.2006734 ]
             [0.72592593]
             [0.05521886]
             [0.01818182]]

        Note that the baseline values sum to 1.0. This is because each represents the
        portion of all responses that fall in that row. This baseline value is the
        denominator of the `._column_index` computation.

        Baseline is a straightforward function of the *unconditional row margin*.
        Unconditional here means that both valid and invalid responses (to the
        columns-var question) are included. This ensures that the baseline is not
        distorted by a large number of missing responses to the columns-question.
        """
        # --- uncond_row_margin is a 1D ndarray of the weighted total observation count
        # --- involving each valid row. Counts consider both valid and invalid columns,
        # --- but are only produced for valid rows.
        uncond_row_margin = np.sum(self._counts_with_missings, axis=1)[
            self._valid_row_idxs
        ]
        return uncond_row_margin[:, None] / np.sum(uncond_row_margin)

    @lazyproperty
    def _table_proportion_variances(self):
        """2D ndarray of np.float64 cell proportion variance for each cell of matrix."""
        p = self._weighted_counts / self.table_margin
        return p * (1 - p)


class _CatXMrMatrix(_CatXCatMatrix):
    """Represents a CAT x MR slice.

    Its `._counts` is a 3D ndarray of np.float64 with axes (rows, cols, selected/not),
    like:

        [[[1002.52343241 1247.791605  ]
          [ 765.95079804 1484.36423937]
          [ 656.43937497 1593.87566244]]

         [[1520.23482091 2573.22762247]
          [1291.0925792  2802.36986418]
          [1595.44412365 2498.01831973]]

         [[ 908.65667501 2254.62623471]
          [ 841.76439186 2321.51851785]
          [1603.79596755 1559.48694217]]

         [[ 746.89008236 1753.26322241]
          [ 721.38248086 1778.7708239 ]
          [1255.87038944 1244.28291533]]

         [[   9.83166357   25.9551254 ]
          [   8.23140253   27.55538645]
          [  22.214956     13.57183298]]]
    """

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def rows_base(self):
        """2D np.float64 ndarray of row-wise unweighted-N for this matrix.

        An X_MR matrix has a distinct row-base for each cell. This is because not all
        responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each X_MR cell is the sum of its selected and unselected
        unweighted counts.
        """
        return np.sum(self._unweighted_counts, axis=2)

    @lazyproperty
    def rows_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of this matrix.

        A matrix with an MR columns dimension has a distinct rows-margin for each cell.
        This is because not all column responses (subvars) are necessarily offered to
        each respondent. The weighted-count for each X_MR cell is the sum of its
        selected and unselected weighted counts.
        """
        return np.sum(self._weighted_counts, axis=2)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """1D np.float64 unweighted N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table base.
        """
        # --- unweighted-counts is (nrows, ncols, selected/not) so axis 1 is preserved
        # --- to provide a distinct value for each MR subvar.
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float64 ndarray of weighted-N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table margin.
        """
        # --- weighted-counts is (rows, cols, selected/not) so axis 1 is preserved to
        # --- provide a distinct value for each MR subvar.
        return np.sum(self._weighted_counts, axis=(0, 2))

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, :, 0]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        If the cube-result has no weight, these values are the same as the
        unweighted-counts.
        """
        return self._weighted_counts[:, :, 0]

    @lazyproperty
    def _baseline(self):
        """2D np.float64 (or NaN) ndarray of baseline value for each matrix cell.

        Its shape is (nrows, ncols) which corresponds to CAT_X_MR_SUBVAR.

        The baseline value is compared with the column-proportion value for each cell to
        form the column-index value. The baseline is a function of the unconditional row
        margin, which is the sum of counts across both valid and missing columns.

        For CAT_X_MR, `uncond_row_margin` sums across the MR_CAT (selected, not,
        missing) dimension to include missing values (an MR_SUBVAR element is never
        "missing": true).
        """
        # --- counts_with_missings.shape is (nall_rows, ncols, selected/not/missing).
        # --- axes[1] corresponds to the MR_SUBVAR dimension, in which there are never
        # --- "missing" subvars (so nall_cols always equals ncols for that dimension
        # --- type). uncond_row_margin selects only valid rows, retains all columns and
        # --- reduces the selected/not/missing axis by summing those three counts. Its
        # --- shape is (nrows, ncols).
        uncond_row_margin = np.sum(self._counts_with_missings, axis=2)[
            self._valid_row_idxs
        ]
        # --- uncond_table_margin sums across rows, producing 1D array of size ncols,
        # --- (although all its values are always the same).
        uncond_table_margin = np.sum(uncond_row_margin, axis=0)
        # --- division produces a 2D matrix of shape (nrows, ncols) ---
        return uncond_row_margin / uncond_table_margin

    @lazyproperty
    def _table_proportion_variances(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._weighted_counts[:, :, 0] / self.table_margin
        return p * (1 - p)


class _MrXCatMatrix(BaseCubeResultMatrix):
    """Represents an MR_X_CAT slice.

    Its `._counts` is a 3D ndarray of np.float64 with axes (rows, sel/not, cols), like:

        [[[ 39  44  24  35]
          [389 447 266 394]]

         [[ 34  36  29  24]
          [394 455 261 405]]

         [[357 415 241 371]
          [ 71  76  49  58]]

         [[  0   0   0   0]
          [428 491 290 429]]]
    """

    @lazyproperty
    def columns_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of this matrix.

        An MR_X matrix has a distinct column-margin for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. The
        weighted-count for each MR_X cell is the sum of its selected and unselected
        weighted counts.
        """
        return np.sum(self._weighted_counts, axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 1))

    @lazyproperty
    def rows_base(self):
        """1D ndarray of np.float64 unweighted-N for each matrix row.

        Only selected values contribute to the rows-base.
        """
        return np.sum(self._unweighted_counts[:, 0, :], axis=1)

    @lazyproperty
    def rows_margin(self):
        """1D np.float64 ndarray of weighted-N for each matrix row.

        Only selected values contribute to the rows-margin.
        """
        return np.sum(self._weighted_counts[:, 0, :], axis=1)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """1D np.float64 ndarray of unweighted N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct base, since not all of
        the multiple responses were necessarily offered to all respondents. The base for
        each row indicates the number of respondents who were offered that option.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float64 ndarray of weighted-N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct table margin, since not
        all of the multiple responses were necessarily offered to all respondents. The
        table-margin for each row indicates the weighted number of respondents who were
        offered that option.
        """
        return np.sum(self._weighted_counts, axis=(1, 2))

    @lazyproperty
    def table_stderrs(self):
        """2D np.float64 ndarray of table-percent std-error for each matrix cell.

        Standard error is sqrt(variance/N).
        """
        return np.sqrt(self._table_proportion_variances / self.table_margin[:, None])

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, 0, :]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        When the cube-result has no weight, in which case these values are the same as
        the unweighted-counts.
        """
        return self._weighted_counts[:, 0, :]

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each row in matrix.

        `._baseline` is the denominator of the column-index and represents the
        proportion of the overall row-count present in each row. A cell with
        a column-proportion exactly equal to this basline will have a column-index of
        100.

        The shape of the return value is (nrows, 1). A baseline for a 4 x 3 matrix looks
        something like this:

            [[0.17935204]
             [0.33454989]
             [0.50762388]
             [0.80331259]
             [0.7996507 ]]

        Baseline is a function of the *unconditional row margin*. Unconditional here
        means that both valid and invalid responses (to the columns-var question) are
        included. This ensures that the baseline is not distorted by a large number of
        missing responses to the columns-question.
        """
        # --- unconditional row-margin is a 1D ndarray of size nrows computed by:
        # --- 1. summing across all columns: np.sum(self._counts_with_missings, axis=2)
        # --- 2. taking only selected counts: [:, 0]
        # --- 3. taking only valid rows: [self._valid_row_idxs]
        uncond_row_margin = np.sum(self._counts_with_missings, axis=2)[:, 0][
            self._valid_row_idxs
        ]
        # --- The "total" (uncond_row_table_margin) is a 1D ndarray of size nrows. Each
        # --- sum includes only valid rows (MR_SUBVAR, axis 0), selected and unselected
        # --- but not missing counts ([0:2]) of the MR_CAT axis (axis 1), and all column
        # --- counts, both valid and missing (axis 2). The rows axis (0) is preserved
        # --- because each MR subvar has a distinct table margin.
        uncond_row_table_margin = np.sum(
            self._counts_with_missings[self._valid_row_idxs][:, 0:2], axis=(1, 2)
        )
        # --- inflate shape to (nrows, 1) for later calculation convenience ---
        return (uncond_row_margin / uncond_row_table_margin)[:, None]

    @lazyproperty
    def _table_proportion_variances(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._weighted_counts[:, 0, :] / self.table_margin[:, None]
        return p * (1 - p)


class _MrXMrMatrix(_CatXCatMatrix):
    """Represents an MR x MR slice.

    Its `._counts` is a 4D ndarray with axes (rows, sel/not, cols, sel/not), like:

        [[[[2990.03485848 4417.96127006]
           [2713.94318797 4694.05294056]
           [2847.96860219 4560.02752634]]

          [[1198.10181578 3436.90253993]
           [ 914.47846452 3720.52589119]
           [2285.79620941 2349.2081463 ]]]


         [[[2626.08325048 5180.55485426]
           [2396.04310657 5410.59499817]
           [3503.08635211 4303.55175262]]

          [[1562.05342378 2674.30895573]
           [1232.37854592 3003.98383359]
           [1630.67845949 2605.68392002]]]


         [[[3370.04923406 5278.54391705]
           [3033.71862569 5614.87452542]
           [3312.56140096 5336.03175016]]

          [[ 818.0874402  2576.31989293]
           [ 594.7030268  2799.70430633]
           [1821.20341065 1573.20392249]]]


         [[[1822.67560537 2883.99243344]
           [1616.70492531 3089.96311351]
           [1735.59793395 2971.07010487]]

          [[2365.46106889 4970.87137654]
           [2011.71672718 5324.61571825]
           [3398.16687766 3938.16556777]]]]

    """

    @lazyproperty
    def columns_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of matrix.

        An MR_X matrix has a distinct columns-margin for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. Each
        MR_X_MR cell has four counts: sel-sel, sel-not, not-sel, and not-not. Only
        sel-sel and not-sel contribute to the columns-margin.
        """
        return np.sum(self._weighted_counts[:, :, :, 0], axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the row MR; only selecteds are considered for
        the columns dimension.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=(0, 1))

    @lazyproperty
    def rows_base(self):
        """2D np.float64 ndarray of unweighted-N for this matrix.

        An MR_X matrix has a distinct row-base for each cell, the sum of sel-sel and
        sel-not for each cell
        """
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=2)

    @lazyproperty
    def rows_margin(self):
        """2D np.float64 ndarray of weighted-N for each cell of matrix.

        An X_MR matrix has a distinct rows-margin for each cell. Each MR_X_MR cell has
        four counts: sel-sel, sel-not, not-sel, and not-not. Only sel-sel and sel-not
        contribute to the rows-margin.
        """
        # --- sum of (sel-sel, sel-not) ---
        return np.sum(self._weighted_counts[:, 0, :, :], axis=2)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the column MR; only selecteds are considered for
        the rows dimension.
        """
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """2D np.float64 ndarray of distinct unweighted N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-base.
        """
        # --- unweighted_counts is 4D of shape (nrows, 2, ncols, 2):
        # --- (MR_SUBVAR (nrows), MR_CAT (sel/not), MR_SUBVAR (ncols), MR_CAT (sel/not))
        # --- Reduce the second and fourth axes with sum() producing 2D (nrows, ncols).
        # --- This sums (selected, selected), (selected, not), (not, selected) and
        # --- (not, not) cells of the subtable for each matrix cell.
        return np.sum(self._unweighted_counts, axis=(1, 3))

    @lazyproperty
    def table_margin(self):
        """2D np.float64 ndarray of weighted N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-margin.
        """
        # --- Reduce second and fourth axes (the two MR_CAT dimensions) with sum()
        # --- producing 2D (nrows, ncols). This sums the (selected, selected),
        # --- (selected, not), (not, selected) and (not, not) cells of the subtable for
        # --- each matrix cell.
        return np.sum(self._weighted_counts, axis=(1, 3))

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, 0, :, 0]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each valid matrix cell.

        When the cube-result has no weight, these values are the same as the
        unweighted-counts. Only *selected* counts contribute to these values.
        """
        return self._weighted_counts[:, 0, :, 0]

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each matrix cell.

        The shape is (nrows, ncols) and all values in a given row are the same. So
        really there are only nrows distinct baseline values, but the returned shape
        makes calculating column-index in a general way more convenient.
        """
        # --- `counts_with_missings` for MR_X_MR is 4D of size (nrows, 3, ncols, 3)
        # --- (MR_SUBVAR, MR_CAT, MR_SUBVAR, MR_CAT). Unconditional row margin:
        # --- * Takes all rows and all cols (axes 0 & 2), because MR_SUBVAR dimension
        # ---   can contain only valid elements (no such thing as "missing": true
        # ---   subvar).
        # --- * Sums selected + unselected + missing categories in second MR_CAT
        # ---   dimension (columns MR, axes[3]). Including missings here fulfills
        # ---   "unconditional" characteristic of margin.
        # --- * Takes only those totals associated with selected categories of first
        # ---    MR_CAT dimension (rows MR). ("counts" for MR are "selected" counts).
        # --- Produces a 2D (nrows, ncols) array.
        uncond_row_margin = np.sum(self._counts_with_missings[:, 0:2], axis=3)[:, 0]
        # --- Unconditional table margin is also 2D (nrows, ncols) but the values for
        # --- all columns in a row have the same value; basically each row has
        # --- a distinct table margin.
        uncond_table_margin = np.sum(self._counts_with_missings[:, 0:2], axis=(1, 3))
        # --- baseline is produced by dividing uncond_row_margin by uncond_table_margin.
        return uncond_row_margin / uncond_table_margin

    @lazyproperty
    def _table_proportion_variances(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._weighted_counts[:, 0, :, 0] / self.table_margin
        return p * (1 - p)
