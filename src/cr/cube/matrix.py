# encoding: utf-8

"""A matrix is the 2D cube-data partition used by a slice.

The `Assembler` object provides the external interface for this module. Its name derives
from its role to "assemble" a finished 2D array ("matrix") for a particular measure from
the base measure values and inserted subtotals, to reorder the rows and columns
according to the dimension *order* transforms, and to hide rows and columns that are
either hidden by the user or "pruned" because they contain no observations.
"""

import numpy as np

from cr.cube.collator import ExplicitOrderCollator, PayloadOrderCollator
from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class Assembler(object):
    """Provides measure and margin methods for a cube-slice.

    An assembled matrix is a 2D ndarray reflecting all ordering, insertion, and hiding
    transforms applied to the dimensions. An assembled margin is often a 1D ndarray
    which is similarly formed from inserted values, ordered, and value hiding applied.

    `cube` is the `cr.cube.Cube` object containing the data for this matrix. Note that
    not all the data in `cube` will necessarily be used by this matrix. When `cube` is
    more than 2-dimensional, it is "sliced" and each slice gets its own matrix (and
    `_Slice` object).

    `dimensions` is a pair (2-tuple) of (rows_dimension, columns_dimension) Dimension
    objects. These are always the last two dimensions of `cube` but may and often do
    have transformations applied that are not present on the `cube` dimensions from
    which they derive.

    `slice_idx` is an int offset indicating which portion of `cube` data to use for this
    matrix. There is one slice for each element of the first cube dimension (the "table"
    dimension) when the cube has more than two dimensions.
    """

    def __init__(self, cube, dimensions, slice_idx):
        self._cube = cube
        self._dimensions = dimensions
        self._slice_idx = slice_idx

    @lazyproperty
    def columns_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each slice column/cell."""
        # --- an MR_X slice produces a 2D column-base (each cell has its own N) ---
        if self._rows_dimension.dimension_type == DT.MR_SUBVAR:
            return self._assemble_matrix(
                _SumSubtotals.blocks(self._cube_result_matrix, "columns_base")
            )

        # --- otherwise columns-base is a vector ---
        return self._assemble_vector(
            self._cube_result_matrix.columns_base,
            self._column_subtotals,
            self._column_order,
        )

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each cell."""
        return self._assemble_matrix(
            _SumSubtotals.blocks(self._cube_result_matrix, "unweighted_counts")
        )

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each cell."""
        return self._assemble_matrix(
            _SumSubtotals.blocks(self._cube_result_matrix, "weighted_counts")
        )

    def _assemble_matrix(self, blocks):
        """Return 2D ndarray matrix assembled from `blocks`.

        The assembled matrix includes inserted vectors (rows and columns), has hidden
        vectors removed, and is ordered by whatever sort method is applied in the
        dimension transforms.
        """
        # --- These are assembled into a single 2D array, and then rearranged based on
        # --- row and column orders. All insertion, ordering, and hiding transforms are
        # --- reflected in the row and column orders. They each include (negative)
        # --- insertion idxs, hidden and pruned vector indices have been removed, and
        # --- the ordering method has been applied to determine the sequence each idx
        # --- appears in. This directly produces a final array that is exactly the
        # --- desired output.
        return np.block(blocks)[np.ix_(self._row_order, self._column_order)]

    def _assemble_vector(self, base_vector, subtotals, order):
        """Return 1D ndarray of `base_vector` with inserted `subtotals`, in `order`.

        Each subtotal value is the result of applying np.sum to the addends extracted
        from `base_vector` according the the `addend_idxs` property of each subtotal in
        `subtotals`. The returned array is arranged by `order`, including possibly
        removing hidden or pruned values.
        """
        # TODO: This only works for "sum" subtotals, which is all that it needs so far,
        # but a fuller solution will probably get the subtotal values from a
        # _BaseSubtotals subclass.
        vector_subtotals = np.array(
            [np.sum(base_vector[subtotal.addend_idxs]) for subtotal in subtotals]
        )
        return np.hstack([base_vector, vector_subtotals])[order]

    @lazyproperty
    def _column_order(self):
        """1D np.int64 ndarray of signed int idx for each assembled column.

        Negative values represent inserted subtotal-column locations.
        """
        order = self._dimension_order(self._columns_dimension, self._empty_column_idxs)
        return order[order >= 0] if self._prune_subtotal_columns else order

    @lazyproperty
    def _column_subtotals(self):
        """Sequence of _Subtotal object for each inserted column."""
        return self._columns_dimension.subtotals

    @lazyproperty
    def _columns_dimension(self):
        """The `Dimension` object representing column elements in this matrix."""
        return self._dimensions[1]

    @lazyproperty
    def _cube_result_matrix(self):
        """_BaseCubeResultMatrix subclass object appropriate to this cube-slice.

        This matrix object encapsulates cube-result array parsing and MR multi-value
        differences and provides a foundational set of second-order analysis measure and
        margin arrays.
        """
        return _BaseCubeResultMatrix.factory(
            self._cube, self._dimensions, self._slice_idx
        )

    def _dimension_order(self, dimension, empty_idxs):
        """1D np.int64 ndarray of signed int idx for each vector in `dimension`.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in `dimension`.
        """
        collation_method = dimension.collation_method

        # --- this will become a significantly more sophisticated dispatch when
        # --- sort-by-value is implemented in later commits. In particular, the
        # --- signature for the sort-by-value collator differs from the payload and
        # --- explicit-order collators. So we'll defer refactoring this until that
        # --- collator is added. (sc)
        display_order = (
            ExplicitOrderCollator.display_order(dimension, empty_idxs)
            if collation_method == CM.EXPLICIT_ORDER
            else PayloadOrderCollator.display_order(dimension, empty_idxs)
        )

        # --- Returning as np.array suits its intended purpose, which is to participate
        # --- in an np._ix() call. It works fine as a sequence too for any alternate
        # --- use. Specifying int type prevents failure when there are zero elements.
        return np.array(display_order, dtype=int)

    @lazyproperty
    def _empty_column_idxs(self):
        """tuple of int index for each column with (unweighted) N = 0.

        These columns are subject to pruning, depending on a user setting in the
        dimension.
        """
        pruning_base = self._cube_result_matrix.columns_pruning_base
        return tuple(i for i, N in enumerate(pruning_base) if N == 0)

    @lazyproperty
    def _empty_row_idxs(self):
        """tuple of int index for each row with N = 0.

        These rows are subject to pruning, depending on a user setting in the dimension.
        """
        pruning_base = self._cube_result_matrix.rows_pruning_base
        return tuple(i for i, N in enumerate(pruning_base) if N == 0)

    @lazyproperty
    def _prune_subtotal_columns(self):
        """True if subtotal columns need to be pruned, False otherwise.

        Subtotal-columns need to be pruned when all base-rows are pruned.
        """
        if not self._rows_dimension.prune:
            return False

        return len(self._empty_row_idxs) == len(self._rows_dimension.element_ids)

    @lazyproperty
    def _prune_subtotal_rows(self):
        """True if subtotal rows need to be pruned, False otherwise.

        Subtotal-rows need to be pruned when all base-columns are pruned.
        """
        if not self._columns_dimension.prune:
            return False

        return len(self._empty_column_idxs) == len(self._columns_dimension.element_ids)

    @lazyproperty
    def _row_order(self):
        """1D np.int64 ndarray of signed int idx for each assembled row.

        Negative values represent inserted subtotal-row locations.
        """
        order = self._dimension_order(self._rows_dimension, self._empty_row_idxs)
        if self._prune_subtotal_rows:
            return np.array([idx for idx in order if idx >= 0], dtype=int)
        return order

    @lazyproperty
    def _rows_dimension(self):
        """The `Dimension` object representing row elements in this matrix."""
        return self._dimensions[0]


# === SUBTOTALS OBJECTS ===


class _BaseSubtotals(object):
    """Base class for Subtotals objects."""

    def __init__(self, cube_result_matrix, measure_propname):
        self._cube_result_matrix = cube_result_matrix
        self._measure_propname = measure_propname

    @classmethod
    def blocks(cls, cube_result_matrix, measure_propname=None):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.
        """
        return cls(cube_result_matrix, measure_propname)._blocks

    @lazyproperty
    def _base_values(self):
        """2D ndarray of "body" values from cube-result matrix."""
        raise NotImplementedError(
            "`%s` must implement `._base_values`" % type(self).__name__
        )

    @lazyproperty
    def _blocks(self):
        """base, row and col insertion, and intersection matrices."""
        return [
            [self._base_values, self._subtotal_columns],
            [self._subtotal_rows, self._intersections],
        ]

    @lazyproperty
    def _column_subtotals(self):
        """Sequence of _Subtotal object for each subtotal in columns-dimension."""
        return self._cube_result_matrix.columns_dimension.subtotals

    @lazyproperty
    def _dtype(self):
        """Numpy data-type for result matrices, used for empty arrays."""
        return np.float64

    def _intersection(self, row_subtotal, column_subtotal):
        """Value for this row/column subtotal intersection."""
        raise NotImplementedError(
            "`%s` must implement `._intersection()`" % type(self).__name__
        )

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        return np.array(
            [
                self._intersection(row_subtotal, column_subtotal)
                for row_subtotal in self._row_subtotals
                for column_subtotal in self._column_subtotals
            ],
            dtype=self._dtype,
        ).reshape(len(self._row_subtotals), len(self._column_subtotals))

    @lazyproperty
    def _ncols(self):
        """int count of columns in base-matrix."""
        return self._base_values.shape[1]

    @lazyproperty
    def _nrows(self):
        """int count of rows in base-matrix."""
        return self._base_values.shape[0]

    @lazyproperty
    def _row_subtotals(self):
        """Sequence of _Subtotal object for each subtotal in rows-dimension."""
        return self._cube_result_matrix.rows_dimension.subtotals

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of values for `subtotal` column."""
        raise NotImplementedError(
            "`%s` must implement `._subtotal_column()`" % type(self).__name__
        )

    @lazyproperty
    def _subtotal_columns(self):
        """(n_rows, n_col_subtotals) matrix of subtotal columns."""
        subtotals = self._column_subtotals

        if len(subtotals) == 0:
            return np.empty((self._nrows, 0), dtype=self._dtype)

        return np.hstack(
            [
                self._subtotal_column(subtotal).reshape(self._nrows, 1)
                for subtotal in subtotals
            ]
        )

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of values for `subtotal` row."""
        return np.sum(self._base_values[subtotal.addend_idxs, :], axis=0)

    @lazyproperty
    def _subtotal_rows(self):
        """(n_row_subtotals, n_cols) ndarray of subtotal rows."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.empty((0, self._ncols), dtype=self._dtype)

        return np.vstack(
            [self._subtotal_row(subtotal) for subtotal in self._row_subtotals]
        )


class _SumSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by np.sum() on addends, primarily counts."""

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of table-stderr for each cell of cube-result matrix."""
        return getattr(self._cube_result_matrix, self._measure_propname)

    @lazyproperty
    def _dtype(self):
        """Numpy data-type for result matrices, used for empty arrays."""
        return self._base_values.dtype

    def _intersection(self, row_subtotal, column_subtotal):
        """Sum for this row/column subtotal intersection."""
        return np.sum(self._subtotal_row(row_subtotal)[column_subtotal.addend_idxs])

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of np.nan values."""
        return np.sum(self._base_values[:, subtotal.addend_idxs], axis=1)


# === CUBE-RESULT MATRIX OBJECTS ===


class _BaseCubeResultMatrix(object):
    """Base class for all cube-result matrix (2D second-order analyzer) objects."""

    def __init__(self, dimensions, weighted_counts, unweighted_counts):
        self._dimensions = dimensions
        self._weighted_counts = weighted_counts
        self._unweighted_counts = unweighted_counts

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return a base-matrix object of appropriate type for `cube`."""
        # --- means cube gets one of the means-matrix types ---
        if cube.has_means:
            return cls._means_matrix_factory(cube, dimensions, slice_idx)

        # --- everything else gets a more conventional matrix ---
        return cls._regular_matrix_factory(cube, dimensions, slice_idx)

    @lazyproperty
    def column_proportions(self):
        """2D ndarray of np.float64 between 0.0 and 1.0.

        The value represents the ratio of each cell count to the total count (margin)
        for its column.
        """
        raise NotImplementedError

    @lazyproperty
    def columns_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each matrix column/cell."""
        raise NotImplementedError(
            "`%s` must implement `.columns_base`" % type(self).__name__
        )

    @lazyproperty
    def columns_dimension(self):
        """The `Dimension` object representing column elements of this matrix."""
        return self._dimensions[1]

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column."""
        raise NotImplementedError(
            "`%s` must implement `.columns_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def rows_dimension(self):
        """The `Dimension` object representing row elements of this matrix."""
        return self._dimensions[0]

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(
            "`%s` must implement `.rows_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def table_margin(self):
        """np.float/int64 scalar or a 1D or 2D np.float/int64 ndarray table margin.

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
        )

    @staticmethod
    def _extract_counts_for_matrix_creation(cube, slice_idx):
        """Returns a tuple of cube counts, prepared for matrix construction.

        Depending on the type of the cube, we need to extract the proper counts for the
        counstruction of a particular slice (matrix). In case of cubes that have more
        then 2 dimensions, we only need a particular slice (a particular selected
        element of the 0th dimension).

        If, in addition to being >2D cube, the 0th dimension is multiple response, we
        need to extract only the selected counts, since we're "just" dealing with the
        tabulation.
        """

        # --- If we have a cube with more than 2 dimensions we need to extract the
        # --- appropriate slice (element of the 0th dimension).
        if cube.ndim > 2:

            # --- If 0th dimension of a >2D cube is MR, we only need the "Selected"
            # --- counts, because it's "just" used to tabulate.
            if cube.dimension_types[0] == DT.MR:
                return (
                    cube.counts[slice_idx][0],
                    cube.unweighted_counts[slice_idx][0],
                    cube.counts_with_missings[slice_idx][0],
                )

            return (
                cube.counts[slice_idx],
                cube.unweighted_counts[slice_idx],
                cube.counts_with_missings[slice_idx],
            )

        return (cube.counts, cube.unweighted_counts, cube.counts_with_missings)

    @classmethod
    def _means_matrix_factory(cls, cube, dimensions, slice_idx):
        """ -> matrix object appropriate to means `cube`."""
        dimension_types = cube.dimension_types[-2:]

        if dimension_types == (DT.MR, DT.MR):
            # --- this MEANS_MR_X_MR case hasn't arisen yet ---
            raise NotImplementedError(
                "MR x MR with means is not implemented"
            )  # pragma: no cover

        MatrixCls = (
            _MrXCatMeansMatrix
            if dimension_types[0] == DT.MR
            else _CatXMrMeansMatrix
            if dimension_types[1] == DT.MR
            else _CatXCatMeansMatrix
        )
        counts, unweighted_counts = (
            (cube.counts[slice_idx], cube.unweighted_counts[slice_idx])
            if cube.ndim == 3
            else (cube.counts, cube.unweighted_counts)
        )
        return MatrixCls(dimensions, counts, unweighted_counts)

    @staticmethod
    def _regular_matrix_class(dimension_types):
        """Return _BaseCubeResultMatrix subclass appropriate to `dimension_types`."""
        return (
            _MrXMrMatrix
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatMatrix
            if dimension_types[0] == DT.MR
            else _CatXMrMatrix
            if dimension_types[1] == DT.MR
            else _CatXCatMatrix
        )

    @staticmethod
    def _regular_matrix_counts_slice(cube, slice_idx):
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
    def _regular_matrix_factory(cls, cube, dimensions, slice_idx):
        """ -> matrix object for non-means slice."""
        MatrixCls = cls._regular_matrix_class(cube.dimension_types[-2:])
        return MatrixCls(dimensions, *cls._sliced_counts(cube, slice_idx))

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
        i = cls._regular_matrix_counts_slice(cube, slice_idx)
        return (cube.counts[i], cube.unweighted_counts[i], cube.counts_with_missings[i])


class _CatXCatMatrix(_BaseCubeResultMatrix):
    """Matrix for CAT_X_CAT cubes and base class for most other matrix classes.

    Despite the name, this matrix is used for CA_SUBVAR and CA_CAT dimension too, since
    these behave the same from a base-matrix perspective.

    `counts_with_missings` is the raw weighted counts array, needed to compute the
    column-index.
    """

    def __init__(
        self, dimensions, weighted_counts, unweighted_counts, counts_with_missings=None
    ):
        super(_CatXCatMatrix, self).__init__(
            dimensions, weighted_counts, unweighted_counts
        )
        self._counts_with_missings = counts_with_missings

    @lazyproperty
    def columns_base(self):
        """1D ndarray of np.int64 unweighted-N for each matrix column."""
        return np.sum(self.unweighted_counts, axis=0)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each column.
        """
        return np.sum(self._unweighted_counts, axis=0)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each row.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each valid matrix cell.

        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        return self._weighted_counts


class _CatXCatMeansMatrix(_CatXCatMatrix):
    """Cat-x-cat matrix for means measure.

    A means matrix has an array of mean values instead of a `counts` array.
    """

    def __init__(self, dimensions, means, unweighted_counts):
        super(_CatXCatMeansMatrix, self).__init__(dimensions, None, unweighted_counts)
        self._means = means

    @lazyproperty
    def weighted_counts(self):
        """2D ndarray of np.nan for each valid matrix cell.

        Weighted-counts have no meaning for a means matrix (although unweighted counts
        do).
        """
        return np.full(self._means.shape, np.nan)


class _CatXMrMatrix(_CatXCatMatrix):
    """Represents a CAT x MR slice.

    Its `._counts` is a 3D ndarray with axes (rows, cols, selected/not), like:

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

    Each value is np.float64 if the cube-result is weighted (as in this example), or
    np.int64 if unweighted.
    """

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, :, 0]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        return self._weighted_counts[:, :, 0]


class _CatXMrMeansMatrix(_CatXMrMatrix):
    """Basis for CAT_X_MR slice having mean measure instead of counts."""


class _MrXCatMatrix(_CatXCatMatrix):
    """Represents an MR_X_CAT slice.

    Its `._counts` is a 3D ndarray with axes (rows, sel/not, cols), like:

        [[[ 39  44  24  35]
          [389 447 266 394]]

         [[ 34  36  29  24]
          [394 455 261 405]]

         [[357 415 241 371]
          [ 71  76  49  58]]

         [[  0   0   0   0]
          [428 491 290 429]]]

    Each value is np.float64, or np.int64 if the cube-result is unweighted (as in this
    example).
    """

    @lazyproperty
    def columns_base(self):
        """2D np.int64 ndarray of unweighted-N for this matrix.

        An MR_X matrix has a distinct column-base for each cell. This is because not all
        responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each MR_X cell is the sum of its selected and unselected
        unweighted counts.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(0, 1))

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR rows
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, 0, :]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        return self._weighted_counts[:, 0, :]


class _MrXCatMeansMatrix(_MrXCatMatrix):
    """MR_X_CAT slice with means measure instead of counts.

    Note that its (weighted) counts are all set to zero. A means slice still has
    meaningful unweighted counts.
    """


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
    def columns_base(self):
        """2D np.int64 ndarray of unweighted-N for this matrix.

        An MR_X_MR matrix has a distinct column-base for each cell. This is because not
        all responses (subvars) are necessarily presented to each respondent. The
        unweighted-count for each MR_X cell is the sum of its selected and unselected
        unweighted counts.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=1)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the row MR; only selecteds are considered for
        the columns dimension.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=(0, 1))

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the column MR; only selecteds are considered for
        the rows dimension.
        """
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=(1, 2))

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each valid matrix cell.

        A valid matrix cell is one whose row and column elements are both non-missing.
        """
        return self._unweighted_counts[:, 0, :, 0]

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each valid matrix cell.

        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts. Only *selected* counts
        contribute to these values.
        """
        return self._weighted_counts[:, 0, :, 0]
