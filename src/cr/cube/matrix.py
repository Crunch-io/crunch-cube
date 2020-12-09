# encoding: utf-8

"""A matrix is the 2D cube-data partition used by a slice.

The `Assembler` object provides the external interface for this module. Its name derives
from its role to "assemble" a finished 2D array ("matrix") for a particular measure from
the base measure values and inserted subtotals, to reorder the rows and columns
according to the dimension *order* transforms, and to hide rows and columns that are
either hidden by the user or "pruned" because they contain no observations.
"""

import numpy as np
from scipy.stats import norm
from scipy.stats.contingency import expected_freq

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
    def column_index(self):
        """2D np.float64 ndarray of column-index "percentage" for each table cell."""
        return self._assemble_matrix(
            _NanSubtotals.blocks(self._cube_result_matrix, "column_index")
        )

    @lazyproperty
    def column_labels(self):
        """1D str ndarray of column name for each matrix column.

        These are suitable for use as column headings; labels for subtotal columns
        appear in the sequence and labels are ordered to correspond with their
        respective data column.
        """
        return self._dimension_labels(self._columns_dimension, self._column_order)

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
    def columns_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each column element.

        A value of np.nan appears for a column element without a numeric-value. All
        subtotal columns have a value of np.nan (subtotals have no numeric value).
        """
        elements = self._columns_dimension.valid_elements
        return np.array(
            [
                (elements[idx].numeric_value if idx >= 0 else np.nan)
                for idx in self._column_order
            ]
        )

    @lazyproperty
    def columns_margin(self):
        """1D/2D np.float64 ndarray of weighted-N for each column of this slice."""
        # --- an MR_X slice produces a 2D columns-margin (each cell has its own N) ---
        if self._rows_dimension.dimension_type == DT.MR_SUBVAR:
            return self._assemble_matrix(
                _SumSubtotals.blocks(self._cube_result_matrix, "columns_margin")
            )

        # --- otherwise columns-base is a vector ---
        return self._assemble_vector(
            self._cube_result_matrix.columns_margin,
            self._column_subtotals,
            self._column_order,
        )

    @lazyproperty
    def inserted_column_idxs(self):
        """tuple of int index of each subtotal column in slice."""
        return tuple(i for i, col_idx in enumerate(self._column_order) if col_idx < 0)

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each subtotal row in slice."""
        return tuple(i for i, row_idx in enumerate(self._row_order) if row_idx < 0)

    @lazyproperty
    def means(self):
        """2D optional np.float64 ndarray of mean for each cell.

        Raises `ValueError` if the cube-result does not include a means cube-measure.
        """
        raise NotImplementedError

    @lazyproperty
    def pvalues(self):
        """2D np.float64/np.nan ndarray of p-value for each matrix cell."""
        return 2 * (1 - norm.cdf(np.abs(self.zscores)))

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of row name for each matrix row.

        These are suitable for use as row headings; labels for subtotal rows appear in
        the sequence and labels are ordered to correspond with their respective data
        row.
        """
        return self._dimension_labels(self._rows_dimension, self._row_order)

    @lazyproperty
    def rows_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each slice row/cell."""
        # --- an X_MR slice produces a 2D row-base (each cell has its own N) ---
        if self._columns_dimension.dimension_type == DT.MR_SUBVAR:
            return self._assemble_matrix(
                _SumSubtotals.blocks(self._cube_result_matrix, "rows_base")
            )

        # --- otherwise rows-base is a vector ---
        return self._assemble_vector(
            self._cube_result_matrix.rows_base, self._row_subtotals, self._row_order
        )

    @lazyproperty
    def rows_dimension_fills(self):
        """tuple of RGB str like "#def032" fill color for each row in slice."""
        elements = self._rows_dimension.valid_elements
        return tuple(
            (elements[idx].fill if idx >= 0 else None) for idx in self._row_order
        )

    @lazyproperty
    def rows_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each row element.

        A value of np.nan appears for a row element without a numeric-value. All
        subtotal rows have a value of np.nan (subtotals have no numeric value).
        """
        elements = self._rows_dimension.valid_elements
        return np.array(
            [
                (elements[idx].numeric_value if idx >= 0 else np.nan)
                for idx in self._row_order
            ]
        )

    @lazyproperty
    def rows_margin(self):
        """1D/2D np.float64 ndarray of weighted-N for each column of this slice.

        This value is a 2D array if one or both of the slice dimensions are MR. This is
        because each cell has a distinct margin in such a slice.
        """
        # --- an X_MR slice produces a 2D rows-margin (each cell has its own N) ---
        if self._columns_dimension.dimension_type == DT.MR_SUBVAR:
            return self._assemble_matrix(
                _SumSubtotals.blocks(self._cube_result_matrix, "rows_margin")
            )

        # --- otherwise rows-margin is a vector ---
        return self._assemble_vector(
            self._cube_result_matrix.rows_margin, self._row_subtotals, self._row_order
        )

    @lazyproperty
    def table_base(self):
        """Scalar, 1D, or 2D ndarray of np.int64 unweighted-N for this slice.

        This value has four distinct forms, depending on the slice dimensions:

            * MR_X_MR - 2D ndarray with a distinct table-base value per cell.
            * MR_X - 1D ndarray of value per *row* when only rows dimension is MR.
            * X_MR - 1D ndarray of value per *column* when only columns dimension is MR.
            * CAT_X_CAT - scalar float value when slice has no MR dimension.

        """
        # --- an MR dimension cannot have subtotals, and a non-scalar table base only
        # --- arises in an MR dimension case, so a non-scalar table-base never has
        # --- subtotals and we need only operate on the base values. There is no need to
        # --- assemble with subtotals, however the values of a non-scalar table-base do
        # --- still need to be ordered (which includes hiding).
        base_table_base = self._cube_result_matrix.table_base

        # --- MR_X_MR slice produces a 2D table-base (each cell has its own N) ---
        if (
            self._rows_dimension.dimension_type == DT.MR_SUBVAR
            and self._columns_dimension.dimension_type == DT.MR_SUBVAR
        ):
            return base_table_base[np.ix_(self._row_order, self._column_order)]

        # --- CAT_X_MR slice produces a 1D table-base (each column has its own N) ---
        if self._columns_dimension.dimension_type == DT.MR_SUBVAR:
            return base_table_base[self._column_order]

        # --- MR_X_CAT slice produces a 1D table-base (each row has its own N) ---
        if self._rows_dimension.dimension_type == DT.MR_SUBVAR:
            return base_table_base[self._row_order]

        # --- CAT_X_CAT produces scalar table-base ---
        table_base = self._cube_result_matrix.table_base
        return table_base

    @lazyproperty
    def table_margin(self):
        """Scalar, 1D, or 2D ndarray of np.float64 weighted-N for this slice.

        This value has four distinct forms, depending on the slice dimensions:

            * MR_X_MR - 2D ndarray with a distinct table-margin value per cell.
            * MR_X - 1D ndarray of value per *row* when only rows dimension is MR.
            * X_MR - 1D ndarray of value per *column* when only columns dimension is MR.
            * CAT_X_CAT - scalar float value when slice has no MR dimension.

        """
        # NOTE: This all works out even though it doesn't include subtotals and isn't
        # assembled, because an MR dimension can't have subtotals. Because the
        # dimensionality of table-margin "follows" the MR-dimension(s), it follows that
        # a table-margin vector or matrix never has subtotals.

        base_table_margin = self._cube_result_matrix.table_margin

        # --- MR_X_MR slice produces a 2D table-margin (each cell has its own N) ---
        if (
            self._rows_dimension.dimension_type == DT.MR_SUBVAR
            and self._columns_dimension.dimension_type == DT.MR_SUBVAR
        ):
            return base_table_margin[np.ix_(self._row_order, self._column_order)]

        # --- MR_X_CAT slice produces a 1D table-margin (each row has its own N) ---
        if self._rows_dimension.dimension_type == DT.MR_SUBVAR:
            return base_table_margin[self._row_order]

        # --- CAT_X_MR slice produces a 1D table-margin (each column has its own N) ---
        if self._columns_dimension.dimension_type == DT.MR_SUBVAR:
            return base_table_margin[self._column_order]

        # --- CAT_X_CAT produces scalar table-margin ---
        return base_table_margin

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

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        # --- punt except on CAT_X_CAT (if any MR dimensions it gets np.nans) ---
        if (
            self._rows_dimension.dimension_type == DT.MR_SUBVAR
            or self._columns_dimension.dimension_type == DT.MR_SUBVAR
        ):
            return self._assemble_matrix(
                _NanSubtotals.blocks(self._cube_result_matrix, "zscores")
            )

        return self._assemble_matrix(_ZscoreSubtotals.blocks(self._cube_result_matrix))

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

    def _dimension_labels(self, dimension, order):
        """1D str ndarray of name for each vector of `dimension`.

        These are suitable for use as row/column headings; labels for subtotals appear
        in the sequence specified by `order`, such that labels are ordered to correspond
        with their respective data vector.
        """
        return np.array(
            [e.label for e in dimension.valid_elements]
            + [s.label for s in dimension.subtotals]
        )[order]

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
    def _row_subtotals(self):
        """Sequence of _Subtotal object for each inserted row."""
        return self._rows_dimension.subtotals

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
    def _blocks(self):
        """base, row and col insertion, and intersection matrices."""
        return [
            [self._base_values, self._subtotal_columns],
            [self._subtotal_rows, self._intersections],
        ]

    @lazyproperty
    def _base_counts(self):
        """2D np.float64 ndarray of weighted-count for each cell of base matrix."""
        return self._cube_result_matrix.weighted_counts

    @lazyproperty
    def _base_values(self):
        """2D ndarray of "body" values from cube-result matrix."""
        if self._measure_propname is None:
            raise NotImplementedError(
                "`%s` must implement `._base_values`" % type(self).__name__
            )
        return getattr(self._cube_result_matrix, self._measure_propname)

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
        raise NotImplementedError(
            "`%s` must implement `._subtotal_row()`" % type(self).__name__
        )

    @lazyproperty
    def _subtotal_rows(self):
        """(n_row_subtotals, n_cols) ndarray of subtotal rows."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.empty((0, self._ncols), dtype=self._dtype)

        return np.vstack(
            [self._subtotal_row(subtotal) for subtotal in self._row_subtotals]
        )

    @lazyproperty
    def _table_margin(self):
        """Scalar or ndarray table-margin of cube-result matrix."""
        return self._cube_result_matrix.table_margin


class _NanSubtotals(_BaseSubtotals):
    """Subtotal blocks for measures that cannot meaningfully be subtotaled.

    Each subtotal value (and intersection value) is `np.nan`.
    """

    def _intersection(self, row_subtotal, column_subtotal):
        """Unconditionally return np.nan for each intersection cell."""
        return np.nan

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of np.nan values."""
        return np.full(self._nrows, np.nan)

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of np.nan values."""
        return np.full(self._ncols, np.nan)


class _SumSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by np.sum() on addends, primarily counts."""

    @lazyproperty
    def _dtype(self):
        """Numpy data-type for result matrices, used for empty arrays."""
        return self._base_values.dtype

    def _intersection(self, row_subtotal, column_subtotal):
        """Sum for this row/column subtotal intersection."""
        return np.sum(self._subtotal_row(row_subtotal)[column_subtotal.addend_idxs])

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of values for `subtotal` column."""
        return np.sum(self._base_values[:, subtotal.addend_idxs], axis=1)

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of values for `subtotal` row."""
        return np.sum(self._base_values[subtotal.addend_idxs, :], axis=0)


class _ZscoreSubtotals(_BaseSubtotals):
    """Computes subtotal values for the z-score measure.

    This is only operative for a CAT_X_CAT cube-result; an MR dimension causes all
    subtotals to be computed (elsewhere) as `np.nan`.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of z-score for each cell of cube-result matrix."""
        return self._cube_result_matrix.zscores

    def _intersection(self, row_subtotal, column_subtotal):
        """Return value for intersection of `row_subtotal` and `column_subtotal`."""
        row_subtotal_counts = np.sum(
            self._base_counts[row_subtotal.addend_idxs, :], axis=0
        )

        # --- the weighted-counts version of this intersection cell ---
        intersection_count = np.sum(row_subtotal_counts[column_subtotal.addend_idxs])

        # --- subtotal-margin is scalar, like 547 ---
        subtotal_margin = np.sum(row_subtotal_counts)

        # --- opposite-margin is scalar because no MR dimensions ---
        opposite_margin = np.sum(
            self._cube_result_matrix.columns_margin[column_subtotal.addend_idxs]
        )

        # --- table_margin is scalar because no MR dimensions ---
        table_margin = self._table_margin

        # --- expected_count is scalar ---
        expected_count = opposite_margin * subtotal_margin / table_margin

        # --- residuals is scalar ---
        residuals = intersection_count - expected_count

        # --- variance is scalar (I think) ---
        variance = (
            opposite_margin
            * subtotal_margin
            * ((table_margin - opposite_margin) * (table_margin - subtotal_margin))
            / table_margin ** 3
        )

        # --- result is scalar ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return residuals / np.sqrt(variance)

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of zscore `subtotal` value."""
        # --- the weighted-counts version of this subtotal-column ---
        subtotal_counts = np.sum(self._base_counts[:, subtotal.addend_idxs], axis=1)

        # --- subtotal-margin is scalar, like 547 ---
        subtotal_margin = np.sum(subtotal_counts)

        # --- base-rows-margin is 1D because no MR dimensions ---
        opposing_margin = self._cube_result_matrix.rows_margin

        # --- table_margin is scalar because no MR dimensions ---
        table_margin = self._table_margin

        # --- expected_counts is 1D ---
        expected_counts = opposing_margin * subtotal_margin / table_margin

        # --- residuals is 1D, like: [ 11.04819 -37.72836  43.35049 -16.67031]
        residuals = subtotal_counts - expected_counts

        # --- variance is 1D, like: [12.413837 49.173948 53.980069 29.783739] ---
        variance = (
            opposing_margin
            * subtotal_margin
            * ((table_margin - opposing_margin) * (table_margin - subtotal_margin))
            / table_margin ** 3
        )

        # --- result is scalar or 1D, depending on dimensionality of residuals ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return residuals / np.sqrt(variance)

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of z-score `subtotal` value."""
        # --- the weighted-counts version of this subtotal-row ---
        subtotal_counts = np.sum(self._base_counts[subtotal.addend_idxs, :], axis=0)

        # --- subtotal-margin is scalar, like 547 ---
        subtotal_margin = np.sum(subtotal_counts)

        # --- base-cols-margin is 1D because no MR dimensions ---
        opposing_margin = self._cube_result_matrix.columns_margin

        # --- table_margin is scalar because no MR dimensions ---
        table_margin = self._table_margin

        # --- expected_counts is 1D ---
        expected_counts = opposing_margin * subtotal_margin / table_margin

        # --- residuals is 1D, like: [ 11.04819 -37.72836  43.35049 -16.67031]
        residuals = subtotal_counts - expected_counts

        # --- variance is 1D, like: [12.413837 49.173948 53.980069 29.783739] ---
        variance = (
            opposing_margin
            * subtotal_margin
            * ((table_margin - opposing_margin) * (table_margin - subtotal_margin))
            / table_margin ** 3
        )

        # --- result is scalar or 1D, depending on dimensionality of residuals ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return residuals / np.sqrt(variance)


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
        return self.weighted_counts / self.columns_margin

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
    def columns_margin(self):
        """1D/2D np.float64 ndarray of weighted-N for each column of matrix."""
        raise NotImplementedError(
            "`%s` must implement `.columns_margin`" % type(self).__name__
        )

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column."""
        raise NotImplementedError(
            "`%s` must implement `.columns_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def rows_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each matrix row/cell."""
        raise NotImplementedError(
            "`%s` must implement `.rows_base`" % type(self).__name__
        )

    @lazyproperty
    def rows_dimension(self):
        """The `Dimension` object representing row elements of this matrix."""
        return self._dimensions[0]

    @lazyproperty
    def rows_margin(self):
        """1D/2D np.int64 ndarray of weighted-N for each matrix row/cell."""
        raise NotImplementedError(
            "`%s` must implement `.rows_margin`" % type(self).__name__
        )

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row."""
        raise NotImplementedError(
            "`%s` must implement `.rows_pruning_base`" % type(self).__name__
        )

    @lazyproperty
    def table_base(self):
        """Scalar, 1D, or 2D ndarray of np.int64 unweighted-N for this slice.

        This value has four distinct forms, depending on the subclass.
        """
        raise NotImplementedError(
            "`%s` must implement `.table_base`" % type(self).__name__
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

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each valid matrix cell."""
        raise NotImplementedError(
            "`%s` must implement `.unweighted_counts" % type(self).__name__
        )

    @lazyproperty
    def weighted_counts(self):
        """2D np.float/int64 ndarray of weighted-count for each valid matrix cell.

        The cell values are np.int64 when the cube-result has no weight, in which case
        these values are the same as the unweighted-counts.
        """
        raise NotImplementedError(
            "`%s` must implement `.weighted_counts`" % type(self).__name__
        )

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        raise NotImplementedError(
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
            return np.zeros(counts.shape)

        expected_counts = rowsum * colsum / total
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)

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

    @lazyproperty
    def _valid_row_idxs(self):
        """ndarray-style index for only valid (non-missing) rows.

        Suitable for indexing a raw measure array to include only valid rows.
        """
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)


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
    def columns_base(self):
        """1D ndarray of np.int64 unweighted-N for each matrix column."""
        return np.sum(self.unweighted_counts, axis=0)

    @lazyproperty
    def columns_margin(self):
        """1D ndarray of np.float64 (or np.int64) weighted N for each matrix column."""
        return np.sum(self.weighted_counts, axis=0)

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each column.
        """
        return np.sum(self._unweighted_counts, axis=0)

    @lazyproperty
    def rows_base(self):
        """1D ndarray of np.int64 unweighted-N for each matrix row."""
        return np.sum(self.unweighted_counts, axis=1)

    @lazyproperty
    def rows_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each matrix row.

        Values are `np.int64` when the source cube-result is not weighted.
        """
        return np.sum(self.weighted_counts, axis=1)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        Because this matrix has no MR dimension, this is simply the sum of unweighted
        counts for each row.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def table_base(self):
        """np.int64 count of actual respondents who answered both questions.

        Each dimension of a CAT_X_CAT matrix represents a categorical question. Only
        responses that include answers to both those questions appear as entries in the
        valid elements of those dimensions. The sum total of all valid answers is the
        sample size, aka "N" or "base".
        """
        return np.sum(self.unweighted_counts)

    @lazyproperty
    def table_margin(self):
        """Scalar np.float/int64 weighted-N for overall table.

        This is the weighted count of respondents who provided a valid response to
        both questions. Because both dimensions are CAT, the table-margin value is the
        same for all cells of the matrix. Value is np.int64 when source cube-result is
        unweighted.
        """
        return np.sum(self._weighted_counts)

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

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        counts = self._weighted_counts

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
    def rows_base(self):
        """2D np.int64 ndarray of row-wise unweighted-N for this matrix.

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
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        These values include both the selected and unselected counts of the MR columns
        dimension.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """1D np.int64 unweighted N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table base.
        """
        # --- unweighted-counts is (nrows, ncols, selected/not) so axis 1 is preserved
        # --- to provide a distinct value for each MR subvar.
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table margin.
        """
        # --- weighted-counts is (rows, cols, selected/not) so axis 1 is preserved to
        # --- provide a distinct value for each MR subvar.
        return np.sum(self._weighted_counts, axis=(0, 2))

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

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean each cell's
        value is.
        """
        return self._array_type_std_res(
            self._weighted_counts[:, :, 0],
            self.table_margin,
            np.sum(self._weighted_counts, axis=2),
            np.sum(self._weighted_counts[:, :, 0], axis=0),
        )

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
    def table_base(self):
        """1D np.int64 ndarray of unweighted N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct base, since not all of
        the multiple responses were necessarily offered to all respondents. The base for
        each row indicates the number of respondents who were offered that option.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct table margin, since not
        all of the multiple responses were necessarily offered to all respondents. The
        table-margin for each row indicates the weighted number of respondents who were
        offered that option.

        The values are np.int64 when the source cube-result is unweighted.
        """
        return np.sum(self._weighted_counts, axis=(1, 2))

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

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of z-score for each matrix cell."""
        return self._array_type_std_res(
            self._weighted_counts[:, 0, :],
            self.table_margin[:, None],
            np.sum(self._weighted_counts[:, 0, :], axis=1)[:, None],
            np.sum(self._weighted_counts, axis=1),
        )

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
        """1D np.int64 ndarray of unweighted-N for each matrix column.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the row MR; only selecteds are considered for
        the columns dimension.
        """
        return np.sum(self._unweighted_counts[:, :, :, 0], axis=(0, 1))

    @lazyproperty
    def rows_base(self):
        """2D np.int64 ndarray of unweighted-N for this matrix.

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
        """1D np.int64 ndarray of unweighted-N for each matrix row.

        Because both dimensions of this matrix are MR, this includes both selected and
        unselected counts, but only for the column MR; only selecteds are considered for
        the rows dimension.
        """
        return np.sum(self._unweighted_counts[:, 0, :, :], axis=(1, 2))

    @lazyproperty
    def table_base(self):
        """2D np.int64 ndarray of distinct unweighted N for each cell of matrix.

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
        """2D np.float/int64 ndarray of weighted N for each cell of matrix.

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

    @lazyproperty
    def zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean each cell's
        value is.
        """
        return self._array_type_std_res(
            self._weighted_counts[:, 0, :, 0],
            self.table_margin,
            np.sum(self._weighted_counts, axis=3)[:, 0, :],
            np.sum(self._weighted_counts, axis=1)[:, :, 0],
        )

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
