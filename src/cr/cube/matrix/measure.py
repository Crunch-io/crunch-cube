# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

import numpy as np

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
    def column_unweighted_bases(self):
        """_ColumnUnweightedBases measure object for this cube-result."""
        return _ColumnUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_weighted_bases(self):
        """_ColumnWeightedBases measure object for this cube-result."""
        return _ColumnWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_unweighted_bases(self):
        """_RowUnweightedBases measure object for this cube-result."""
        return _RowUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_weighted_bases(self):
        """_RowWeightedBases measure object for this cube-result."""
        return _RowWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_unweighted_bases(self):
        """_TableUnweightedBases measure object for this cube-result."""
        return _TableUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_weighted_bases(self):
        """_TableWeightedBases measure object for this cube-result."""
        return _TableWeightedBases(self._dimensions, self, self._cube_measures)

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
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values. This default implementation assumes the
        subclass will implement each block separately.
        """
        return [
            [self._base_values, self._subtotal_columns],
            [self._subtotal_rows, self._intersections],
        ]

    @lazyproperty
    def _base_values(self):
        """2D np.int64 ndarray of column-wise proportions denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `._base_values`" % type(self).__name__
        )

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `._intersections`" % type(self).__name__
        )

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.int64 ndarray of inserted column proportions denominator value.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `._subtotal_columns`" % type(self).__name__
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.int64 ndarray of column-proportions denominator for subtotal rows.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `._subtotal_rows`" % type(self).__name__
        )

    @lazyproperty
    def _unweighted_cube_counts(self):
        """_BaseUnweightedCubeCounts subclass instance for this measure.

        Provides cube measures associated with unweighted counts, including
        unweighted-counts and cell, vector, and table bases.
        """
        return self._cube_measures.unweighted_cube_counts

    @lazyproperty
    def _weighted_cube_counts(self):
        """_BaseWeightedCubeCounts subclass instance for this measure.

        Provides cube measures associated with weighted counts, including
        weighted-counts and cell, vector, and table margins.
        """
        return self._cube_measures.weighted_cube_counts


class _ColumnUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the column-bases measure for a matrix.

    Column-bases is a 2D np.int64 ndarray of unweighted-N "basis" for each matrix cell.
    Depending on the dimensionality of the underlying cube-result some or all of these
    values may be the same.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.int64 ndarray of column-wise proportions denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        return self._unweighted_cube_counts.column_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        # --- the strategy here is to broadcast one row of the subtotal-columns to the
        # --- shape of the intersections. This works in the X_CAT case because each row
        # --- of subtotal-columns is the same. In the X_MR case there can be no subtotal
        # --- columns and so it is just an empty row that is broadcast.
        shape = SumSubtotals.intersections(self._base_values, self._dimensions).shape
        columns_base = self._subtotal_columns[0]
        return np.broadcast_to(columns_base, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.int64 ndarray of inserted column proportions denominator value.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- Summing works on columns because column-proportion denominators add along
        # --- that axis, like column-proportions denominator of a subtotal of two
        # --- columns each with a base of 25 is indeed 50. This doesn't work on rows
        # --- though, see below. This wouldn't work on MR-columns but there can be no
        # --- subtotal columns on an MR dimension (X_MR slice) so that case never
        # --- arises.
        return SumSubtotals.subtotal_columns(self._base_values, self._dimensions)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.int64 ndarray of column-proportions denominator for subtotal rows.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- the strategy here is simply to broadcast the columns_base to the shape of
        # --- the subtotal-rows matrix because a subtotal-row value has the same
        # --- column-base as all other cells in that column. Note that this initial
        # --- subtotal-rows matrix is used only for its shape (and when it is empty)
        # --- because it computes the wrong cell values for this case.
        subtotal_rows = SumSubtotals.subtotal_rows(self._base_values, self._dimensions)

        # --- in the "no-row-subtotals" case, short-circuit with a (0, ncols) array
        # --- return value, both because that is the right answer, but also because the
        # --- non-empty columns-base array cannot be broadcast into that shape. dtype
        # --- must be `int` to avoid changing type of assembled array to float.
        if subtotal_rows.shape[0] == 0:
            return subtotal_rows

        return np.broadcast_to(
            self._unweighted_cube_counts.columns_base, subtotal_rows.shape
        )


class _ColumnWeightedBases(_BaseSecondOrderMeasure):
    """Provides the column-weighted-bases measure for a matrix.

    Column-weighted-bases is a 2D np.float64 ndarray of the weighted "base", aka.
    "denominator" for the column-proportion of each cell. This measure is generally only
    interesting where the rows dimension is MR, causing each column cell to have a
    distinct proportions denominator. In the CAT_X case, the denominator is the same for
    each cell in a particular column.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of column-proportion denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        return self._weighted_cube_counts.column_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth (and final) "block" required by the assembler.
        """
        # --- the strategy here is to broadcast one row of the subtotal-columns to the
        # --- shape of the intersections. This works in the X_CAT case because each row
        # --- of subtotal-columns is the same. In the X_MR case there can be no subtotal
        # --- columns so an empty row is broadcast to shape.
        shape = SumSubtotals.intersections(self._base_values, self._dimensions).shape
        intersections_row = self._subtotal_columns[0]
        return np.broadcast_to(intersections_row, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted-column denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- Summing works on columns because column-proportion denominators add along
        # --- that axis.
        return SumSubtotals.subtotal_columns(self._base_values, self._dimensions)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of inserted-row denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- the strategy here is simply to broadcast the columns_base to the shape of
        # --- the subtotal-rows matrix because these don't add. Note that this initial
        # --- subtotal-rows matrix is used only for its shape because it computes the
        # --- wrong values.
        subtotal_rows = SumSubtotals.subtotal_rows(self._base_values, self._dimensions)
        # --- in the "no-row-subtotals" case, short-circuit with a (0, ncols) array
        # --- return value, both because that is the right answer, but also because the
        # --- non-empty columns-base array cannot be broadcast into that shape.
        if subtotal_rows.shape[0] == 0:
            return subtotal_rows

        return np.broadcast_to(
            self._weighted_cube_counts.columns_margin, subtotal_rows.shape
        )


class _RowUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the row-unweighted-bases measure for a matrix.

    row-unweighted-bases is a 2D np.int64 ndarray of the unweighted row-proportion
    denominator (base) for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.int64 ndarray of row-proportions denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        return self._unweighted_cube_counts.row_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- the strategy here is to broadcast one column of the subtotal-rows to the
        # --- shape of the intersections. This works in the CAT_X case because each
        # --- column of subtotal-rows is the same. In the MR_X case there can be no
        # --- subtotal rows so just an empty column is broadcast.
        shape = SumSubtotals.intersections(self._base_values, self._dimensions).shape
        intersection_column = self._subtotal_rows[:, 0]
        return np.broadcast_to(intersection_column[:, None], shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.int64 ndarray of column-subtotal row-proportions denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- the strategy here is simply to broadcast the rows_base to the shape of
        # --- the subtotal-columns matrix because these don't add. Note that this
        # --- initial subtotal-columns matrix is used only for its shape because it
        # --- computes the wrong values.
        subtotal_columns = SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions
        )
        # --- in the "no-column-subtotals" case, short-circuit with an (nrows, 0) array
        # --- return value, both because that is the right answer, but also because the
        # --- non-empty rows-base array cannot be broadcast into that "empty" shape.
        if subtotal_columns.shape[1] == 0:
            return subtotal_columns

        return np.broadcast_to(
            self._unweighted_cube_counts.rows_base[:, None], subtotal_columns.shape
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.int64 ndarray of row-subtotal row-proportions denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- Summing works on rows because row-proportion denominators add along this
        # --- axis. This wouldn't work on MR-rows but there can be no subtotals on an
        # --- MR rows dimension (or any MR dimension) so that case never arises.
        return SumSubtotals.subtotal_rows(self._base_values, self._dimensions)


class _RowWeightedBases(_BaseSecondOrderMeasure):
    """Provides the row-weighted-bases measure for a matrix.

    row-weighted-bases is a 2D np.float64 ndarray of the (weighted) row-proportion
    denominator (base) for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of row-proportion denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        return self._weighted_cube_counts.row_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- to broadcast one column of the subtotal-rows to the shape of the
        # --- intersections. This works in the CAT_X case because each column of
        # --- subtotal-rows is the same. In the MR_X case there can be no subtotal rows
        # --- so an empty column is broadcast.
        shape = SumSubtotals.intersections(self._base_values, self._dimensions).shape
        intersection_column = self._subtotal_rows[:, 0]
        return np.broadcast_to(intersection_column[:, None], shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of column-subtotal row-proportions denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- broadcast the rows_margin to the shape of the subtotal-columns matrix
        # --- because rows-margin doesn't add in this direction. Note this initial
        # --- subtotal-columns matrix is used only for its shape because it computes
        # --- the wrong values.
        subtotal_columns = SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions
        )
        # --- in the "no-column-subtotals" case, short-circuit with an (nrows, 0) array
        # --- return value, both because that is the right answer, but also because the
        # --- non-empty rows-margin array cannot be broadcast into that "empty" shape.
        if subtotal_columns.shape[1] == 0:
            return subtotal_columns

        return np.broadcast_to(
            self._weighted_cube_counts.rows_margin[:, None], subtotal_columns.shape
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of row-subtotal row-proportions denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- Summing works on rows because row-proportion denominators add along this
        # --- axis. This wouldn't work on MR-rows but there can be no subtotals on an
        # --- MR dimension (MR_X slice) so that case never arises.
        return SumSubtotals.subtotal_rows(self._base_values, self._dimensions)


class _TableUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the table-unweighted-bases measure for a matrix.

    table-unweighted-bases is a 2D np.int64 ndarray of the denominator, or "base" of the
    unweighted table-proportion for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.int64 ndarray of unweighted table-proportion denominator for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        return self._unweighted_cube_counts.table_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- There are only two cases. Note that intersections can only occur when
        # --- there are *both* row-subtotals and column-subtotals. Since an MR dimension
        # --- can have no subtotals, this rules out all but the CAT_X_CAT case. We need
        # --- the shape of the intersections array for all cases, and we get it from
        # --- SumSubtotals. Note that in general, the summed values it returns are wrong
        # --- for this case, but the shape and dtype are right and when empty, it gives
        # --- the right answer directly.
        intersections = SumSubtotals.intersections(self._base_values, self._dimensions)
        shape = intersections.shape

        # --- if intersections is empty, return it, because it is the right shape and
        # --- because the table-base cannot be broadcast to an empty shape if it is an
        # --- array value (which it will be when the slice has an MR dimension).
        if 0 in shape:
            return intersections

        # --- Otherwise, we know that table-base must be a scalar (only CAT_X_CAT can
        # --- have intersections, so fill the intersections shape with that scalar.
        return np.broadcast_to(self._unweighted_cube_counts.table_base, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.int64 ndarray of inserted-column table-proportions denominator value.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- There are three cases. For all of them we need the shape of the
        # --- subtotal-rows array, which we can get from SumSubtotals. Note that in
        # --- general, the summed values it returns are wrong for this case, but the
        # --- shape and dtype are right and when empty, it gives the right answer
        # --- directly.
        subtotal_columns = SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions
        )

        # --- Case 1: in the "no-row-subtotals" case, short-circuit with an (nrows, 0)
        # --- array return value, both because that is the right answer, but also
        # --- because the non-empty table-base value cannot be broadcast into that
        # --- shape. Note that because an X_MR cube can have no column subtotals, this
        # --- automatically takes care of the CAT_X_MR and MR_X_MR cases.
        if subtotal_columns.shape[1] == 0:
            return subtotal_columns

        table_base = self._unweighted_cube_counts.table_base
        shape = subtotal_columns.shape

        # TODO: Resolve this abstraction leakage from _BaseUnweightedCounts where the
        # table-margin (for MR_X_CAT) is a (column) vector instead of a scalar and
        # column subtotals can still occur. When `.table_base` becomes a min-max range,
        # we might need something like `.table_base_column` that is (nrows, 1) so this
        # "rotation" is performed in `_MrXCatUnweightedCounts`. This same shape
        # diversity happens with `._subtotal_rows` below, but since that vector is a row
        # it is handled without special-casing.

        # --- Case 2: in the "vector table-base" (MR_X_CAT) case, rotate the vector into
        # --- a "column" and broadcast it into the subtotal-columns shape.
        if isinstance(table_base, np.ndarray):
            return np.broadcast_to(table_base[:, None], shape)

        # --- Case 3: in the "scalar table-base" (CAT_X_CAT) case, simply fill the
        # --- subtotal-columns shape with that scalar.
        return np.broadcast_to(table_base, shape)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.int64 ndarray of inserted-row table-proportions denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- There are three cases. For all of them we need the shape of the
        # --- subtotal-rows array, which we can get from SumSubtotals.subtotal_rows().
        # --- Note that in general, the summed values it returns are wrong for this
        # --- case, but the shape and dtype are right and when empty, it gives the
        # --- right answer directly.
        subtotal_rows = SumSubtotals.subtotal_rows(self._base_values, self._dimensions)

        # --- Case 1: in the "no-row-subtotals" case, short-circuit with a (0, ncols)
        # --- array return value, both because that is the right answer, but also
        # --- because the non-empty table-base value cannot be broadcast into that
        # --- shape. Note that because an MR_X cube can have no row subtotals, this
        # --- automatically takes care of the MR_X_CAT and MR_X_MR case.
        if subtotal_rows.shape[0] == 0:
            return subtotal_rows

        # --- Case 2 & 3: in the "scalar table-base" (CAT_X_CAT) case, fill the
        # --- subtotal-rows shape with the scalar value. The same numpy operation also
        # --- works for the vector table-base (CAT_X_MR) case because the vector
        # --- table-base is a "row" of base values.
        return np.broadcast_to(
            self._unweighted_cube_counts.table_base, subtotal_rows.shape
        )


class _TableWeightedBases(_BaseSecondOrderMeasure):
    """Provides the table-weighted-bases measure for a matrix.

    table-weighted-bases is a 2D np.float64 ndarray of the (weighted) table-proportion
    denominator (base) for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of weighted table-proportion denominator per cell."""
        return self._weighted_cube_counts.table_bases

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- There are only two cases. Note that intersections can only occur when
        # --- there are *both* row-subtotals and column-subtotals. Since an MR dimension
        # --- can have no subtotals, this rules out all but the CAT_X_CAT case. We need
        # --- the shape of the intersections array for both cases, and we get it from
        # --- SumSubtotals. Note that in general, the summed values it returns are wrong
        # --- for this case, but the shape and dtype are right and when empty, it gives
        # --- the right answer directly.
        intersections = SumSubtotals.intersections(self._base_values, self._dimensions)
        shape = intersections.shape

        # --- if intersections is empty, return it, both because it is the correct
        # --- return value (it has the right "empty" shape) and because an array
        # --- table-margin (occuring when slice has an MR dimension) cannot be broadcast
        # --- to an empty shape.
        if 0 in shape:
            return intersections

        # --- Otherwise, we know that table-base must be a scalar (only CAT_X_CAT can
        # --- have intersections), so fill the intersections shape with that scalar.
        return np.broadcast_to(self._weighted_cube_counts.table_margin, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted-column table-proportion denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- There are three cases. For all of them we need the shape of the
        # --- subtotal-rows array, which we can get from SumSubtotals. Note that in
        # --- general, the summed values it returns are wrong for this case, but the
        # --- shape and dtype are right and when empty, it gives the right answer
        # --- directly.
        subtotal_columns = SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions
        )

        # --- Case 1: in the "no-row-subtotals" case, short-circuit with an (nrows, 0)
        # --- array return value, both because that is the right answer, but also
        # --- because the non-empty table-base value cannot be broadcast into that
        # --- shape. Note that because an X_MR cube can have no column subtotals, this
        # --- automatically takes care of the CAT_X_MR and MR_X_MR cases.
        if subtotal_columns.shape[1] == 0:
            return subtotal_columns

        table_margin = self._weighted_cube_counts.table_margin
        shape = subtotal_columns.shape

        # TODO: Resolve this abstraction leakage from _BaseWeightedCounts where the
        # table-margin (for MR_X_CAT) is a (column) vector instead of a scalar and
        # therefore cannot be directly broadcast. When `.table_margin` becomes a min-max
        # range, we might need something like `.table_margin_column` that is (nrows, 1)
        # such that this "rotation" is performed in `_MrXCatWeightedCounts`. This same
        # shape diversity happens with `._subtotal_rows` below, but since that vector is
        # a row it is handled without special-casing.

        # --- Case 2: in the "vector table-base" (MR_X_CAT) case, rotate the vector into
        # --- a "column" and broadcast it into the subtotal-columns shape.
        if isinstance(table_margin, np.ndarray):
            return np.broadcast_to(table_margin[:, None], shape)

        # --- Case 3: in the "scalar table-base" (CAT_X_CAT) case, simply fill the
        # --- subtotal-columns shape with that scalar.
        return np.broadcast_to(table_margin, shape)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of inserted-row table-proportion denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- There are three cases. For all of them we need the shape of the
        # --- subtotal-rows array, which we can get from SumSubtotals.subtotal_rows().
        # --- Note that in general, the summed values it returns are wrong for this
        # --- case, but the shape and dtype are right and when empty, it gives the
        # --- right answer directly.
        subtotal_rows = SumSubtotals.subtotal_rows(self._base_values, self._dimensions)

        # --- Case 1: in the "no-row-subtotals" case, short-circuit with a (0, ncols)
        # --- array return value, both because that is the right answer, but also
        # --- because the non-empty table-base value cannot be broadcast into that
        # --- shape. Note that because an MR_X cube can have no row subtotals, this
        # --- automatically takes care of the MR_X_CAT and MR_X_MR case.
        if subtotal_rows.shape[0] == 0:
            return subtotal_rows

        # --- Case 2 & 3: in the "scalar table-base" (CAT_X_CAT) case, fill the
        # --- subtotal-rows shape with the scalar value. The same numpy operation also
        # --- works for the vector table-base (CAT_X_MR) case because the vector
        # --- table-base is a "row" of base values.
        return np.broadcast_to(
            self._weighted_cube_counts.table_margin, subtotal_rows.shape
        )


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
        return SumSubtotals.blocks(
            self._weighted_cube_counts.weighted_counts, self._dimensions
        )
