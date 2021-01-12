# encoding: utf-8

"""Provides subtotalling services according to a variety of strategies.

A matrix can have inserted vectors (row or column subtotals) that summarize two or more
other vectors by ostensibly "adding" them. Simple addition works for counts, but more
sophisticated methods are required for higher-order measures.

This module provides the various strategies required for computing subtotals and is
primarily used by measure objects as a collaborator to handle this aspect.
"""

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty


class _BaseSubtotals(object):
    """Base class for Subtotals objects."""

    def __init__(self, base_values, dimensions):
        self._base_values = base_values
        self._dimensions = dimensions

    @classmethod
    def blocks(cls, base_values, dimensions):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.
        """
        return cls(base_values, dimensions)._blocks

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
        return self._columns_dimension.subtotals

    @lazyproperty
    def _columns_dimension(self):
        """Dimension object for matrix columns."""
        return self._dimensions[1]

    @lazyproperty
    def _dtype(self):
        """Numpy data-type for result matrices, used for empty arrays."""
        return np.float64

    def _intersection(self, row_subtotal, column_subtotal):
        """Value for this row/column subtotal intersection."""
        raise NotImplementedError(
            "`%s` must implement `._intersection()`" % type(self).__name__
        )  # pragma: no cover

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
        return self._rows_dimension.subtotals

    @lazyproperty
    def _rows_dimension(self):
        """Dimension object for rows of matrix."""
        return self._dimensions[0]

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of values for `subtotal` column."""
        raise NotImplementedError(
            "`%s` must implement `._subtotal_column()`" % type(self).__name__
        )  # pragma: no cover

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
        )  # pragma: no cover

    @lazyproperty
    def _subtotal_rows(self):
        """(n_row_subtotals, n_cols) ndarray of subtotal rows."""
        subtotals = self._row_subtotals

        if len(subtotals) == 0:
            return np.empty((0, self._ncols), dtype=self._dtype)

        return np.vstack(
            [self._subtotal_row(subtotal) for subtotal in self._row_subtotals]
        )


class NanSubtotals(_BaseSubtotals):
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


class SumSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by np.sum() on addends, primarily counts."""

    @classmethod
    def intersections(cls, base_values, dimensions):
        """Return (n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        return cls(base_values, dimensions)._intersections

    @classmethod
    def subtotal_columns(cls, base_values, dimensions):
        """Return (n_base_rows, n_col_subtotals) ndarray of subtotal columns."""
        return cls(base_values, dimensions)._subtotal_columns

    @classmethod
    def subtotal_rows(cls, base_values, dimensions):
        """Return (n_row_subtotals, n_base_cols) ndarray of subtotal rows."""
        return cls(base_values, dimensions)._subtotal_rows

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


class TableStdErrSubtotals(_BaseSubtotals):
    """Computes subtotal values for the table-stderrs measure."""

    def __init__(self, base_values, dimensions, cube_result_matrix):
        super(TableStdErrSubtotals, self).__init__(base_values, dimensions)
        self._cube_result_matrix = cube_result_matrix

    @classmethod
    def blocks(cls, base_values, dimensions, cube_result_matrix):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.
        """
        return cls(base_values, dimensions, cube_result_matrix)._blocks

    @lazyproperty
    def _base_counts(self):
        """2D np.float64 ndarray of weighted-count for each cell of base matrix."""
        return self._cube_result_matrix.weighted_counts

    def _intersection(self, row_subtotal, column_subtotal):
        """Return value for intersection of `row_subtotal` and `column_subtotal`."""
        # --- column of base subtotal counts is 1D, like [44 148 283 72] ---
        # --- the inserted-row for counts measure, like: [159 172 107 272] ---
        row_subtotal_counts = np.sum(
            self._base_counts[row_subtotal.addend_idxs, :], axis=0
        )
        intersection_count = np.sum(row_subtotal_counts[column_subtotal.addend_idxs])

        # --- `p` is intersection-table-proportion ---
        p = intersection_count / self._table_margin
        table_proportion_variance = p * (1 - p)

        return np.sqrt(table_proportion_variance / self._table_margin)

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of table-stderr `subtotal` value."""
        # --- column of base subtotal counts is 1D, like [44 148 283 72] ---
        subtotal_counts = np.sum(
            self._base_counts[:, subtotal.addend_idxs],
            axis=1,
        )

        # --- `p` is subtotal-table-proportions ---
        p = subtotal_counts / self._table_margin
        table_proportion_variance = p * (1 - p)

        return np.sqrt(table_proportion_variance / self._table_margin)

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of table-stderr `subtotal` value."""
        # --- row of base subtotal counts is 1D, like [435 392 260 162] ---
        subtotal_counts = np.sum(self._base_counts[subtotal.addend_idxs, :], axis=0)

        # --- `p` is subtotal-table-proportions ---
        p = subtotal_counts / self._table_margin
        table_proportion_variance = p * (1 - p)

        return np.sqrt(table_proportion_variance / self._table_margin)

    @lazyproperty
    def _table_margin(self):
        """Scalar or ndarray table-margin of cube-result matrix."""
        return self._cube_result_matrix.table_margin


class ZscoreSubtotals(_BaseSubtotals):
    """Computes subtotal values for the z-score measure.

    This is only operative for a CAT_X_CAT cube-result; an MR dimension causes all
    subtotals to be computed (elsewhere) as `np.nan`.
    """

    def __init__(self, base_values, dimensions, cube_result_matrix):
        super(ZscoreSubtotals, self).__init__(base_values, dimensions)
        self._cube_result_matrix = cube_result_matrix

    @classmethod
    def blocks(cls, base_values, dimensions, cube_result_matrix):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.
        """
        return cls(base_values, dimensions, cube_result_matrix)._blocks

    @lazyproperty
    def _base_counts(self):
        """2D np.float64 ndarray of weighted-count for each cell of base matrix."""
        return self._cube_result_matrix.weighted_counts

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

    @lazyproperty
    def _table_margin(self):
        """Scalar or ndarray table-margin of cube-result matrix."""
        return self._cube_result_matrix.table_margin
