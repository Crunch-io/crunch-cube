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
            return np.empty((self._nrows, 0))

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
            return np.empty((0, self._ncols))

        return np.vstack(
            [self._subtotal_row(subtotal) for subtotal in self._row_subtotals]
        )


class NanSubtotals(_BaseSubtotals):
    """Subtotal blocks for measures that cannot meaningfully be subtotaled.

    Each subtotal value (and intersection value) is `np.nan`.
    """

    filler = np.nan

    def _intersection(self, row_subtotal, column_subtotal):
        """Unconditionally return np.nan for each intersection cell."""
        return self.filler

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of np.nan values."""
        return np.full(self._nrows, self.filler)

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of np.nan values."""
        return np.full(self._ncols, self.filler)


class NoneSubtotals(NanSubtotals):
    filler = None


class SumSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by adding and subtracting terms for subtotals.

    In addition to `base_values` and `dimensions`, `SumSubtotals` have
    properties for `diff_cols_nan` and `diff_rows_nan` which allow for columns/
    rows where subtotals have subtrahends to be overridden with np.nan. These are
    used for measures such as bases (and measures derived from bases, like proportions)
    that are not computed along the same direction as a subtotal difference. Example:
    a subtotal difference in the row dimension does not have a valid row proportion.

    One way of thinking about this is that when calculating proportions, users are
    only interested in the direction of proportions where the difference is equal
    to the sum of the addends minus the sum of the subtrahends. But when calculating
    proportions along a row or column, the proportions are only additive in one
    direction, so subtotal differences in the other direction don't work. For example,
    when you go along a row and you are calculating column percents, each column
    has a different base, so the proportions don't add up.

    Example showing column percents. We do want to calculate column proportion for
    `a-b`, but not `c-d` because `c-d` is a subtotal difference in the column
    dimension.
    ```
    |     | c-d | c   | d   | c+d  |
    |-----|-----|-----|-----|------|
    | a-b | -   | 20  | 50  | 25   |
    | a   | -   | 60  | 75  | 62.5 |
    | b   | -   | 40  | 25  | 37.5 |
    | a+b | -   | 100 | 100 | 100  |
    ```

    Another way to think about it is that a "base" is a positive thing, so you can't
    really subtract out the subtrahends. One option would be to add both the addends and
    the subtrahends, but ultimately we decided that users would be confused by
    by these values, and they really only want to see the base and proportions in the
    opposing direction. Therefore we set the corresponding direction to nan.
    """

    def __init__(
        self, base_values, dimensions, diff_cols_nan=False, diff_rows_nan=False
    ):
        super(SumSubtotals, self).__init__(base_values, dimensions)
        self._diff_cols_nan = diff_cols_nan
        self._diff_rows_nan = diff_rows_nan

    @classmethod
    def blocks(cls, base_values, dimensions, diff_cols_nan=False, diff_rows_nan=False):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.

        Keyword arguments:
        `diff_cols_nan` -- Overrides subtotal differences in the columns direction eg for
        column bases (default False)
        `diff_rows_nan` -- Overrides subtotal differences in the rows direction eg for
        row bases (default False)
        """
        return cls(base_values, dimensions, diff_cols_nan, diff_rows_nan)._blocks

    @classmethod
    def intersections(
        cls, base_values, dimensions, diff_cols_nan=False, diff_rows_nan=False
    ):
        """Return (n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.

        Keyword arguments:
        `diff_cols_nan` -- Overrides subtotal differences in the columns direction eg for
        column bases (default False)
        `diff_rows_nan` -- Overrides subtotal differences in the rows direction eg for
        row bases (default False)
        """
        return cls(base_values, dimensions, diff_cols_nan, diff_rows_nan)._intersections

    @classmethod
    def subtotal_columns(
        cls, base_values, dimensions, diff_cols_nan=False, diff_rows_nan=False
    ):
        """Return (n_base_rows, n_col_subtotals) ndarray of subtotal columns.

        Keyword arguments:
        `diff_cols_nan` -- Overrides subtotal differences in the columns direction eg for
        column bases (default False)
        `diff_rows_nan` -- Overrides subtotal differences in the rows direction eg for
        row bases (default False)
        """
        return cls(
            base_values, dimensions, diff_cols_nan, diff_rows_nan
        )._subtotal_columns

    @classmethod
    def subtotal_rows(
        cls, base_values, dimensions, diff_cols_nan=False, diff_rows_nan=False
    ):
        """Return (n_row_subtotals, n_base_cols) ndarray of subtotal rows.

        Keyword arguments:
        `diff_cols_nan` -- Overrides subtotal differences in the columns direction eg for
        column bases (default False)
        `diff_rows_nan` -- Overrides subtotal differences in the rows direction eg for
        row bases (default False)
        """
        return cls(base_values, dimensions, diff_cols_nan, diff_rows_nan)._subtotal_rows

    def _intersection(self, row_subtotal, column_subtotal):
        """Sum and Diff for this row/column subtotal intersection."""

        col_has_subs = len(column_subtotal.subtrahend_idxs) > 0
        row_has_subs = len(row_subtotal.subtrahend_idxs) > 0

        if (
            # --- Intersections of subtotal differences are undefined ---
            (col_has_subs and row_has_subs)
            # --- Also need to respect diff_cols_nan/diff_rows_nan
            or (col_has_subs and self._diff_cols_nan)
            or (row_has_subs and self._diff_rows_nan)
        ):
            return np.nan

        addend_sum = np.sum(
            self._subtotal_row(row_subtotal)[column_subtotal.addend_idxs]
        )
        subtrahend_sum = np.sum(
            self._subtotal_row(row_subtotal)[column_subtotal.subtrahend_idxs]
        )
        return addend_sum - subtrahend_sum

    def _subtotal_column(self, subtotal):
        """Return (n_rows,) ndarray of values for `subtotal` column."""
        if self._diff_cols_nan and len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._base_values.shape[0], np.nan)

        addend_sum = np.sum(self._base_values[:, subtotal.addend_idxs], axis=1)
        subtrahend_sum = np.sum(self._base_values[:, subtotal.subtrahend_idxs], axis=1)
        return addend_sum - subtrahend_sum

    def _subtotal_row(self, subtotal):
        """Return (n_cols,) ndarray of values for `subtotal` row."""
        if self._diff_rows_nan and len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._base_values.shape[1], np.nan)

        addend_sum = np.sum(self._base_values[subtotal.addend_idxs, :], axis=0)
        subtrahend_sum = np.sum(self._base_values[subtotal.subtrahend_idxs, :], axis=0)
        return addend_sum - subtrahend_sum


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
        # --- Cannot calculate if there are any subtrahends ---
        if (
            len(row_subtotal.subtrahend_idxs) > 0
            or len(column_subtotal.subtrahend_idxs) > 0
        ):
            return np.nan

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
        # --- Cannot calculate if there are any subtrahends ---
        if len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._nrows, np.nan)

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
        # --- Cannot calculate if there are any subtrahends ---
        if len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._ncols, np.nan)

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
        # --- Cannot calculate if there are any subtrahends ---
        if (
            len(row_subtotal.subtrahend_idxs) > 0
            or len(column_subtotal.subtrahend_idxs) > 0
        ):
            return np.nan

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
        # --- Cannot calculate if there are any subtrahends ---
        if len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._nrows, np.nan)

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
        # --- Cannot calculate if there are any subtrahends ---
        if len(subtotal.subtrahend_idxs) > 0:
            return np.full(self._ncols, np.nan)

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
