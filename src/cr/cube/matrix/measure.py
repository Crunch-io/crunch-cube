# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

import numpy as np
from scipy.stats import t

from cr.cube.enums import DIMENSION_TYPE as DT, MARGINAL_ORIENTATION as MO
from cr.cube.matrix.cubemeasure import CubeMeasures
from cr.cube.matrix.subtotals import (
    SumSubtotals,
    NanSubtotals,
    NoneSubtotals,
    ZscoreSubtotals,
)
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
    def column_comparable_counts(self):
        """_ColumnComparableCounts measure object for this cube-result."""
        return _ColumnComparableCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_proportions(self):
        """_ColumnProportions measure object for this cube-result."""
        return _ColumnProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_share_sum(self):
        """_ColumnShareSum measure object for this cube-result"""
        return _ColumnShareSum(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_unweighted_bases(self):
        """_ColumnUnweightedBases measure object for this cube-result."""
        return _ColumnUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_weighted_bases(self):
        """_ColumnWeightedBases measure object for this cube-result."""
        return _ColumnWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def columns_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        return self._cube_measures.unweighted_cube_counts.columns_base

    @lazyproperty
    def columns_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        return self._cube_measures.unweighted_cube_counts.columns_pruning_base

    @lazyproperty
    def columns_scale_mean(self):
        """_ScaleMean for columns measure object for this cube-result."""
        return _ScaleMean(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_scale_mean_stddev(self):
        """_ScaleMeanStddev for columns measure object for this cube-result."""
        return _ScaleMeanStddev(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_scale_median(self):
        """_ScaleMedian for columns measure object for this cube-result."""
        return _ScaleMedian(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def means(self):
        """_Means measure object for this cube-result"""
        return _Means(self._dimensions, self, self._cube_measures)

    def pairwise_indices(self, alpha, only_larger):
        """_PairwiseIndices measure object for this cube-result"""
        return _PairwiseIndices(
            self._dimensions, self, self._cube_measures, alpha, only_larger
        )

    def pairwise_means_indices(self, alpha, only_larger):
        """_PairwiseIndices measure object for this cube-result"""
        return _PairwiseMeansIndices(
            self._dimensions, self, self._cube_measures, alpha, only_larger
        )

    def pairwise_p_vals_for_subvar(self, subvar_idx):
        """_PairwiseSigPVals measure object for this cube-result and selected subvar"""
        return _PairwiseSigPVals(
            self._dimensions, self, self._cube_measures, subvar_idx
        )

    def pairwise_t_stats_for_subvar(self, subvar_idx):
        """_PairwiseSigTStats measure object for this cube-result and selected subvar"""
        return _PairwiseSigTStats(
            self._dimensions, self, self._cube_measures, subvar_idx
        )

    def pairwise_significance_means_p_vals(self, column_idx):
        """_PairwiseMeansSigPVals measure object for this cube-result.

        The `column_idx` is the reference column on which calculate the pairwise sig
        test for mean measure. E.G. a single category of a categorical variable or
        a subvariable reference for an array type.
        """
        return _PairwiseMeansSigPVals(
            self._dimensions, self, self._cube_measures, column_idx
        )

    def pairwise_significance_means_t_stats(self, column_idx):
        """_PairwiseMeansSigTStats measure object for this cube-result."""
        return _PairwiseMeansSigTStats(
            self._dimensions, self, self._cube_measures, column_idx
        )

    @lazyproperty
    def row_comparable_counts(self):
        """_RowComparableCounts measure object for this cube-result."""
        return _RowComparableCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_proportions(self):
        """_RowProportions measure object for this cube-result."""
        return _RowProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_share_sum(self):
        """_RowShareSum measure object for this cube-result"""
        return _RowShareSum(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_unweighted_bases(self):
        """_RowUnweightedBases measure object for this cube-result."""
        return _RowUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_weighted_bases(self):
        """_RowWeightedBases measure object for this cube-result."""
        return _RowWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def rows_pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix row."""
        return self._cube_measures.unweighted_cube_counts.rows_pruning_base

    @lazyproperty
    def rows_scale_mean(self):
        """_ScaleMean for rows measure object for this cube-result."""
        return _ScaleMean(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_scale_mean_stddev(self):
        """_ScaleMeanStddev for rpws measure object for this cube-result."""
        return _ScaleMeanStddev(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_scale_median(self):
        """_ScaleMedian for rows measure object for this cube-result."""
        return _ScaleMedian(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def sums(self):
        """_Sums measure object for this cube-result"""
        return _Sums(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def stddev(self):
        """_StdDev measure object for this cube-result"""
        return _StdDev(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_unweighted_bases(self):
        """_TableUnweightedBases measure object for this cube-result."""
        return _TableUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_weighted_bases(self):
        """_TableWeightedBases measure object for this cube-result."""
        return _TableWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def total_share_sum(self):
        """_TotalShareSum measure object for this cube-result"""
        return _TotalShareSum(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this cube-result."""
        return _UnweightedCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def weighted_counts(self):
        """_WeightedCounts measure object for this cube-result."""
        return _WeightedCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def zscores(self):
        """_Zscores measure object for this cube-result."""
        return _Zscores(self._dimensions, self, self._cube_measures)

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
        """2D np.float64 ndarray of column-wise proportions denominator for each cell.

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
        """2D np.float64 ndarray of inserted column proportions denominator value.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `._subtotal_columns`" % type(self).__name__
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of column-proportions denominator for subtotal rows.

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


class _ColumnComparableCounts(_BaseSecondOrderMeasure):
    """Provides the column-comparable count measure for a matrix.

    Column-comparable Counts is a 2D np.float64 ndarray of the counts, with values in the
    column of a subtotal difference overridden as np.nan because they do not share the
    same base and so are "not comparable" when calculating other measures along the
    column, like column proportions.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.

        Column-comparable counts are the same as counts, exccept that they are undefined
        for columns of subtotal differences.
        """
        return SumSubtotals.blocks(
            self._weighted_cube_counts.weighted_counts,
            self._dimensions,
            diff_cols_nan=True,
        )


class _ColumnProportions(_BaseSecondOrderMeasure):
    """Provides the column-proportions measure for a matrix.

    Column-proportions is a 2D np.float64 ndarray of the proportion of its column margin
    contributed by the weighted count of each matrix cell.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.

        Column-proportions are counts divided by the column base, except that they are
        undefined for columns with subtotal differences.
        """
        count_blocks = self._second_order_measures.column_comparable_counts.blocks
        weighted_base_blocks = self._second_order_measures.column_weighted_bases.blocks

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    count_blocks[0][0] / weighted_base_blocks[0][0],
                    # --- inserted columns ---
                    count_blocks[0][1] / weighted_base_blocks[0][1],
                ],
                [
                    # --- inserted rows ---
                    count_blocks[1][0] / weighted_base_blocks[1][0],
                    # --- intersections ---
                    count_blocks[1][1] / weighted_base_blocks[1][1],
                ],
            ]


class _ColumnShareSum(_BaseSecondOrderMeasure):
    """Provides the column share of sum measure for a matrix.

    Column share sum is the sum of each subvar divided by the TOTAL number of col items.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        sums_blocks = SumSubtotals.blocks(
            self._cube_measures.cube_sum.sums,
            self._dimensions,
            diff_cols_nan=True,
            diff_rows_nan=True,
        )
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    sums_blocks[0][0] / np.sum(sums_blocks[0][0], axis=0),
                    # --- inserted columns ---
                    sums_blocks[0][1] / np.sum(sums_blocks[0][1], axis=0),
                ],
                [
                    # --- inserted rows ---
                    sums_blocks[1][0] / np.sum(sums_blocks[1][0], axis=0),
                    # --- intersections ---
                    sums_blocks[1][1] / np.sum(sums_blocks[1][1], axis=0),
                ],
            ]


class _ColumnUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the column-bases measure for a matrix.

    Column-bases is a 2D np.float64 ndarray of unweighted-N "basis" for each matrix cell.
    Depending on the dimensionality of the underlying cube-result some or all of these
    values may be the same.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of column-wise proportions denominator for each cell.

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
        shape = SumSubtotals.intersections(
            self._base_values, self._dimensions, diff_cols_nan=True
        ).shape
        columns_base = self._subtotal_columns[0]
        return np.broadcast_to(columns_base, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted column proportions denominator value.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- Summing works on columns because column-proportion denominators add along
        # --- that axis, like column-proportions denominator of a subtotal of two
        # --- columns each with a base of 25 is indeed 50. This doesn't work on rows
        # --- though, see below. This wouldn't work on MR-columns but there can be no
        # --- subtotal columns on an MR dimension (X_MR slice) so that case never
        # --- arises.
        return SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions, diff_cols_nan=True
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of column-proportions denominator for subtotal rows.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- the strategy here is simply to broadcast the columns_base to the shape of
        # --- the subtotal-rows matrix because a subtotal-row value has the same
        # --- column-base as all other cells in that column. Note that this initial
        # --- subtotal-rows matrix is used only for its shape (and when it is empty)
        # --- because it computes the wrong cell values for this case.
        subtotal_rows = SumSubtotals.subtotal_rows(
            self._base_values, self._dimensions, diff_cols_nan=True
        )

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
        shape = SumSubtotals.intersections(
            self._base_values, self._dimensions, diff_cols_nan=True
        ).shape
        intersections_row = self._subtotal_columns[0]
        return np.broadcast_to(intersections_row, shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted-column denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- Summing works on columns because column-proportion denominators add along
        # --- that axis.
        return SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions, diff_cols_nan=True
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of inserted-row denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- the strategy here is simply to broadcast the columns_base to the shape of
        # --- the subtotal-rows matrix because these don't add. Note that this initial
        # --- subtotal-rows matrix is used only for its shape because it computes the
        # --- wrong values.
        subtotal_rows = SumSubtotals.subtotal_rows(
            self._base_values, self._dimensions, diff_cols_nan=True
        )
        # --- in the "no-row-subtotals" case, short-circuit with a (0, ncols) array
        # --- return value, both because that is the right answer, but also because the
        # --- non-empty columns-base array cannot be broadcast into that shape.
        if subtotal_rows.shape[0] == 0:
            return subtotal_rows

        return np.broadcast_to(
            self._weighted_cube_counts.columns_margin, subtotal_rows.shape
        )


class _Means(_BaseSecondOrderMeasure):
    """Provides the mean measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(
            self._cube_measures.cube_means.means, self._dimensions
        )


class _PairwiseIndices(_BaseSecondOrderMeasure):
    """Provides pairwise significance indices measure for matrix."""

    def __init__(
        self, dimensions, second_order_measures, cube_measures, alpha, only_larger
    ):
        super(_PairwiseIndices, self).__init__(
            dimensions, second_order_measures, cube_measures
        )
        self._alpha = alpha
        self._only_larger = only_larger

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        pairwise_indices = np.array(
            [self._pairwise_indices(v.p_vals, v.t_stats) for v in self._values]
        ).T
        return NoneSubtotals.blocks(pairwise_indices, self._dimensions)

    def _pairwise_indices(self, p_vals, t_stats):
        """1D ndarray containing tuples of int pairwise indices of each column."""
        significance = p_vals < self._alpha
        if self._only_larger:
            significance = np.logical_and(t_stats < 0, significance)
        col_significance = np.empty((len(significance),), dtype=object)
        col_significance[:] = [tuple(np.where(sig_row)[0]) for sig_row in significance]
        return col_significance

    @lazyproperty
    def _values(self):
        """list of _PairwiseSigPVals tests objects.

        Result has as many elements as there are columns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        return [
            _PairwiseSigPVals(
                self._dimensions,
                self._second_order_measures,
                self._cube_measures,
                col_idx,
            )
            for col_idx in range(
                self._cube_measures.cube_overlaps.selected_bases.shape[1]
            )
        ]


class _PairwiseMeansIndices(_PairwiseIndices):
    """Provides pairwise means significance indices measure for matrix."""

    @lazyproperty
    def _values(self):
        """list of _PairwiseMeansSigPVals tests objects.
        Result has as many elements as there are columns in the slice. Each
        significance test contains `p_vals` and `t_stats` significance tests.
        """
        return [
            _PairwiseMeansSigPVals(
                self._dimensions,
                self._second_order_measures,
                self._cube_measures,
                col_idx,
            )
            for col_idx in range(self._cube_measures.cube_means.means.shape[1])
        ]


class _PairwiseSigTStats(_BaseSecondOrderMeasure):
    """Provides pairwise significance t-stats measure for matrix and selected subvar.

    Pairwise significance is calculated for each selected subvar (column) separately.
    """

    def __init__(
        self, dimensions, second_order_measures, cube_measures, selected_subvar_idx
    ):
        super(_PairwiseSigTStats, self).__init__(
            dimensions, second_order_measures, cube_measures
        )
        self._selected_subvar_idx = selected_subvar_idx

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(self.t_stats, self._dimensions)

    @lazyproperty
    def _n_rows(self):
        """int number of rows in the matrix."""
        return self._cube_measures.cube_overlaps.valid_bases.shape[0]

    @lazyproperty
    def _n_subvars(self):
        """int number of columns (subvariables) in the matrix."""
        return self._cube_measures.cube_overlaps.valid_bases.shape[1]

    @lazyproperty
    def t_stats(self):
        """2D ndarray of float64 representing t-stats for pairwise MR testing.

        For each (category) row, we calculate the test statistic of the overlap between
        columns (subvariables) of the crossing MR variable. To do that we have to
        iterate across all categories (rows), and then across all subvars (columns).
        Each of these iterations produces a single number for the test statistic, so we
        end up with n_rows x n_cols 2-dimensional ndarray.
        """
        return np.array(
            [
                [
                    _PairwiseSignificaneBetweenSubvariablesHelper(
                        self._second_order_measures.column_proportions.blocks[0][0],
                        self._cube_measures.cube_overlaps,
                        row_idx,
                        self._selected_subvar_idx,
                        subvar_idx,
                    ).t_stats
                    for subvar_idx in range(self._n_subvars)
                ]
                for row_idx in range(self._n_rows)
            ]
        )


class _PairwiseSigPVals(_PairwiseSigTStats):
    """Provides pairwise significance p-vals measure for matrix and selected subvar.

    Pairwise significance is calculated for each selected subvar (column) separately.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(self.p_vals, self._dimensions)

    @lazyproperty
    def p_vals(self):
        """2D ndarray of float64 representing p-vals for pairwise MR testing.

        For each (category) row, we calculate the test significance of the overlap
        between columns (subvariables) of the crossing MR variable. To do that we have
        to iterate across all categories (rows), and then across all subvars (columns).
        Each of these iterations produces a single number for the significance, so we
        end up with n_rows x n_cols 2-dimensional ndarray.
        """
        return np.array(
            [
                [
                    _PairwiseSignificaneBetweenSubvariablesHelper(
                        self._second_order_measures.column_proportions.blocks[0][0],
                        self._cube_measures.cube_overlaps,
                        row_idx,
                        self._selected_subvar_idx,
                        subvar_idx,
                    ).p_vals
                    for subvar_idx in range(self._n_subvars)
                ]
                for row_idx in range(self._n_rows)
            ]
        )


class _PairwiseMeansSigTStats(_BaseSecondOrderMeasure):
    """Provides pairwise means significance t-stats measure for matrix.
    Pairwise significance is calculated for each selected column separately.
    """

    def __init__(
        self, dimensions, second_order_measures, cube_measures, selected_column_idx
    ):
        super(_PairwiseMeansSigTStats, self).__init__(
            dimensions, second_order_measures, cube_measures
        )
        self._selected_column_idx = selected_column_idx

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(self.t_stats, self._dimensions)

    @lazyproperty
    def t_stats(self):
        """2D ndarray of float64 representing t-stats for means pairwise testing.
        Calculate the level of significance for the difference of two means from the
        selected column different from a hypothesized value.
        t = (x̄1 - x̄2 - μ) √ (N / s2)
        s2 = var1 + var2 - 2 * covar12 - covar12 in this specific case is 0.
        """
        if self._selected_column_idx < 0:
            return np.full(self._cube_measures.cube_stddev.stddev.shape, np.nan)

        means = self._cube_measures.cube_means.means
        variance = np.power(self._cube_measures.cube_stddev.stddev, 2)
        col_bases = self._cube_measures.unweighted_cube_counts.column_bases

        combined_variance = variance[:, self._selected_column_idx] + variance.T
        diff = means.T - means[:, self._selected_column_idx]
        n = col_bases[:, self._selected_column_idx] + col_bases.T
        return diff.T * np.sqrt(n.T / combined_variance.T)


class _PairwiseMeansSigPVals(_PairwiseMeansSigTStats):
    """Provides pairwise means significance p-vals measure for matrix.
    Pairwise significance is calculated for each selected column separately.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(self.p_vals, self._dimensions)

    @lazyproperty
    def p_vals(self):
        """2D ndarray of float64 representing p-vals for means pairwise testing."""
        col_bases = self._cube_measures.unweighted_cube_counts.column_bases
        n = col_bases[:, self._selected_column_idx] + col_bases.T
        df = 2 * (n - 1)
        return 2 * (1 - t.cdf(abs(self.t_stats), df=df.T))


class _RowComparableCounts(_BaseSecondOrderMeasure):
    """Provides the row-comparable count measure for a matrix.

    row-comparable Counts is a 2D np.float64 ndarray of the counts, with values in the
    row of a subtotal difference overridden as np.nan because they do not share the
    same base and so are "not comparable" when calculating other measures along the
    row, like row proportions.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.

        Row-comparable counts are the same as counts, exccept that they are undefined
        for rows of subtotal differences.
        """
        return SumSubtotals.blocks(
            self._weighted_cube_counts.weighted_counts,
            self._dimensions,
            diff_rows_nan=True,
        )


class _RowProportions(_BaseSecondOrderMeasure):
    """Provides the row-proportions measure for a matrix.

    Row-proportions is a 2D np.float64 ndarray of the proportion of its row margin
    contributed by the weighted count of each matrix cell.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.

        Row-proportions are counts divided by the row comparable counts.
        """
        count_blocks = self._second_order_measures.row_comparable_counts.blocks
        weighted_base_blocks = self._second_order_measures.row_weighted_bases.blocks

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    count_blocks[0][0] / weighted_base_blocks[0][0],
                    # --- inserted columns ---
                    count_blocks[0][1] / weighted_base_blocks[0][1],
                ],
                [
                    # --- inserted rows ---
                    count_blocks[1][0] / weighted_base_blocks[1][0],
                    # --- intersections ---
                    count_blocks[1][1] / weighted_base_blocks[1][1],
                ],
            ]


class _RowShareSum(_BaseSecondOrderMeasure):
    """Provides the row share of sum measure for a matrix.

    Row share sum is the sum of each subvar divided by the TOTAL number of row items.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        sums_blocks = SumSubtotals.blocks(
            self._cube_measures.cube_sum.sums,
            self._dimensions,
            diff_cols_nan=True,
            diff_rows_nan=True,
        )
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    (sums_blocks[0][0].T / np.sum(sums_blocks[0][0], axis=1)).T,
                    # --- inserted columns ---
                    (sums_blocks[0][1].T / np.sum(sums_blocks[0][1], axis=1)).T,
                ],
                [
                    # --- inserted rows ---
                    (sums_blocks[1][0].T / np.sum(sums_blocks[1][0], axis=1)).T,
                    # --- intersections ---
                    (sums_blocks[1][1].T / np.sum(sums_blocks[1][1], axis=1)).T,
                ],
            ]


class _RowUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the row-unweighted-bases measure for a matrix.

    row-unweighted-bases is a 2D np.float64 ndarray of the unweighted row-proportion
    denominator (base) for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of row-proportions denominator for each cell.

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
        shape = SumSubtotals.intersections(
            self._base_values, self._dimensions, diff_rows_nan=True
        ).shape
        intersection_column = self._subtotal_rows[:, 0]
        return np.broadcast_to(intersection_column[:, None], shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of column-subtotal row-proportions denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- the strategy here is simply to broadcast the rows_base to the shape of
        # --- the subtotal-columns matrix because these don't add. Note that this
        # --- initial subtotal-columns matrix is used only for its shape because it
        # --- computes the wrong values.
        subtotal_columns = SumSubtotals.subtotal_columns(
            self._base_values, self._dimensions, diff_rows_nan=True
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
        """2D np.float64 ndarray of row-subtotal row-proportions denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- Summing works on rows because row-proportion denominators add along this
        # --- axis. This wouldn't work on MR-rows but there can be no subtotals on an
        # --- MR rows dimension (or any MR dimension) so that case never arises.
        return SumSubtotals.subtotal_rows(
            self._base_values, self._dimensions, diff_rows_nan=True
        )


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
            self._base_values, self._dimensions, diff_rows_nan=True
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
        return SumSubtotals.subtotal_rows(
            self._base_values, self._dimensions, diff_rows_nan=True
        )


class _Sums(_BaseSecondOrderMeasure):
    """Provides the sum measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return SumSubtotals.blocks(
            self._cube_measures.cube_sum.sums,
            self._dimensions,
            diff_rows_nan=True,
            diff_cols_nan=True,
        )


class _StdDev(_BaseSecondOrderMeasure):
    """Provides the stddev measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(
            self._cube_measures.cube_stddev.stddev, self._dimensions
        )


class _TableUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the table-unweighted-bases measure for a matrix.

    table-unweighted-bases is a 2D np.float64 ndarray of the denominator, or "base" of
    the unweighted table-proportion for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D float64 ndarray of unweighted table-proportion denominator for each cell.

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
        """2D np.float64 ndarray of inserted-column table-proportions denominator value.

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
        """2D np.float64 ndarray of inserted-row table-proportions denominator values.

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


class _TotalShareSum(_BaseSecondOrderMeasure):
    """Provides the row share of sum measure for a matrix.

    Row share sum is the sum of each subvar divided by the TOTAL number of row items.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        sums_blocks = SumSubtotals.blocks(
            self._cube_measures.cube_sum.sums,
            self._dimensions,
            diff_cols_nan=True,
            diff_rows_nan=True,
        )
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    sums_blocks[0][0] / np.sum(sums_blocks[0][0]),
                    # --- inserted columns ---
                    sums_blocks[0][1] / np.sum(sums_blocks[0][1]),
                ],
                [
                    # --- inserted rows ---
                    sums_blocks[1][0] / np.sum(sums_blocks[1][0]),
                    # --- intersections ---
                    sums_blocks[1][1] / np.sum(sums_blocks[1][1]),
                ],
            ]


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        diff_nans = self._unweighted_cube_counts.diff_nans
        return SumSubtotals.blocks(
            self._unweighted_cube_counts.unweighted_counts,
            self._dimensions,
            diff_cols_nan=diff_nans,
            diff_rows_nan=diff_nans,
        )


class _WeightedCounts(_BaseSecondOrderMeasure):
    """Provides the weighted-counts measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        diff_nans = self._weighted_cube_counts.diff_nans
        return SumSubtotals.blocks(
            self._weighted_cube_counts.weighted_counts,
            self._dimensions,
            diff_cols_nan=diff_nans,
            diff_rows_nan=diff_nans,
        )


class _Zscores(_BaseSecondOrderMeasure):
    """Provides the zscore measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        base_values = self._weighted_cube_counts.zscores
        dimension_types = tuple(d.dimension_type for d in self._dimensions)
        if DT.MR_SUBVAR in dimension_types:
            return NanSubtotals.blocks(base_values, self._dimensions)
        return ZscoreSubtotals.blocks(self._weighted_cube_counts, self._dimensions)


# === MARGINALS ===


class _BaseMarginal(object):
    """Base class for all (second-order) marginal objects."""

    def __init__(self, dimensions, second_order_measures, cube_measures, orientation):
        self._dimensions = dimensions
        self._second_order_measures = second_order_measures
        self._cube_measures = cube_measures
        self._orientation = orientation

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" making up this measure.

        These are the base-values and the subtotals.
        """
        raise NotImplementedError(  # pragma: no cover
            "%s must implement `.blocks`" % type(self).__name__
        )

    @lazyproperty
    def orientation(self):
        """String, either "row" or "column", the orientation the marginal."""
        return self._orientation

    @lazyproperty
    def is_defined(self):
        """Bool, indicating whether the marginal can be calculated.

        Defaults to True, but designed to be overridden by subclasses if necessary.
        """
        return True

    def _apply_along_orientation(self, func1d, arr, *args, **kwargs):
        """Wrapper around `np.apply_along_axis` useful for marginal calculation

        Gets the axis from the `.orientation` property and also returns an empty
        1d array of floats when the dimension is length 0 (rather than failing).
        """
        axis = 1 if self.orientation == MO.ROWS else 0

        if arr.shape[1 - axis] == 0:
            return np.array([], dtype=np.float64)

        return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    @lazyproperty
    def _counts(self):
        """List of 2 ndarrays count blocks, in their original payload order

        The counts come from the count measure blocks that are comparable in the correct
        orientation, and include the base values and the subtotals for the orientation.
        """
        if self.orientation == MO.ROWS:
            # --- Use *row* comparable counts ---
            counts = self._second_order_measures.row_comparable_counts.blocks
            # --- Get base values & *row* subtotals
            return [counts[0][0], counts[1][0]]
        else:
            # --- Use *column* comparable counts ---
            counts = self._second_order_measures.column_comparable_counts.blocks
            # --- Get base values & *column* subtotals
            return [counts[0][0], counts[0][1]]


class _BaseScaledCountMarginal(_BaseMarginal):
    """A base class for marginals that depend on the scaled counts."""

    @lazyproperty
    def _opposing_numeric_values(self):
        """ndarray of numeric_values from the opposing dimension"""
        return (
            np.array(self._dimensions[1].numeric_values, dtype=np.float64)
            if self.orientation == MO.ROWS
            else np.array(self._dimensions[0].numeric_values, dtype=np.float64)
        )


class _ScaleMean(_BaseScaledCountMarginal):
    """Provides the scale mean marginals for a matrix if available.

    The rows/columns median is a 1D np.float64 ndarray, the same dimension as the
    opposing dimension, and has has the weighted mean of the opposing dimension's
    numeric values.

    It is None when there are no numeric values for the opposing dimension. If
    there are any numeric values defined, but for a given row/column none of the
    categories with numeric values have positive counts it will be np.nan.
    """

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the scale mean.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError(
                "%s-scale-mean is undefined if no numeric values are defined on "
                "opposing dimension." % self.orientation.value
            )

        return [
            self._apply_along_orientation(
                self._weighted_mean, proportion, values=self._opposing_numeric_values
            )
            for proportion in self._proportions
        ]

    @lazyproperty
    def is_defined(self):
        """True if any have numeric values"""
        return not np.all(np.isnan(self._opposing_numeric_values))

    @lazyproperty
    def _proportions(self):
        """List of 2 ndarray matrixes of the relevant proportion blocks"""
        if self.orientation == MO.ROWS:
            # --- Use *row* proportions ---
            props = self._second_order_measures.row_proportions.blocks
            # --- Get base values & *row* subtotals
            return [props[0][0], props[1][0]]
        else:
            # --- Use *column* proportions ---
            props = self._second_order_measures.column_proportions.blocks
            # --- Get base values & *column* subtotals
            return [props[0][0], props[0][1]]

    @staticmethod
    def _weighted_mean(proportions, values):
        """Calculate the weighted mean from proportions and numeric values

        Both proportions and values are 1D ndarrays
        """
        inner = np.nansum(values * proportions)
        # --- denominator isn't necessarily 1 because need to remove the proportions
        # --- that don't have numeric values defined.
        not_a_nan_mask = ~np.isnan(values)
        denominator = np.sum(proportions[not_a_nan_mask])
        return inner / denominator


class _ScaleMedian(_BaseScaledCountMarginal):
    """Provides the scale median marginals for a matrix if available.

    The rows/columns median is a 1D np.float64 ndarray, the same dimension as the
    opposing dimension, and has has the weighted median of the opposing dimension's
    numeric values.

    It is None when there are no numeric values for the opposing dimension. If
    there are any numeric values defined, but for a given row/column none of the
    categories with numeric values have positive counts it will be np.nan.
    """

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the scale median.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError(
                "%s-scale-median is undefined if no numeric values are defined on "
                "opposing dimension." % self.orientation.value
            )

        return [
            self._apply_along_orientation(
                self._weighted_median, count, sorted_values=self._sorted_values
            )
            for count in self._sorted_counts
        ]

    @lazyproperty
    def is_defined(self):
        """True if any have numeric values"""
        return not np.all(np.isnan(self._opposing_numeric_values))

    @lazyproperty
    def _sorted_counts(self):
        """List of 2 ndarrays count blocks, sorted by `._values_sort_order`

        These values are the `._counts` sorted in the same order as the numeric
        values will be so that they can be used in `_weighted_median()`.
        """
        axis = 1 if self.orientation == MO.ROWS else 0
        return (count.take(self._values_sort_order, axis) for count in self._counts)

    @lazyproperty
    def _sorted_values(self):
        """ndarray of numeric values, sorted for use in `._weighted_median()`"""
        return self._opposing_numeric_values[self._values_sort_order]

    @lazyproperty
    def _values_sort_order(self):
        """ndarray of indexes of the order to sort numeric values in

        Provides the sort order (removing np.nan values) of numeric values for the
        opposing dimension to use when sorting both the numeric values themselves and
        also the the counts so that the order between these remains consistent.
        """
        values = self._opposing_numeric_values
        sort_idx = values.argsort()
        nan_idx = np.argwhere(np.isnan(values))
        # take the sorted indices and remove the nans
        # `assume_unique=True` preserves the order of the array
        return np.setdiff1d(sort_idx, nan_idx, assume_unique=True)

    @staticmethod
    def _weighted_median(sorted_counts, sorted_values):
        """Calculate the median given a set of values and their frequency in the data

        `sorted_values` must be sorted in order and the `sorted_counts` must be sorted to
        match that order.
        """
        # --- Convert nans to 0, as we don't want them to contribute to median. Counts
        # --- can possibly be nans for subtotal differences along the orientation we're
        # --- calculating.
        sorted_counts = np.nan_to_num(sorted_counts)
        # --- TODO: We could convert to int, so that it matches old behavior exactly
        # --- Should we do this? Seems better to use true weights.
        # sorted_counts = sorted_counts.astype("int64")
        cumulative_counts = np.cumsum(sorted_counts)
        # --- If no valid numeric value has counts, median is nan
        if cumulative_counts[-1] == 0:
            return np.nan
        # --- Find the point at which the count that contains 50% of total counts
        cumulative_prop = cumulative_counts / cumulative_counts[-1]
        median_idx = np.argmax(cumulative_prop >= 0.5)
        # --- If it's exactly 50%, take the mean of the two nearest values
        # --- TODO: Does this behave okay for large numbers (or is precision a problem)?
        if cumulative_prop[median_idx] == 0.5:
            return np.mean(sorted_values[[median_idx, median_idx + 1]])
        else:
            return sorted_values[median_idx]


class _ScaleMeanStddev(_BaseScaledCountMarginal):
    """Provides the std. dev of the marginal scale means."""

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the scale mean standard deviation.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError(
                "%s-scale-mean-standard-deviation is undefined if no numeric values "
                "are defined on opposing dimension." % self.orientation.value
            )

        return [
            self._stddev_func(count, self._opposing_numeric_values, scale_means)
            for count, scale_means in zip(self._counts, self._scale_means)
        ]

    @lazyproperty
    def is_defined(self):
        """True if any have numeric values"""
        return not np.all(np.isnan(self._opposing_numeric_values))

    @staticmethod
    def _columns_weighted_mean_stddev(counts, values, scale_mean):
        """Calculate the scale mean std dev for columns orientation"""
        if counts.shape[1] == 0:
            return np.array([], dtype=float)
        # --- Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is
        # --- the mean of the data.
        not_a_nan_index = ~np.isnan(values)
        row_dim_numeric_values = values[not_a_nan_index]
        numerator = (
            counts[not_a_nan_index, :]
            * pow(
                np.broadcast_to(
                    row_dim_numeric_values, counts[not_a_nan_index, :].T.shape
                )
                - scale_mean.reshape(-1, 1),
                2,
            ).T
        )
        denominator = np.sum(counts[not_a_nan_index, :], axis=0)
        variance = np.nansum(numerator, axis=0) / denominator
        return np.sqrt(variance)

    @staticmethod
    def _rows_weighted_mean_stddev(counts, values, scale_mean):
        """Calculate the scale mean std dev for rows orientation"""
        if counts.shape[0] == 0:
            return np.array([], dtype=float)
        # --- Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is
        # --- the mean of the data.
        not_a_nan_index = ~np.isnan(values)
        col_dim_numeric = values[not_a_nan_index]

        numerator = counts[:, not_a_nan_index] * pow(
            np.broadcast_to(col_dim_numeric, counts[:, not_a_nan_index].shape)
            - scale_mean.reshape(-1, 1),
            2,
        )
        denominator = np.sum(counts[:, not_a_nan_index], axis=1)
        variance = np.nansum(numerator, axis=1) / denominator
        return np.sqrt(variance)

    @lazyproperty
    def _scale_means(self):
        """list of 2 np.ndarray blocks of the scale mean for this orientation"""
        return (
            self._second_order_measures.rows_scale_mean.blocks
            if self.orientation == MO.ROWS
            else self._second_order_measures.columns_scale_mean.blocks
        )

    @lazyproperty
    def _stddev_func(self):
        """function for calculating standard deviation along this orientation."""
        return (
            self._rows_weighted_mean_stddev
            if self.orientation == MO.ROWS
            else self._columns_weighted_mean_stddev
        )


# === PAIRWISE HELPERS ===


class _PairwiseSignificaneBetweenSubvariablesHelper(object):
    """Helper for calculating overlaps significance between subvariables."""

    def __init__(self, column_proportions, cube_overlaps, row_idx, idx_a, idx_b):
        self._column_proportions = column_proportions
        self._cube_overlaps = cube_overlaps
        self._row_idx = row_idx
        self._idx_a = idx_a
        self._idx_b = idx_b

    @lazyproperty
    def p_vals(self):
        """float64 significance p-vals for the selected subvariables."""
        return (
            0.0
            if self._idx_a == self._idx_b
            else 2 * (1 - t.cdf(abs(self.t_stats), df=self._df - 2))
        )

    @lazyproperty
    def t_stats(self):
        """float64 significance t-stats for the selected subvariables."""
        if self._idx_a == self._idx_b:
            return 0.0

        Sa, Sb, Sab = self._selected_counts
        Na, Nb, Nab = self._valid_counts
        pa, pb, pab = Sa / Na, Sb / Nb, Sab / Nab
        col_prop_a = self._column_proportions[self._row_idx, self._idx_a]
        col_prop_b = self._column_proportions[self._row_idx, self._idx_b]

        # ---Subtract the selected column from the "variable" column, to get
        # ---the correct sign of the test statistic (hence b-a, and not a-b).
        return (col_prop_b - col_prop_a) / np.sqrt(
            1 / self._df * (pa * (1 - pa) + pb * (1 - pb) + 2 * pa * pb - 2 * pab)
        )

    @lazyproperty
    def _df(self):
        """int representing degrees of freedom for the CDF distribution.

        This is the count of non-overlapping cases of subvariables a and b.
        """
        Na, Nb, Nab = self._valid_counts
        return Na + Nb - Nab

    @lazyproperty
    def _selected_counts(self):
        """tuple(int/float64) of selected counts for subvars a, b, and combined (a^b).

        Most often these numbers will be int, because that's how the database counts
        if the responses are of the "Selected" category (row[idx] == 1). They can only
        be float64 when the overlaps result is weighted.
        """
        return (
            self._cube_overlaps.selected_bases[self._row_idx, self._idx_a, self._idx_a],
            self._cube_overlaps.selected_bases[self._row_idx, self._idx_b, self._idx_b],
            self._cube_overlaps.selected_bases[self._row_idx, self._idx_a, self._idx_b],
        )

    @lazyproperty
    def _valid_counts(self):
        """tuple(int/float64) of valid counts for subvars a, b, and combined (a^b).

        Most often these numbers will be int, because that's how the database counts
        if the responses are different than the "Missing" category (row[idx] != -1).
        They can only be float64 when the overlaps result is weighted.
        """
        return (
            self._cube_overlaps.valid_bases[self._row_idx, self._idx_a, self._idx_a],
            self._cube_overlaps.valid_bases[self._row_idx, self._idx_b, self._idx_b],
            self._cube_overlaps.valid_bases[self._row_idx, self._idx_a, self._idx_b],
        )
