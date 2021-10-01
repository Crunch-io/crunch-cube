# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

import numpy as np
from scipy.stats import t, norm

from cr.cube.enums import DIMENSION_TYPE as DT, MARGINAL_ORIENTATION as MO
from cr.cube.matrix.cubemeasure import CubeMeasures
from cr.cube.matrix.subtotals import (
    SumSubtotals,
    NanSubtotals,
)
from cr.cube.smoothing import Smoother
from cr.cube.util import lazyproperty


class SecondOrderMeasures:
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
    def column_index(self):
        """_ColumnIndex measure object for this cube-result."""
        return _ColumnIndex(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_proportions(self):
        """_ColumnProportions measure object for this cube-result."""
        return _ColumnProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_proportion_variances(self):
        """_ColumnProportions measure object for this cube-result."""
        return _ColumnProportionVariances(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_share_sum(self):
        """_ColumnShareSum measure object for this cube-result"""
        return _ColumnShareSum(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_std_err(self):
        """_ColumnStandardError measure object for this cube-result."""
        return _ColumnStandardError(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_unweighted_bases(self):
        """_ColumnUnweightedBases measure object for this cube-result."""
        return _ColumnUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def column_weighted_bases(self):
        """_ColumnWeightedBases measure object for this cube-result."""
        return _ColumnWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def columns_table_proportion(self):
        """_MarginTableProportion for measure object for columns of this cube-result.

        The name is a bit of a mouthful, but each component is meaningful.
        - "columns": Indicates it is a marginal in the "columns" orientation (kind of
        like a stripe in the shape of a row).
        - "table": Indicates that it is the proportion for the whole table, meaning it
        uses the `columns_table_weighted_base` as the denominator.
        - "proportion": It is a proportion, the `columns_weighted_base` divided
        by the `columns_table_weighted_base`

        Note that there is an implied "weighted" here, which we do not spell out because
        we never calculate unweighted proportions.
        """
        return _MarginTableProportion(
            self._dimensions, self, self._cube_measures, MO.COLUMNS
        )

    @lazyproperty
    def columns_pruning_mask(self):
        """1D bool ndarray indicating if each matrix column is empty."""
        return self._cube_measures.unweighted_cube_counts.columns_pruning_mask

    @lazyproperty
    def columns_scale_mean(self):
        """_ScaleMean for columns measure object for this cube-result."""
        return _ScaleMean(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_scale_mean_stddev(self):
        """_ScaleMeanStddev for columns measure object for this cube-result."""
        return _ScaleMeanStddev(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_scale_mean_stderr(self):
        """_ScaleMeanStderr for rows measure object for this cube-result."""
        return _ScaleMeanStderr(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_scale_median(self):
        """_ScaleMedian for columns measure object for this cube-result."""
        return _ScaleMedian(self._dimensions, self, self._cube_measures, MO.COLUMNS)

    @lazyproperty
    def columns_table_unweighted_base(self):
        """_MarginTableBase measure object for this cube-result for columns (unweighted).

        The name is a mouthful, but each component is important.

        * "columns": Indicates it is a marginal in the "columns" orientation (kind of
          like a stripe in the shape of a row).
        * "table": Indicates that it is the base for the whole table. When the
          `.table_unweighted_base` exists (CAT X CAT), it is a repetition of that, but when
          the columns are array (and therefore we can't sum across them), each cell has
          its own value.
        * "unweighted": Indicates that weights are not used
        * "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)
        """
        return _MarginTableBase(
            self._dimensions,
            self,
            self._cube_measures,
            MO.COLUMNS,
            self._cube_measures.unweighted_cube_counts,
        )

    @lazyproperty
    def columns_table_weighted_base(self):
        """_MarginTableBase measure object for this cube-result for columns (weighted).

        The name is a mouthful, but each component is important.

        * "columns": Indicates it is a marginal in the "columns" orientation (kind of
          like a stripe in the shape of a row).
        * "table": Indicates that it is the base for the whole table. When the
          `.table_weighted_base` exists (CAT X CAT), it is a repetition of that, but when
          the columns are array (and therefore we can't sum across them), each cell has
          its own value.
        * "weighted": Indicates that weights are used
        * "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)
        """
        return _MarginTableBase(
            self._dimensions,
            self,
            self._cube_measures,
            MO.COLUMNS,
            self._cube_measures.weighted_cube_counts,
        )

    @lazyproperty
    def columns_unweighted_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        return _MarginUnweightedBase(
            self._dimensions, self, self._cube_measures, MO.COLUMNS
        )

    @lazyproperty
    def columns_weighted_base(self):
        """_MarginWeightedBase for columns measure object for this cube-result.

        Formerly called the 1D columns-margin, this is the weighted base of the column
        margin, the sum across the columns of the weighted counts for non-MR or selected
        & non-selected for MR.
        """
        return _MarginWeightedBase(
            self._dimensions, self, self._cube_measures, MO.COLUMNS
        )

    @lazyproperty
    def means(self):
        """_Means measure object for this cube-result"""
        return _Means(self._dimensions, self, self._cube_measures)

    def pairwise_p_vals_for_subvar(self, subvar_idx):
        """_PairwiseSigPValsForSubvar measure object for this cube-result"""
        return _PairwiseSigPValsForSubvar(
            self._dimensions, self, self._cube_measures, subvar_idx
        )

    def pairwise_t_stats_for_subvar(self, subvar_idx):
        """_PairwiseSigTStatsForSubvar measure object for this cube-result"""
        return _PairwiseSigTStatsForSubvar(
            self._dimensions, self, self._cube_measures, subvar_idx
        )

    def pairwise_p_vals(self, column_idx):
        """_PairwiseSigPvals measure object for this cube-result."""
        return _PairwiseSigPvals(
            self._dimensions, self, self._cube_measures, column_idx
        )

    def pairwise_t_stats(self, column_idx):
        """_PairwiseSigTstats measure object for this cube-result."""
        return _PairwiseSigTstats(
            self._dimensions, self, self._cube_measures, column_idx
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
    def population_proportions(self):
        """_PopulationProportions measure object for this cube-result.

        Population proportions choose between the correct proportion, to calculate the
        population, generally the table-proportion, but when there is a categorical
        date we want to hold the population constant for all time periods instead of
        dividing them.
        """
        return _PopulationProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def population_std_err(self):
        """_PopulationStandardError measure object for this cube-result.

        Population standard-errors choose between the correct std-err, to calculate the
        population, generally the table-std-err, but when there is a categorical
        date we want to hold the population constant for all time periods instead of
        dividing them.
        """
        return _PopulationStandardError(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def pvalues(self):
        """_Pvalues measure object for this cube-result."""
        return _Pvalues(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_comparable_counts(self):
        """_RowComparableCounts measure object for this cube-result."""
        return _RowComparableCounts(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_proportions(self):
        """_RowProportions measure object for this cube-result."""
        return _RowProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_proportion_variances(self):
        """_RowProportions measure object for this cube-result."""
        return _RowProportionVariances(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_share_sum(self):
        """_RowShareSum measure object for this cube-result"""
        return _RowShareSum(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_std_err(self):
        """_RowStandardError measure object for this cube-result."""
        return _RowStandardError(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_unweighted_bases(self):
        """_RowUnweightedBases measure object for this cube-result."""
        return _RowUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def row_weighted_bases(self):
        """_RowWeightedBases measure object for this cube-result."""
        return _RowWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def rows_pruning_mask(self):
        """1D bool ndarray indicating if each matrix row is empty."""
        return self._cube_measures.unweighted_cube_counts.rows_pruning_mask

    @lazyproperty
    def rows_scale_mean(self):
        """_ScaleMean for rows measure object for this cube-result."""
        return _ScaleMean(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_scale_mean_stddev(self):
        """_ScaleMeanStddev for rows measure object for this cube-result."""
        return _ScaleMeanStddev(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_scale_mean_stderr(self):
        """_ScaleMeanStderr for rows measure object for this cube-result."""
        return _ScaleMeanStderr(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_scale_median(self):
        """_ScaleMedian for rows measure object for this cube-result."""
        return _ScaleMedian(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def rows_table_proportion(self):
        """_MarginTableProportion for measure object for this cube-result.

        The name is a bit of a mouthful, but each component is meaningful.
        - "rows": Indicates it is a marginal in the "rows" orientation (kind of
        like a stripe in the shape of a column).
        - "table": Indicates that it is the proportion for the whole table, meaning it
        uses the `rows_table_weighted_base` as the denominator.
        - "proportion": It is a proportion, the `rows_weighted_base` divided
        by the `rows_table_weighted_base`

        Note that there is an implied "weighted" here, which we do not spell out because
        we never calculate unweighted proportions.
        """
        return _MarginTableProportion(
            self._dimensions, self, self._cube_measures, MO.ROWS
        )

    @lazyproperty
    def rows_table_unweighted_base(self):
        """_MarginTableBase measure object for this cube-result for rows (unweighted).

        The name is a mouthful, but each component is important.

        * "rows": Indicates it is a marginal in the "rows" orientation (kind of like a
          stripe in the shape of a column).
        * "table": Indicates that it is the base for the whole table. When the
          `.table_unweighted_base` exists (CAT X CAT), it is a repetition of that, but when
          the rows are array (and therefore we can't sum across them), each cell has
          its own value.
        * "unweighted": Indicates that weights are not used
        * "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)
        """
        return _MarginTableBase(
            self._dimensions,
            self,
            self._cube_measures,
            MO.ROWS,
            self._cube_measures.unweighted_cube_counts,
        )

    @lazyproperty
    def rows_table_weighted_base(self):
        """_MarginTableBase measure object for this cube-result for rows (weighted).

        The name is a mouthful, but each component is important.

        - "rows": Indicates it is a marginal in the "rows" orientation (kind of like a
          stripe in the shape of a column).
        - "table": Indicates that it is the base for the whole table. When the
          `.table_weighted_base` exists (CAT X CAT), it is a repetition of that, but when
          the rows are array (and therefore we can't sum across them), each cell has
          its own value.
        - "weighted": Indicates that weights are used
        - "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)
        """
        return _MarginTableBase(
            self._dimensions,
            self,
            self._cube_measures,
            MO.ROWS,
            self._cube_measures.weighted_cube_counts,
        )

    @lazyproperty
    def rows_unweighted_base(self):
        """1D np.float64 ndarray of unweighted-N for each matrix column."""
        return _MarginUnweightedBase(
            self._dimensions, self, self._cube_measures, MO.ROWS
        )

    @lazyproperty
    def rows_weighted_base(self):
        """_MarginWeightedBase for rows measure object for this cube-result.

        Formerly called the 1D rows-margin, this is the weighted base of the column
        margin, the sum across the rows of the weighted counts for non-MR or selected
        & non-selected for MR.
        """
        return _MarginWeightedBase(self._dimensions, self, self._cube_measures, MO.ROWS)

    @lazyproperty
    def smoothed_column_index(self):
        """_ColumnIndexSmoothed measure object for this cube-result.

        If smoothing is defined in the dimension transform a _ColumnIndexSmoothed object
        will be returned, otherwise fallback to unsmoothed measure object.
        """
        return _ColumnIndexSmoothed(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def smoothed_column_proportions(self):
        """_ColumnProportionsSmoothed measure object for this cube-result.

        If smoothing is defined in the dimension transform a _ColumnProportionsSmoothed
        object will be returned, otherwise fallback to unsmoothed measure object.
        """
        return _ColumnProportionsSmoothed(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def smoothed_columns_scale_mean(self):
        """_ScaleMeanSmoothed for columns measure object for this cube-result.

        If smoothing is defined in the dimension transform a _ScaleMeanSmoothed object
        will be returned, otherwise fallback to unsmoothed measure object.
        """
        return _ScaleMeanSmoothed(
            self._dimensions, self, self._cube_measures, MO.COLUMNS
        )

    @lazyproperty
    def smoothed_means(self):
        """_MeansSmoothed measure object for this cube-result

        If smoothing is defined in the dimension transform a _MeansSmoothed object will
        be returned, otherwise fallback to unsmoothed measure object.
        """
        return _MeansSmoothed(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def sums(self):
        """_Sums measure object for this cube-result"""
        return _Sums(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def stddev(self):
        """_StdDev measure object for this cube-result"""
        return _StdDev(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_proportions(self):
        """_TableProportions measure object for this cube-result."""
        return _TableProportions(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_proportion_variances(self):
        """_TableProportionVariances measure object for this cube-result."""
        return _TableProportionVariances(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_std_err(self):
        """_TableStandardError measure object for this cube-result."""
        return _TableStandardError(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_unweighted_base(self):
        """_TableBase measure object for this cube-result (unweighted).

        A scalar value that is the unweighted base for the whole table. It is only
        defined when both dimensions are CAT and is equal to the sum of all the counts.
        """
        return _TableBase(
            self._dimensions,
            self,
            self._cube_measures,
            self._cube_measures.unweighted_cube_counts,
        )

    @lazyproperty
    def table_unweighted_bases(self):
        """_TableUnweightedBases measure object for this cube-result."""
        return _TableUnweightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_unweighted_bases_range(self):
        """_TableBasesRange measure object for this cube-result (unweighted)."""
        return _TableBasesRange(
            self._dimensions,
            self,
            self._cube_measures,
            self._cube_measures.unweighted_cube_counts,
        )

    @lazyproperty
    def table_weighted_base(self):
        """_TableBase measure object for this cube-result (weighted).

        A scalar value that is the weighted base for the whole table. It is only defined
        when both dimensions are CAT and is equal to the sum of all the counts.
        """
        return _TableBase(
            self._dimensions,
            self,
            self._cube_measures,
            self._cube_measures.weighted_cube_counts,
        )

    @lazyproperty
    def table_weighted_bases(self):
        """_TableWeightedBases measure object for this cube-result."""
        return _TableWeightedBases(self._dimensions, self, self._cube_measures)

    @lazyproperty
    def table_weighted_bases_range(self):
        """_TableBasesRange measure object for this cube-result (weighted)."""
        return _TableBasesRange(
            self._dimensions,
            self,
            self._cube_measures,
            self._cube_measures.weighted_cube_counts,
        )

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


class _BaseSecondOrderMeasure:
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
        """2D np.float64 ndarray of measure's value for each cell.

        This is the first "block" and has the shape of the cube-measure (no insertions).
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._base_values`"
        )

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._intersections`"
        )

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of measure values for subtotal columns.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._subtotal_columns`"
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of measure values for subtotal rows.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._subtotal_rows`"
        )

    @lazyproperty
    def _unweighted_cube_counts(self):
        """_BaseCubeCounts subclass instance for this measure.

        Provides cube measures associated with weighted counts, including
        weighted-counts and cell, vector, and table margins.
        """
        return self._cube_measures.unweighted_cube_counts

    @lazyproperty
    def _weighted_cube_counts(self):
        """_BaseCubeCounts subclass instance for this measure.

        Provides cube measures associated with weighted counts, including
        weighted-counts and cell, vector, and table margins.
        """
        return self._cube_measures.weighted_cube_counts


class _SmoothedMeasure(_BaseSecondOrderMeasure):
    """Mixin providing `._smoother` property for smoothed measures."""

    @lazyproperty
    def _smoother(self):
        """BaseSmoother subtype object providing smoothing as specified in spec."""
        return Smoother.factory(self._dimensions[-1])


class _ColumnComparableCounts(_BaseSecondOrderMeasure):
    """Provides the column-comparable count measure for a matrix.

    Column-Comparable-Counts is a 2D np.float64 ndarray of the counts, defined only when
    the row dimension is not an array. The values in a subtotal difference column are
    overridden as np.nan because they do not share the same base and so are "not
    comparable" when calculating other measures along the column (across the rows),
    like rows-weighted-base.

    Raises a ValueError when the row is an array.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure."""
        if not self.is_defined:
            raise ValueError(
                "column_comparable_counts not defined across subvariables."
            )

        return SumSubtotals.blocks(
            self._weighted_cube_counts.counts,
            self._dimensions,
            diff_rows_nan=True,
        )

    @lazyproperty
    def is_defined(self):
        """Bool indicating whether column comparable counts are defined

        We cannot sum counts across subvariable dimensions.
        """
        return not self._dimensions[1].dimension_type in DT.ARRAY_TYPES


class _ColumnIndex(_BaseSecondOrderMeasure):
    """Provides the column-index measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure."""
        return NanSubtotals.blocks(self._column_index, self._dimensions)

    @lazyproperty
    def _column_index(self):
        """Column-index base values.

        Column-index answers the question "are respondents in this row-category more or
        less likely than the overall table population to choose the answer represented
        by this column?". For example, if the row is "Hispanic" and the column is
        home-ownership, a value of 100 indicates hispanics are no less and no more
        likely to own their home than the overall population. If that value was 150, it
        would indicate hispanics are 50% more likely to own their home than the general
        population (or the population surveyed anyway).
        """
        counts = self._second_order_measures.weighted_counts.blocks[0][0]
        weighted_base = self._second_order_measures.column_weighted_bases.blocks[0][0]
        proportions = counts / weighted_base
        baseline = self._cube_measures.unconditional_cube_counts.baseline
        return 100 * (proportions / baseline)


class _ColumnIndexSmoothed(_ColumnIndex, _SmoothedMeasure):
    """Provides the smoothed column-index measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        It applies the smoothing algorithm to the base column_index values.
        """
        smoother = self._smoother
        return NanSubtotals.blocks(
            smoother.smooth(self._column_index), self._dimensions
        )


class _ColumnProportions(_BaseSecondOrderMeasure):
    """Provides the column-proportions measure for a matrix.

    Column-proportions is a 2D np.float64 ndarray of the proportion of its column margin
    contributed by the weighted count of each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D ndarray np.float64 of the base values column proportions (1st block)."""
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._count_blocks[0][0] / self._weighted_base_blocks[0][0]

    @lazyproperty
    def _count_blocks(self):
        """List of four 2D ndarray of weighted counts."""
        return self._second_order_measures.weighted_counts.blocks

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._count_blocks[1][1] / self._weighted_base_blocks[1][1]

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of column proportions values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._count_blocks[0][1] / self._weighted_base_blocks[0][1]

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of column proportions values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._count_blocks[1][0] / self._weighted_base_blocks[1][0]

    @lazyproperty
    def _weighted_base_blocks(self):
        """List of four 2D ndarray of column weighted bases."""
        return self._second_order_measures.column_weighted_bases.blocks


class _ColumnProportionsSmoothed(_ColumnProportions, _SmoothedMeasure):
    """Provides the smoothed column-proportions measure for a matrix.

    Column-proportions is a 2D np.float64 ndarray of the proportion of its column margin
    contributed by the weighted count of each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D ndarray np.float64 of base values column proportions smoothed values."""
        smoother = self._smoother
        return smoother.smooth(super(_ColumnProportionsSmoothed, self)._base_values)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of subtotal rows column proportions smoothed values."""
        smoother = self._smoother
        return smoother.smooth(super(_ColumnProportionsSmoothed, self)._subtotal_rows)


class _ColumnProportionVariances(_BaseSecondOrderMeasure):
    """Provides the variance of the column-proportions measure for a matrix.

    Column-proportions-variance is a 2D np.float64 ndarray of p * (1 - p) where p is the
    column-proportions.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        p = self._second_order_measures.column_proportions.blocks

        return [
            [
                # --- base values ---
                p[0][0] * (1 - p[0][0]),
                # --- inserted columns ---
                p[0][1] * (1 - p[0][1]),
            ],
            [
                # --- inserted rows ---
                p[1][0] * (1 - p[1][0]),
                # --- intersections ---
                p[1][1] * (1 - p[1][1]),
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
                    sums_blocks[0][0] / np.nansum(sums_blocks[0][0], axis=0),
                    # --- inserted columns ---
                    sums_blocks[0][1] / np.nansum(sums_blocks[0][1], axis=0),
                ],
                [
                    # --- inserted rows ---
                    sums_blocks[1][0] / np.nansum(sums_blocks[1][0], axis=0),
                    # --- intersections ---
                    sums_blocks[1][1] / np.nansum(sums_blocks[1][1], axis=0),
                ],
            ]


class _ColumnStandardError(_BaseSecondOrderMeasure):
    """Provides the standard errors of the column-proportions measure for a matrix.

    Column-standard-errors is a 2D np.float64 ndarray of the column proportion variance
    divided by the column weighted bases.
    """

    @lazyproperty
    def blocks(self):
        """list of lists of the four 2D "blocks" making up this measure."""
        variance_blocks = self._second_order_measures.column_proportion_variances.blocks
        weighted_base_blocks = self._second_order_measures.column_weighted_bases.blocks

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    np.sqrt(variance_blocks[0][0] / weighted_base_blocks[0][0]),
                    # --- inserted columns ---
                    np.sqrt(variance_blocks[0][1] / weighted_base_blocks[0][1]),
                ],
                [
                    # --- inserted rows ---
                    np.sqrt(variance_blocks[1][0] / weighted_base_blocks[1][0]),
                    # --- intersections ---
                    np.sqrt(variance_blocks[1][1] / weighted_base_blocks[1][1]),
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
        # --- non-empty columns-base array cannot be broadcast into that shape.
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
        # --- of subtotal-columns is the same. In the X_ARRAY case there can be no
        # --- subtotal columns so an empty row is broadcast to shape.
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

        return np.broadcast_to(self._base_values[0, :], subtotal_rows.shape)


class _Means(_BaseSecondOrderMeasure):
    """Provides the mean measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        return NanSubtotals.blocks(
            self._cube_measures.cube_means.means, self._dimensions
        )


class _MeansSmoothed(_Means, _SmoothedMeasure):
    """Provides the smoothed mean measure for a matrix."""

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        smoother = self._smoother
        return NanSubtotals.blocks(
            smoother.smooth(self._cube_measures.cube_means.means),
            self._dimensions,
        )


class _PairwiseSigTStatsForSubvar(_BaseSecondOrderMeasure):
    """Provides pairwise significance t-stats measure for matrix and selected subvar.

    Pairwise significance is calculated for each selected subvar (column) separately.
    """

    def __init__(
        self, dimensions, second_order_measures, cube_measures, selected_subvar_idx
    ):
        super(_PairwiseSigTStatsForSubvar, self).__init__(
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


class _PairwiseSigPValsForSubvar(_PairwiseSigTStatsForSubvar):
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
        t = (x̄1 - x̄2 - (μ1 - μ2)) / √ ((s1 / N1) + (s2 / N2))
        """
        if self._selected_column_idx < 0:
            return np.full(self._cube_measures.cube_stddev.stddev.shape, np.nan)

        means = self._cube_measures.cube_means.means
        variance = np.power(self._cube_measures.cube_stddev.stddev, 2)
        col_bases = self._cube_measures.unweighted_cube_counts.column_bases
        idx = self._selected_column_idx

        ref_means = np.broadcast_to(means[:, [idx]], means.shape)
        ref_variance = np.broadcast_to(variance[:, [idx]], variance.shape)
        ref_col_bases = np.broadcast_to(col_bases[:, [idx]], col_bases.shape)

        return (means - ref_means) / np.sqrt(
            (variance / col_bases) + (ref_variance / ref_col_bases)
        )


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
        return 2 * (1 - t.cdf(abs(self.t_stats), df=self._df))

    @lazyproperty
    def _df(self):
        """A np.float64 ndarray of the degrees of freedom for the Pairwise mean test

        We use the Welch's T Test, which has a complicated formula for the degrees of
        freedom to account for the fact that we are allowing the variances to be
        different between the distributions.

        Formula is:
        df = ( (s1/N1) + (s2/N2) )^2 / ( (s1/N1)^2/(N1-1) + (s2/N2)^2/(N2-1) )
        """
        variance = np.power(self._cube_measures.cube_stddev.stddev, 2)
        col_bases = self._cube_measures.unweighted_cube_counts.column_bases
        idx = self._selected_column_idx

        ref_variance = np.broadcast_to(variance[:, [idx]], variance.shape)
        ref_col_bases = np.broadcast_to(col_bases[:, [idx]], col_bases.shape)

        numerator = np.power((variance / col_bases) + (ref_variance / ref_col_bases), 2)
        denominator1 = np.power(variance / col_bases, 2) / (col_bases - 1)
        denominator2 = np.power(ref_variance / ref_col_bases, 2) / (ref_col_bases - 1)

        return numerator / (denominator1 + denominator2)


class _PairwiseSigTstats(_BaseSecondOrderMeasure):
    """Provides pairwise significance t-stats measure for matrix.

    The paired samples t-test is used to compare the means between two related groups
    of samples. This measure class take care of compute the paired test given the
    selected column.
    """

    def __init__(
        self, dimensions, second_order_measures, cube_measures, selected_column_idx
    ):
        super(_PairwiseSigTstats, self).__init__(
            dimensions, second_order_measures, cube_measures
        )
        self._selected_column_idx = selected_column_idx

    @lazyproperty
    def _base_values(self):
        """2D ndarray np.float64 of the t-stats for the base values (1st block)

        These are the base values t-stats considering the eventual selection on a
        subtotal column. For a normal column selected, this property compute the t-stats
        pairing each column with the selected one from the base column proportions.
        +====+====+====+====+
        | C1 | C2 | C3 | S1 |
        +====+====+====+====+
        |0.3 |0.4 |0.5 |0.4 |
        +----+----+----+----+
        |0.2 |0.1 |0.3 |0.7 |
        +----+----+----+----+
        Considering C3 the selected column we compute the t-stats comparing C3-C1,
        C3-C2, C3-C3. If the selected column is a subtotal (S1 in this example) the
        comparison will be between S1-C1, S1-C2 and S1-C3. The final shape of the
        t-stats will be the same of the base values block of the column proportions,
        in this example (2,3).
        """
        # --- Use "body" reference values for base values
        (ref_props, ref_bases) = self._reference_values(0)
        return self._calculate_t_stats(
            self._proportions[0][0], self._bases[0][0], ref_props, ref_bases
        )

    @lazyproperty
    def _bases(self):
        """2D array of 2D ndarray "blocks" for the column unweighted bases"""
        return self._second_order_measures.column_unweighted_bases.blocks

    def _reference_values(self, block_index):
        """Tuple of the reference proportions and bases for

        Because the comparison of interest is between columns, the shape of the reference
        is determined by whether we're in the "body" of the table (the base values and
        subtotal columns), or the "inserted" rows (inserted rows & intersections).
        This takes the column index and gets the needed references for the body.

        The block_index parameter chooses between the body (block_index=0) and inserted
        rows (block_index=1).
        """
        col_idx = self._selected_column_idx
        if col_idx < 0:
            props = self._proportions[block_index][1]
            bases = self._bases[block_index][1]
        else:
            props = self._proportions[block_index][0]
            bases = self._bases[block_index][0]

        return (props[:, [col_idx]], bases[:, [col_idx]])

    def _calculate_t_stats(self, props, bases, ref_props, ref_bases):
        """Calculates the 2D ndarray of np.float64 values for the tstats

        This method calculates the t test on dependent samples, comparing each column
        with the selected one.
        """
        if props.size == 0:
            return props
        var_props = props * (1.0 - props) / bases
        ref_var_props = ref_props * (1.0 - ref_props) / ref_bases
        diff = props - ref_props
        # --- Absolute value to handle subtotal differences (can't get square root of
        # --- negative number)
        se_diff = np.sqrt(np.abs(var_props + ref_var_props))
        return diff / se_diff

    @lazyproperty
    def _intersections(self):
        "2D ndarray np.float64 of the t-stats for the intersections (4th block)" ""
        # --- Use "inserted" reference values for intersections
        (ref_props, ref_variance) = self._reference_values(1)
        return self._calculate_t_stats(
            self._proportions[1][1], self._bases[1][1], ref_props, ref_variance
        )

    @lazyproperty
    def _proportions(self):
        """2D ndarray np.float64 of the t-stats for the column proportions"""
        return self._second_order_measures.column_proportions.blocks

    @lazyproperty
    def _subtotal_columns(self):
        """2D ndarray np.float64 of the values for the subtotal columns (2nd block)"""
        # --- Use "body" reference values for inserted columns
        (ref_props, ref_variance) = self._reference_values(0)
        return self._calculate_t_stats(
            self._proportions[0][1], self._bases[0][1], ref_props, ref_variance
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D ndarray np.float64 of the t-stats for the subtotal rows (3rd block)"""
        # --- Use "inserted" reference values for inserted rows
        (ref_props, ref_variance) = self._reference_values(1)
        return self._calculate_t_stats(
            self._proportions[1][0], self._bases[1][0], ref_props, ref_variance
        )


class _PairwiseSigPvals(_PairwiseSigTstats):
    """Provides pairwise significance p-vals measure for matrix.

    Pairwise significance is calculated for each selected column separately.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        col_idx = self._selected_column_idx
        t_stats = self._second_order_measures.pairwise_t_stats(col_idx).blocks
        column_bases = self._second_order_measures.column_unweighted_bases.blocks
        body_selected_base = self._selected_columns_base(0)
        ins_selected_base = self._selected_columns_base(1)

        return [
            [
                self._p_vals(t_stats[0][0], column_bases[0][0], body_selected_base),
                self._p_vals(t_stats[0][1], column_bases[0][1], body_selected_base),
            ],
            [
                self._p_vals(t_stats[1][0], column_bases[1][0], ins_selected_base),
                self._p_vals(t_stats[1][1], column_bases[1][1], ins_selected_base),
            ],
        ]

    def _p_vals(self, t_stats, columns_base, selected_columns_base):
        """2D ndarray of float64 representing p-vals for pairwise col testing.

        P values are calculated considering the cumulative distribution function
        evaluated at the t_stats values with specific degrees of freedom.
        """
        df = (columns_base + selected_columns_base - 2) if t_stats.size > 0 else 0
        return 2 * (1 - t.cdf(abs(t_stats), df=df))

    def _selected_columns_base(self, table_index):
        """1D int64 ndarray of the selected columns base values.

        In case of selected subtotal column the column base selection will be done on
        the column base subtotal values instead of the columns_base base values.

        The parameter `table_index` chooses whether the reference comes from the body
        (the base_values or subtotal_columns) or the insertions (the subtotal_rows or
        intersections). By choosing between them, we get the bases in a shape that we
        don't have to broadcast.
        """
        col_idx = self._selected_column_idx
        column_bases = self._second_order_measures.column_unweighted_bases.blocks
        return (
            column_bases[table_index][1][:, [col_idx]]
            if col_idx < 0
            else column_bases[table_index][0][:, [col_idx]]
        )


class _PopulationProportions(_BaseSecondOrderMeasure):
    """Provides the cell-specific fraction of population

    If any of the dimensions (rows or columns) is a categorical date the appropriate
    percentages are used, to calculate the population counts, so as to be the total
    amount among categorical dates. Otherwise, table percents are used to calculate
    population counts.

    Otherwise, table percents are used to calculate population counts. We do not have
    to special case array variables, because in this case, the table proportion is
    equivalent to the correct proportion (eg if the rows are categorical array suvars
    then the table_propoortion and row_proportions are equal).
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        return (
            self._second_order_measures.row_proportions.blocks
            if self._dimensions[-2].dimension_type == DT.CAT_DATE
            else self._second_order_measures.column_proportions.blocks
            if self._dimensions[-1].dimension_type == DT.CAT_DATE
            else self._second_order_measures.table_proportions.blocks
        )


class _PopulationStandardError(_BaseSecondOrderMeasure):
    """Provides the cell-specific standard error for proportion used for population

    If any of the dimensions (rows or columns) is a categorical date the appropriate
    percentages are used, to calculate the population counts, so as to be the total
    amount among categorical dates. Otherwise, table percents are used to calculate
    population counts.

    We do not have to special case array variables, because in this case, the table
    proportion is equivalent to the correct proportion (eg if the rows are categorical
    array suvars then the table_propoortion and row_proportions are equal).

    This provides the standard error that corresponds to that so that it can be used
    to calculate the MOE.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        return (
            self._second_order_measures.row_std_err.blocks
            if self._dimensions[-2].dimension_type == DT.CAT_DATE
            else self._second_order_measures.column_std_err.blocks
            if self._dimensions[-1].dimension_type == DT.CAT_DATE
            else self._second_order_measures.table_std_err.blocks
        )


class _Pvalues(_BaseSecondOrderMeasure):
    """p-value measure for the matrix

    A p-value is a measure of the probability that an observed difference could have
    occurred just by random chance. The lower the p-value, the greater the
    statistical significance of the observed difference. It is derived from the
    normal distribution's cdf and the zscore.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure."""
        zscore_blocks = self._second_order_measures.zscores.blocks

        return [
            [
                self._calculate_pval(zscore_blocks[0][0]),
                self._calculate_pval(zscore_blocks[0][1]),
            ],
            [
                self._calculate_pval(zscore_blocks[1][0]),
                self._calculate_pval(zscore_blocks[1][1]),
            ],
        ]

    def _calculate_pval(self, zscores):
        """np.array of floats of the pvalues calculated form np.array of zscores"""
        if 0 in zscores.shape:
            return zscores
        return 2 * (1 - norm.cdf(np.abs(zscores)))


class _RowComparableCounts(_BaseSecondOrderMeasure):
    """Provides the row-comparable count measure for a matrix.

    Row-Comparable-Counts is a 2D np.float64 ndarray of the counts, defined only when
    the column dimension is not an array. The values in a subtotal difference row are
    overridden as np.nan because they do not share the same base and so are "not
    comparable" when calculating other measures along the row (across the columns),
    like columns-weighted-base.

    Raises a ValueError when the column is an array.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure."""
        if not self.is_defined:
            raise ValueError("row_comparable_counts not defined across subvariables.")

        return SumSubtotals.blocks(
            self._weighted_cube_counts.counts,
            self._dimensions,
            diff_cols_nan=True,
        )

    @lazyproperty
    def is_defined(self):
        """Bool indicating whether row comparable counts are defined

        We cannot sum counts across subvariable dimensions.
        """
        return not self._dimensions[0].dimension_type in DT.ARRAY_TYPES


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

        Row-proportions are row comparable counts divided by the row weighted bases.
        """
        count_blocks = self._second_order_measures.weighted_counts.blocks
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


class _RowProportionVariances(_BaseSecondOrderMeasure):
    """Provides the variance of the row-proportions measure for a matrix.

    Row-proportions-variance is a 2D np.float64 ndarray of p * (1 - p) where p is the
    row-proportions.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        p = self._second_order_measures.row_proportions.blocks

        return [
            [
                # --- base values ---
                p[0][0] * (1 - p[0][0]),
                # --- inserted columns ---
                p[0][1] * (1 - p[0][1]),
            ],
            [
                # --- inserted rows ---
                p[1][0] * (1 - p[1][0]),
                # --- intersections ---
                p[1][1] * (1 - p[1][1]),
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
                    (sums_blocks[0][0].T / np.nansum(sums_blocks[0][0], axis=1)).T,
                    # --- inserted columns ---
                    (sums_blocks[0][1].T / np.nansum(sums_blocks[0][0], axis=1)).T,
                ],
                [
                    # --- inserted rows ---
                    (sums_blocks[1][0].T / np.nansum(sums_blocks[1][0], axis=1)).T,
                    # --- intersections ---
                    (sums_blocks[1][1].T / np.nansum(sums_blocks[1][1], axis=1)).T,
                ],
            ]


class _RowStandardError(_BaseSecondOrderMeasure):
    """Provides the standard errors of the row-proportions measure for a matrix.

    Row-standard-errors is a 2D np.float64 ndarray of the row proportion variance
    divided by the row weighted bases.
    """

    @lazyproperty
    def blocks(self):
        """list of lists of the four 2D "blocks" making up this measure."""
        variance_blocks = self._second_order_measures.row_proportion_variances.blocks
        weighted_base_blocks = self._second_order_measures.row_weighted_bases.blocks

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    np.sqrt(variance_blocks[0][0] / weighted_base_blocks[0][0]),
                    # --- inserted columns ---
                    np.sqrt(variance_blocks[0][1] / weighted_base_blocks[0][1]),
                ],
                [
                    # --- inserted rows ---
                    np.sqrt(variance_blocks[1][0] / weighted_base_blocks[1][0]),
                    # --- intersections ---
                    np.sqrt(variance_blocks[1][1] / weighted_base_blocks[1][1]),
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
        # --- subtotal-rows is the same. In the ARRAY_X case there can be no subtotal rows
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

        return np.broadcast_to(self._base_values[:, 0][:, None], subtotal_columns.shape)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of row-subtotal row-proportions denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- Summing works on rows because row-proportion denominators add along this
        # --- axis. This wouldn't work on MR-rows but there can be no subtotals on an
        # --- ARRAY dimension (ARRAY_X slice) so that case never arises.
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


class _TableProportions(_BaseSecondOrderMeasure):
    """Provides the table-proportions measure for a matrix.

    Table-proportions is a 2D np.float64 ndarray of the proportion of its count
    contributed by the weighted count of each matrix cell.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.

        Table-proportions are weighted counts divided by the table weighted bases.
        """
        count_blocks = self._second_order_measures.weighted_counts.blocks
        weighted_base_blocks = self._second_order_measures.table_weighted_bases.blocks

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


class _TableProportionVariances(_BaseSecondOrderMeasure):
    """Provides the variance of the table-proportions measure for a matrix.

    Table-proportions-variance is a 2D np.float64 ndarray of p * (1 - p) where p is the
    table-proportions.
    """

    @lazyproperty
    def blocks(self):
        """Nested list of the four 2D ndarray "blocks" making up this measure.

        These are the base-values, the column-subtotals, the row-subtotals, and the
        subtotal intersection-cell values.
        """
        p = self._second_order_measures.table_proportions.blocks

        return [
            [
                # --- base values ---
                p[0][0] * (1 - p[0][0]),
                # --- inserted columns ---
                p[0][1] * (1 - p[0][1]),
            ],
            [
                # --- inserted rows ---
                p[1][0] * (1 - p[1][0]),
                # --- intersections ---
                p[1][1] * (1 - p[1][1]),
            ],
        ]


class _TableStandardError(_BaseSecondOrderMeasure):
    """Provides the standard errors of the table-proportions measure for a matrix.

    Table-standard-errors is a 2D np.float64 ndarray of the table proportion variance
    divided by the table weighted bases.
    """

    @lazyproperty
    def blocks(self):
        variance_blocks = self._second_order_measures.table_proportion_variances.blocks
        weighted_base_blocks = self._second_order_measures.table_weighted_bases.blocks

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return [
                [
                    # --- base values ---
                    np.sqrt(variance_blocks[0][0] / weighted_base_blocks[0][0]),
                    # --- inserted columns ---
                    np.sqrt(variance_blocks[0][1] / weighted_base_blocks[0][1]),
                ],
                [
                    # --- inserted rows ---
                    np.sqrt(variance_blocks[1][0] / weighted_base_blocks[1][0]),
                    # --- intersections ---
                    np.sqrt(variance_blocks[1][1] / weighted_base_blocks[1][1]),
                ],
            ]


class _TableUnweightedBases(_BaseSecondOrderMeasure):
    """Provides the table-unweighted-bases measure for a matrix.

    table-unweighted-bases is a 2D np.float64 ndarray of the (weighted) table-proportion
    denominator (base) for each matrix cell.
    """

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of weighted table-proportion denominator per cell."""
        return self._unweighted_cube_counts.table_bases

    @lazyproperty
    def _intersections_shape(self):
        """Tuple of (number of row insertions, number of column insertions)"""
        return tuple(len(dimension.subtotals) for dimension in self._dimensions)

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- There are two cases.
        # --- Case 1: Intersections are empty, we can broadcast any value of the
        # --- right dtype and it will be correct, because it is empty.
        # --- Case 2: There is an intersection, so both dimensions have subtotals,
        # --- meaning they must both be CAT (because ARRAYs don't have subtotals).

        # --- In either case, we can broadcast any value from the base values to the
        # --- shape and have the correct answer.
        return np.broadcast_to(self._base_values[0, 0], self._intersections_shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted-column table-proportion denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- There are two cases.
        # --- Case 1: There are no column-subtotals, we can broadcast any column np.array
        # --- of the right dtype and it will become the right answer because it is empty.
        # --- Case 2: There are column-subtotals, so we know that the columns dimension
        # --- is not ARRAY, and therefore each column is the same. Therefore, we can
        # --- get a column from the base values (rotate it to appease numpy), and then
        # --- broadcast it to the correct shape.

        # --- In either case, we can broadcast a rotated column from the base values to
        # --- the shape and have the correct answer.
        shape = (self._base_values.shape[0], self._intersections_shape[1])
        return np.broadcast_to(self._base_values[:, 0][:, None], shape)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of inserted-row table-proportion denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- There are two cases.
        # --- Case 1: There are no row-subtotal, we can broadcast any row np.array
        # --- of the right dtype and it will become the right answer because it is empty.
        # --- Case 2: There are row-subtotals, so we know that the columns dimension
        # --- is not ARRAY, and therefore each row is the same. Therefore, we can
        # --- get a row from the base values, and then broadcast it to the correct shape.

        # --- In either case, we can broadcast a row from the base values to
        # --- the shape and have the correct answer.
        shape = (self._intersections_shape[0], self._base_values.shape[1])
        return np.broadcast_to(self._base_values[0, :], shape)


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
    def _intersections_shape(self):
        """Tuple of (number of row insertions, number of column insertions)"""
        return tuple(len(dimension.subtotals) for dimension in self._dimensions)

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of intersection values.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        This is the fourth and final "block" required by the assembler.
        """
        # --- There are two cases.
        # --- Case 1: Intersections are empty, we can broadcast any value of the
        # --- right dtype and it will be correct, because it is empty.
        # --- Case 2: There is an intersection, so both dimensions have subtotals,
        # --- meaning they must both be CAT (because ARRAYs don't have subtotals).

        # --- In either case, we can broadcast any value from the base values to the
        # --- shape and have the correct answer.
        return np.broadcast_to(self._base_values[0, 0], self._intersections_shape)

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of inserted-column table-proportion denominator values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        # --- There are two cases.
        # --- Case 1: There are no column-subtotals, we can broadcast any column np.array
        # --- of the right dtype and it will become the right answer because it is empty.
        # --- Case 2: There are column-subtotals, so we know that the columns dimension
        # --- is not ARRAY, and therefore each column is the same. Therefore, we can
        # --- get a column from the base values (rotate it to appease numpy), and then
        # --- broadcast it to the correct shape.

        # --- In either case, we can broadcast a rotated column from the base values to
        # --- the shape and have the correct answer.
        shape = (self._base_values.shape[0], self._intersections_shape[1])
        return np.broadcast_to(self._base_values[:, 0][:, None], shape)

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of inserted-row table-proportion denominator values.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        # --- There are two cases.
        # --- Case 1: There are no row-subtotal, we can broadcast any row np.array
        # --- of the right dtype and it will become the right answer because it is empty.
        # --- Case 2: There are row-subtotals, so we know that the columns dimension
        # --- is not ARRAY, and therefore each row is the same. Therefore, we can
        # --- get a row from the base values, and then broadcast it to the correct shape.

        # --- In either case, we can broadcast a row from the base values to
        # --- the shape and have the correct answer.
        shape = (self._intersections_shape[0], self._base_values.shape[1])
        return np.broadcast_to(self._base_values[0, :], shape)


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
                    sums_blocks[0][0] / np.nansum(sums_blocks[0][0]),
                    # --- inserted columns ---
                    sums_blocks[0][1] / np.nansum(sums_blocks[0][0]),
                ],
                [
                    # --- inserted rows ---
                    sums_blocks[1][0] / np.nansum(sums_blocks[1][0]),
                    # --- intersections ---
                    sums_blocks[1][1] / np.nansum(sums_blocks[1][1]),
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
            self._unweighted_cube_counts.counts,
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
            self._weighted_cube_counts.counts,
            self._dimensions,
            diff_cols_nan=diff_nans,
            diff_rows_nan=diff_nans,
        )


class _Zscores(_BaseSecondOrderMeasure):
    """Provides the zscore measure for a matrix.

    A z-score is also known as a *standard score* and is the number of standard
    deviations above (positive) or below (negative) the population mean each cell's
    value is.
    """

    @lazyproperty
    def blocks(self):
        """2D array of the four 2D "blocks" making up this measure."""
        dimension_types = tuple(d.dimension_type for d in self._dimensions)
        if DT.MR_SUBVAR in dimension_types:
            return NanSubtotals.blocks(self._base_values, self._dimensions)
        return [
            [self._base_values, self._subtotal_columns],
            [self._subtotal_rows, self._intersections],
        ]

    @lazyproperty
    def _base_values(self):
        """2D np.float64 ndarray of zscore for each body cell."""
        return self._calculate_zscores(
            self._second_order_measures.weighted_counts.blocks[0][0],
            self._second_order_measures.table_weighted_bases.blocks[0][0],
            self._second_order_measures.row_weighted_bases.blocks[0][0],
            self._second_order_measures.column_weighted_bases.blocks[0][0],
        )

    def _calculate_zscores(self, counts, table_bases, row_bases, column_bases):
        """2D np.float64 ndarray of zscore value for each cell of the matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        if self._is_defective:
            return np.full(counts.shape, np.nan)

        # --- to avoid precision errors, check if it'll be 0/0 before other
        # --- calculations and set to nan
        if np.all(table_bases == row_bases) or np.all(table_bases == column_bases):
            return np.full(counts.shape, np.nan)

        expected_counts = row_bases * column_bases / table_bases
        variance = (
            row_bases
            * column_bases
            * (table_bases - row_bases)
            * (table_bases - column_bases)
            / table_bases ** 3
        )
        return (counts - expected_counts) / np.sqrt(variance)

    @lazyproperty
    def _intersections(self):
        """(n_row_subtotals, n_col_subtotals) ndarray of zscores for intersections.

        An intersection value arises where a row-subtotal crosses a column-subtotal.
        """
        return self._calculate_zscores(
            self._second_order_measures.weighted_counts.blocks[1][1],
            self._second_order_measures.table_weighted_bases.blocks[1][1],
            self._second_order_measures.row_weighted_bases.blocks[1][1],
            self._second_order_measures.column_weighted_bases.blocks[1][1],
        )

    @lazyproperty
    def _is_defective(self):
        """Bool indicating whether the matrix is defective

        Consider it "defective" if it doesn't have at least two rows and two columns
        in the base values that are "full" of data. In this case we won't calculate
        zscores for any block.
        """
        counts = self._second_order_measures.weighted_counts.blocks[0][0]
        return not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2

    @lazyproperty
    def _subtotal_columns(self):
        """2D np.float64 ndarray of zscore values.

        This is the second "block" and has the shape (n_rows, n_col_subtotals).
        """
        return self._calculate_zscores(
            self._second_order_measures.weighted_counts.blocks[0][1],
            self._second_order_measures.table_weighted_bases.blocks[0][1],
            self._second_order_measures.row_weighted_bases.blocks[0][1],
            self._second_order_measures.column_weighted_bases.blocks[0][1],
        )

    @lazyproperty
    def _subtotal_rows(self):
        """2D np.float64 ndarray of zscores for subtotal rows.

        This is the third "block" and has the shape (n_row_subtotals, n_cols).
        """
        return self._calculate_zscores(
            self._second_order_measures.weighted_counts.blocks[1][0],
            self._second_order_measures.table_weighted_bases.blocks[1][0],
            self._second_order_measures.row_weighted_bases.blocks[1][0],
            self._second_order_measures.column_weighted_bases.blocks[1][0],
        )


# === MARGINALS ===


class _BaseMarginal:
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
            f"{type(self).__name__} must implement `.blocks`"
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
            # --- Use *column* comparable counts (going across rows) ---
            counts = self._second_order_measures.column_comparable_counts.blocks
            # --- Get base values & *row* subtotals
            return [counts[0][0], counts[1][0]]
        # --- Use *row* comparable counts (going across columns) ---
        counts = self._second_order_measures.row_comparable_counts.blocks
        # --- Get base values & *column* subtotals
        return [counts[0][0], counts[0][1]]

    @lazyproperty
    def _counts_are_defined(self):
        """Bool indicating whether counts are defined for this orientation

        Counts are undefined across subvariable dimensions.
        """
        if self.orientation == MO.ROWS:
            return self._second_order_measures.column_comparable_counts.is_defined
        return self._second_order_measures.row_comparable_counts.is_defined


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


class _MarginTableProportion(_BaseMarginal):
    """The 'margin-table-proportion', table proportion of the marginal values

    The name is a bit of a mouthful, but each component is meaningful.
    * "margin": Indicates it is a marginal eg 1D (either in rows or columns orientation)
    * "table": Indicates that it is the proportion for the whole table, meaning it uses
      the weighted `_MarginTableBase` as the denominator.
    * "proportion": It is a proportion, eg the `_MarginWeightedBase` divided by the
      weighted `_MarginTableBase`

    Note that there is an implied "weighted" here, which we do not spell out because we
    never calculate unweighted proportions.

    Also note we don't actually use the `_MarginWeightedBase` as the numerator here
    because we want values for subtotal differences. See `._proportion_numerators`
    """

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the summed count margin.

        These are the base-values and the subtotals.
        """
        return [
            self._proportion_numerators[0] / self._proportion_denominators[0],
            self._proportion_numerators[1] / self._proportion_denominators[1],
        ]

    @lazyproperty
    def is_defined(self):
        """True if counts are defined."""
        return self._counts_are_defined

    @lazyproperty
    def _proportion_numerators(self):
        """list of ndarray of counts to be used as numerator for the margin proportions

        We want the numerator to exist even along subtotal differences, so we cannot use
        the "rows/columns -table-weighted-base". This is because of the dual nature of
        bases on non-MR variables, where bases are just the sum of counts and there's no
        "selected" diension to sum over. Therefore, the "margin-weighted-base" could
        also be called the "margin-weighted-count" if you think of the margin as a
        stripe, and for the proportions we want to use the count as the numerator. For
        non-array variables (all situations where there is a 1D table-weighted-base),
        this "base" and "count" are equal for the base values, but we set the base to
        `NaN` for subtotal differences for bases, but we actually want the values for
        counts. Since this is the only place we use this, and there are already a lot of
        overlapping names, we just calculate the numerators privately in this class.
        """
        counts = self._second_order_measures.weighted_counts.blocks

        if self.orientation == MO.ROWS:
            # --- Now get only the base values and relevant row subtotals
            counts = [counts[0][0], counts[1][0]]
        else:
            # --- Now get only the base values and relevant column subtotals
            counts = [counts[0][0], counts[0][1]]

        return [self._apply_along_orientation(np.sum, count) for count in counts]

    @lazyproperty
    def _proportion_denominators(self):
        """list of ndarray 1D table margins to be used as denominator"""
        if self.orientation == MO.ROWS:
            return self._second_order_measures.rows_table_weighted_base.blocks
        return self._second_order_measures.columns_table_weighted_base.blocks


class _MarginTableBase(_BaseMarginal):
    """The 'margin-table-bases', or the denominator for margin-table-proportion

    A consistently 1D form of what used to be called the table_margin/table_base. The
    name is a mouthful, but each component is important.

    * "margin": Indicates it is a marginal eg 1D (either in rows or columns orientation)
    * "table": Indicates that it is the base for the whole table. When the
      `.table_weighted_base` exists (CAT X CAT), it is a repetition of that, but when
      the corresponding dimension is an array (and therefore we can't sum across them),
      each cell has its own value.
    * "base": Indicates that it is the base, not necessarily the counts (eg the sum
      of selected and non-selected for MR variables)

    Depending on the cube_counts passed in, can be weighted or unweighted.
    """

    def __init__(
        self, dimensions, second_order_measures, cube_measures, orientation, cube_counts
    ):
        super(_MarginTableBase, self).__init__(
            dimensions, second_order_measures, cube_measures, orientation
        )
        self._cube_counts = cube_counts

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the summed count margin."""
        if not self.is_defined:
            raise ValueError(
                "Could not calculate margin-table-base across subvariables"
            )

        # --- There are 2 cases if defined.
        # --- 1) The corresponding dimension is array, in which case there can be no subtotals
        # --- and so repeating any value to the shape of the subtotals gives the correct
        # --- empty values.
        # --- 2) The corresponding dimension is not array, in which case, the table-margin
        # --- is always the same, so repeating any value to the shape is correct.
        # --- Therefore we can just repeat the first value to the shape.
        return [
            self._base_values,
            np.repeat([self._base_values[0]], self._subtotal_shape),
        ]

    @lazyproperty
    def is_defined(self):
        """Bool indicating whether the margin is defined in this orientation

        True if the cube measure has the base values we need (eg the corresponding
        dimension is not array).
        """
        return self._base_values is not None

    @lazyproperty
    def _base_values(self):
        """Optional np.ndarray of float margin-table-base from the cube measure"""
        return (
            self._cube_counts.rows_table_base
            if self.orientation == MO.ROWS
            else self._cube_counts.columns_table_base
        )

    @lazyproperty
    def _subtotal_shape(self):
        """Int indicating the number of subtotals given the orientation"""
        if self.orientation == MO.ROWS:
            return len(self._dimensions[0].subtotals)
        return len(self._dimensions[1].subtotals)


class _MarginUnweightedBase(_BaseMarginal):
    """The 'margin-unweighted base', a 1D weighted base in the margin

    This is the sum of the unweighted counts across a non-array dimension, It is called
    "Unweighted N" when put in the margin of  the web app. Since we cannot add across
    array variables, we cannot reduce the dimensionality to get to a margin when there
    are arrays, each cell has it's own value.
    """

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the summed count margin.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError("Cannot calculate base across subvariables dimension.")

        # --- Because we know that the corresponding dimension is not array (otherwise
        # --- the margin-unweighted-base is not defined), we know that the 2D row/column
        # --- unweighted base is a repetition in the orientation we need. Therefore,
        # --- we can get a row/column of the 2D dimension and use it so that we don't
        # --- have to recalculate the subtotals.
        if self.orientation == MO.ROWS:
            bases = self._second_order_measures.row_unweighted_bases.blocks
            return [bases[0][0][:, 0], bases[1][0][:, 0]]
        bases = self._second_order_measures.column_unweighted_bases.blocks
        return [bases[0][0][0, :], bases[0][1][0, :]]

    @lazyproperty
    def is_defined(self):
        """True if counts are defined."""
        return self._counts_are_defined


class _MarginWeightedBase(_BaseMarginal):
    """The 'margin-weighted base', a 1D weighted base in the margin

    This is the sum of the weighted counts across a non-array dimension, It is called
    "Weighted N" when put in the margin of  the web app. Since we cannot add across array
    variables, we cannot reduce the dimensionality to get to a margin when there are
    arrays, each cell has it's own value.
    """

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the summed count margin.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError(
                "Could not calculate weighted-base-margin across subvariables"
            )

        # --- Because we know that the corresponding dimension is not array (otherwise
        # --- the margin-weighted-base is not defined), we know that the 2D row/column
        # --- weighted base is a repetition in the orientation we need. Therefore,
        # --- we can get a row/column of the 2D dimension and use it so that we don't
        # --- have to recalculate the subtotals.
        if self.orientation == MO.ROWS:
            bases = self._second_order_measures.row_weighted_bases.blocks
            return [bases[0][0][:, 0], bases[1][0][:, 0]]
        bases = self._second_order_measures.column_weighted_bases.blocks
        return [bases[0][0][0, :], bases[0][1][0, :]]

    @lazyproperty
    def is_defined(self):
        """Bool indicating whether the margin is defined in this orientation

        True if the cube measure has the base values we need (eg the corresponding
        dimension is not array).
        """
        return self._base_values is not None

    @lazyproperty
    def _base_values(self):
        """Optional np.ndarray of float margin-base from the cube measure"""
        return (
            self._cube_measures.weighted_cube_counts.rows_base
            if self.orientation == MO.ROWS
            else self._cube_measures.weighted_cube_counts.columns_base
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
                f"{self.orientation.value}-scale-mean is undefined if no numeric values"
                " are defined on opposing dimension."
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


class _ScaleMeanSmoothed(_ScaleMean, _SmoothedMeasure):
    """Provides the scale mean marginals smoothed for a matrix if available."""

    @lazyproperty
    def _proportions(self):
        """List of 2 ndarray of the relevant proportion blocks"""
        smoother = self._smoother
        props = self._second_order_measures.column_proportions.blocks
        # --- Get base values & *column* subtotals
        return [
            smoother.smooth(props[0][0]),
            smoother.smooth(props[0][1]),
        ]


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
                f"{self.orientation.value}-scale-median is undefined if no numeric values are defined on "
                "opposing dimension."
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
                f"{self.orientation.value}-scale-mean-standard-deviation is undefined if no numeric values "
                "are defined on opposing dimension."
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


class _ScaleMeanStderr(_BaseScaledCountMarginal):
    """Provides the std. error of the marginal scale means."""

    @lazyproperty
    def blocks(self):
        """List of the 2 1D ndarray "blocks" of the scale mean standard deviation.

        These are the base-values and the subtotals.
        """
        if not self.is_defined:
            raise ValueError(
                f"{self.orientation.value}-scale-mean-standard-error is undefined if "
                "no numeric values are defined on opposing dimension or if the "
                "corresponding dimension has no margin."
            )

        return [
            self._scale_mean_stddev.blocks[0] / np.sqrt(self._margin.blocks[0]),
            self._scale_mean_stddev.blocks[1] / np.sqrt(self._margin.blocks[1]),
        ]

    @lazyproperty
    def is_defined(self):
        """True if both the scale mean is defined and the margin is defined."""
        return self._scale_mean_stddev.is_defined and self._margin.is_defined

    @lazyproperty
    def _margin(self):
        """The margin for the opposing dimension, used in the denominator."""
        return (
            self._second_order_measures.rows_weighted_base
            if self.orientation == MO.ROWS
            else self._second_order_measures.columns_weighted_base
        )

    @lazyproperty
    def _scale_mean_stddev(self):
        """The scale mean std-dev for the opposing dimension, used in the nunmerator."""
        return (
            self._second_order_measures.rows_scale_mean_stddev
            if self.orientation == MO.ROWS
            else self._second_order_measures.columns_scale_mean_stddev
        )


# === SCALAR TABLE VALUES ===


class _BaseTableValue:
    """Base class for all (second-order) table values

    Table values are values that apply to the whole table, which are often scalars, but
    not necessarily (eg the table-weighted-base-range is a length 2 array, but these
    values apply to the whole table.)
    """

    def __init__(self, dimensions, second_order_measures, cube_measures):
        self._dimensions = dimensions
        self._second_order_measures = second_order_measures
        self._cube_measures = cube_measures

    @lazyproperty
    def is_defined(self):
        """Whether scalar is defined. Defaults to True, but designed to be overridden"""
        return True

    @lazyproperty
    def value(self):
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `.value`"
        )


class _TableBase(_BaseTableValue):
    """The 'table-weighted-base', a scalar base for the whole table

    The table-weighted-base is only defined when both rows and columns dimensions
    are non-array and is equal to the sum of all counts. It was formerly known as the
    'table-margin' when it was a scalar (which was also only when neither dimension is
    an arary).
    """

    def __init__(self, dimensions, second_order_measures, cube_measures, cube_counts):
        super(_TableBase, self).__init__(
            dimensions,
            second_order_measures,
            cube_measures,
        )
        self._cube_counts = cube_counts

    @lazyproperty
    def is_defined(self):
        """Bool indicating whether the table-margin is defined

        It is defined (as a scalar) only when available on the weighted_cube_counts
        (eg neither dimension is an array)
        """
        return self._cube_counts.table_base is not None

    @lazyproperty
    def value(self):
        """float of the table-margin"""
        if not self.is_defined:
            raise ValueError(
                "Cannot sum across subvariables dimension for table base scalar"
            )
        return self._cube_counts.table_base


class _TableBasesRange(_BaseTableValue):
    """The (unpruned) 'table-weighted-bases-range', a length 2 np.array of min and max.

    The range of the table-weighted-bases *before* pruning, eg removing rows and columns
    that are hidden or empty.
    """

    def __init__(self, dimensions, second_order_measures, cube_measures, cube_counts):
        super(_TableBasesRange, self).__init__(
            dimensions,
            second_order_measures,
            cube_measures,
        )
        self._cube_counts = cube_counts

    @lazyproperty
    def value(self):
        """[min, max] np.float64 ndarray range of the table-weighted-base"""
        bases = self._cube_counts.table_bases
        return np.array([np.min(bases), np.max(bases)])


# === PAIRWISE HELPERS ===


class _PairwiseSignificaneBetweenSubvariablesHelper:
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
