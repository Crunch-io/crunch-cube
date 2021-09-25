# encoding: utf-8

"""The `Assembler` object provides the external interface for this module.

Its name derives from its role to "assemble" a finished 2D array ("matrix") for a
particular measure from the base measure values and inserted subtotals, to reorder the
rows and columns according to the dimension *order* transforms, and to hide rows and
columns that are either hidden by the user or "pruned" because they contain no
observations.
"""

import numpy as np

from cr.cube.collator import (
    ExplicitOrderCollator,
    PayloadOrderCollator,
    SortByValueCollator,
)
from cr.cube.enums import (
    COLLATION_METHOD as CM,
    DIMENSION_TYPE as DT,
    MARGINAL,
    MARGINAL_ORIENTATION as MO,
    MEASURE as M,
)
from cr.cube.matrix.measure import SecondOrderMeasures
from cr.cube.matrix.subtotals import SumSubtotals
from cr.cube.util import lazyproperty


class Assembler:
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
    def column_aliases(self):
        """1D str ndarray of column alias for each matrix row.

        These are suitable for use as column headings; alias for subtotal columns appear
        in the sequence and alias are ordered to correspond with their respective data
        column.
        """
        return self._dimension_aliases(self._columns_dimension, self._column_order)

    @lazyproperty
    def column_codes(self):
        """1D int ndarray of column codes for each matrix row.

        These are suitable for use as column headings; codes for subtotal columns appear
        in the sequence and codes are ordered to correspond with their respective data
        column.
        """
        return self._dimension_codes(self._columns_dimension, self._column_order)

    @lazyproperty
    def column_comparable_counts(self):
        """2D np.float64 ndarray of counts comparable in the column dimension.

        Comparable in the "column" dimension means that it's not a column subtotal
        difference because these are not considered comparable since their bases
        aren't the same. The values along these subtotal differences are defined as
        np.nan.
        """
        return self._assemble_matrix(self._measures.column_comparable_counts.blocks)

    @lazyproperty
    def column_index(self):
        """2D np.float64 ndarray of column-index "percentage" for each table cell."""
        return self._assemble_matrix(self._measures.column_index.blocks)

    @lazyproperty
    def column_labels(self):
        """1D str ndarray of column name for each matrix column.

        These are suitable for use as column headings; labels for subtotal columns
        appear in the sequence and labels are ordered to correspond with their
        respective data column.
        """
        return self._dimension_labels(self._columns_dimension, self._column_order)

    @lazyproperty
    def column_proportions(self):
        """2D np.float64 ndarray of column-proportion for each matrix cell.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        column the cell appears in (aka. column-margin). Generally a number between 0.0
        and 1.0 inclusive, but subtotal differences can be between -1.0 and 1.0
        inclusive.
        """
        return self._assemble_matrix(self._measures.column_proportions.blocks)

    @lazyproperty
    def column_proportion_variances(self):
        """2D ndarray of np.float64 column-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.column_proportion_variances.blocks)

    @lazyproperty
    def column_share_sum(self):
        """2D optional np.float64 ndarray of column share of sum for each cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.
        """
        return self._assemble_matrix(self._measures.column_share_sum.blocks)

    @lazyproperty
    def column_std_err(self):
        """2D np.float64 ndarray of std-error of column-percent for each matrix cell."""
        return self._assemble_matrix(self._measures.column_std_err.blocks)

    @lazyproperty
    def column_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted col-proportions denominator per cell."""
        return self._assemble_matrix(self._measures.column_unweighted_bases.blocks)

    @lazyproperty
    def column_weighted_bases(self):
        """2D np.float64 ndarray of column-proportions denominator for each cell."""
        return self._assemble_matrix(self._measures.column_weighted_bases.blocks)

    @lazyproperty
    def columns_base(self):
        """1D/2D np.float64 ndarray of unweighted-N for each slice column/cell."""
        # --- an MR_X slice produces a 2D columns-base (each cell has its own N) ---
        # --- This is really just another way to call the columns_weighted_bases ---
        # --- TODO: Should column_base only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a
        # --- "column_base_range"
        if not self._measures.columns_unweighted_base.is_defined:
            return self.column_unweighted_bases

        # --- otherwise columns-base is a vector ---
        return self._assemble_marginal(self._measures.columns_unweighted_base)

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
        # --- This is really just another way to call the columns_weighted_bases ---
        # --- TODO: Should column_margin only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a
        # --- "column_margin_range"
        if not self._measures.columns_weighted_base.is_defined:
            return self.column_weighted_bases

        # --- otherwise columns-base is a vector ---
        return self._assemble_marginal(self._measures.columns_weighted_base)

    @lazyproperty
    def columns_margin_proportion(self):
        """1D/2D np.float64 ndarray of weighted-prop for each column of this slice."""
        # --- an MR_X slice produces a 2D columns-margin (each cell has its own N) ---
        # --- TODO: Should colums_margin_proportion only be defined when it's 1D? This
        # --- requires changes to exporter to use the bases to give a
        # --- "columns_margin_range"
        if not self._measures.columns_table_proportion.is_defined:
            return self._assemble_matrix(
                SumSubtotals.blocks(
                    self.columns_margin / self.table_weighted_bases,
                    self._dimensions,
                )
            )

        # --- otherwise columns-margin-proportion is a maginal ---
        return self._assemble_marginal(self._measures.columns_table_proportion)

    @lazyproperty
    def columns_scale_mean(self):
        """Optional 1D column scale mean marginal for this cube-result"""
        return self._assemble_marginal(self._measures.columns_scale_mean)

    @lazyproperty
    def columns_scale_mean_stddev(self):
        """Optional 1D column scale mean std dev marginal for this cube-result"""
        return self._assemble_marginal(self._measures.columns_scale_mean_stddev)

    @lazyproperty
    def columns_scale_mean_stderr(self):
        """Optional 1D column scale mean std err marginal for this cube-result"""
        return self._assemble_marginal(self._measures.columns_scale_mean_stderr)

    @lazyproperty
    def columns_scale_median(self):
        """Optional 1D column scale median marginal for this cube-result"""
        return self._assemble_marginal(self._measures.columns_scale_median)

    @lazyproperty
    def derived_column_idxs(self):
        """tuple(int) of derived column elements' indexes, can be empty.

        An element is derived if it's a subvariable of a multiple response dimension,
        which has been produced by the zz9, and inserted into the response data.

        All other elements, including regular MR and CA subvariables, as well as
        categories of CAT dimensions, are not derived. Subtotals are also not derived
        in this sense, because they're not even part of the data (elements).
        """
        return self._derived_element_idxs(self._columns_dimension, self._column_order)

    @lazyproperty
    def derived_row_idxs(self):
        """tuple(int) of derived row elements' indexes, can be empty.

        An element is derived if it's a subvariable of a multiple response dimension,
        which has been produced by the zz9, and inserted into the response data.

        All other elements, including regular MR and CA subvariables, as well as
        categories of CAT dimensions, are not derived. Subtotals are also not derived
        in this sense, because they're not even part of the data (elements).
        """
        return self._derived_element_idxs(self._rows_dimension, self._row_order)

    @lazyproperty
    def diff_row_idxs(self):
        """tuple(int) of difference row elements' indexes, can be empty."""
        return self._diff_element_idxs(self._rows_dimension, self._row_order)

    @lazyproperty
    def diff_column_idxs(self):
        """tuple(int) of difference column elements' indexes, can be empty."""
        return self._diff_element_idxs(self._columns_dimension, self._column_order)

    @lazyproperty
    def inserted_column_idxs(self):
        """tuple of int index of each subtotal column in slice."""
        # --- insertions have a negative idx in their order sequence ---
        return tuple(i for i, col_idx in enumerate(self._column_order) if col_idx < 0)

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each subtotal row in slice."""
        # --- insertions have a negative idx in their order sequence ---
        return tuple(i for i, row_idx in enumerate(self._row_order) if row_idx < 0)

    @lazyproperty
    def means(self):
        """2D optional np.float64 ndarray of mean for each cell.

        Raises `ValueError` if the cube-result does not include a means cube-measure.
        """
        return self._assemble_matrix(self._measures.means.blocks)

    def pairwise_indices(self, alpha, only_larger):
        """2D optional ndarray of tuple of int column-idxs pairwise-t threshold.

        Raises `ValueError if the cube-result does not include `overlaps` cube-measures.
        """
        return np.array(
            [
                self._pairwise_indices(
                    self.pairwise_significance_p_vals(col),
                    self.pairwise_significance_t_stats(col),
                    alpha,
                    only_larger,
                )
                for col in range(len(self._column_order))
            ]
        ).T

    def pairwise_means_indices(self, alpha, only_larger):
        """2D optional ndarray of tuple of int column-idxs means pairwise-t threshold.

        Raises `ValueError if the cube-result does not include `means` cube-measures.
        """
        return np.array(
            [
                self._pairwise_indices(
                    self.pairwise_significance_means_p_vals(col),
                    self.pairwise_significance_means_t_stats(col),
                    alpha,
                    only_larger,
                )
                for col in range(len(self._column_order))
            ]
        ).T

    def pairwise_significance_p_vals(self, column_idx):
        """2D optional np.float64 ndarray of overlaps-p_vals matrices for subvar idx.

        For cubes where the last dimension is categorical, column idxs represent
        specific categories.

        For cubes where the last dimension is a multiple response, each subvariable
        pairwise significance matrix is a 2D ndarray of the p-vals for the selected
        subvariable index (the selected column).

        Raises `ValueError if the cube-result does not include `overlaps`
        and `valid_overlaps` cube-measures.
        """
        if self._cube_has_overlaps:
            # If overlaps are defined, calculate significance based on them
            return self._assemble_matrix(
                self._measures.pairwise_p_vals_for_subvar(column_idx).blocks
            )
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_p_vals(base_column_idx).blocks
        )

    def pairwise_significance_t_stats(self, column_idx):
        """2D optional np.float64 ndarray of overlaps-t_stats matrices for subvar idx.

        For cubes where the last dimension is categorical, column idxs represent
        specific categories.

        For cubes where the last dimension is a multiple response, each subvariable
        pairwise significance matrix is a 2D ndarray of the t-stats for the selected
        subvariable index (the selected column).

        Raises `ValueError if the cube-result does not include `overlaps`
        and `valid_overlaps` cube-measures.
        """
        if self._cube_has_overlaps:
            # If overlaps are defined, calculate significance based on them
            return self._assemble_matrix(
                self._measures.pairwise_t_stats_for_subvar(column_idx).blocks
            )
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_t_stats(base_column_idx).blocks
        )

    def pairwise_significance_means_p_vals(self, column_idx):
        """2D optional np.float64 ndarray of mean difference p_vals for column idx.

        Raises `ValueError if the cube-result does not include `mean` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_significance_means_p_vals(base_column_idx).blocks
        )

    def pairwise_significance_means_t_stats(self, column_idx):
        """2D optional np.float64 ndarray of mean difference t_stats for column idx.

        Raises `ValueError if the cube-result does not include `mean` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_significance_means_t_stats(base_column_idx).blocks
        )

    @lazyproperty
    def population_proportions(self):
        """2D np.float64 ndarray of proportions

        The proportion used to calculate proportion counts depends on the dimension
        types.
        """
        return self._assemble_matrix(self._measures.population_proportions.blocks)

    @lazyproperty
    def population_std_err(self):
        """2D np.float64 ndarray of standard errors

        The proportion used to calculate proportion counts depends on the dimension
        types.
        """
        return self._assemble_matrix(self._measures.population_std_err.blocks)

    @lazyproperty
    def pvalues(self):
        """2D np.float64/np.nan ndarray of p-value for each matrix cell."""
        return self._assemble_matrix(self._measures.pvalues.blocks)

    @lazyproperty
    def row_comparable_counts(self):
        """2D np.float64 ndarray of counts comparable in the row dimension.

        Comparable in the "row" dimension means that it's not a row subtotal
        difference because these are not considered comparable since their bases
        aren't the same. The values along these subtotal differences are defined as
        np.nan.
        """
        return self._assemble_matrix(self._measures.row_comparable_counts.blocks)

    @lazyproperty
    def row_aliases(self):
        """1D str ndarray of row alias for each matrix row.

        These are suitable for use as row headings; alias for subtotal rows appear in
        the sequence and alias are ordered to correspond with their respective data
        row.
        """
        return self._dimension_aliases(self._rows_dimension, self._row_order)

    @lazyproperty
    def row_codes(self):
        """1D int ndarray of row codes for each matrix row.

        These are suitable for use as row headings; codes for subtotal rows appear in
        the sequence and codes are ordered to correspond with their respective data
        row.
        """
        return self._dimension_codes(self._rows_dimension, self._row_order)

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of row name for each matrix row.

        These are suitable for use as row headings; labels for subtotal rows appear in
        the sequence and labels are ordered to correspond with their respective data
        row.
        """
        return self._dimension_labels(self._rows_dimension, self._row_order)

    @lazyproperty
    def row_proportions(self):
        """2D np.float64 ndarray of row-proportion for each matrix cell.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        row the cell appears in (aka. row-margin). Generally a number between 0.0 and
        1.0 inclusive, but subtotal differences can be between -1.0 and 1.0 inclusive.
        """
        return self._assemble_matrix(self._measures.row_proportions.blocks)

    @lazyproperty
    def row_proportion_variances(self):
        """2D ndarray of np.float64 row-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.row_proportion_variances.blocks)

    @lazyproperty
    def row_share_sum(self):
        """2D optional np.float64 ndarray of row share of sum for each cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.
        """
        return self._assemble_matrix(self._measures.row_share_sum.blocks)

    @lazyproperty
    def row_std_err(self):
        """2D np.float64 ndarray of std-error of row-percent for each matrix cell."""
        return self._assemble_matrix(self._measures.row_std_err.blocks)

    @lazyproperty
    def row_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted row-proportions denominator per cell."""
        return self._assemble_matrix(self._measures.row_unweighted_bases.blocks)

    @lazyproperty
    def row_weighted_bases(self):
        """2D np.float64 ndarray of row-proportions denominator for each cell."""
        return self._assemble_matrix(self._measures.row_weighted_bases.blocks)

    @lazyproperty
    def rows_base(self):
        """1D/2D np.float64 ndarray of unweighted-N for each slice row/cell."""
        # --- an X_ARRAY slice produces a 2D row-base (each cell has its own N) ---
        # --- TODO: Should rows_base only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a "rows_base_range"
        if not self._measures.rows_unweighted_base.is_defined:
            return self.row_unweighted_bases

        # --- otherwise rows-base is a vector ---
        return self._assemble_marginal(self._measures.rows_unweighted_base)

    @lazyproperty
    def rows_dimension_fills(self):
        """tuple of RGB str like "#def032" fill color for each row in slice."""
        elements = self._rows_dimension.valid_elements
        subtotals = self._rows_dimension.subtotals
        return tuple(
            # ---Subtotals have negative sequential indexes (-1, -2, ..., -m)---
            # ---To index them properly, we need to convert those indexes to---
            # ---zero based positive indexes (0, 1, ... m - 1) i.e. -idx - 1---
            (elements[idx].fill if idx >= 0 else subtotals[idx + len(subtotals)].fill)
            for idx in self._row_order
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
        # --- This is really just another way to call the row_weighted_bases ---
        # --- TODO: Should rows_margin only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a "rows_margin_range"
        if not self._measures.rows_weighted_base.is_defined:
            return self.row_weighted_bases

        # --- otherwise rows-margin is a vector ---
        return self._assemble_marginal(self._measures.rows_weighted_base)

    @lazyproperty
    def rows_margin_proportion(self):
        """1D/2D np.float64 ndarray of weighted-proportion for each slice row/cell."""
        # --- an X_MR slice produces a 2D rows-margin (each cell has its own N) ---
        # --- TODO: Should rows_margin_proportion only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a "rows_margin_range"
        if not self._measures.rows_table_proportion.is_defined:
            return self._assemble_matrix(
                SumSubtotals.blocks(
                    self.rows_margin / self.table_weighted_bases,
                    self._dimensions,
                )
            )

        # --- otherwise rows-margin is a vector ---
        return self._assemble_marginal(self._measures.rows_table_proportion)

    @lazyproperty
    def rows_scale_mean(self):
        """Optional 1D Row scale median marginal for this cube-result"""
        return self._assemble_marginal(self._measures.rows_scale_mean)

    @lazyproperty
    def rows_scale_mean_stddev(self):
        """Optional 1D row scale mean std dev marginal for this cube-result"""
        return self._assemble_marginal(self._measures.rows_scale_mean_stddev)

    @lazyproperty
    def rows_scale_mean_stderr(self):
        """Optional 1D row scale mean std dev marginal for this cube-result"""
        return self._assemble_marginal(self._measures.rows_scale_mean_stderr)

    @lazyproperty
    def rows_scale_median(self):
        """Optional 1D Row scale median marginal for this cube-result"""
        return self._assemble_marginal(self._measures.rows_scale_median)

    @lazyproperty
    def smoothed_column_index(self):
        """2D np.float64 ndarray of smoothed column-index for each table cell."""
        return self._assemble_matrix(self._measures.smoothed_column_index.blocks)

    @lazyproperty
    def smoothed_column_proportions(self):
        """2D np.float64 ndarray of smoothed column-proportion for each matrix cell.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        column the cell appears in (aka. column-margin). Generally a number between 0.0
        and 1.0 inclusive, but subtotal differences can be between -1.0 and 1.0
        inclusive.
        """
        return self._assemble_matrix(self._measures.smoothed_column_proportions.blocks)

    @lazyproperty
    def smoothed_columns_scale_mean(self):
        """Optional 1D smoothed column scale mean marginal for this cube-result"""
        return self._assemble_marginal(self._measures.smoothed_columns_scale_mean)

    @lazyproperty
    def smoothed_means(self):
        """2D optional np.float64 ndarray of smoothed mean for each cell.

        Raises `ValueError` if the cube-result does not include a means cube-measure.
        """
        return self._assemble_matrix(self._measures.smoothed_means.blocks)

    @lazyproperty
    def sums(self):
        """2D optional np.float64 ndarray of sum for each cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.
        """
        return self._assemble_matrix(self._measures.sums.blocks)

    @lazyproperty
    def stddev(self):
        """2D optional np.float64 ndarray of stddev for each cell.

        Raises `ValueError` if the cube-result does not include a stddev cube-measure.
        """
        return self._assemble_matrix(self._measures.stddev.blocks)

    @lazyproperty
    def table_base(self):
        """Scalar, 1D, or 2D ndarray of np.float64 unweighted-N for this slice.

        This value has four distinct forms, depending on the slice dimensions:

            * ARR_X_ARR - 2D ndarray with a distinct table-base value per cell.
            * ARR_X - 1D ndarray of value per *row* when only rows dimension is ARR.
            * X_ARR - 1D ndarray of value per *column* when only columns dimension is ARR
            * CAT_X_CAT - scalar float value when slice has no MR dimension.

        """
        if self._measures.table_unweighted_base.is_defined:
            return self._measures.table_unweighted_base.value
        if self._measures.columns_table_unweighted_base.is_defined:
            return self._assemble_marginal(self._measures.columns_table_unweighted_base)
        if self._measures.rows_table_unweighted_base.is_defined:
            return self._assemble_marginal(self._measures.rows_table_unweighted_base)
        return self.table_unweighted_bases

    @lazyproperty
    def table_base_range(self):
        """[min, max] np.float64 ndarray range of the table_margin (table-weighted-base)

        A CAT_X_CAT has a scalar for all table-weighted-bases, but arrays have more than
        one table-weighted-base. This collapses all the values them to the range, and
        it is "unpruned", meaning that it is calculated before any hiding or removing
        of empty rows/columns.
        """
        return self._measures.table_unweighted_bases_range.value

    @lazyproperty
    def table_margin(self):
        """Scalar, 1D, or 2D ndarray of np.float64 weighted-N for this slice.

        This value has four distinct forms, depending on the slice dimensions:

            * CAT_X_CAT - scalar float value when slice has no ARRAY dimension.
            * ARRAY_X - 1D ndarray of value per *row* when only rows dimension is ARRAY.
            * X_ARRAY - 1D ndarray of value per *column* when only column is ARRAY.
            * ARRAY_X_ARRAY - 2D ndarray with a distinct table-margin value per cell.
        """
        if self._measures.table_weighted_base.is_defined:
            return self._measures.table_weighted_base.value
        if self._measures.columns_table_weighted_base.is_defined:
            return self._assemble_marginal(self._measures.columns_table_weighted_base)
        if self._measures.rows_table_weighted_base.is_defined:
            return self._assemble_marginal(self._measures.rows_table_weighted_base)
        return self.table_weighted_bases

    @lazyproperty
    def table_proportion_variances(self):
        """2D ndarray of np.float64 table-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.table_proportion_variances.blocks)

    @lazyproperty
    def table_proportions(self):
        """2D np.float64 ndarray of table-proportion for each matrix cell.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        row the cell appears in (aka. table-margin). Generally a number between 0.0 and
        1.0 inclusive, but subtotal differences can be between -1.0 and 1.0 inclusive.
        """
        return self._assemble_matrix(self._measures.table_proportions.blocks)

    @lazyproperty
    def table_std_err(self):
        """2D np.float64 ndarray of std-error of table-percent for each matrix cell."""
        return self._assemble_matrix(self._measures.table_std_err.blocks)

    @lazyproperty
    def table_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted table-proportion denominator per cell."""
        return self._assemble_matrix(self._measures.table_unweighted_bases.blocks)

    @lazyproperty
    def table_weighted_bases(self):
        """2D np.float64 ndarray of weighted table-proportion denominator per cell."""
        return self._assemble_matrix(self._measures.table_weighted_bases.blocks)

    @lazyproperty
    def total_share_sum(self):
        """2D optional np.float64 ndarray of total share of sum for each cell.

        Raises `ValueError` if the cube-result does not include a sum
        cube-measure.
        """
        return self._assemble_matrix(self._measures.total_share_sum.blocks)

    @lazyproperty
    def table_margin_range(self):
        """[min, max] np.float64 ndarray range of the table_margin (table-weighted-base)

        A CAT_X_CAT has a scalar for all table-weighted-bases, but arrays have more than
        one table-weighted-base. This collapses all of the values to a range, and
        it is "unpruned", meaning that it is calculated before any hiding or removing
        of empty rows/columns.
        """
        return self._measures.table_weighted_bases_range.value

    @lazyproperty
    def unweighted_counts(self):
        """2D np.float64 ndarray of unweighted-count for each cell."""
        return self._assemble_matrix(self._measures.unweighted_counts.blocks)

    @lazyproperty
    def weighted_counts(self):
        """2D np.float64 ndarray of weighted-count for each cell."""
        return self._assemble_matrix(self._measures.weighted_counts.blocks)

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        return self._assemble_matrix(self._measures.zscores.blocks)

    def _assemble_marginal(self, marginal):
        """Optional 1D ndarray created from a marginal.

        The assembled marginal is the shape of either a row or column (determined by
        `marginal.orientation`), and with the ordering that's applied to those
        dimensions.

        It is None when the marginal is not defined (`marginal._is_defined`).
        """
        if not marginal.is_defined:
            return None

        order = (
            self._row_order if marginal.orientation == MO.ROWS else self._column_order
        )

        return np.hstack(marginal.blocks)[order]

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

    def _assemble_vector(self, base_vector, subtotals, order, diffs_nan=False):
        """Return 1D ndarray of `base_vector` with inserted `subtotals`, in `order`.

        Each subtotal value is the result of applying np.sum to the addends and
        subtrahends extracted from `base_vector` according to the `addend_idxs`
        and `subtrahend_idxs` property of each subtotal in `subtotals`. The returned
        array is arranged by `order`, including possibly removing hidden or pruned
        values.
        """
        # TODO: This works for "sum" and "diff" subtotals, because either we set to
        # nan or add & subtract, but a fuller solution will probably get the subtotal
        # values from a _BaseSubtotals subclass.
        vector_subtotals = np.array(
            [
                np.nan
                if diffs_nan and len(subtotal.subtrahend_idxs) > 0
                else np.sum(base_vector[subtotal.addend_idxs])
                - np.sum(base_vector[subtotal.subtrahend_idxs])
                for subtotal in subtotals
            ]
        )
        return np.hstack([base_vector, vector_subtotals])[order]

    @lazyproperty
    def _column_order(self):
        """1D np.int64 ndarray of signed int idx for each assembled column.

        Negative values represent inserted subtotal-column locations.
        """
        return _BaseOrderHelper.column_display_order(self._dimensions, self._measures)

    @lazyproperty
    def _column_subtotals(self):
        """Sequence of _Subtotal object for each inserted column."""
        return self._columns_dimension.subtotals

    @lazyproperty
    def _columns_dimension(self):
        """The `Dimension` object representing column elements in this matrix."""
        return self._dimensions[1]

    @lazyproperty
    def _cube_has_overlaps(self):
        """True if overlaps are defined and the last dimension is MR, False otherwise"""
        return (
            self._dimensions[-1].dimension_type == DT.MR
            and self._cube.overlaps is not None
            and self._cube.valid_overlaps is not None
        )

    def _derived_element_idxs(self, dimension, order):
        """Return tuple(int) of derived elements' indices for a dimension.

        Subtotals cannot be derived elements. Only some elements (subvariables) can.
        """
        n_subtotals = len(dimension.valid_elements)
        derivs = [e.derived for e in dimension.valid_elements] + [False] * n_subtotals
        return tuple(np.where(np.array(derivs)[order])[0])

    def _diff_element_idxs(self, dimension, order):
        """Return tuple(int) of difference elements' indices for a dimension.

        Valid elements cannot be differences. Only some subtotals can.
        """
        n_valids = len(dimension.valid_elements)
        diffs = [False] * n_valids + [e.is_difference for e in dimension.subtotals]
        return tuple(np.where(np.array(diffs)[order])[0])

    def _dimension_aliases(self, dimension, order):
        """1D str ndarray of alias for each vector of `dimension`.

        These are suitable for use as row/column headings; alias for subtotals appear
        in the sequence specified by `order`, such that aliases are ordered to
        correspond with their respective data vector.
        """
        return np.array(dimension.element_aliases + dimension.subtotal_aliases)[order]

    def _dimension_codes(self, dimension, order):
        """1D str ndarray of codes for each vector of `dimension`.

        These are suitable for use as row/column headings; codes for subtotals appear
        in the sequence specified by `order`, such that codes are ordered to
        correspond with their respective data vector.
        """
        return np.array(dimension.element_ids + dimension.insertion_ids)[order]

    def _dimension_labels(self, dimension, order):
        """1D str ndarray of name for each vector of `dimension`.

        These are suitable for use as row/column headings; labels for subtotals appear
        in the sequence specified by `order`, such that labels are ordered to correspond
        with their respective data vector.
        """
        return np.array(dimension.element_labels + dimension.subtotal_labels)[order]

    @lazyproperty
    def _measures(self):
        """SecondOrderMeasures collection object for this cube-result."""
        return SecondOrderMeasures(self._cube, self._dimensions, self._slice_idx)

    @staticmethod
    def _pairwise_indices(p_vals, t_stats, alpha, only_larger):
        """1D ndarray containing tuples of int pairwise indices of each column."""
        significance = p_vals < alpha
        if only_larger:
            significance = np.logical_and(t_stats < 0, significance)
        col_significance = np.empty((len(significance),), dtype=object)
        col_significance[:] = [tuple(np.where(sig_row)[0]) for sig_row in significance]
        return col_significance

    @lazyproperty
    def _row_order(self):
        """1D np.int64 ndarray of signed int idx for each assembled row.

        Negative values represent inserted subtotal-row locations.
        """
        return _BaseOrderHelper.row_display_order(self._dimensions, self._measures)

    @lazyproperty
    def _row_subtotals(self):
        """Sequence of _Subtotal object for each inserted row."""
        return self._rows_dimension.subtotals

    @lazyproperty
    def _rows_dimension(self):
        """The `Dimension` object representing row elements in this matrix."""
        return self._dimensions[0]


class _BaseOrderHelper:
    """Base class for ordering helpers."""

    def __init__(self, dimensions, second_order_measures):
        self._dimensions = dimensions
        self._second_order_measures = second_order_measures

    @classmethod
    def column_display_order(cls, dimensions, second_order_measures):
        """1D np.int64 ndarray of signed int idx for each column of measure matrix.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        columns-dimension.
        """
        # --- This is essentially a factory method. There is no sort-columns-by-value
        # --- yet, and both explicit and payload ordering are handled by
        # --- _ColumnOrderHelper, so there's not much to this yet, just keeping
        # --- form consistent with `.row_display_order()` and we'll elaborate this when
        # --- we add sort-by-value to columns.
        return _ColumnOrderHelper(dimensions, second_order_measures)._display_order

    @classmethod
    def row_display_order(cls, dimensions, second_order_measures):
        """1D np.int64 ndarray of signed int idx for each row of measure matrix.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        collation_method = dimensions[0].order_spec.collation_method
        dim_type = dimensions[1].dimension_type
        HelperCls = (
            _SortRowsByBaseColumnHelper
            if collation_method == CM.OPPOSING_ELEMENT
            else _SortRowsByDerivedColumnHelper
            if collation_method == CM.OPPOSING_INSERTION and dim_type in DT.ARRAY_TYPES
            else _SortRowsByInsertedColumnHelper
            if collation_method == CM.OPPOSING_INSERTION
            else _SortRowsByLabelHelper
            if collation_method == CM.LABEL
            else _SortRowsByMarginalHelper
            if collation_method == CM.MARGINAL
            else _RowOrderHelper
        )

        return HelperCls(dimensions, second_order_measures)._display_order

    @lazyproperty
    def _display_order(self):
        """1D np.int64 ndarray of signed int idx for each vector of dimension.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        # --- Returning as np.array suits its intended purpose, which is to participate
        # --- in an np._ix() call. It works fine as a sequence too for any alternate
        # --- use. Specifying int type prevents failure when there are zero elements.
        if self._prune_subtotals:
            return np.array([idx for idx in self._order if idx >= 0], dtype=int)
        return np.array(self._order, dtype=int)

    @lazyproperty
    def _columns_dimension(self):
        """The `Dimension` object representing column elements in the matrix."""
        return self._dimensions[1]

    @lazyproperty
    def _empty_column_idxs(self):
        """tuple of int index for each column with (unweighted) N = 0.

        These columns are subject to pruning, depending on a user setting in the
        dimension.
        """
        # --- subset because numpy.where returns tuple to allow for multiple axes
        return tuple(np.where(self._second_order_measures.columns_pruning_mask)[0])

    @lazyproperty
    def _empty_row_idxs(self):
        """tuple of int index for each row with N = 0.

        These rows are subject to pruning, depending on a user setting in the dimension.
        """
        # --- subset because numpy.where returns tuple to allow for multiple axes
        return tuple(np.where(self._second_order_measures.rows_pruning_mask)[0])

    @lazyproperty
    def _measure(self):
        """Second-order measure object providing values for sort.

        Returns None when the measure in the `order_spec` is not enable for sorting.

        This property is not used by some subclasses.
        """
        # --- Sometimes we sort by a measure that is a monotonic transformation of
        # --- another measure that is already calculated by the assembler. Because
        # --- monotonic transformations preserve order, the sorting is correct.
        propname_by_measure = {
            M.COLUMN_BASE_UNWEIGHTED: "column_unweighted_bases",
            M.COLUMN_BASE_WEIGHTED: "column_weighted_bases",
            M.COLUMN_INDEX: "column_index",
            M.COLUMN_PERCENT: "column_proportions",
            M.COLUMN_PERCENT_MOE: "column_std_err",  # monotonic transform
            M.COLUMN_SHARE_SUM: "column_share_sum",
            M.COLUMN_STDDEV: "column_proportion_variances",  # monotonic transform
            M.COLUMN_STDERR: "column_std_err",
            M.MEAN: "means",
            M.POPULATION: "population_proportions",  # montonic transform
            M.POPULATION_MOE: "population_std_err",  # monotonic transform
            M.PVALUES: "pvalues",
            M.ROW_BASE_UNWEIGHTED: "row_unweighted_bases",
            M.ROW_BASE_WEIGHTED: "row_weighted_bases",
            M.ROW_PERCENT: "row_proportions",
            M.ROW_PERCENT_MOE: "row_std_err",  # monotonic transform
            M.ROW_SHARE_SUM: "row_share_sum",
            M.ROW_STDDEV: "row_proportion_variances",  # montonic transform
            M.ROW_STDERR: "row_std_err",
            M.STDDEV: "stddev",
            M.SUM: "sums",
            M.TABLE_PERCENT: "table_proportions",
            M.TABLE_PERCENT: "table_proportions",
            M.TABLE_PERCENT_MOE: "table_std_err",  # monotonic transform
            M.TABLE_STDDEV: "table_proportion_variances",  # monotonic transform
            M.TABLE_STDERR: "table_std_err",
            M.TABLE_BASE_UNWEIGHTED: "table_unweighted_bases",
            M.TABLE_BASE_WEIGHTED: "table_weighted_bases",
            M.TOTAL_SHARE_SUM: "total_share_sum",
            M.UNWEIGHTED_COUNT: "unweighted_counts",
            M.UNWEIGHTED_VALID_COUNT: "unweighted_counts",
            M.WEIGHTED_COUNT: "weighted_counts",
            M.WEIGHTED_VALID_COUNT: "weighted_counts",
            M.Z_SCORE: "zscores",
        }
        measure = self._order_spec.measure
        measure_propname = propname_by_measure.get(measure)

        if measure_propname is None:
            raise NotImplementedError(
                f"sort-by-value for measure '{measure}' is not yet supported"
            )

        return getattr(self._second_order_measures, measure_propname)

    @lazyproperty
    def _order(self):
        """tuple of signed int idx for each sorted vector of measure matrix.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._order`"
        )

    @lazyproperty
    def _order_spec(self):
        """_OrderSpec object for dimension being sorted.

        Provides access to ordering details like measure and sort-direction.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._order_spec`"
        )

    @lazyproperty
    def _prune_subtotals(self):
        """True if subtotal vectors need to be pruned, False otherwise."""
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._prune_subtotals`"
        )

    @lazyproperty
    def _rows_dimension(self):
        """The `Dimension` object representing row elements in the matrix."""
        return self._dimensions[0]


class _ColumnOrderHelper(_BaseOrderHelper):
    """Encapsulates the complexity of the various kinds of column ordering."""

    @lazyproperty
    def _order(self):
        """tuple of signed int idx for each column of measure matrix.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        CollatorCls = (
            ExplicitOrderCollator
            if self._order_spec.collation_method == CM.EXPLICIT_ORDER
            else PayloadOrderCollator
        )
        return CollatorCls.display_order(
            self._columns_dimension, self._empty_column_idxs
        )

    @lazyproperty
    def _order_spec(self):
        """_OrderSpec object for the columns dimension.

        Provides access to ordering details like measure and sort-direction.
        """
        return self._columns_dimension.order_spec

    @lazyproperty
    def _prune_subtotals(self):
        """True if subtotal columns need to be pruned, False otherwise.

        Subtotal-columns need to be pruned when all base-rows are pruned. Subtotal
        columns are only subject to pruning when row-pruning is specified in the
        request.
        """
        return (
            len(self._empty_row_idxs) == len(self._rows_dimension.element_ids)
            if self._rows_dimension.prune
            else False
        )


class _RowOrderHelper(_BaseOrderHelper):
    """Encapsulates the complexity of the various kinds of row ordering."""

    @lazyproperty
    def _order(self):
        """tuple of signed int idx for each row of measure matrix.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        CollatorCls = (
            ExplicitOrderCollator
            if self._order_spec.collation_method == CM.EXPLICIT_ORDER
            else PayloadOrderCollator
        )
        return CollatorCls.display_order(self._rows_dimension, self._empty_row_idxs)

    @lazyproperty
    def _order_spec(self):
        """_OrderSpec object for this row dimension.

        Provides access to ordering details like measure and sort-direction.
        """
        return self._rows_dimension.order_spec

    @lazyproperty
    def _prune_subtotals(self):
        """True if subtotal rows need to be pruned, False otherwise.

        Subtotal-rows need to be pruned when all base-columns are pruned. Subtotal rows
        only subject to pruning when column-pruning is specified in the request.
        """
        return (
            len(self._empty_column_idxs) == len(self._columns_dimension.element_ids)
            if self._columns_dimension.prune
            else False
        )


class _BaseSortRowsByValueHelper(_RowOrderHelper):
    """A base class that orders elements by a set of values.

    This class is intended only to serve as a base for the other sort-by-value classes
    which must all provide their own implentations of `_element_values` and
    `_subtotal_values`.

    If `_order` encouters a ValueError, it falls back to the payload order.
    """

    @lazyproperty
    def _element_values(self):
        """Sequence of body values that form the basis for sort order.

        Must be implemented by child classes.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._element_values`"
        )

    @lazyproperty
    def _order(self):
        """tuple of int element-idx specifying ordering of dimension elements."""
        try:
            return SortByValueCollator.display_order(
                self._rows_dimension,
                self._element_values,
                self._subtotal_values,
                self._empty_row_idxs,
            )
        except ValueError:
            return PayloadOrderCollator.display_order(
                self._rows_dimension, self._empty_row_idxs
            )

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of subtotal values that form the basis for sort order.

        Must be implemented by child classes.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._subtotal_values`"
        )


class _SortRowsByBaseColumnHelper(_BaseSortRowsByValueHelper):
    """Orders elements by the values of an opposing base (not a subtotal) vector.

    This would be like "order rows in descending order by value of 'Strongly Agree'
    column. An opposing-element ordering is only available on a matrix, because only
    a matrix dimension has an opposing dimension.
    """

    @lazyproperty
    def _column_idx(self):
        """int index of column whose values the sort is based on."""
        column_element_ids = self._columns_dimension.element_ids
        sort_column_id = self._order_spec.element_id
        # --- Need to translate the element id to the shimmed element id
        sort_column_id = self._columns_dimension.translate_element_id(sort_column_id)
        return column_element_ids.index(sort_column_id)

    @lazyproperty
    def _element_values(self):
        """Sequence of body values that form the basis for sort order.

        There is one value per row and values appear in payload (dimension) element
        order. These are only the "base" values and do not include insertions.
        """
        measure_base_values = self._measure.blocks[0][0]
        return measure_base_values[:, self._column_idx]

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal values that contribute to the sort basis.

        There is one value per row subtotal and values appear in payload (dimension)
        insertion order.
        """
        measure_subtotal_rows = self._measure.blocks[1][0]
        return measure_subtotal_rows[:, self._column_idx]


class _SortRowsByDerivedColumnHelper(_SortRowsByBaseColumnHelper):
    """Orders elements by the values of an opposing "derived" element vector.

    "Derived" columns are insertions that are calculated by zz9 and put into the
    cube result before this cr.cube library gets them. These insertions are always
    on a subvariable dimension, an example would be to sort "'Coke' or 'Diet Coke'"
    MR array column. An opposing-element ordering is only available on a matrix,
    because only a matrix dimension has an opposing dimension.

    For the most part, we treat these elements as regular "base" columns, in the cr.cube
    library, but because the user-facing language treats these as subtotals, we allow
    specifying sort-by-value as if these were true subtotals.
    """

    @lazyproperty
    def _column_idx(self):
        """int index of column whose values the sort is based on."""
        column_element_ids = self._columns_dimension.element_ids
        sort_column_id = self._order_spec.insertion_id
        # --- Need to translate the element id to the shimmed element id
        sort_column_id = self._columns_dimension.translate_element_id(sort_column_id)
        return column_element_ids.index(sort_column_id)


class _SortRowsByInsertedColumnHelper(_BaseSortRowsByValueHelper):
    """Orders rows by the values in an inserted column.

    This would be like "order rows in descending order by value of 'Top 3' subtotal
    column. An opposing-insertion ordering is only available on a matrix because only
    a matrix dimension has an opposing dimension.
    """

    @lazyproperty
    def _element_values(self):
        """Sequence of insertion "body" values that form the basis for sort order.

        There is one value per base row and values appear in payload order. These are
        only the "base" insertion values and do not include intersections.
        """
        insertion_base_values = self._measure.blocks[0][1]
        return insertion_base_values[:, self._insertion_idx]

    @lazyproperty
    def _insertion_idx(self):
        """int index of insertion whose values the sort is based on."""
        return self._columns_dimension.insertion_ids.index(
            self._order_spec.insertion_id
        )

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal intersection values that contribute to sort basis.

        There is one value per row subtotal and values appear in payload (dimension)
        insertion order.
        """
        intersections = self._measure.blocks[1][1]
        return intersections[:, self._insertion_idx]


class _SortRowsByLabelHelper(_BaseSortRowsByValueHelper):
    """Orders elements by the values of their labels (from the dimension)."""

    @lazyproperty
    def _element_values(self):
        """Sequence of body labels that form the basis for sort order.

        There is one value per row and values appear in payload (dimension) element
        order. These are only the "base" values and do not include insertions.
        """
        return np.array(self._dimensions[0].element_labels)

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal labels that contribute to the sort basis.

        There is one value per row subtotal and labels appear in payload (dimension)
        insertion order.
        """
        return np.array(self._dimensions[0].subtotal_labels)


class _SortRowsByMarginalHelper(_BaseSortRowsByValueHelper):
    """Orders elements by the values of an opposing marginal vector.

    This would be like "order rows in descending order by value of rows scale mean.
    column. An opposing-marginal ordering is only available on a matrix, because only
    a matrix dimension has an opposing dimension.
    """

    @lazyproperty
    def _element_values(self):
        """Sequence of body values that form the basis for sort order.

        There is one value per row and values appear in payload (dimension) element
        order. These are only the "base" values and do not include insertions.
        """
        return self._marginal.blocks[0]

    @lazyproperty
    def _marginal(self):
        """Marginal object providing values for sort."""
        marginal = self._order_spec.marginal
        marginal_propname = {
            MARGINAL.BASE: "rows_unweighted_base",
            MARGINAL.MARGIN: "rows_weighted_base",
            MARGINAL.MARGIN_PROPORTION: "rows_table_proportion",
            MARGINAL.SCALE_MEAN: "rows_scale_mean",
            MARGINAL.SCALE_MEAN_STDDEV: "rows_scale_mean_stddev",
            MARGINAL.SCALE_MEAN_STDERR: "rows_scale_mean_stderr",
            MARGINAL.SCALE_MEDIAN: "rows_scale_median",
        }.get(marginal)

        if marginal_propname is None:
            raise NotImplementedError(
                f"sort-by-value for marginal '{marginal}' is not yet supported"
            )

        return getattr(self._second_order_measures, marginal_propname)

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal values that contribute to the sort basis.

        There is one value per row subtotal and values appear in payload (dimension)
        insertion order.
        """
        return self._marginal.blocks[1]
