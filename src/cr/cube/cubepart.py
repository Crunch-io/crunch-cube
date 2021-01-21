# encoding: utf-8

"""Cube-partition objects.

A cube-partition allows cubes of various dimensionality to be processed in a uniform
way. For example, a 2D cube becomes a `_Slice` object, but a 3D cube is "sliced" into
a sequence of `_Slice` objects; a `_Slice` object corresponds to a crosstab, and can be
operated on consistently whether it is "alone" or one of a sequence that came from a 3D
cube.

Cube-partition objects are typically used for display of secondary analysis, often in an
Excel spreadsheet but also other formats.

The three types of cube partition are the *slice*, *strand*, and *nub*, which are 2D,
1D, and 0D respectively.
"""

from __future__ import division

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.min_base_size_mask import MinBaseSizeMask
from cr.cube.matrix import Assembler
from cr.cube.measures.pairwise_significance import PairwiseSignificance
from cr.cube.noa.smoothing import SingleSidedMovingAvgSmoother
from cr.cube.scalar import MeansScalar
from cr.cube.stripe import TransformedStripe
from cr.cube.util import lazyproperty

# ---This is the quantile of the normal Cumulative Distribution Function (CDF) at
# ---probability 97.5% (p=.975), since the computed confidence interval
# ---is ±2.5% (.025) on each side of the CDF.
Z_975 = 1.959964


class CubePartition(object):
    """A slice, a strand, or a nub drawn from a cube-response.

    These represent 2, 1, or 0 dimensions of a cube, respectively.
    """

    def __init__(self, cube, transforms=None):
        self._cube = cube
        self._transforms_arg = transforms

    @classmethod
    def factory(
        cls,
        cube,
        slice_idx=0,
        transforms=None,
        population=None,
        ca_as_0th=None,
        mask_size=0,
    ):
        """Return slice, strand, or nub object appropriate to passed parameters."""
        if cube.ndim == 0:
            return _Nub(cube)
        if cube.ndim == 1 or ca_as_0th:
            return _Strand(
                cube, transforms, population, ca_as_0th, slice_idx, mask_size
            )
        return _Slice(cube, slice_idx, transforms, population, mask_size)

    @lazyproperty
    def cube_index(self):
        """Offset of this partition's cube in its CubeSet.

        Used to differentiate certain partitions like a filtered rows-summary strand.
        """
        return self._cube.cube_index

    @lazyproperty
    def dimension_types(self):
        """Sequence of member of `cr.cube.enum.DIMENSION_TYPE` for each dimension.

        Items appear in rows-dimension, columns-dimension order.
        """
        return tuple(d.dimension_type for d in self._dimensions)

    def evaluate(self, measure_expr):
        """Return 1D/2D ndarray result of evaluating `measure_expr`.

        `measure_expr` contains the function to apply and its parameters, like::

            {
                 "function": "one_sided_moving_avg",
                 "base_measure": "col_percent",
                 "window": 3
            }

        This expression specifies application of the `one_sided_moving_avg` function to
        the `col_percent` base-measure, with a sliding window of 3 periods.
        """
        function = measure_expr.get("function", None)
        if function != "one_sided_moving_avg":
            raise NotImplementedError("Function {} is not available.".format(function))
        return SingleSidedMovingAvgSmoother(self, measure_expr).values

    @lazyproperty
    def has_means(self):
        """True if cube-result includes means values."""
        return self._cube.has_means

    @lazyproperty
    def ndim(self):
        """int count of dimensions for this partition."""
        return len(self._dimensions)

    @lazyproperty
    def population_fraction(self):
        """population fraction of the cube"""
        return self._cube.population_fraction

    @lazyproperty
    def selected_category_labels(self):
        """Tuple of str: names of any and all underlying categories in 'Selected'."""
        return tuple(
            s["name"]
            for d in self._dimensions
            for s in d.selected_categories
            if s.get("name")
        )

    @lazyproperty
    def shape(self):
        """Tuple of int vector counts for this partition.

        Not to be confused with `numpy.ndarray.shape`, this represent the count of rows
        and columns respectively, in this partition. It does not necessarily represent
        the shape of any underlying `numpy.ndarray` object that may arise in the
        implementation of the cube partition. In particular, the value of any count in
        the shape can be zero.

        A _Slice has a shape like (2, 3) representing (row-count, col-count). A _Strand
        has a shape like (5,) which represents its row-count. The shape of a _Nub is
        unconditionally () (an empty tuple).
        """
        raise NotImplementedError(
            "must be implemented by each subclass"
        )  # pragma: no cover

    @lazyproperty
    def variable_name(self):
        """str representing the name of the superheading variable."""
        return self._dimensions[0 if self.ndim < 2 else 1].name

    @lazyproperty
    def _alpha(self):
        """float confidence-interval threshold for pairwise-t (sig) tests."""
        return self._alpha_values[0]

    @lazyproperty
    def _alpha_alt(self):
        """Alternate float confidence-interval threshold or None.

        This is an optional secondary confidence interval allowing two-level
        significance testing. Value is None if no alternate alpha was specified by user.
        """
        return self._alpha_values[1]

    @lazyproperty
    def _alpha_values(self):
        """Pair (tuple) of confidence-interval thresholds to be used for t-tests.

        The second value is optional and is None when no secondary alpha value was
        defined for the cube-set.
        """
        value = self._transforms_dict.get("pairwise_indices", {}).get("alpha")

        # --- handle omitted, None, [], (), {}, "", 0, and 0.0 cases ---
        if not value:
            return (0.05, None)

        # --- reject invalid types ---
        if not isinstance(value, (float, list, tuple)):
            raise TypeError(
                "transforms.pairwise_indices.alpha, when defined, must be a list of 1 "
                "or 2 float values between 0.0 and 1.0 exclusive. Got %r" % value
            )

        # --- legacy float "by-itself" case ---
        if isinstance(value, float):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    "alpha value, when provided, must be between 0.0 and 1.0 "
                    "exclusive. Got %r" % value
                )
            return (value, None)

        # --- sequence case ---
        for x in value[:2]:
            if not isinstance(x, float) or not 0.0 < x < 1.0:
                raise ValueError(
                    "transforms.pairwise_indices.alpha must be a list of 1 or 2 float "
                    "values between 0.0 and 1.0 exclusive. Got %r" % value
                )

        if len(value) == 1:
            return (value[0], None)

        return tuple(sorted(value[:2]))

    @lazyproperty
    def _dimensions(self):
        """tuple of Dimension object for each dimension in cube-partition."""
        raise NotImplementedError(
            "must be implemented by each subclass"
        )  # pragma: no cover

    @lazyproperty
    def _only_larger(self):
        """True if only the larger of reciprocal pairwise-t values should appear.

        In general, pairwise-t tests are reciprocal. That is, if A is significant with
        respect to B, then B is significant with respect to A. Having a letter in both
        columns can produce a cluttered appearance. When this flag is set by the user,
        only the cell in the reciprocal pair having the largest value gets a letter.
        Defaults to True unless explicitly set False.
        """
        return (
            False
            if self._transforms_dict.get("pairwise_indices", {}).get(
                "only_larger", True
            )
            is False
            else True
        )

    @lazyproperty
    def _transforms_dict(self):
        """dict holding transforms for this partition, provided as `transforms` arg.

        This value is an empty dict (`{}`) when no transforms were specified on
        construction.
        """
        return {} if self._transforms_arg is None else self._transforms_arg


class _Slice(CubePartition):
    """2D cube partition.

    A slice represents the cross-tabulation of two dimensions, often, but not
    necessarily contributed by two different variables. A single CA variable has two
    dimensions which can be crosstabbed in a slice.
    """

    def __init__(self, cube, slice_idx, transforms, population, mask_size):
        super(_Slice, self).__init__(cube, transforms)
        self._slice_idx = slice_idx
        self._population = population
        self._mask_size = mask_size

    # ---interface ---------------------------------------------------

    @lazyproperty
    def column_index(self):
        """2D np.float64 ndarray of column-index "percentage".

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the rows variable.
        """
        return self._assembler.column_index

    @lazyproperty
    def column_labels(self):
        """1D str ndarray of name for each column, for use as column headings."""
        return self._assembler.column_labels

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def column_proportions(self):
        """2D np.float64 ndarray of column-proportion for each matrix cell.

        This is the proportion of the weighted-N (aka. margin) of its column that the
        *weighted-count* in each cell represents, a number between 0.0 and 1.0.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.counts / self.columns_margin

    @lazyproperty
    def column_proportions_moe(self):
        """1D/2D np.float64 ndarray of margin-of-error (MoE) for columns proportions.

        The values are represented as fractions, analogue to the `column_proportions`
        property. This means that the value of 3.5% will have the value 0.035.
        The values can be np.nan when the corresponding percentage is also np.nan, which
        happens when the respective columns margin is 0.
        """
        return Z_975 * self.column_std_err

    @lazyproperty
    def column_std_dev(self):
        """standard deviation for column percentages

        `std_deviation = sqrt(variance)`
        """
        return np.sqrt(self._column_variances)

    @lazyproperty
    def column_std_err(self):
        """standard error for column percentages

        `std_error = sqrt(variance/N)`
        """
        return np.sqrt(self._column_variances / self.columns_margin)

    @lazyproperty
    def column_unweighted_bases(self):
        """2D np.int64 ndarray of unweighted column-proportion denominator per cell."""
        return self._assembler.column_unweighted_bases

    @lazyproperty
    def column_weighted_bases(self):
        """2D np.float64 ndarray of column-proportion denominator for each cell."""
        return self._assembler.column_weighted_bases

    @lazyproperty
    def columns_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each column/cell of slice.

        This array is 2D (a distinct base for each cell) when the rows dimension is MR,
        because each MR-subvariable has its own unweighted N. This is because not every
        possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        return self._assembler.columns_base

    @lazyproperty
    def columns_dimension_name(self):
        """str name assigned to columns-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._columns_dimension.name

    @lazyproperty
    def columns_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` describing columns dimension."""
        return self._columns_dimension.dimension_type

    @lazyproperty
    def columns_margin(self):
        """1D or 2D np.float/int64 ndarray of weighted-N for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the rows dimension
        is MR, because each MR-subvariable has its own weighted N. This is because not
        every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        return self._assembler.columns_margin

    @lazyproperty
    def columns_scale_mean(self):
        """Optional 1D np.float64 ndarray of scale mean for each column.

        The returned vector is to be interpreted as a summary *row*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has an assigned numeric value.
        """
        if not self._rows_have_numeric_value:
            return None

        inner = np.nansum(
            self._rows_dimension_numeric_values[:, None] * self.column_proportions,
            axis=0,
        )
        not_a_nan_mask = ~np.isnan(self._rows_dimension_numeric_values)
        denominator = np.sum(self.column_proportions[not_a_nan_mask, :], axis=0)
        return inner / denominator

    @lazyproperty
    def columns_scale_mean_margin(self):
        """Optional float overall mean of column-scale values.

        This value is the "margin" of the `.columns_scale_mean` vector and might
        typically appear in the cell immediately to the right of the
        `.columns_scale_mean` summary-row. It is similar to a "table-total" value, in
        that it is a scalar that might appear in the lower right-hand corner of a table,
        but note that it does *not* represent the overall table in that
        `.rows_scale_mean_margin` will not have the same value (except by chance).
        This value derives from the numeric values of the row elements whereas its
        counterpart `.rows_scale_mean_margin` derives from the numeric values of the
        column elements.

        This value is `None` if no row has an assigned numeric-value.
        """
        if not self._rows_have_numeric_value:
            return None

        rows_margin = self.rows_margin
        # TODO: This is a hack for X_MR slices, where rows-margin is 2D. Figure out a
        # better way to do this, perhaps using ranges.
        if rows_margin.ndim > 1:
            rows_margin = rows_margin[:, 0]

        scale_total = np.nansum(self._rows_dimension_numeric_values * rows_margin)
        not_a_nan_mask = ~np.isnan(self._rows_dimension_numeric_values)
        scale_count = np.sum(rows_margin[not_a_nan_mask])
        return scale_total / scale_count

    @lazyproperty
    def columns_scale_mean_pairwise_indices(self):
        """Sequence of column-idx tuples indicating pairwise-t result of scale-means.

        The sequence contains one tuple for each column. The indicies in a column's
        tuple each identify *another* of the columns who's scale-mean is
        pairwise-significant to that of the tuple's column. Pairwise significance is
        computed based on the more restrictive (lesser-value) threshold specified in the
        analysis.
        """
        return PairwiseSignificance.scale_mean_pairwise_indices(
            self, self._alpha, self._only_larger
        )

    @lazyproperty
    def columns_scale_mean_pairwise_indices_alt(self):
        """Optional sequence of column-idx tuples indicating pairwise-t of scale-means.

        This value is `None` if no secondary threshold value (alpha) was specified in
        the analysis. Otherwise, it is the same calculation as
        `.columns_scale_mean_pairwise_indices` computed using the less restrictive
        (greater-valued) threshold.
        """
        if self._alpha_alt is None:
            return None

        return PairwiseSignificance.scale_mean_pairwise_indices(
            self, self._alpha_alt, self._only_larger
        )

    @lazyproperty
    def columns_scale_mean_stddev(self):
        """Optional 1D np.float64 ndarray of scale-mean std-deviation for each column.

        The returned vector (1D array) is to be interpreted as a summary *row*. Also
        note that the underlying scale values are based on the numeric values of the
        opposing *rows-dimension* elements.

        This value is `None` if no row element has been assigned a numeric value.
        """
        return (
            np.sqrt(self._columns_scale_mean_variance)
            if self._rows_have_numeric_value
            else None
        )

    @lazyproperty
    def columns_scale_mean_stderr(self):
        """Optional 1D np.float64 ndarray of scale-mean standard-error for each row.

        The returned vector is to be interpreted as a summary *row*. Also note that the
        underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has been assigned a numeric value.
        """
        return (
            self.columns_scale_mean_stddev / np.sqrt(self.columns_margin)
            if self._rows_have_numeric_value
            else None
        )

    @lazyproperty
    def columns_scale_median(self):
        """Optional 1D np.int/float64 ndarray of scale median for each column.

        The returned vector is to be interpreted as a summary *row*. Also note that the
        underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has been assigned a numeric value.
        """
        if not self._rows_have_numeric_value:
            return None

        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric_values)
        numeric_values = self._rows_dimension_numeric_values[not_a_nan_index]
        counts = self.counts[not_a_nan_index, :].astype("int64")
        scale_median = np.array(
            [
                self._median(np.repeat(numeric_values, counts[:, i]))
                for i in range(counts.shape[1])
            ]
        )
        return scale_median

    @lazyproperty
    def columns_scale_median_margin(self):
        """Optional scalar numeric median of all column-scale values.

        This value is the "margin" of the `.columns_scale_median` vector and might
        typically appear in the cell immediately to the right of the
        `.columns_scale_median` summary-row. It is similar to a "table-total" value, in
        that it is a scalar that might appear in the lower right-hand corner of a table,
        but note that it does *not* represent the overall table in that
        `.rows_scale_median_margin` will not have the same value (except by chance).
        This value derives from the numeric values of the row elements whereas its
        counterpart `.rows_scale_median_margin` derives from the numeric values of
        the column elements.

        This value is `None` if no row has an assigned numeric-value.
        """
        if not self._rows_have_numeric_value:
            return None

        rows_margin = self.rows_margin

        # --- hack to accommodate 2D rows-margin of an X_MR slice ---
        if len(rows_margin.shape) > 1:
            rows_margin = rows_margin[:, 0]

        not_a_nan_mask = ~np.isnan(self._rows_dimension_numeric_values)
        numeric_values = self._rows_dimension_numeric_values[not_a_nan_mask]
        counts = rows_margin[not_a_nan_mask].astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of weighted cube counts."""
        return self._assembler.weighted_counts

    @lazyproperty
    def description(self):
        """str description of this slice, which it takes from its rows-dimension."""
        return self._rows_dimension.description

    @lazyproperty
    def inserted_column_idxs(self):
        """tuple of int index of each subtotal column in slice."""
        return self._assembler.inserted_column_idxs

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each subtotal row in slice."""
        return self._assembler.inserted_row_idxs

    @lazyproperty
    def is_empty(self):
        return any(s == 0 for s in self.shape)

    @lazyproperty
    def means(self):
        """2D optional np.float64 ndarray of mean value for each table cell.

        Cell value is `np.nan` for each cell corresponding to an inserted subtotal
        (mean of addend cells cannot simply be added to get the mean of the subtotal).

        Raises `ValueError` if the cube-result does not include a means cube-measure.
        """
        return self._assembler.means

    @lazyproperty
    def min_base_size_mask(self):
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def name(self):
        """str name assigned to this slice.

        A slice takes the name of its rows-dimension.
        """
        return self.rows_dimension_name

    @lazyproperty
    def pairwise_indices(self):
        """2D ndarray of tuple of int column-idxs meeting pairwise-t threshold.

        Like::

            [
               [(1, 3, 4), (), (0,), (), ()],
               [(2,), (1, 2), (), (), (0, 3)],
               [(), (), (), (), ()],
            ]

        Has the same shape as `.counts`. Each int represents the offset of another
        column in the same row with a confidence interval meeting the threshold defined
        for this analysis.
        """
        return PairwiseSignificance.pairwise_indices(
            self, self._alpha, self._only_larger
        )

    @lazyproperty
    def pairwise_indices_alt(self):
        """2D ndarray of tuple of int column-idxs meeting alternate threshold.

        This value is None if no alternate threshold has been defined.
        """
        return (
            None
            if self._alpha_alt is None
            else PairwiseSignificance.pairwise_indices(
                self, self._alpha_alt, self._only_larger
            )
        )

    @lazyproperty
    def pairwise_significance_tests(self):
        """tuple of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are columns in the slice. Each
        significance test contains `p_vals` and `t_stats` (ndarrays that represent
        probability values and statistical scores).
        """
        return tuple(
            PairwiseSignificance(self).values[column_idx]
            for column_idx in range(len(self.column_labels))
        )

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def population_counts_moe(self):
        """2D np.float64 ndarray of population-count margin-of-error (MoE) per cell.

        The values are represented as population estimates, analogue to the
        `population_counts` property. This means that the values will be presented by
        actual estimated counts of the population. The values can be np.nan when the
        corresponding percentage is also np.nan, which happens when the respective
        table margin is 0.
        """
        total_filtered_population = self._population * self._cube.population_fraction
        return Z_975 * total_filtered_population * self.table_std_err

    @lazyproperty
    def pvals(self):
        """2D optional np.float64 ndarray of p-value for each cell.

        A p-value is a measure of the probability that an observed difference could have
        occurred just by random chance. The lower the p-value, the greater the
        statistical significance of the observed difference.

        A cell value of np.nan indicates a meaningful p-value could not be computed for
        that cell.
        """
        return self._assembler.pvalues

    @lazyproperty
    def residual_test_stats(self):
        """Exposes pvals and zscores (with HS) stacked together

        Public method used as cube_method for the SOA API
        """
        return np.stack([self.pvals, self.zscores])

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of name for each row, suitable for use as row headings."""
        return self._assembler.row_labels

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def row_proportions(self):
        """2D np.float64 ndarray of row-proportion for each matrix cell.

        Each row-proportion value is that cell's fraction of the weighted-N for the row
        (aka. rows_margin), a number between 0.0 and 1.0.
        """
        rows_margin = self.rows_margin
        with np.errstate(divide="ignore", invalid="ignore"):
            # --- numpy broadcasting only works "vertically", so in the 1D margin case,
            # --- we need to "rotate" (transpose) the counts such that rows are running
            # --- "vertically", and then re-transpose the result to get back
            # --- (rows, cols) orientation.
            if rows_margin.ndim == 1:
                return (self.counts.T / rows_margin).T
            return self.counts / rows_margin

    @lazyproperty
    def row_proportions_moe(self):
        """2D np.float64 ndarray of margin-of-error (MoE) for rows proportions.

        The values are represented as percentage-fractions, analogue to the
        `row_proportions` property. This means that the value of 3.5% will have the
        value 0.035. The values can be np.nan when the corresponding percentage is also
        np.nan, which happens when the respective table margin is 0.
        """
        return Z_975 * self.row_std_err

    @lazyproperty
    def row_std_dev(self):
        """2D np.float64 ndarray of standard deviation for row percentages."""
        return np.sqrt(self._row_variance)

    @lazyproperty
    def row_std_err(self):
        """2D np.float64 ndarray of standard errors for row percentages."""
        rows_margin = (
            self.rows_margin  # --- no need to cast (MR dim involved)
            if len(self.rows_margin.shape) > 1
            else self.rows_margin[:, np.newaxis]  # --- cast to actual vector column
        )
        return np.sqrt(self._row_variance / rows_margin)

    @lazyproperty
    def row_unweighted_bases(self):
        """2D np.int64 ndarray of unweighted row-proportion denominator per cell."""
        return self._assembler.row_unweighted_bases

    @lazyproperty
    def row_weighted_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each table cell."""
        return self._assembler.row_weighted_bases

    @lazyproperty
    def rows_base(self):
        """1D/2D np.int64 ndarray of unweighted-N for each row/cell of slice.

        This array is 2D (a distinct base for each cell) when the columns dimension is
        MR, because each MR-subvariable has its own unweighted N. This is because not
        every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        return self._assembler.rows_base

    @lazyproperty
    def rows_dimension_description(self):
        """str description assigned to rows-dimension.

        Reflects the resolved dimension-description transform cascade.
        """
        return self._rows_dimension.description

    @lazyproperty
    def rows_dimension_fills(self):
        """tuple of optional RGB str like "#def032" fill color for each row in slice.

        The values reflect the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions and hidden rows. A value of `None` indicates the
        default fill, possibly determined by a theme or template.
        """
        return self._assembler.rows_dimension_fills

    @lazyproperty
    def rows_dimension_name(self):
        """str name assigned to rows-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._rows_dimension.name

    @lazyproperty
    def rows_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` specifying type of rows dimension."""
        return self._rows_dimension.dimension_type

    @lazyproperty
    def rows_margin(self):
        """1D or 2D np.float/int64 ndarray of weighted-N for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the columns
        dimension is MR, because each MR-subvariable has its own weighted N. This is
        because not every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        return self._assembler.rows_margin

    @lazyproperty
    def rows_scale_mean(self):
        """Optional 1D np.float64 ndarray of scale mean for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has an assigned numeric value.
        """
        if not self._columns_have_numeric_value:
            return None

        column_numeric_values = self._columns_dimension_numeric_values
        inner = np.nansum(column_numeric_values * self.row_proportions, axis=1)
        not_a_nan_index = ~np.isnan(column_numeric_values)
        denominator = np.sum(self.row_proportions[:, not_a_nan_index], axis=1)
        return inner / denominator

    @lazyproperty
    def rows_scale_mean_margin(self):
        """Optional float overall mean of row-scale values.

        This value is the "margin" of the `.rows_scale_mean` vector and might typically
        appear in the cell immediately below the `.rows_scale_mean` summary-column. It
        is similar to a "table-total" value, in that it is a scalar that might appear in
        the lower right-hand corner of a table, but note that it does *not* represent
        the overall table in that `.columns_scale_mean_margin` will not have the same
        value (except by chance). This value derives from the numeric values of the
        column elements whereas its counterpart `.columns_scale_mean_margin` derives
        from the numeric values of the row elements.

        This value is `None` if no column has an assigned numeric-value.
        """
        if not self._columns_have_numeric_value:
            return None

        columns_margin = self.columns_margin
        # TODO: This is a hack for MR_X slices, where columns-margin is 2D. Figure out
        # a better way to do this, perhaps using ranges.
        if len(columns_margin.shape) > 1:
            columns_margin = columns_margin[0]

        scale_total = np.nansum(self._columns_dimension_numeric_values * columns_margin)
        not_a_nan_mask = ~np.isnan(self._columns_dimension_numeric_values)
        scale_count = np.sum(columns_margin[not_a_nan_mask])
        return scale_total / scale_count

    @lazyproperty
    def rows_scale_mean_stddev(self):
        """Optional 1D np.float64 ndarray of std-deviation of scale-mean for each row.

        The returned vector (1D array) is to be interpreted as a summary *column*. Also
        note that the underlying scale values are based on the numeric values of the
        opposing *columns-dimension* elements.

        This value is `None` if no column elements have an assigned numeric value.
        """
        return (
            np.sqrt(self._rows_scale_mean_variance)
            if self._columns_have_numeric_value
            else None
        )

    @lazyproperty
    def rows_scale_mean_stderr(self):
        """Optional 1D np.float64 ndarray of standard-error of scale-mean for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has an assigned numeric value.
        """
        return (
            self.rows_scale_mean_stddev / np.sqrt(self.rows_margin)
            if self._columns_have_numeric_value
            else None
        )

    @lazyproperty
    def rows_scale_median(self):
        """Optional 1D np.int/float64 ndarray of scale median for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has an assigned numeric value.
        """
        if not self._columns_have_numeric_value:
            return None

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric_values)
        numeric_values = self._columns_dimension_numeric_values[not_a_nan_index]
        counts = self.counts[:, not_a_nan_index].astype("int64")
        scale_median = np.array(
            [
                self._median(np.repeat(numeric_values, counts[i, :]))
                for i in range(counts.shape[0])
            ]
        )
        return scale_median

    @lazyproperty
    def rows_scale_median_margin(self):
        """Optional scalar numeric median of all row-scale values.

        This value is the "margin" of the `.rows_scale_median` vector and might
        typically appear in the cell immediately below the `.rows_scale_median`
        summary-column. It is similar to a "table-total" value, in that it is a scalar
        that might appear in the lower right-hand corner of a table, but note that it
        does *not* represent the overall table in that `.columns_scale_mean_margin` will
        not have the same value (except by chance). This value derives from the numeric
        values of the column elements whereas its counterpart
        `.columns_scale_median_margin` derives from the numeric values of the row
        elements.

        This value is `None` if no column has an assigned numeric-value.
        """
        if not self._columns_have_numeric_value:
            return None

        columns_margin = self.columns_margin

        # --- hack to accommodate the 2D columns-margin of an MR_X slice ---
        if len(columns_margin.shape) > 1:
            columns_margin = columns_margin[0]

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric_values)
        numeric_values = self._columns_dimension_numeric_values[not_a_nan_index]
        counts = columns_margin[not_a_nan_index].astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def smoothed_dimension_dict(self):
        """dict, smoothed column dimension definition"""
        # TODO:  remove this property when the smoother gets the base measure directly
        # from the matrix later on.
        return self._columns_dimension._dimension_dict

    @lazyproperty
    def summary_pairwise_indices(self):
        return PairwiseSignificance(
            self, self._alpha, self._only_larger
        ).summary_pairwise_indices

    @lazyproperty
    def table_base(self):
        """Scalar or 1D/2D np.int64 ndarray of unweighted-N for table.

        This value is scalar when the slice has no MR dimensions, 1D when the slice has
        one MR dimension (either MR_X or X_MR), and 2D for an MR_X_MR slice.

        The caller must know the dimensionality of the slice in order to correctly
        interpret a 1D value for this property.
        """
        return self._assembler.table_base

    @lazyproperty
    def table_base_unpruned(self):
        """np.int64 scalar or a 1D or 2D ndarray of np.int64 representing table base.

        This value includes hidden vectors, those with either a hide transform on
        their element or that have been pruned (because their base (N) is zero). This
        does not affect a scalar value but when the return value is an ndarray, the
        shape may be different than the array returned by `.table_base`.

        A multiple-response (MR) dimension produces an array of table-base values
        because each element (subvariable) of the dimension represents a logically
        distinct question which may not have been asked of all respondents. When both
        dimensions are MR, the return value is a 2D ndarray. A CAT_X_CAT slice produces
        a scalar value for this property.
        """
        return self._assembler.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        """Scalar or 1D/2D np.float/int64 ndarray of weighted-N table.

        This value is scalar when the slice has no MR dimensions, 1D when the slice has
        one MR dimension (either MR_X or X_MR), and 2D for an MR_X_MR slice.

        The caller must know the dimensionality of the slice in order to correctly
        interpret a 1D value for this property.
        """
        return self._assembler.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        """np.float/int64 scalar or a 1D or 2D ndarray of np.float/int64 table margin.

        This value includes hidden vectors, those with either a hide transform on
        their element or that have been pruned (because their base (N) is zero). Also,
        it does not include inserted subtotals. This
        does not affect a scalar value but when the return value is an ndarray, the
        shape may be different than the array returned by `.table_margin`.

        A matrix with a multiple-response (MR) dimension produces an array of
        table-margin values because each element (subvariable) of the dimension
        represents a logically distinct question which may not have been asked of all
        respondents. When both dimensions are MR, the return value is a 2D ndarray.
        A CAT_X_CAT matrix produces a scalar value for this property.
        """
        return self._assembler.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Provides differentiated name for each stacked table of a 3D cube."""
        if self._cube.ndim < 3:
            return None

        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        """2D ndarray of np.float64 fraction of table count each cell contributes."""
        table_margin = self.table_margin
        rows_dimension_type = self._rows_dimension.dimension_type

        # --- table-margin can be scalar, 2D, or two cases of 1D. A scalar, 2D, and one
        # --- of the 1D table-margins broadcast fine, but the MR-X-CAT case needs
        # --- transposition to get the broadcasting right.
        with np.errstate(divide="ignore", invalid="ignore"):
            if rows_dimension_type == DT.MR_SUBVAR and table_margin.ndim == 1:
                return (self.counts.T / table_margin).T
            return self.counts / table_margin

    @lazyproperty
    def table_proportions_moe(self):
        """1D/2D np.float64 ndarray of margin-of-error (MoE) for table proportions.

        The values are represented as fractions, analogue to the `table_proportions`
        property. This means that the value of 3.5% will have the value 0.035.  The
        values can be np.nan when the corresponding percentage is also np.nan, which
        happens when the respective table margin is 0.
        """
        return Z_975 * self.table_std_err

    @lazyproperty
    def table_std_dev(self):
        """2D np.float64 ndarray of std-dev of table-percent for each table cell."""
        return np.sqrt(self._table_proportion_variance)

    @lazyproperty
    def table_std_err(self):
        """2D optional np.float64 ndarray of std-error of table-percent for each cell.

        A cell value can be np.nan under certain conditions.
        """
        return self._assembler.table_stderrs

    @lazyproperty
    def table_unweighted_bases(self):
        """2D np.int64 ndarray of unweighted table-proportion denominator per cell."""
        return self._assembler.table_unweighted_bases

    @lazyproperty
    def table_weighted_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        return self._assembler.table_weighted_bases

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted count for each slice matrix cell."""
        return self._assembler.unweighted_counts

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        return self._assembler.zscores

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _assembler(self):
        """The Assembler object for this slice.

        The assembler dispatches all second-order measure calculations and insertion
        construction, and orders the result matrix, including removing hidden vectors.
        """
        return Assembler(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def _column_variances(self):
        """Variance for column percentages."""
        p = self.counts / self.columns_margin
        return p * (1 - p)

    @lazyproperty
    def _columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _columns_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each column element.

        A value of np.nan appears for a column element without a numeric-value. All
        subtotal rows have a value of np.nan (subtotals have no numeric value).
        """
        return self._assembler.columns_dimension_numeric_values

    @lazyproperty
    def _columns_have_numeric_value(self):
        """True when one or more column elements have an assigned numeric-value."""
        return not np.all(np.isnan(self._columns_dimension_numeric_values))

    @lazyproperty
    def _columns_scale_mean_variance(self):
        """Optional 1D np.float64 ndarray of scale-mean variance for each column."""
        if not self._rows_have_numeric_value:
            return None

        # --- Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is
        # --- the mean of the data.
        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric_values)
        row_dim_numeric_values = self._rows_dimension_numeric_values[not_a_nan_index]
        numerator = (
            self.counts[not_a_nan_index, :]
            * pow(
                np.broadcast_to(
                    row_dim_numeric_values, self.counts[not_a_nan_index, :].T.shape
                )
                - self.columns_scale_mean.reshape(-1, 1),
                2,
            ).T
        )
        denominator = np.sum(self.counts[not_a_nan_index, :], axis=0)
        return np.nansum(numerator, axis=0) / denominator

    @lazyproperty
    def _dimensions(self):
        """tuple of (rows_dimension, columns_dimension) Dimension objects."""
        return tuple(
            dimension.apply_transforms(transforms)
            for dimension, transforms in zip(
                self._cube.dimensions[-2:], self._transform_dicts
            )
        )

    def _median(self, values):
        return np.median(values) if values.size != 0 else np.nan

    @lazyproperty
    def _row_variance(self):
        """2D np.float64 ndarray of row-percentage variance for each cell."""
        # --- rows-margin is a vector that represents a column (to the right of the
        # --- crosstab), or a table-like structure, in the case of Multiple Response.
        # --- We need to divide all values in the crosstab by it and therefore need to
        # --- cast it to an actual column, when it's not a table already, because
        # --- of how NumPy does broadcasting.
        rows_margin = (
            self.rows_margin  # --- No need to inflate (MR dim involved)
            if len(self.rows_margin.shape) > 1
            else self.rows_margin[:, np.newaxis]  # --- Inflate for correct broadcasting
        )
        p = self.counts / rows_margin
        return p * (1 - p)

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each row element.

        A value of np.nan appears for a row element without a numeric-value. All
        subtotal rows have a value of np.nan (subtotals have no numeric value).
        """
        return self._assembler.rows_dimension_numeric_values

    @lazyproperty
    def _rows_have_numeric_value(self):
        """True when one or more row elements have an assigned numeric-value."""
        return not np.all(np.isnan(self._rows_dimension_numeric_values))

    @lazyproperty
    def _rows_scale_mean_variance(self):
        """Optional 1D np.float64 ndarray of scale-mean variance for each row."""
        if not self._columns_have_numeric_value:
            return None

        # --- Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is
        # --- the mean of the data.
        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric_values)
        col_dim_numeric = self._columns_dimension_numeric_values[not_a_nan_index]

        numerator = self.counts[:, not_a_nan_index] * pow(
            np.broadcast_to(col_dim_numeric, self.counts[:, not_a_nan_index].shape)
            - self.rows_scale_mean.reshape(-1, 1),
            2,
        )
        denominator = np.sum(self.counts[:, not_a_nan_index], axis=1)
        return np.nansum(numerator, axis=1) / denominator

    @lazyproperty
    def _table_proportion_variance(self):
        """2D ndarray of np.float64 table-proportion variance for each matrix cell."""
        p = self.table_proportions
        return p * (1 - p)

    @lazyproperty
    def _transform_dicts(self):
        """Pair of dict (rows_dimension_transforms, columns_dimension_transforms).

        Resolved from the `transforms` argument provided on construction, it always has
        two members, even when one or both dimensions have no transforms. The transforms
        item is an empty dict (`{}`) when no transforms are specified for that
        dimension.
        """
        return (
            self._transforms_dict.get("rows_dimension", {}),
            self._transforms_dict.get("columns_dimension", {}),
        )


class _Strand(CubePartition):
    """1D cube-partition.

    A strand can arise from a 1D cube (non-CA univariate), or as a partition of
    a CA-cube (CAs are 2D) into a sequence of 1D partitions, one for each subvariable.
    """

    def __init__(self, cube, transforms, population, ca_as_0th, slice_idx, mask_size):
        super(_Strand, self).__init__(cube, transforms)
        self._population = population
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx
        self._mask_size = mask_size

    @lazyproperty
    def bases(self):
        """Sequence of weighted base for each row."""
        return tuple(np.broadcast_to(self.table_margin, self.shape))

    @lazyproperty
    def counts(self):
        """tuple, 1D cube counts."""
        return tuple(row.count for row in self._stripe.rows)

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each inserted row in this strand.

        Suitable for use in applying different formatting (e.g. Bold) to inserted rows.
        Provided index values correspond to measure values as-delivered by this strand,
        after any re-ordering specified in a transform.
        """
        return tuple(i for i, row in enumerate(self._stripe.rows) if row.is_inserted)

    @lazyproperty
    def is_empty(self):
        return any(s == 0 for s in self.shape)

    @lazyproperty
    def means(self):
        return tuple(row.mean for row in self._stripe.rows)

    @lazyproperty
    def min_base_size_mask(self):
        mask = self.table_base < self._mask_size
        shape = self.shape
        return (
            mask
            if self.table_base.shape == shape
            else np.logical_or(np.zeros(shape, dtype=bool), mask)
        )

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def population_counts(self):
        return tuple(
            self._table_proportions_as_array
            * self._population
            * self._cube.population_fraction
        )

    @lazyproperty
    def population_moe(self):
        """1D np.float64 ndarray of population margin-of-error (MoE) for table percents.

        The values are represented as population estimates, analogue to the
        `population_counts` property. This means that the values will be presented by
        actual estimated counts of the population The values can be np.nan when the
        corresponding percentage is also np.nan, which happens when the respective
        table margin is 0.
        """
        total_filtered_population = self._population * self._cube.population_fraction
        return Z_975 * total_filtered_population * self.standard_error

    @lazyproperty
    def row_count(self):
        return len(self._stripe.rows)

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._stripe.rows)

    @lazyproperty
    def rows_base(self):
        return np.array([row.base for row in self._stripe.rows])

    @lazyproperty
    def rows_dimension_fills(self):
        """sequence of RGB str like "#def032" fill colors for row elements.

        The values reflect the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions and hidden rows.
        """
        return tuple(row.fill for row in self._stripe.rows)

    @lazyproperty
    def rows_dimension_name(self):
        """str name assigned to rows-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._rows_dimension.name

    @lazyproperty
    def rows_dimension_type(self):
        """Member of DIMENSION_TYPE enum describing type of rows dimension."""
        return self._rows_dimension.dimension_type

    @lazyproperty
    def rows_margin(self):
        return np.array([row.count for row in self._stripe.rows])

    @lazyproperty
    def scale_mean(self):
        """float mean of numeric-value applied to elements, or None.

        This value is `None` when no row-elements have a numeric-value assigned.
        """
        # ---return None when no row-element has been assigned a numeric value. This
        # ---avoids a division-by-zero error.
        if np.all(np.isnan(self._numeric_values)):
            return None

        # ---produce operands with rows without numeric values removed. Notably, this
        # ---excludes subtotal rows.
        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask]

        # ---calculate numerator and denominator---
        total_numeric_value = np.sum(numeric_values * counts)
        total_count = np.sum(counts)

        # ---overall scale-mean is the quotient---
        return total_numeric_value / total_count

    @lazyproperty
    def scale_median(self):
        """np.int64 the median of scales

        The median is calculated using the standard algebra applied to the numeric
        values repeated for each related counts value
        """
        if np.all(np.isnan(self._numeric_values)):
            return None
        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask].astype("int64")
        unwrapped_numeric_values = np.repeat(numeric_values, counts)
        return np.median(unwrapped_numeric_values)

    @lazyproperty
    def scale_std_dev(self):
        """np.float64, the standard deviation of scales"""
        if np.all(np.isnan(self._numeric_values)):
            return None
        return np.sqrt(self.var_scale_mean)

    @lazyproperty
    def scale_std_err(self):
        """np.float64, the standard error of scales"""
        if np.all(np.isnan(self._numeric_values)):
            return None
        counts = self._counts_as_array[self._numeric_values_mask]
        return np.sqrt(self.var_scale_mean / np.sum(counts))

    @lazyproperty
    def shape(self):
        """Tuple of int vector counts for this partition.

        A _Strand has a shape like (5,) which represents its row-count.

        Not to be confused with `numpy.ndarray.shape`, this represent the count of rows
        in this strand. It does not necessarily represent the shape of any underlying
        `numpy.ndarray` object In particular, the value of its row-count can be zero.
        """
        return (self.row_count,)

    @lazyproperty
    def smoothed_dimension_dict(self):
        """dict, row dimension definition"""
        return self._rows_dimension._dimension_dict

    @lazyproperty
    def standard_deviation(self):
        """np.ndarray percentages standard deviation"""
        return np.sqrt(self._variance)

    @lazyproperty
    def standard_error(self):
        """np.ndarray percentages standard error"""
        if self.dimension_types[0] == DT.MR:
            return np.sqrt(self._variance / self.bases)
        return np.sqrt(self._variance / np.sum(self.rows_margin))

    @lazyproperty
    def table_proportions_moe(self):
        """1D np.float64 ndarray of margin-of-error (MoE) for table proportions.

        The values are represented as fractions, analogue to the `table_proportions`
        property. This means that the value of 3.5% will have the value 0.035. The
        values can be np.nan when the corresponding proportion is also np.nan, which
        happens when the respective columns margin is 0.
        """
        return Z_975 * self.standard_error

    @lazyproperty
    def table_base(self):
        """1D, single-element ndarray (like [3770])."""
        # For MR strands, table base is also a strand, since subvars never collapse.
        # We need to keep the ordering and hiding as in rows dimension. All this
        # information is already accessible in the underlying rows property
        # of the `_stripe`.
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_base for row in self._stripe.rows])

        # TODO: shouldn't this just be the regular value for a strand? Maybe change to
        # that if exporter always knows when it's getting this from a strand. The
        # ndarray "wrapper" seems like unnecessary baggage when we know it will always
        # be a scalar.
        return self._stripe.table_base_unpruned

    @lazyproperty
    def table_base_unpruned(self):
        return self._stripe.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        # For MR strands, table base is also a strand, since subvars never collapse.
        # We need to keep the ordering and hiding as in rows dimension. All this
        # information is already accessible in the underlying rows property
        # of the `_stripe`.
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_margin for row in self._stripe.rows])

        return self._stripe.table_margin_unpruned

    @lazyproperty
    def table_margin_unpruned(self):
        return self._stripe.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Only for CA-as-0th case, provides differentiated names for stacked tables."""
        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return tuple(self._table_proportions_as_array * 100)

    @lazyproperty
    def table_proportions(self):
        return tuple(row.table_proportions for row in self._stripe.rows)

    @lazyproperty
    def title(self):
        """The str display name of this strand, suitable for use as a column heading.

        `Strand.name` is the rows-dimension name, which is suitable for use as a title
        of the row-headings. However, a strand can also appear as a *column* and this
        value is a suitable name for such a column.
        """
        return self._cube.title

    @lazyproperty
    def unweighted_bases(self):
        """Sequence of base count for each row, before weighting.

        When the rows dimension is multiple-response, each value is different,
        reflecting the base for that individual subvariable. In all other cases, the
        table base is repeated for each row.
        """
        return tuple(np.broadcast_to(self.table_base, self.shape))

    @lazyproperty
    def unweighted_counts(self):
        """tuple of int unweighted count for each row of stripe."""
        return tuple(row.unweighted_count for row in self._stripe.rows)

    @lazyproperty
    def var_scale_mean(self):
        if np.all(np.isnan(self._numeric_values)):
            return None

        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask]

        return np.nansum(counts * pow((numeric_values - self.scale_mean), 2)) / np.sum(
            counts
        )

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _counts_as_array(self):
        """1D ndarray of count for each row."""
        return np.array([row.count for row in self._stripe.rows_including_hidden])

    @lazyproperty
    def _dimensions(self):
        """tuple of (row,) Dimension object."""
        return (self._rows_dimension,)

    @lazyproperty
    def _numeric_values(self):
        """Array of numeric-value for each element in rows dimension.

        The items in the array can be numeric or np.nan, which appears for an inserted
        row (subtotal) or where the row-element has been assigned no numeric value.
        """
        return np.array(
            [row.numeric_value for row in self._stripe.rows_including_hidden]
        )

    @lazyproperty
    def _numeric_values_mask(self):
        """np.ndarray boolean elements for each element in rows dimension."

        This array contains True or False according to the nan in the numeric_values
        array
        """
        is_a_number_mask = ~np.isnan(self._numeric_values)
        return is_a_number_mask

    @lazyproperty
    def _rows_dimension(self):
        """Dimension object for the single dimension of this strand."""
        return self._cube.dimensions[-1].apply_transforms(self._row_transforms_dict)

    @lazyproperty
    def _row_transforms_dict(self):
        """Transforms dict for the single (rows) dimension of this strand."""
        return self._transforms_dict.get("rows_dimension", {})

    @lazyproperty
    def _stripe(self):
        """The post-transforms 1D data-partition for this strand."""
        return TransformedStripe.stripe(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )

    @lazyproperty
    def _table_proportions_as_array(self):
        return np.array([row.table_proportions for row in self._stripe.rows])

    @lazyproperty
    def _variance(self):
        """variance for cell percentages

        `variance = p * (1-p)`
        """
        p = self._table_proportions_as_array
        return p * (1 - p)


class _Nub(CubePartition):
    """0D slice."""

    @lazyproperty
    def is_empty(self):
        return False if self.unweighted_count else True

    @lazyproperty
    def means(self):
        return self._scalar.means

    @lazyproperty
    def table_base(self):
        return self._scalar.table_base

    @lazyproperty
    def unweighted_count(self):
        return self._cube.unweighted_counts

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _dimensions(self):
        return ()

    @lazyproperty
    def _scalar(self):
        """The pre-transforms data-array for this slice."""
        return MeansScalar(self._cube.counts, self._cube.unweighted_counts)
