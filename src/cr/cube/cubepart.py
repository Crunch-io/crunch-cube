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

import math
import numpy as np
from tabulate import tabulate

from cr.cube.collator import PayloadOrderCollator
from cr.cube.enums import (
    CUBE_MEASURE as CM,
    DIMENSION_TYPE as DT,
    ORDER_FORMAT,
    MARGINAL_ORIENTATION as MO,
)
from cr.cube.matrix.assembler import _BaseOrderHelper
from cr.cube.matrix.measure import SecondOrderMeasures
from cr.cube.matrix.subtotals import SumSubtotals
from cr.cube.min_base_size_mask import MinBaseSizeMask
from cr.cube.measures.pairwise_significance import PairwiseSignificance
from cr.cube.scalar import MeansScalar
from cr.cube.stripe.assembler import _BaseOrderHelper as stripe_BaseOrderHelper
from cr.cube.stripe.measure import StripeMeasures
from cr.cube.util import lazyproperty

# ---This is the quantile of the normal Cumulative Distribution Function (CDF) at
# ---probability 97.5% (p=.975), since the computed confidence interval
# ---is ±2.5% (.025) on each side of the CDF.
Z_975 = 1.959964


class CubePartition:
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
                f"transforms.pairwise_indices.alpha, when defined, must be a list of 1 "
                f"or 2 float values between 0.0 and 1.0 exclusive. Got {repr(value)}"
            )

        # --- legacy float "by-itself" case ---
        if isinstance(value, float):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    "alpha value, when provided, must be between 0.0 and 1.0 "
                    f"exclusive. Got {repr(value)}"
                )
            return (value, None)

        # --- sequence case ---
        for x in value[:2]:
            if not isinstance(x, float) or not 0.0 < x < 1.0:
                raise ValueError(
                    f"transforms.pairwise_indices.alpha must be a list of 1 or 2 float "
                    f"values between 0.0 and 1.0 exclusive. Got {repr(value)}"
                )

        if len(value) == 1:
            return (value[0], None)

        return tuple(sorted(value[:2]))

    @lazyproperty
    def _available_measures(self):
        """sorted list of available CUBE_MEASURE members in the cube response."""
        return sorted(list(self._cube.available_measures), key=lambda el: el.name)

    @lazyproperty
    def _default_contents(self):
        """1D/2D np.float64 ndarray of counts, means or sums, if available."""
        measure = self._available_measures[0]
        return getattr(
            self, {CM.COUNT: "counts", CM.MEAN: "means", CM.SUM: "sums"}[measure]
        )

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

    def __repr__(self):
        """Provide text representation suitable for working at console.

        Falls back to a default repr on exception, such as might occur in
        unit tests where object need not otherwise be provided with all
        instance variable values.
        """
        try:
            dimensionality = " x ".join(dt.name for dt in self.dimension_types)
            title = (
                f"{type(self).__name__}(name='{self.name}', "
                f"dimension_types='{dimensionality}')"
            )
            contents = [
                [row_label] + row.tolist()
                for row_label, row in zip(self.row_labels, self._default_contents)
            ]
            return (
                f"{title}"
                f"\nShowing: {self._available_measures[0].name}"
                f"\n{tabulate(contents, [''] + self.column_labels.tolist())}"
                f"\nAvailable measures: {str(self._available_measures)}"
            )
        except Exception:
            return super(_Slice, self).__repr__()  # noqa

    # ---interface ---------------------------------------------------

    @lazyproperty
    def column_aliases(self):
        """1D str ndarray of alias for each column, for use as column headings."""
        dim = self._dimensions[1]
        return np.array(dim.element_aliases + dim.subtotal_aliases)[self._column_order]

    @lazyproperty
    def column_codes(self):
        """1D int ndarray of code for each column, for use as column headings."""
        dim = self._dimensions[1]
        return np.array(dim.element_ids + dim.insertion_ids)[self._column_order]

    @lazyproperty
    def column_index(self):
        """2D np.float64 ndarray of column-index "percentage".

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the rows variable.
        """
        return self._assemble_matrix(self._measures.column_index.blocks)

    @lazyproperty
    def column_labels(self):
        """1D str ndarray of name for each column, for use as column headings."""
        dim = self._dimensions[1]
        return np.array(dim.element_labels + dim.subtotal_labels)[self._column_order]

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def column_proportions(self):
        """2D np.float64 ndarray of column-proportion for each matrix cell.

        This is the proportion of the weighted-N (aka. weighted base) of its column
        that the *weighted-count* in each cell represents, generally a number between
        0.0 and 1.0. Note that within an inserted subtotal vector involving differences,
        the values can range between -1.0 and 1.0.
        """
        return self._assemble_matrix(self._measures.column_proportions.blocks)

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
    def column_share_sum(self):
        """2D optional np.float64 ndarray of column share sum value for each table cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.

        Column share of sum is the sum of each subvar item divided by the TOTAL number
        of column items.
        """
        try:
            return self._assemble_matrix(self._measures.column_share_sum.blocks)
        except ValueError:
            raise ValueError(
                "`.column_share_sum` is undefined for a cube-result without a sum "
                "measure"
            )

    @lazyproperty
    def column_proportion_variances(self):
        """2D ndarray of np.float64 column-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.column_proportion_variances.blocks)

    @lazyproperty
    def column_std_dev(self):
        """standard deviation for column percentages

        `std_deviation = sqrt(variance)`
        """
        return np.sqrt(self.column_proportion_variances)

    @lazyproperty
    def column_std_err(self):
        """standard error for column percentages

        `std_error = sqrt(variance/N)`
        """
        return self._assemble_matrix(self._measures.column_std_err.blocks)

    @lazyproperty
    def column_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted col-proportion denominator per cell."""
        return self._assemble_matrix(self._measures.column_unweighted_bases.blocks)

    @lazyproperty
    def column_weighted_bases(self):
        """2D np.float64 ndarray of column-proportion denominator for each cell."""
        return self._assemble_matrix(self._measures.column_weighted_bases.blocks)

    @lazyproperty
    def columns_base(self):
        """1D/2D np.float64 ndarray of unweighted-N for each column/cell of slice.

        This array is 2D (a distinct base for each cell) when the rows dimension is MR,
        because each MR-subvariable has its own unweighted N. This is because not every
        possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
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
    def columns_squared_base(self):
        """1D np.float64 ndarray of squared weights, summed for each column.

        This is a measure that needs to be asked from zz9 explicitly. It is only used
        in the calculation of the pairwise comparisons, where weights are applied, in
        order to adjust for the "design effect" of the study (reduce the inflated Nw).
        """
        if not self._measures.columns_squared_base.is_defined:
            return None

        return self._assemble_marginal(self._measures.columns_squared_base)

    @lazyproperty
    def columns_dimension_description(self):
        """str description assigned to columns-dimension."""
        return self._dimensions[1].description

    @lazyproperty
    def columns_dimension_name(self):
        """str name assigned to columns-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._dimensions[1].name

    @lazyproperty
    def columns_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` describing columns dimension."""
        return self._dimensions[1].dimension_type

    @lazyproperty
    def columns_margin(self):
        """1D or 2D np.float64 ndarray of weighted-N for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the rows dimension
        is MR, because each MR-subvariable has its own weighted N. This is because not
        every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
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
        """1D or 2D np.float64 ndarray of weighted-proportion for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the rows dimension
        is MR, because each MR-subvariable has its own weighted N. This is because not
        every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
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
        """Optional 1D np.float64 ndarray of scale mean for each column.

        The returned vector is to be interpreted as a summary *row*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has an assigned numeric value.
        """
        return self._assemble_marginal(self._measures.columns_scale_mean)

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

        # TODO: This is a hack for X_Array slices, where rows-margin is undefined.
        # I think this probably shouldn't be defined across arrays, but to minimize
        # test damage, we use the first column of the weighted bases, which is
        # equal to the rows_margin for CAT_X_CAT and always exists for others.
        rows_margin = self.row_weighted_bases[:, 0]

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
        return self._assemble_marginal(self._measures.columns_scale_mean_stddev)

    @lazyproperty
    def columns_scale_mean_stderr(self):
        """Optional 1D np.float64 ndarray of scale-mean standard-error for each row.

        The returned vector is to be interpreted as a summary *row*. Also note that the
        underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has a numeric value assigned or if
        the columns-weighted-base is `None` (eg an array variable in the row dim).
        """
        return self._assemble_marginal(self._measures.columns_scale_mean_stderr)

    @lazyproperty
    def columns_scale_median(self):
        """Optional 1D np.float64 ndarray of scale median for each column.

        The returned vector is to be interpreted as a summary *row*. Also note that the
        underlying scale values are based on the numeric values of the opposing
        *rows-dimension* elements.

        This value is `None` if no row element has been assigned a numeric value.
        """
        return self._assemble_marginal(self._measures.columns_scale_median)

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

        # TODO: This is a hack for X_Array slices, where rows-margin is undefined.
        # I think this probably shouldn't be defined across arrays, but to minimize
        # test damage, we use the first column of the weighted bases, which is
        # equal to the rows_margin for CAT_X_CAT and always exists for others.
        rows_margin = self.row_weighted_bases[:, 0]

        not_a_nan_mask = ~np.isnan(self._rows_dimension_numeric_values)
        numeric_values = self._rows_dimension_numeric_values[not_a_nan_mask]
        counts = np.nan_to_num(rows_margin[not_a_nan_mask]).astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of weighted cube counts."""
        return self._assemble_matrix(self._measures.weighted_counts.blocks)

    weighted_counts = counts

    @lazyproperty
    def description(self):
        """str description of this slice, which it takes from its rows-dimension."""
        return self._rows_dimension.description

    @lazyproperty
    def has_scale_means(self):
        """True if the slice has valid columns scale mean."""
        return True if self.columns_scale_mean is not None else False

    @lazyproperty
    def inserted_column_idxs(self):
        """tuple of int index of each subtotal column in slice."""
        # --- insertions have a negative idx in their order sequence ---
        return tuple(i for i, col_idx in enumerate(self._column_order) if col_idx < 0)

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each subtotal row in slice."""
        # --- insertions have a negative idx in their order sequence ---
        return tuple(
            i for i, row_idx in enumerate(self._row_order_signed_indexes) if row_idx < 0
        )

    @lazyproperty
    def derived_column_idxs(self):
        """tuple of int index of each derived column-element in slice.

        An element is derived if it's a subvariable of a multiple response dimension,
        which has been produced by the zz9, and inserted into the response data.

        All other elements, including regular MR and CA subvariables, as well as
        categories of CAT dimensions, are not derived. Subtotals are also not derived
        in this sense, because they're not even part of the data (elements).
        """
        return self._derived_element_idxs(self._dimensions[1], self._column_order)

    @lazyproperty
    def derived_row_idxs(self):
        """tuple of int index of each derived row-element in slice.

        An element is derived if it's a subvariable of a multiple response dimension,
        which has been produced by the zz9, and inserted into the response data.

        All other elements, including regular MR and CA subvariables, as well as
        categories of CAT dimensions, are not derived. Subtotals are also not derived
        in this sense, because they're not even part of the data (elements).
        """
        return self._derived_element_idxs(
            self._rows_dimension, self._row_order_signed_indexes
        )

    @lazyproperty
    def diff_column_idxs(self):
        """tuple of int index of each difference column-element in slice."""
        return self._diff_element_idxs(self._dimensions[1], self._column_order)

    @lazyproperty
    def diff_row_idxs(self):
        """tuple of int index of each difference row-element in slice."""
        return self._diff_element_idxs(
            self._rows_dimension, self._row_order_signed_indexes
        )

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
        try:
            return self._assemble_matrix(self._measures.means.blocks)
        except ValueError:
            raise ValueError(
                "`.means` is undefined for a cube-result without a mean measure"
            )

    @lazyproperty
    def min_base_size_mask(self):
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def name(self):
        """str name assigned to this slice.

        A slice takes the name of its rows-dimension.
        """
        return self.rows_dimension_name

    @staticmethod
    def _pairwise_indices(p_vals, t_stats, alpha, only_larger):
        """1D ndarray containing tuples of int pairwise indices of each column."""
        with np.errstate(divide="ignore", invalid="ignore"):
            significance = p_vals < alpha
            if only_larger:
                significance = np.logical_and(t_stats < 0, significance)
            col_significance = np.empty((len(significance),), dtype=object)
            col_significance[:] = [
                tuple(np.where(sig_row)[0]) for sig_row in significance
            ]
            return col_significance

    def _pairwise_means_indices(self, alpha, only_larger):
        """2D optional ndarray of tuple of int column-idxs means pairwise-t threshold.

        Raises `ValueError if the cube-result does not include `means` cube-measures.
        """
        return np.array(
            [
                self._pairwise_indices(
                    self._pairwise_significance_means_p_vals(col),
                    self._pairwise_significance_means_t_stats(col),
                    alpha,
                    only_larger,
                )
                for col in range(len(self._column_order))
            ]
        ).T

    def _pairwise_significance_p_vals(self, column_idx):
        """2D optional np.float64 ndarray of overlaps-p_vals matrices for subvar idx.

        For cubes where the last dimension is categorical, column idxs represent
        specific categories.

        For cubes where the last dimension is a multiple response, each subvariable
        pairwise significance matrix is a 2D ndarray of the p-vals for the selected
        subvariable index (the selected column).

        Raises `ValueError if the cube-result does not include `overlaps`
        and `valid_overlaps` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        if self._cube_has_overlaps:
            # If overlaps are defined, calculate significance based on them
            return self._assemble_matrix(
                self._measures.pairwise_p_vals_for_subvar(base_column_idx).blocks
            )
        return self._assemble_matrix(
            self._measures.pairwise_p_vals(base_column_idx).blocks
        )

    def _pairwise_significance_t_stats(self, column_idx):
        """2D optional np.float64 ndarray of overlaps-t_stats matrices for subvar idx.

        For cubes where the last dimension is categorical, column idxs represent
        specific categories.

        For cubes where the last dimension is a multiple response, each subvariable
        pairwise significance matrix is a 2D ndarray of the t-stats for the selected
        subvariable index (the selected column).

        Raises `ValueError if the cube-result does not include `overlaps`
        and `valid_overlaps` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        if self._cube_has_overlaps:
            # If overlaps are defined, calculate significance based on them
            return self._assemble_matrix(
                self._measures.pairwise_t_stats_for_subvar(base_column_idx).blocks
            )
        return self._assemble_matrix(
            self._measures.pairwise_t_stats(base_column_idx).blocks
        )

    def _pairwise_significance_means_p_vals(self, column_idx):
        """2D optional np.float64 ndarray of mean difference p_vals for column idx.

        Raises `ValueError if the cube-result does not include `mean` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_significance_means_p_vals(base_column_idx).blocks
        )

    def _pairwise_significance_means_t_stats(self, column_idx):
        """2D optional np.float64 ndarray of mean difference t_stats for column idx.

        Raises `ValueError if the cube-result does not include `mean` cube-measures.
        """
        base_column_idx = self._column_order[column_idx]
        return self._assemble_matrix(
            self._measures.pairwise_significance_means_t_stats(base_column_idx).blocks
        )

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
        return np.array(
            [
                self._pairwise_indices(
                    self._pairwise_significance_p_vals(col),
                    self._pairwise_significance_t_stats(col),
                    self._alpha,
                    self._only_larger,
                )
                for col in range(len(self._column_order))
            ]
        ).T

    @lazyproperty
    def pairwise_indices_alt(self):
        """2D ndarray of tuple of int column-idxs meeting alternate threshold.

        This value is None if no alternate threshold has been defined.
        """

        if self._alpha_alt is None:
            return None

        return np.array(
            [
                self._pairwise_indices(
                    self._pairwise_significance_p_vals(col),
                    self._pairwise_significance_t_stats(col),
                    self._alpha_alt,
                    self._only_larger,
                )
                for col in range(len(self._column_order))
            ]
        ).T

    @lazyproperty
    def pairwise_means_indices(self):
        """Optional 2D ndarray of tuple column-idxs significance threshold for mean.

        Like::

            [
               [(1, 3, 4), (), (0,), (), ()],
               [(2,), (1, 2), (), (), (0, 3)],
               [(), (), (), (), ()],
            ]

        Has the same shape as `.means`. Each int represents the offset of another
        column in the same row with a confidence interval meeting the threshold defined
        for this analysis.
        """
        try:
            return self._pairwise_means_indices(self._alpha, self._only_larger)
        except ValueError:
            raise ValueError(
                "`.pairwise_means_indices` is undefined for a cube-result "
                "without a mean measure"
            )

    @lazyproperty
    def pairwise_means_indices_alt(self):
        """2D ndarray of tuple of column-idxs meeting alternate threshold for mean.

        This value is None if no alternate threshold has been defined.
        """
        if self._alpha_alt is None:
            return None
        try:
            return self._pairwise_means_indices(self._alpha_alt, self._only_larger)
        except ValueError:
            raise ValueError(
                "`.pairwise_means_indices_alt` is undefined for a cube-result "
                "without a mean measure"
            )

    def pairwise_significance_p_vals(self, column_idx):
        """2D ndarray of pairwise-significance p-vals matrices for column idx."""
        return self._pairwise_significance_p_vals(column_idx)

    def pairwise_significance_t_stats(self, column_idx):
        """return 2D ndarray of pairwise-significance t-stats for selected column."""
        return self._pairwise_significance_t_stats(column_idx)

    def pairwise_significance_means_p_vals(self, column_idx):
        """Optional 2D ndarray of means significance p-vals matrices for column idx."""
        # Significance of means difference is available only is cube contains means.
        try:
            return self._pairwise_significance_means_p_vals(column_idx)
        except ValueError:
            raise ValueError(
                "`.pairwise_significance_means_p_vals` is undefined for a cube-result "
                "without a mean measure"
            )

    def pairwise_significance_means_t_stats(self, column_idx):
        """Optional 2D ndarray of means significance t-stats matrices for column idx."""
        # Significance of means difference is available only is cube contains means.
        try:
            return self._pairwise_significance_means_t_stats(column_idx)
        except ValueError:
            raise ValueError(
                "`.pairwise_significance_means_t_stats` is undefined for a cube-result "
                "without a mean measure"
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
    def payload_order(self):
        """1D np.int64 ndarray of signed int idx respecting the payload order.

        Positive integers indicate the 1-indexed position in payload of regular
        elements, while negative integers are the subtotal insertions.

        Needed for reordering color palette in exporter.
        """
        empty_rows_idxs = tuple(np.where(self._measures.rows_pruning_mask)[0])
        po = PayloadOrderCollator(self._rows_dimension, empty_rows_idxs).payload_order
        return tuple(po)

    @lazyproperty
    def population_proportions(self):
        """2D np.float64 ndarray of proportions

        The proportion used to calculate proportion counts depends on the dimension
        types.
        """
        population_proportions = self._assemble_matrix(
            self._measures.population_proportions.blocks
        )
        # Diff subtotals not allowed in population measure
        if self.diff_row_idxs:
            population_proportions[self.diff_row_idxs, :] = np.nan
        if self.diff_column_idxs:
            population_proportions[:, self.diff_column_idxs] = np.nan
        return population_proportions

    @lazyproperty
    def population_counts(self):
        """2D np.float64 ndarray of population counts per cell.

        The (estimated) population count is computed based on the `population` value
        provided when the Slice is created (`._population`). It is also adjusted to
        account for any filters that were applied as part of the query
        (`._cube.population_fraction`).

        `._population` and `_cube.population_fraction` are both scalars and so do not
        affect sort order.
        """
        return (
            self.population_proportions
            * self._population
            * self._cube.population_fraction
        )

    @lazyproperty
    def population_std_err(self):
        """2D np.float64 ndarray of standard errors

        The proportion used to calculate proportion counts depends on the dimension
        types.
        """
        return self._assemble_matrix(self._measures.population_std_err.blocks)

    @lazyproperty
    def population_counts_moe(self):
        """2D np.float64 ndarray of population-count margin-of-error (MoE) per cell.

        The values are represented as population estimates, analogue to the
        `population_counts` property. This means that the values will be presented by
        actual estimated counts of the population. The values can be np.nan when the
        corresponding percentage is also np.nan, which happens when the
        respective margin is 0.

        When calculating the estimates of categorical dates, the total populatioin is
        not "divided" between its categories, but rather considered constant for all
        categorical dates (or waves). Hence, the different standard errors will be
        applied in these specific cases (like the `row_std_err` or `column_std_err`).
        If categorical dates are not involved, the standard `table_std_err` is used.
        """
        std_err = self.population_std_err
        total_filtered_population = self._population * self._cube.population_fraction
        return Z_975 * total_filtered_population * std_err

    @lazyproperty
    def pvals(self):
        """2D optional np.float64 ndarray of p-value for each cell.

        A p-value is a measure of the probability that an observed difference could have
        occurred just by random chance. The lower the p-value, the greater the
        statistical significance of the observed difference.

        A cell value of np.nan indicates a meaningful p-value could not be computed for
        that cell.
        """
        return self._assemble_matrix(self._measures.pvalues.blocks)

    pvalues = pvals

    @lazyproperty
    def residual_test_stats(self):
        """Exposes pvals and zscores (with HS) stacked together

        Public method used as cube_method for the SOA API
        """
        return np.stack([self.pvals, self.zscores])

    @lazyproperty
    def row_aliases(self):
        """1D str ndarray of row alias for each matrix row.

        These are suitable for use as row headings; alias for subtotal rows appear in
        the sequence and alias are ordered to correspond with their respective data
        row.
        """
        dim = self._dimensions[0]
        return np.array(dim.element_aliases + dim.subtotal_aliases)[
            self._row_order_signed_indexes
        ]

    @lazyproperty
    def row_codes(self):
        """1D int ndarray of row codes for each matrix row.

        These are suitable for use as row headings; codes for subtotal rows appear in
        the sequence and codes are ordered to correspond with their respective data
        row.
        """
        dim = self._dimensions[0]
        return np.array(dim.element_ids + dim.insertion_ids)[
            self._row_order_signed_indexes
        ]

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of row name for each matrix row.

        These are suitable for use as row headings; labels for subtotal rows appear in
        the sequence and labels are ordered to correspond with their respective data
        row.
        """
        dim = self._dimensions[0]
        return np.array(dim.element_labels + dim.subtotal_labels)[
            self._row_order_signed_indexes
        ]

    def row_order(self, format=ORDER_FORMAT.SIGNED_INDEXES):
        """1D np.int64 ndarray of idx for each assembled row of matrix.

        If order format is `SIGNED_INDEXES` negative values represent inserted
        subtotal-row locations; for `BOGUS_IDS` insertios are represented by
        `ins_{insertion_id}` string.

        Indices appear in the order rows are to appear in the final result.

        Needed for reordering color palette in exporter.
        """
        if format == ORDER_FORMAT.BOGUS_IDS:
            return _BaseOrderHelper.row_display_order(
                self._dimensions, self._measures, format=ORDER_FORMAT.BOGUS_IDS
            )
        return self._row_order_signed_indexes

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def row_proportions(self):
        """2D np.float64 ndarray of row-proportion for each matrix cell.

        This is the proportion of the weighted-N (aka. weighted base) of its row
        that the *weighted-count* in each cell represents, generally a number between
        0.0 and 1.0. Note that within an inserted subtotal vector involving differences,
        the values can range between -1.0 and 1.0.
        """
        return self._assemble_matrix(self._measures.row_proportions.blocks)

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
    def row_share_sum(self):
        """2D optional np.float64 ndarray of row share sum value for each table cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.

        Row share of sum is the sum of each subvar item divided by the TOTAL number of
        row items.
        """
        try:
            return self._assemble_matrix(self._measures.row_share_sum.blocks)
        except ValueError:
            raise ValueError(
                "`.row_share_sum` is undefined for a cube-result without a sum measure"
            )

    @lazyproperty
    def row_proportion_variances(self):
        """2D ndarray of np.float64 row-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.row_proportion_variances.blocks)

    @lazyproperty
    def row_std_dev(self):
        """2D np.float64 ndarray of standard deviation for row percentages."""
        return np.sqrt(self.row_proportion_variances)

    @lazyproperty
    def row_std_err(self):
        """2D np.float64 ndarray of standard errors for row percentages."""
        return self._assemble_matrix(self._measures.row_std_err.blocks)

    @lazyproperty
    def row_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted row-proportion denominator per cell."""
        return self._assemble_matrix(self._measures.row_unweighted_bases.blocks)

    @lazyproperty
    def row_weighted_bases(self):
        """2D np.float64 ndarray of row-proportion denominator for each table cell."""
        return self._assemble_matrix(self._measures.row_weighted_bases.blocks)

    @lazyproperty
    def rows_base(self):
        """1D/2D np.float64 ndarray of unweighted-N for each row/cell of slice.

        This array is 2D (a distinct base for each cell) when the columns dimension is
        MR, because each MR-subvariable has its own unweighted N. This is because not
        every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        # --- an X_ARRAY slice produces a 2D row-base (each cell has its own N) ---
        # --- TODO: Should rows_base only be defined when it's 1D? This would
        # --- require changes to exporter to use the bases to give a "rows_base_range"
        if not self._measures.rows_unweighted_base.is_defined:
            return self.row_unweighted_bases

        # --- otherwise rows-base is a vector ---
        return self._assemble_marginal(self._measures.rows_unweighted_base)

    @lazyproperty
    def rows_dimension_alias(self):
        """str alias assigned to rows-dimension."""
        return self._rows_dimension.alias

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
        elements = self._rows_dimension.valid_elements
        subtotals = self._rows_dimension.subtotals
        return tuple(
            # ---Subtotals have negative sequential indexes (-1, -2, ..., -m)---
            # ---To index them properly, we need to convert those indexes to---
            # ---zero based positive indexes (0, 1, ... m - 1) i.e. -idx - 1---
            (elements[idx].fill if idx >= 0 else subtotals[idx + len(subtotals)].fill)
            for idx in self._row_order_signed_indexes
        )

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
        """1D or 2D np.float64 ndarray of weighted-N for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the columns
        dimension is MR, because each MR-subvariable has its own weighted N. This is
        because not every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
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
        """1D or 2D np.float64 ndarray of weighted-proportion for each column of slice.

        This array is 2D (a distinct margin value for each cell) when the columns
        dimension is MR, because each MR-subvariable has its own weighted N. This is
        because not every possible response is necessarily offered to every respondent.

        In all other cases, the array is 1D, containing one value for each column.
        """
        # --- an X_MR slice produces a 2D rows-margin (each cell has its own N) ---
        # --- TODO: Should rows_margin_proportion only be defined when it's 1D? This
        # --- would require changes to exporter to use the bases to give a
        # --- "rows_margin_range"
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
        """Optional 1D np.float64 ndarray of scale mean for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has an assigned numeric value.
        """
        return self._assemble_marginal(self._measures.rows_scale_mean)

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

        # TODO: This is a hack for X_Array slices, where columns-margin is undefined.
        # I think this measure probably also shouldn't be defined across arrays, but to
        # minimize test damage, we use the first row of the row weighted bases, which is
        # equal to the columnss_margin for CAT_X_CAT and always exists for others.
        columns_margin = self.column_weighted_bases[0, :]

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
        return self._assemble_marginal(self._measures.rows_scale_mean_stddev)

    @lazyproperty
    def rows_scale_mean_stderr(self):
        """Optional 1D np.float64 ndarray of standard-error of scale-mean for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has a numeric value assigned or if
        the rows-weighted-base is `None` (eg an array variable in the column dim).
        """
        return self._assemble_marginal(self._measures.rows_scale_mean_stderr)

    @lazyproperty
    def rows_scale_median(self):
        """Optional 1D np.float64 ndarray of scale median for each row.

        The returned vector is to be interpreted as a summary *column*. Also note that
        the underlying scale values are based on the numeric values of the opposing
        *columns-dimension* elements.

        This value is `None` if no column element has an assigned numeric value.
        """
        return self._assemble_marginal(self._measures.rows_scale_median)

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

        # TODO: This is a hack for X_Array slices, where columns-margin is undefined.
        # I think this measure probably also shouldn't be defined across arrays, but to
        # minimize test damage, we use the first row of the row weighted bases, which is
        # equal to the columnss_margin for CAT_X_CAT and always exists for others.
        columns_margin = self.column_weighted_bases[0, :]

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric_values)
        numeric_values = self._columns_dimension_numeric_values[not_a_nan_index]
        counts = np.nan_to_num(columns_margin[not_a_nan_index]).astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def smoothed_column_index(self):
        """2D np.float64 ndarray of smoothed column-index "percentage".

        If cube has smoothing specification in the transforms it will return the
        column index smoothed according to the algorithm and the parameters
        specified, otherwise it fallbacks to unsmoothed values.
        """
        return self._assemble_matrix(self._measures.smoothed_column_index.blocks)

    @lazyproperty
    def smoothed_column_percentages(self):
        """2D np.float64 ndarray of smoothed column-percentages for each matrix cell.

        If cube has smoothing specification in the transforms it will return the
        column percentages smoothed according to the algorithm and the parameters
        specified, otherwise it fallbacks to unsmoothed values.
        """
        return self.smoothed_column_proportions * 100

    @lazyproperty
    def smoothed_column_proportions(self):
        """2D np.float64 ndarray of smoothed column-proportion for each matrix cell.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        column the cell appears in (aka. column-margin). Generally a number between 0.0
        and 1.0 inclusive, but subtotal differences can be between -1.0 and 1.0
        inclusive.

        If cube has smoothing specification in the transforms it will return the
        column proportions smoothed according to the algorithm and the parameters
        specified, otherwise it fallbacks to unsmoothed values.
        """
        return self._assemble_matrix(self._measures.smoothed_column_proportions.blocks)

    @lazyproperty
    def smoothed_columns_scale_mean(self):
        """Optional 1D np.float64 ndarray of smoothed scale mean for each column.

        If cube has smoothing specification in the transforms it will return the
        column scale mean smoothed according to the algorithm and the parameters
        specified, otherwise it fallbacks to unsmoothed values.
        """
        return self._assemble_marginal(self._measures.smoothed_columns_scale_mean)

    @lazyproperty
    def smoothed_means(self):
        """2D optional np.float64 ndarray of smoothed mean value for each table cell.

        If cube has smoothing specification in the transforms it will return the
        smoothed means according to the algorithm and the parameters specified,
        otherwise it fallbacks to unsmoothed values.
        """
        try:
            return self._assemble_matrix(self._measures.smoothed_means.blocks)
        except ValueError:
            raise ValueError(
                "`.means` is undefined for a cube-result without a mean measure"
            )

    @lazyproperty
    def stddev(self):
        """2D optional np.float64 ndarray of stddev value for each table cell.

        Raises `ValueError` if the cube-result does not include a stddev cube-measure.
        """
        try:
            return self._assemble_matrix(self._measures.stddev.blocks)
        except ValueError:
            raise ValueError(
                "`.stddev` is undefined for a cube-result without a stddev measure"
            )

    @lazyproperty
    def sums(self):
        """2D optional np.float64 ndarray of sum value for each table cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.
        """
        try:
            return self._assemble_matrix(self._measures.sums.blocks)
        except ValueError:
            raise ValueError(
                "`.sums` is undefined for a cube-result without a sum measure"
            )

    @lazyproperty
    def summary_pairwise_indices(self):
        return PairwiseSignificance(
            self, self._alpha, self._only_larger
        ).summary_pairwise_indices

    @lazyproperty
    def tab_label(self):
        """Subvar label of slice id if first dimension is a CA_SUBVAR, '"' otherwise."""
        first_dimension = self._cube.dimensions[0]
        return (
            first_dimension.valid_elements[self._slice_idx].label
            if first_dimension.dimension_type == DT.CA_SUBVAR
            else ""
        )

    @lazyproperty
    def tab_alias(self):
        """Subvar alias of slice id if first dimension is a CA_SUBVAR, '"' otherwise."""
        first_dimension = self._cube.dimensions[0]
        return (
            first_dimension.valid_elements[self._slice_idx].alias
            if first_dimension.dimension_type == DT.CA_SUBVAR
            else ""
        )

    @lazyproperty
    def table_base(self):
        """Scalar or 1D/2D np.float64 ndarray of unweighted-N for table.

        This value is scalar when the slice has no MR dimensions, 1D when the slice has
        one MR dimension (either MR_X or X_MR), and 2D for an MR_X_MR slice.

        The caller must know the dimensionality of the slice in order to correctly
        interpret a 1D value for this property.

        This value has four distinct forms, depending on the slice dimensions:

            * ARR_X_ARR - 2D ndarray with a distinct table-base value per cell.
            * ARR_X - 1D ndarray of value per *row* when only rows dimension is ARR.
            * X_ARR - 1D ndarray of value per *column* when only col dimension is ARR
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
    def table_margin(self):
        """Scalar or 1D/2D np.float64 ndarray of weighted-N table.

        This value is scalar when the slice has no MR dimensions, 1D when the slice has
        one MR dimension (either MR_X or X_MR), and 2D for an MR_X_MR slice.

        The caller must know the dimensionality of the slice in order to correctly
        interpret a 1D value for this property.

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
    def table_name(self):
        """Optional table name for this Slice

        Provides differentiated name for each stacked table of a 3D cube.
        """
        if self._cube.ndim < 3:
            return None
        title = self._cube.name
        valid_elements = self._cube.dimensions[0].valid_elements
        if valid_elements.element_ids:
            table_name = valid_elements[self._slice_idx].label
            return f"{title}: {table_name}"
        return None

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        """2D ndarray of np.float64 fraction of table count each cell contributes.

        This is the proportion of the weighted-count for cell to the weighted-N of the
        row the cell appears in (aka. table-margin). Generally a number between 0.0 and
        1.0 inclusive, but subtotal differences can be between -1.0 and 1.0 inclusive.
        """
        return self._assemble_matrix(self._measures.table_proportions.blocks)

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
    def table_proportion_variances(self):
        """2D ndarray of np.float64 table-proportion variance for each matrix cell."""
        return self._assemble_matrix(self._measures.table_proportion_variances.blocks)

    @lazyproperty
    def table_std_dev(self):
        """2D np.float64 ndarray of std-dev of table-percent for each table cell."""
        return np.sqrt(self.table_proportion_variances)

    @lazyproperty
    def table_std_err(self):
        """2D optional np.float64 ndarray of std-error of table-percent for each cell.

        A cell value can be np.nan under certain conditions.
        """
        return self._assemble_matrix(self._measures.table_std_err.blocks)

    @lazyproperty
    def table_unweighted_bases(self):
        """2D np.float64 ndarray of unweighted table-proportion denominator per cell."""
        return self._assemble_matrix(self._measures.table_unweighted_bases.blocks)

    @lazyproperty
    def table_weighted_bases(self):
        """2D np.float64 ndarray of table-proportion denominator for each cell."""
        return self._assemble_matrix(self._measures.table_weighted_bases.blocks)

    @lazyproperty
    def total_share_sum(self):
        """2D optional np.float64 ndarray of total share sum value for each table cell.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.

        Total share of sum is the sum of each subvar item divided by the TOTAL of items.
        """
        try:
            return self._assemble_matrix(self._measures.total_share_sum.blocks)
        except ValueError:
            raise ValueError(
                "`.total_share_sum` is undefined for a cube-result without a sum "
                "measure"
            )

    @lazyproperty
    def table_base_range(self):
        """[min, max] np.float64 ndarray range of the table_base (table-unweighted-base)

        A CAT_X_CAT has a scalar for all table-unweighted-bases, but arrays have more
        than one table-weighted-base. This collapses all the values them to the range,
        and it is "unpruned", meaning that it is calculated before any hiding or
        removing of empty rows/columns.
        """
        return self._measures.table_unweighted_bases_range.value

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
        """2D np.float64 ndarray of unweighted count for each slice matrix cell."""
        return self._assemble_matrix(self._measures.unweighted_counts.blocks)

    @lazyproperty
    def zscores(self):
        """2D np.float64 ndarray of std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        return self._assemble_matrix(self._measures.zscores.blocks)

    # ---implementation (helpers)-------------------------------------

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
            self._row_order_signed_indexes
            if marginal.orientation == MO.ROWS
            else self._column_order
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
        return np.block(blocks)[
            np.ix_(self._row_order_signed_indexes, self._column_order)
        ]

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
                (
                    np.nan
                    if diffs_nan and len(subtotal.subtrahend_idxs) > 0
                    else np.sum(base_vector[subtotal.addend_idxs])
                    - np.sum(base_vector[subtotal.subtrahend_idxs])
                )
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
    def _columns_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each column element.

        A value of np.nan appears for a column element without a numeric-value. All
        subtotal rows have a value of np.nan (subtotals have no numeric value).
        """
        elements = self._dimensions[1].valid_elements
        return np.array(
            [
                (elements[idx].numeric_value if idx >= 0 else np.nan)
                for idx in self._column_order
            ]
        )

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
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum(numerator, axis=0) / denominator

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

    @lazyproperty
    def _dimensions(self):
        """tuple of (rows_dimension, columns_dimension) Dimension objects."""
        return tuple(
            dimension.apply_transforms(transforms)
            for dimension, transforms in zip(
                self._cube.dimensions[-2:], self._transform_dicts
            )
        )

    @lazyproperty
    def _measures(self):
        """SecondOrderMeasures collection object for this cube-result."""
        return SecondOrderMeasures(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def _row_order_signed_indexes(self):
        """Row order idx with signed idxs."""
        return _BaseOrderHelper.row_display_order(
            self._dimensions, self._measures, format=ORDER_FORMAT.SIGNED_INDEXES
        )

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows_dimension_numeric_values(self):
        """1D optional np.int/float64 ndarray of numeric-value for each row element.

        A value of np.nan appears for a row element without a numeric-value. All
        subtotal rows have a value of np.nan (subtotals have no numeric value).
        """
        elements = self._rows_dimension.valid_elements
        return np.array(
            [
                (elements[idx].numeric_value if idx >= 0 else np.nan)
                for idx in self._row_order_signed_indexes
            ]
        )

    @lazyproperty
    def _rows_have_numeric_value(self):
        """True when one or more row elements have an assigned numeric-value."""
        return not np.all(np.isnan(self._rows_dimension_numeric_values))

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

    def __repr__(self):
        """Provide text representation suitable for working at console.

        Falls back to a default repr on exception, such as might occur in
        unit tests where object need not otherwise be provided with all
        instance variable values.
        """
        try:
            title = (
                f"{type(self).__name__}(name='{self.name}', "
                f"dimension_type='{self.dimension_types[0].name}')"
            )
            contents = [
                [row_label, row]
                for row_label, row in zip(self.row_labels, self._default_contents)
            ]
            return (
                f"{title}"
                f"\nShowing: {self._available_measures[0].name}"
                f"\n{tabulate(contents, ['', self.name])}"
                f"\nAvailable measures: {str(self._available_measures)}"
            )
        except Exception:
            return super(_Strand, self).__repr__()  # noqa

    @lazyproperty
    def weighted_counts(self):
        """1D np.float64 ndarray of weighted count for each row of strand.

        The values are int when the underlying cube-result has no weighting.
        """
        return self._assemble_vector(self._measures.weighted_counts.blocks)

    counts = weighted_counts

    @lazyproperty
    def derived_row_idxs(self):
        """tuple of int index of each derived row-element in this strand.

        Subtotals cannot be derived

        An element is derived if it's a subvariable of a multiple response dimension,
        which has been produced by the zz9, and inserted into the response data.

        All other elements, including regular MR and CA subvariables, as well as
        categories of CAT dimensions, are not derived. Subtotals are also not derived
        in this sense, because they're not even part of the data (elements).
        """
        rows_dim = self._rows_dimension
        n_subtotals = len(rows_dim.subtotals)
        derivs = [e.derived for e in rows_dim.valid_elements] + [False] * n_subtotals
        return tuple(np.where(np.array(derivs)[self._row_order_signed_indexes])[0])

    @lazyproperty
    def diff_row_idxs(self):
        """tuple of int index of each difference row-element in this strand.

        Valid elements are cannot be differences, only some subtotals can.
        """
        rows_dim = self._rows_dimension
        n_valids = len(rows_dim.valid_elements)
        diffs = [False] * n_valids + [e.is_difference for e in rows_dim.subtotals]
        return tuple(np.where(np.array(diffs)[self._row_order_signed_indexes])[0])

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each inserted row in this strand.

        Suitable for use in applying different formatting (e.g. Bold) to inserted rows.
        Provided index values correspond to measure values as-delivered by this strand,
        after any insertion of subtotals, re-ordering, and hiding/pruning of rows
        specified in a transform has been applied.

        Provided index values correspond rows after any insertion of subtotals,
        re-ordering, and hiding/pruning.
        """
        return tuple(
            i for i, row_idx in enumerate(self._row_order_signed_indexes) if row_idx < 0
        )

    @lazyproperty
    def has_scale_means(self):
        """True if the strand has valid scale means."""
        return True if self.scale_mean is not None else False

    @lazyproperty
    def is_empty(self):
        return any(s == 0 for s in self.shape)

    @lazyproperty
    def means(self):
        """1D np.float64 ndarray of mean for each row of strand.

        Raises ValueError when accessed on a cube-result that does not contain a means
        cube-measure.
        """
        try:
            return self._assemble_vector(self._measures.means.blocks)
        except ValueError:
            raise ValueError(
                "`.means` is undefined for a cube-result without a mean measure"
            )

    @lazyproperty
    def min_base_size_mask(self):
        """1D bool ndarray of True for each row that fails to meet min-base spec.

        The "base" is the physical (unweighted) count of respondents to the question.
        When this is lower than a specified threshold, the reliability of the value is
        too low to be meaningful. The threshold is defined by the caller (user).
        """
        return self.unweighted_bases < self._mask_size

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def payload_order(self):
        """1D np.int64 ndarray of signed int idx respecting the payload order.

        Positive integers indicate the 1-indexed position in payload of regular
        elements, while negative integers are the subtotal insertions.

        Needed for reordering color palette in exporter.
        """
        empty_row_idxs = tuple(
            i for i, N in enumerate(self._measures.pruning_base) if N == 0
        )
        return tuple(
            PayloadOrderCollator(self._rows_dimension, empty_row_idxs).payload_order
        )

    @lazyproperty
    def population_counts(self):
        """1D np.float64 ndarray of population count for each row of strand.

        The (estimated) population count is computed based on the `population` value
        provided when the Strand is created. It is also adjusted to account for any
        filters that were applied as part of the query.
        """
        return (
            self.population_proportions
            * self._population
            * self._cube.population_fraction
        )

    @lazyproperty
    def population_counts_moe(self):
        """1D np.float64 ndarray of population margin-of-error (MoE) for table percents.

        The values are represented as population estimates, analogue to the
        `population_counts` property. This means that the values will be presented by
        actual estimated counts of the population The values can be np.nan when the
        corresponding percentage is also np.nan, which happens when the respective
        table margin is 0.
        """
        total_filtered_population = self._population * self._cube.population_fraction
        return Z_975 * total_filtered_population * self.population_proportion_stderrs

    @lazyproperty
    def population_proportions(self):
        """1D np.float64 population-proportion for each row

        Generally equal to the table_proprotions, but because we don't divide the
        population when the row is a CAT_DATE, can also be all 1s. Used to calculate
        the population_counts.
        """
        population_proportions = self._assemble_vector(
            self._measures.population_proportions.blocks
        )
        # Diff subtotals not allowed in population measure
        if self.diff_row_idxs:
            population_proportions[self.diff_row_idxs] = np.nan
        return population_proportions

    @lazyproperty
    def population_proportion_stderrs(self):
        """1D np.float64 population-proportion-standard-error for each row

        Generally equal to the table_proprotion_standard_error, but because we don't
        divide the population when the row is a CAT_DATE, can also be all 0s. Used to
        calculate the population_counts_moe.
        """
        return self._assemble_vector(
            self._measures.population_proportion_stderrs.blocks
        )

    @lazyproperty
    def row_count(self):
        """int count of rows in a returned measure or marginal.

        This count includes inserted rows but not rows that have been hidden/pruned.
        """
        return len(self._row_order_signed_indexes)

    @lazyproperty
    def row_aliases(self):
        """1D str ndarray of alias for each row, for use as row headings."""
        return np.array(
            self._rows_dimension.element_aliases + self._rows_dimension.subtotal_aliases
        )[self._row_order_signed_indexes]

    @lazyproperty
    def row_codes(self):
        """1D int ndarray of code for each row, for use as row headings."""
        return np.array(
            self._rows_dimension.element_ids + self._rows_dimension.insertion_ids
        )[self._row_order_signed_indexes]

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of name for each row, suitable for use as row headings."""
        return np.array(
            self._rows_dimension.element_labels + self._rows_dimension.subtotal_labels
        )[self._row_order_signed_indexes]

    def row_order(self, format=ORDER_FORMAT.SIGNED_INDEXES):
        """1D np.int64 ndarray of idx for each assembled row of stripe.

        If order format is `SIGNED_INDEXES` negative values represent inserted
        subtotal-row locations; for `BOGUS_IDS` insertios are represented by
        `ins_{insertion_id}` string.
        Indices appear in the order rows are to appear in the final result.

        Needed for reordering color palette in exporter.
        """
        # --- specify dtype explicitly to prevent error when display-order is empty. The
        # --- default dtype is float, which cannot be used to index an array.
        if format == ORDER_FORMAT.BOGUS_IDS:
            return self._row_order_bogus_ids
        return self._row_order_signed_indexes

    @lazyproperty
    def rows_base(self):
        """1D np.float64 ndarray of unweighted-N for each row of slice."""
        # --- for a strand, this is the same as unweighted-counts, but needs this
        # --- alternate property so it can be accessed uniformly between a slice and a
        # --- strand.
        return self.unweighted_counts

    @lazyproperty
    def rows_dimension_alias(self):
        """str alias assigned to rows-dimension."""
        return self._rows_dimension.alias

    @lazyproperty
    def rows_dimension_description(self):
        """str description assigned to rows-dimension.

        Reflects the resolved dimension-description transform cascade.
        """
        return self._rows_dimension.description

    @lazyproperty
    def rows_dimension_fills(self):
        """tuple of optional RGB str like "#def032" fill color for each strand row.

        Each value reflects the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions, ordering, and hidden rows. A fill value is `None`
        when no explicit fill color is defined for that row, indicating the default fill
        color for that row should be used, probably coming from a caller-defined theme.
        """
        element_fills = tuple(e.fill for e in self._rows_dimension.valid_elements)
        subtotal_fills = tuple(st.fill for st in self._rows_dimension.subtotals)
        return tuple(
            # ---Subtotals have negative sequential indexes (-1, -2, ..., -m)---
            # ---To index them properly, we need to convert those indexes to---
            # ---zero based positive indexes (0, 1, ... m - 1) i.e. -idx - 1---
            (
                element_fills[idx]
                if idx > -1
                else subtotal_fills[idx + len(subtotal_fills)]
            )
            for idx in self._row_order_signed_indexes
        )

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
        """1D np.float64 ndarray of weighted-N for each row of slice."""
        # --- for a strand, this is the same as (weighted) counts, but needs this
        # --- alternate name so it can be accessed uniformly between a slice and strand.
        return self.counts

    @lazyproperty
    def scale_mean(self):
        """Optional float mean of row numeric-values (scale).

        This value is `None` when no row-elements have a numeric-value assigned. The
        numeric value (aka. "scale") for a row is its count multiplied by the
        numeric-value of its element. For example, if 100 women responded "Very Likely"
        and the numeric-value of the "Very Likely" response (element) was 4, then the
        scale for that row would be 400. The scale mean is the average of those scale
        values over the total count of responses.
        """
        return self._measures.scaled_counts.scale_mean

    @lazyproperty
    def scale_median(self):
        """Optional int/float median of scaled weighted-counts.

        This value is `None` when no rows have a numeric-value assigned.
        """
        return self._measures.scaled_counts.scale_median

    @lazyproperty
    def scale_std_dev(self):
        """Optional np.float64 standard-deviation of scaled weighted counts.

        This value is `None` when no rows have a numeric-value assigned.
        """
        return self._measures.scaled_counts.scale_stddev

    scale_stddev = scale_std_dev

    @lazyproperty
    def scale_std_err(self):
        """Optional np.float64 standard-error of scaled weighted counts.

        This value is `None` when no rows have a numeric-value assigned. The value has
        the same units as the assigned numeric values and indicates the dispersion of
        the scaled-count distribution from its mean (scale-mean).
        """
        return self._measures.scaled_counts.scale_stderr

    scale_stderr = scale_std_err

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
    def share_sum(self):
        """1D np.float64 ndarray of share of sum for each row of strand.

        Raises `ValueError` if the cube-result does not include a sum cube-measure.

        Share of sum is the sum of each subvar item divided by the TOTAL number of
        items.
        """
        try:
            return self._assemble_vector(self._measures.share_sum.blocks)
        except ValueError:
            raise ValueError(
                "`.share_sum` is undefined for a cube-result without a sum measure"
            )

    @lazyproperty
    def smoothed_means(self):
        """1D np.float64 ndarray of smoothed mean for each row of strand.

        If cube has smoothing specification in the transforms it will return the
        smoothed means according to the algorithm and the parameters specified,
        otherwise it fallbacks to unsmoothed values.
        """
        try:
            return self._assemble_vector(self._measures.smoothed_means.blocks)
        except ValueError:
            raise ValueError(
                "`.means` is undefined for a cube-result without a mean measure"
            )

    @lazyproperty
    def stddev(self):
        """1D np.float64 ndarray of stddev for each row of strand.

        Raises ValueError when accessed on a cube-result that does not contain a stddev
        cube-measure.
        """
        try:
            return self._assemble_vector(self._measures.stddev.blocks)
        except ValueError:
            raise ValueError(
                "`.stddev` is undefined for a cube-result without a stddev measure"
            )

    @lazyproperty
    def sums(self):
        """1D np.float64 ndarray of sum for each row of strand.

        Raises ValueError when accessed on a cube-result that does not contain a sum
        cube-measure.
        """
        try:
            return self._assemble_vector(self._measures.sums.blocks)
        except ValueError:
            raise ValueError(
                "`.sums` is undefined for a cube-result without a sum measure"
            )

    @lazyproperty
    def tab_label(self):
        """Subvar label of strand if first dimension is a CA_SUBVAR, '""' otherwise."""
        first_dimension = self._cube.dimensions[0]
        return (
            first_dimension.valid_elements[self._slice_idx].label
            if first_dimension.dimension_type == DT.CA_SUBVAR
            else ""
        )

    @lazyproperty
    def tab_alias(self):
        """Subvar alias of strand if first dimension is a CA_SUBVAR, '""' otherwise."""
        first_dimension = self._cube.dimensions[0]
        return (
            first_dimension.valid_elements[self._slice_idx].alias
            if first_dimension.dimension_type == DT.CA_SUBVAR
            else ""
        )

    @lazyproperty
    def table_base_range(self):
        """[min, max] np.float64 ndarray range of unweighted-N for this stripe.

        A non-MR stripe will have a single base, represented by min and max being the
        same value. Each row of an MR stripe has a distinct base, which is reduced to a
        range in that case.
        """
        return self._measures.unweighted_bases.table_base_range

    @lazyproperty
    def table_margin_range(self):
        """[min, max] np.float64 ndarray range of (total) weighted-N for this stripe.

        A non-MR stripe will have a single margin, represented by min and max being the
        same value. Each row of an MR stripe has a distinct base, which is reduced to a
        range in that case.
        """
        return self._measures.weighted_bases.table_margin_range

    @lazyproperty
    def table_name(self):
        """Optional table name for this strand

        Only for CA-as-0th case, provides differentiated names for stacked tables.
        """
        title = self._cube.name
        valid_elements = self._cube.dimensions[0].valid_elements
        if valid_elements.element_ids:
            table_name = valid_elements[self._slice_idx].label
            return f"{title}: {table_name}"
        return None

    @lazyproperty
    def table_percentages(self):
        """1D np.float64 ndarray of table-percentage for each row.

        Table-percentage is the fraction of the table weighted-N contributed by each
        row, expressed as a percentage (float between 0.0 and 100.0 inclusive).
        """
        return tuple(self.table_proportions * 100)

    @lazyproperty
    def table_proportion_moes(self):
        """1D np.float64 ndarray of table-proportion margin-of-error (MoE) for each row.

        The values are represented as fractions, analogue to the `table_proportions`
        property. This means that the value of 3.5% will have the value 0.035. The
        values can be np.nan when the corresponding proportion is also np.nan, which
        happens when the respective columns margin is 0.
        """
        return Z_975 * self.table_proportion_stderrs

    @lazyproperty
    def table_proportion_stddevs(self):
        """1D np.float64 ndarray of table-proportion std-deviation for each row."""
        return self._assemble_vector(self._measures.table_proportion_stddevs.blocks)

    @lazyproperty
    def table_proportion_stderrs(self):
        """1D np.float64 ndarray of table-proportion std-error for each row."""
        return self._assemble_vector(self._measures.table_proportion_stderrs.blocks)

    @lazyproperty
    def table_proportions(self):
        """1D np.float64 ndarray of fraction of weighted-N contributed by each row.

        The proportion is expressed as a float between 0.0 and 1.0 inclusive.
        """
        return self._assemble_vector(self._measures.table_proportions.blocks)

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
        """1D np.float64 ndarray of base count for each row, before weighting.

        When the rows dimension is multiple-response (MR), each value is different,
        reflecting the base for that individual subvariable. In all other cases, the
        table base is repeated for each row.
        """
        return self._assemble_vector(self._measures.unweighted_bases.blocks)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.float64 ndarray of unweighted count for each row of stripe."""
        return self._assemble_vector(self._measures.unweighted_counts.blocks)

    @lazyproperty
    def weighted_bases(self):
        """1D np.float64 ndarray of table-proportion denominator for each row.

        For a non-MR strand, all values in the array are the same. For an MR strand,
        each value may be different, reflecting the fact that not all response options
        were necessarily presented to all respondents.
        """
        return self._assemble_vector(self._measures.weighted_bases.blocks)

    # ---implementation (helpers)-------------------------------------

    def _assemble_vector(self, blocks):
        """Return 1D ndarray of base_vector with inserted subtotals, in order.

        `blocks` is a pair of two 1D arrays, first the base-values and then the subtotal
        values of the stripe vector. The returned array is sequenced in the computed
        row order including possibly removing hidden or pruned values.
        """
        return np.concatenate(blocks)[self._row_order_signed_indexes]

    @lazyproperty
    def _dimensions(self):
        """tuple of (row,) Dimension object."""
        return (self._rows_dimension,)

    @lazyproperty
    def _rows_dimension(self):
        """Dimension object for the single dimension of this strand."""
        return self._cube.dimensions[-1].apply_transforms(self._row_transforms_dict)

    @lazyproperty
    def _row_transforms_dict(self):
        """Transforms dict for the single (rows) dimension of this strand."""
        return self._transforms_dict.get("rows_dimension", {})

    @lazyproperty
    def _measures(self):
        """StripeMeasures collection object for this stripe."""
        return StripeMeasures(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )

    @lazyproperty
    def _row_order_bogus_ids(self):
        """Row order with bogus ids."""
        return np.array(
            stripe_BaseOrderHelper.display_order(
                self._rows_dimension, self._measures, format=ORDER_FORMAT.BOGUS_IDS
            )
        )

    @lazyproperty
    def _row_order_signed_indexes(self):
        """Row order idx with signed idxs."""
        return np.array(
            stripe_BaseOrderHelper.display_order(
                self._rows_dimension, self._measures, format=ORDER_FORMAT.SIGNED_INDEXES
            ),
            dtype=int,
        )


class _Nub(CubePartition):
    """0D slice."""

    @lazyproperty
    def is_empty(self):
        """True if the partition has no counts, False otherwise"""
        if self.unweighted_count <= 0:
            return True
        return math.isnan(self.unweighted_count)

    @lazyproperty
    def means(self):
        """Float scalar representing the mean."""
        return self._scalar.means

    @lazyproperty
    def table_base(self):
        """Int scalar of the unweighted N of the table."""
        return self._scalar.table_base

    @lazyproperty
    def table_name(self):
        return None

    @lazyproperty
    def unweighted_count(self):
        """Integer scalar of total unweighted count of the table"""
        return self._cube.unweighted_counts

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _dimensions(self):
        return ()

    @lazyproperty
    def _scalar(self):
        """The pre-transforms data-array for this slice."""
        return MeansScalar(self._cube.means, self._cube.unweighted_counts)
