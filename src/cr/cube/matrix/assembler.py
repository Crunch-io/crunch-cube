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
    MEASURE as M,
    ORDER_FORMAT,
)
from cr.cube.util import lazyproperty


class _BaseOrderHelper:
    """Base class for ordering helpers."""

    def __init__(
        self, dimensions, second_order_measures, format=ORDER_FORMAT.SIGNED_INDEXES
    ):
        self._dimensions = dimensions
        self._second_order_measures = second_order_measures
        self._format = format

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
    def row_display_order(cls, dimensions, second_order_measures, format):
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
            else (
                _SortRowsByDerivedColumnHelper
                if collation_method == CM.OPPOSING_INSERTION
                and dim_type in DT.ARRAY_TYPES
                else (
                    _SortRowsByInsertedColumnHelper
                    if collation_method == CM.OPPOSING_INSERTION
                    else (
                        _SortRowsByLabelHelper
                        if collation_method == CM.LABEL
                        else (
                            _SortRowsByMarginalHelper
                            if collation_method == CM.MARGINAL
                            else _RowOrderHelper
                        )
                    )
                )
            )
        )
        return HelperCls(dimensions, second_order_measures, format)._display_order

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
        dtype = None if self._format == ORDER_FORMAT.BOGUS_IDS else int
        if self._prune_subtotals:
            return np.array([idx for idx in self._order if idx >= 0], dtype=dtype)
        return np.array(self._order, dtype=dtype)

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
            self._columns_dimension, self._empty_column_idxs, self._format
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
        return CollatorCls.display_order(
            self._rows_dimension, self._empty_row_idxs, self._format
        )

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
                self._format,
            )
        except ValueError:
            return PayloadOrderCollator.display_order(
                self._rows_dimension, self._empty_row_idxs, self._format
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
