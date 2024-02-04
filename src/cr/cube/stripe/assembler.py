# encoding: utf-8

"""The `StripeAssembler` object provides the external interface for this module.

Its name derives from its role to "assemble" a finished 1D array ("stripe") for a
particular measure from the base measure values and inserted subtotals, to reorder the
rows according to the dimension *order* transforms, and to hide rows that are either
hidden by the user or "pruned" because they contain no observations.
"""

import numpy as np

from cr.cube.collator import (
    ExplicitOrderCollator,
    PayloadOrderCollator,
    SortByValueCollator,
)
from cr.cube.enums import COLLATION_METHOD as CM, ORDER_FORMAT
from cr.cube.util import lazyproperty


# === ORDER HELPERS ===


class _BaseOrderHelper:
    """Base class for ordering helpers."""

    def __init__(self, rows_dimension, measures, format=ORDER_FORMAT.SIGNED_INDEXES):
        self._rows_dimension = rows_dimension
        self._measures = measures
        self._format = format

    @classmethod
    def display_order(cls, rows_dimension, measures, format):
        """1D np.int64 ndarray of signed int idx for each row of stripe.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        order_spec = rows_dimension.order_spec
        HelperCls = (
            _SortByMeasureHelper
            if order_spec.collation_method == CM.UNIVARIATE_MEASURE
            else (
                _SortByLabelHelper
                if order_spec.collation_method == CM.LABEL
                else _OrderHelper
            )
        )
        return HelperCls(rows_dimension, measures, format)._display_order

    @lazyproperty
    def _display_order(self):
        """tuple of signed int idx for each row of stripe.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        raise NotImplementedError(
            f"`{type(self).__name__}` must implement `._display_order`"
        )  # pragma: no cover

    @lazyproperty
    def _empty_row_idxs(self):
        """tuple of int index for each row with N = 0.

        These rows are subject to pruning, depending on a user setting in the dimension.
        """
        return tuple(i for i, N in enumerate(self._measures.pruning_base) if N == 0)

    @lazyproperty
    def _order_spec(self):
        """_OrderSpec object for the stripe dimension.

        Provides access to ordering details like measure and sort-direction.
        """
        return self._rows_dimension.order_spec


class _OrderHelper(_BaseOrderHelper):
    """Encapsulates the complexity of the various kinds of row ordering."""

    @lazyproperty
    def _display_order(self):
        """tuple of signed int idx for each row of stripe.

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


class _BaseSortByValueHelper(_BaseOrderHelper):
    """A base class that orders elements by a set of values.

    This class is intended only to serve as a base for the other sort-by-value classes
    which must all provide their own implentations of `_element_values` and
    `_subtotal_values`.

    If `_display_order` encouters a ValueError, it falls back to the payload order.
    """

    @lazyproperty
    def _display_order(self):
        """tuple of signed int idx for each row of stripe.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        # --- fall back to payload order if sort-by-value failes
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
    def _element_values(self):
        """Sequence of body values that form the basis for sort order.

        Must be implemented by child classes.
        """
        raise NotImplementedError(  # pragma: no cover
            "{type(self).__name__} must implement `._element_values`"
        )

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of subtotal values that form the basis for sort order.

        Must be implemented by child classes.
        """
        raise NotImplementedError(  # pragma: no cover
            f"{type(self).__name__} must implement `._subtotal_values`"
        )


class _SortByLabelHelper(_BaseSortByValueHelper):
    """Orders rows by the value of a specified measure."""

    @lazyproperty
    def _element_values(self):
        """Sequence of labels that form the basis for sort order.

        There is one value per row and values appear in payload (dimension) element
        order. These are only the "base" values and do not include insertions.
        """
        return np.array(self._rows_dimension.element_labels)

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal labels that contribute to the sort basis.

        There is one value per row subtotal and values appear in payload (dimension)
        insertion order.
        """
        return np.array(self._rows_dimension.subtotal_labels)


class _SortByMeasureHelper(_BaseSortByValueHelper):
    """Orders rows by the value of a specified measure."""

    @lazyproperty
    def _element_values(self):
        """Sequence of measure values that form the basis for sort order.

        There is one value per row and values appear in payload (dimension) element
        order. These are only the "base" values and do not include insertions.
        """
        return self._measure.blocks[0]

    @lazyproperty
    def _measure(self):
        """Second-order measure object providing values for sort."""
        measure_keyname = self._order_spec.measure_keyname
        measure_propname = {
            "base_unweighted": "unweighted_bases",
            "base_weighted": "weighted_bases",
            "count_unweighted": "unweighted_counts",
            "count_weighted": "weighted_counts",
            "mean": "means",
            "percent": "table_proportions",
            # --- margin-of-error is strictly proportional to stderr ---
            "percent_moe": "table_proportion_stderrs",
            "percent_stddev": "table_proportion_stddevs",
            "percent_stderr": "table_proportion_stderrs",
            "population": "population_proportions",  # strictly proportional
            "population_moe": "population_proportion_stderrs",  # strictly proportional
            "share_sum": "share_sum",
            "sum": "sums",
        }.get(measure_keyname)

        if measure_propname is None:
            raise ValueError(
                f"sort-by-value for measure '{measure_keyname}' is not yet supported"
            )

        return getattr(self._measures, measure_propname)

    @lazyproperty
    def _subtotal_values(self):
        """Sequence of row-subtotal values that contribute to the sort basis.

        There is one value per row subtotal and values appear in payload (dimension)
        insertion order.
        """
        return self._measure.blocks[1]
