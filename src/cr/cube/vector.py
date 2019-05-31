# encoding: utf-8

"""Vector objects used by frozen-slice.

A vector represents a row or column of data in the overall data matrix. It composes the
element that corresponds to the row or column and so knows the name, element_id, numeric
value, etc. for the row or column.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import norm

from cr.cube.util import lazyproperty


# ===TRANSFORMATION VECTORS===


class _BaseMatrixInsertionVector(object):
    """Base class for matrix insertion vectors.

    There are some differences that arise when there are rows *and* columns, which
    entails the complication of insertion *intersections*.
    """

    def __init__(self, matrix, subtotal):
        self._matrix = matrix
        self._subtotal = subtotal

    @lazyproperty
    def addend_idxs(self):
        return np.array(self._subtotal.addend_idxs)

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def base(self):
        return np.sum(np.array([vec.base for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def base_values(self):
        return np.sum(
            np.array([row.base_values for row in self._addend_vectors]), axis=0
        )

    @lazyproperty
    def column_index(self):
        # TODO: Calculate insertion column index for real. Check with Mike
        return np.array([np.nan] * len(self.values))

    @lazyproperty
    def fill(self):
        """Unconditionally `None` for an insertion vector.

        A `fill` value is normally a str RGB value like "#da09fc", specifying the color
        to use for a chart category or series representing this element. The value
        reflects the resolved element-fill transform cascade. Since an insertion cannot
        (currently) have a fill-transform, the default value of `None` (indicating "use
        default color") is unconditionally returned.
        """
        return None

    @lazyproperty
    def hidden(self):
        """True if vector is pruned.

        Insertions can never be hidden explicitly (for now). They can also almost never
        be pruned, except in the case when all of the opposite vectors are also pruned
        (thus leaving no elements for this insertion vector).
        """
        return self.pruned

    @lazyproperty
    def is_insertion(self):
        return True

    @lazyproperty
    def label(self):
        return self._subtotal.label

    @lazyproperty
    def margin(self):
        return np.sum(np.array([vec.margin for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def means(self):
        return np.array([np.nan])

    @lazyproperty
    def numeric(self):
        return np.nan

    @lazyproperty
    def table_margin(self):
        return self._matrix.table_margin

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)


class InsertionColumn(_BaseMatrixInsertionVector):
    """Represents an inserted (subtotal) column."""

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        Insertions can almost never be pruned, except in the case when all of the
        opposite vectors are also pruned (thus leaving no elements for this
        insertion vector).
        """
        return self._subtotal.prune and not np.any(
            np.array([row.base for row in self._matrix.rows])
        )

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            column
            for i, column in enumerate(self._matrix.columns)
            if i in self._subtotal.addend_idxs
        )


class InsertionRow(_BaseMatrixInsertionVector):
    """Represents an inserted (subtotal) row."""

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        Insertions can almost never be pruned, except in the case when all of the
        opposite vectors are also pruned (thus leaving no elements for this
        insertion vector).
        """
        return self._subtotal.prune and not np.any(
            np.array([column.base for column in self._matrix.columns])
        )

    @lazyproperty
    def pvals(self):
        return np.array([np.nan] * len(self._matrix.columns))

    @lazyproperty
    def zscore(self):
        return np.array([np.nan] * len(self._matrix.columns))

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._matrix.rows)
            if i in self._subtotal.addend_idxs
        )


class _BaseTransformationVector(object):
    """Base class for most transformation vectors."""

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._base_vector.fill

    @lazyproperty
    def hidden(self):
        return self._base_vector.hidden

    @lazyproperty
    def is_insertion(self):
        return self._base_vector.is_insertion

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def means(self):
        return self._base_vector.means

    @lazyproperty
    def numeric(self):
        return self._base_vector.numeric

    @lazyproperty
    def table_base(self):
        return self._base_vector.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_vector.table_margin


class AssembledVector(_BaseTransformationVector):
    """Vector with base, as well as inserted, elements (of the opposite dimension)."""

    def __init__(self, base_vector, opposite_inserted_vectors):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        # TODO: Do for real
        return np.array(
            self._top_base_values
            + self._interleaved_base_values
            + self._bottom_base_values
        )

    @lazyproperty
    def column_index(self):
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_column_index
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def proportions(self):
        return self.values / self.margin

    @lazyproperty
    def pvals(self):
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_pvals
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def table_proportions(self):
        return self.values / self._base_vector.table_margin

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
        )

    @lazyproperty
    def zscore(self):
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_zscore
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def _bottom_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _bottom_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _interleaved_base_values(self):
        base_values = []
        for i in range(len(self._base_vector.base_values)):
            base_values.append(self._base_vector.base_values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.base_values[inserted_vector.addend_idxs]
                    )
                    base_values.append(insertion_value)
        return tuple(base_values)

    @lazyproperty
    def _interleaved_column_index(self):
        # TODO: Replace with real column index values from insertions vectors. This
        # should be something like:
        #   col_ind = (ins1.prop + ins2.prop) / (ins1.baseline + ins2.baseline)
        # ask @mike to confirm
        column_index = []
        for i, value in enumerate(self._base_vector.column_index):
            column_index.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    column_index.append(np.nan)
        return tuple(column_index)

    @lazyproperty
    def _interleaved_pvals(self):
        pvals = []
        for i, value in enumerate(self._base_vector.pvals):
            pvals.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    pvals.append(np.nan)
        return tuple(pvals)

    @lazyproperty
    def _interleaved_values(self):
        values = []
        for i in range(len(self._base_vector.values)):
            values.append(self._base_vector.values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.values[inserted_vector.addend_idxs]
                    )
                    values.append(insertion_value)
        return tuple(values)

    @lazyproperty
    def _interleaved_zscore(self):
        zscore = []
        for i, value in enumerate(self._base_vector.zscore):
            zscore.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    zscore.append(np.nan)
        return tuple(zscore)

    @lazyproperty
    def _top_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _top_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )


class _BaseVectorAfterHiding(_BaseTransformationVector):
    """Reflects a row or column with hidden elements removed."""

    def __init__(self, base_vector):
        self._base_vector = base_vector

    @lazyproperty
    def base(self):
        if not isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base
        return self._base_vector.base[self._visible_element_idxs]

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self._visible_element_idxs]

    @lazyproperty
    def margin(self):
        if not isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin
        return self._base_vector.margin[self._visible_element_idxs]

    @lazyproperty
    def means(self):
        return self._base_vector.means[self._visible_element_idxs]

    @lazyproperty
    def table_proportions(self):
        return self._base_vector.table_proportions[self._visible_element_idxs]

    @lazyproperty
    def values(self):
        return self._base_vector.values[self._visible_element_idxs]


class VectorAfterHiding(_BaseVectorAfterHiding):
    """Reflects a row or column with hidden elements removed."""

    def __init__(self, base_vector, opposite_vectors):
        super(VectorAfterHiding, self).__init__(base_vector)
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index[self._visible_element_idxs]

    @lazyproperty
    def proportions(self):
        return self._base_vector.proportions[self._visible_element_idxs]

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals[self._visible_element_idxs]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore[self._visible_element_idxs]

    @lazyproperty
    def _visible_element_idxs(self):
        """An 1D ndarray of int idxs of non-hidden values, suitable for indexing.

        This value is derived from the opposing vectors collection, based on the hidden
        status of its elements.
        """
        return np.array(
            [
                idx
                for idx, opposite_vector in enumerate(self._opposite_vectors)
                if not opposite_vector.hidden
            ],
            dtype=int,
        )


class OrderedVector(_BaseTransformationVector):
    """In charge of indexing elements properly, after ordering transform."""

    def __init__(self, base_vector, opposing_order):
        self._base_vector = base_vector
        self._opposing_order_arg = opposing_order

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self._opposing_order]

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def _opposing_order(self):
        return (
            slice(None)
            if self._opposing_order_arg is None
            else self._opposing_order_arg
        )

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals

    @lazyproperty
    def values(self):
        return self._base_vector.values[self._opposing_order]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore


# ===OPERAND VECTORS===


class _BaseVector(object):
    """Base class for all vector objects."""

    def __init__(self, element, base_counts):
        self._element = element
        self._base_counts = base_counts

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._element.fill

    @lazyproperty
    def hidden(self):
        """True if vector is hidden.

        Vectors are hidden in two ways:

            1. Explicitly via transforms
            2. Implicitly when the base is 0 (also called pruning)

        This property checks whether a vector needs to be hidden, either implicitly or
        explicitly. It is used when iterating through rows or columns, to form the
        correct result.
        """
        return self._element.is_hidden or (self._element.prune and self.pruned)

    @lazyproperty
    def is_insertion(self):
        return False

    @lazyproperty
    def label(self):
        return self._element.label

    @lazyproperty
    def numeric(self):
        return self._element.numeric_value

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)


class CategoricalVector(_BaseVector):
    """Main staple of all measures.

    Some of the measures it can calculate by itself, others it needs to receive at
    construction time (like table margin and zscores).
    """

    def __init__(
        self, counts, base_counts, element, table_margin, zscore=None, column_index=None
    ):
        super(CategoricalVector, self).__init__(element, base_counts)
        self._counts = counts
        self._table_margin = table_margin
        self._zscore = zscore
        self._column_index = column_index

    @lazyproperty
    def base_values(self):
        if not isinstance(self._base_counts, np.ndarray):
            return np.array([self._base_counts])
        return self._base_counts

    @lazyproperty
    def column_index(self):
        return self._column_index

    @lazyproperty
    def margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin

    @lazyproperty
    def pvals(self):
        return 2 * (1 - norm.cdf(np.abs(self._zscore)))

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def table_proportions(self):
        return self.values / self._table_margin

    @lazyproperty
    def values(self):
        if not isinstance(self._counts, np.ndarray):
            return np.array([self._counts])
        return self._counts

    @lazyproperty
    def zscore(self):
        return self._zscore


class CatXMrVector(CategoricalVector):
    def __init__(
        self, counts, base_counts, label, table_margin, zscore=None, column_index=None
    ):
        super(CatXMrVector, self).__init__(
            counts[0], base_counts[0], label, table_margin, zscore, column_index
        )
        self._all_bases = base_counts
        self._all_counts = counts

    @lazyproperty
    def pruned(self):
        return self.table_base == 0

    @lazyproperty
    def table_base(self):
        return np.sum(self._all_bases)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._all_counts)


class MeansVector(_BaseVector):
    def __init__(self, element, base_counts, means):
        super(MeansVector, self).__init__(element, base_counts)
        self._means = means

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def values(self):
        return self._means


class MeansWithMrVector(MeansVector):
    """MR vector with means for use in a matrix."""

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts[0])

    @lazyproperty
    def table_base(self):
        return self.base


class MultipleResponseVector(CategoricalVector):
    """Handles MR vectors (either rows or columns)

    Needs to handle selected and not-selected properly. Consequently, it calculates the
    right margin (for itself), but receives table margin on construction time (from the
    slice).
    """

    @lazyproperty
    def base(self):
        counts = zip(self._selected_unweighted, self._not_selected_unweighted)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def base_values(self):
        return self._base_counts[0, :]

    @lazyproperty
    def margin(self):
        counts = zip(self._selected, self._not_selected)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def pruned(self):
        return np.all(self.base == 0) or np.all(np.isnan(self.base))

    @lazyproperty
    def values(self):
        return self._selected

    @lazyproperty
    def _not_selected(self):
        return self._counts[1, :]

    @lazyproperty
    def _not_selected_unweighted(self):
        return self._base_counts[1, :]

    @lazyproperty
    def _selected(self):
        return self._counts[0, :]

    @lazyproperty
    def _selected_unweighted(self):
        return self._base_counts[0, :]


# ===STRIPE ROWS===


class StripeInsertionRow(object):
    """Represents an inserted (subtotal) row.

    This row item participates like any other row item, and a lot of its public
    properties are used by the strand object to form aggregate values across all rows.
    """

    def __init__(self, subtotal, base_rows, table_margin):
        self._subtotal = subtotal
        self._base_rows = base_rows
        self._table_margin = table_margin

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def base(self):
        return np.sum(np.array([row.base for row in self._addend_rows]))

    @lazyproperty
    def base_value(self):
        return np.sum(np.array([row.base_value for row in self._addend_rows]))

    @lazyproperty
    def hidden(self):
        """True if subtotal is pruned. Unconditionally False for stripe subtotal row."""
        return False

    @lazyproperty
    def is_insertion(self):
        return True

    @lazyproperty
    def label(self):
        return self._subtotal.label

    @lazyproperty
    def mean(self):
        return np.nan

    @lazyproperty
    def numeric(self):
        return np.nan

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def table_proportions(self):
        return self.value / self.table_margin

    @lazyproperty
    def value(self):
        return np.sum(np.array([row.value for row in self._addend_rows]), axis=0)

    @lazyproperty
    def _addend_rows(self):
        return tuple(
            row
            for i, row in enumerate(self._base_rows)
            if i in self._subtotal.addend_idxs
        )


class _BaseStripeRow(object):
    """Base class for all stripe-row objects.

    Provides attributes drawn from the row element.
    """

    def __init__(self, element):
        self._element = element

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._element.fill

    @lazyproperty
    def hidden(self):
        """True if vector is hidden.

        Vectors are hidden in two ways:

            1. Explicitly via transforms
            2. Implicitly when the base is 0 (also called pruning)

        This property checks whether a vector needs to be hidden, either implicitly or
        explicitly. It is used when iterating through rows or columns, to form the
        correct result.
        """
        return self._element.is_hidden or (self._element.prune and self.pruned)

    @lazyproperty
    def is_insertion(self):
        return False

    @lazyproperty
    def label(self):
        return self._element.label

    @lazyproperty
    def numeric(self):
        return self._element.numeric_value


class CatStripeRow(_BaseStripeRow):
    """Stripe-row for a categorical rows-dimension."""

    def __init__(self, element, count, base_count, table_margin):
        super(CatStripeRow, self).__init__(element)
        # ---count is a numpy numeric (np.int64 or np.float64)---
        self._count = count
        # ---base_count is np.int64---
        self._base_count = base_count
        # ---table_margin is np.int64 or np.float64---
        self._table_margin = table_margin

    @lazyproperty
    def base(self):
        """np.int64 unweighted count for this row."""
        return self._base_count

    @lazyproperty
    def base_value(self):
        return self._base_count

    @lazyproperty
    def margin(self):
        """float total weighted count for this row."""
        return self._count

    @lazyproperty
    def pruned(self):
        """True if this row is "empty"."""
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def table_margin(self):
        """int or float sum of weighted counts for this row's stripe."""
        return self._table_margin

    @lazyproperty
    def table_proportions(self):
        """float between 0.0 and 1.0 quantifying this row's share of stripe total."""
        return self.value / self._table_margin

    @lazyproperty
    def value(self):
        """Weighted count for this row."""
        # TODO: remove ndarray wrapper from this scalar and fix small exporter breakage
        return np.array([self._count])


class MeansStripeRow(_BaseStripeRow):
    """Stripe-row for a non-MR 1D cube-result with means."""

    def __init__(self, element, base_count, mean):
        super(MeansStripeRow, self).__init__(element)
        # ---base_count is np.int64---
        self._base_count = base_count
        # ---mean is np.float64 or np.nan---
        self._mean = mean

    @lazyproperty
    def base(self):
        return self._base_count

    @lazyproperty
    def mean(self):
        return self._mean

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def value(self):
        return self._mean


class MeansMrStripeRow(_BaseStripeRow):
    """Stripe-row for a univariate MR cube-response with means."""

    def __init__(self, element, base_counts, means):
        super(MeansMrStripeRow, self).__init__(element)
        # ---base_counts is an int array like [336 5501]---
        self._base_counts = base_counts
        # ---means is a float array like [2.578 1.639]---
        self._means = means

    @lazyproperty
    def base(self):
        """np.int64 unweighted count of selected."""
        return self._base_count

    @lazyproperty
    def mean(self):
        """float mean of selected."""
        return self._mean

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def table_base(self):
        return self.base

    @lazyproperty
    def _base_count(self):
        """np.int64 unweighted count of selected."""
        return self._base_counts[0]

    @lazyproperty
    def _mean(self):
        return self._means[0]


class MrStripeRow(_BaseStripeRow):
    """Stripe-row for use in MR stripe."""

    def __init__(self, element, counts, base_counts, table_margin):
        super(MrStripeRow, self).__init__(element)
        # ---counts is a float array like [42.0 63.5]---
        self._counts = counts
        # ---base_counts is an int array like [6076 1431]---
        self._base_counts = base_counts
        # ---table_margin is a numpy numeric scalar---
        self._table_margin = table_margin

    @lazyproperty
    def base(self):
        """np.int64 unweighted count of selected."""
        return self._base_count

    @lazyproperty
    def base_value(self):
        """np.int64 ... selected-only."""
        return self._base_count

    @lazyproperty
    def margin(self):
        return self._count

    @lazyproperty
    def pruned(self):
        return self.table_base == 0

    @lazyproperty
    def table_base(self):
        return np.sum(self._both_bases)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._both_counts)

    @lazyproperty
    def table_proportions(self):
        return self.value / self._table_margin

    @lazyproperty
    def value(self):
        # TODO: remove array wrapper and fix minor exporter breakage that results
        return np.array([self._count])

    @lazyproperty
    def _base_count(self):
        """np.int64 unweighted count of selected."""
        return self._base_counts[0]

    @lazyproperty
    def _both_bases(self):
        return self._base_counts

    @lazyproperty
    def _both_counts(self):
        return self._counts

    @lazyproperty
    def _count(self):
        return self._counts[0]
