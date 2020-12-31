# encoding: utf-8

"""A stripe is a 1D data partition, roughly a one-dimension version of matrix.

Each strand object is based on a stripe, which is where most of its values come from.
Unlike a matrix, a stripe does not represent a crosstab and does not have "body" values.
It does however have summary values of some of the same types, like counts and
proportions. It does *not* have pvals or zscores.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class TransformedStripe(object):
    """Stripe that reflects application of all transforms.

    `rows_dimension` is a `cr.cube.dimension.Dimension` object representing the single
    dimension of this stripe. This is referred to as the "rows" dimension because when
    viewed as a "card" in the Crunch UI, the elements of this dimension each appear as
    a distinct row of the card.

    `base_stripe` is one of the low-level, pre-transformation stripe objects defined
    later in the module. The `.stripe()` classmethod uses a factory method to construct
    an instance of the right class depending on the dimension type.

    This object is not meant to be constructed directly, except perhaps in unit tests.
    Use the `.stripe()` classmethod to create a new `TransformedStripe` instance.
    """

    def __init__(self, rows_dimension, base_stripe):
        self._rows_dimension = rows_dimension
        self._base_stripe = base_stripe

    @classmethod
    def stripe(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return a TransformedStripe object constructed from the data in `cube`."""
        base_stripe = _BaseBaseStripe.factory(
            cube, rows_dimension, ca_as_0th, slice_idx
        )
        return cls(rows_dimension, base_stripe)

    @lazyproperty
    def rows(self):
        """Sequence of post-transformation row vectors."""
        return tuple(row for row in self.rows_including_hidden if not row.hidden)

    @lazyproperty
    def rows_including_hidden(self):
        """Sequence of row vectors including those hidden by the user."""
        # --- ordering and insertion transforms are applied here. `._ordered_rows`
        # --- applies any ordering transforms and _StripeInsertionHelper creates and
        # --- interleaves subtotal rows.
        return tuple(
            _StripeInsertionHelper.iter_interleaved_rows(
                self._rows_dimension, self._ordered_rows, self._table_margin
            )
        )

    @lazyproperty
    def table_base_unpruned(self):
        """np.int64 scalar or 1D np.int64 ndarray of table base including "empty" items.

        Value is an np.int64 scalar except when this stripe is multiple-response (MR).
        This value is a 1D ndarray of np.int64 for an MR stripe.
        """
        # TODO: This the name is misleading. It's not only "unpruned" it's
        # "before_hiding" (of either kind, prune or hide-transform). But the real
        # problem is having this interface property at all. The need for this is related
        # to expressing ranges for base and margin in cubes that have an MR dimension.
        # The real solution is to compute ranges in `cr.cube` rather than leaking this
        # sort of internal detail through the interface and making the client compute
        # those for themselves. So this will require reconstructing that "show-ranges"
        # requirement and either adding some sort of a `.range` property that returns
        # a sequence of (min, max) tuples, or maybe just returning margin or base as
        # tuples when appropriate and having something like a `.margin_is_ranges`
        # predicate the client can switch on to control their rendering.
        return self._base_stripe.table_base

    @lazyproperty
    def table_margin_unpruned(self):
        """np.float/int64 scalar or 1D np.float/int64 ndarray of table margin.

        This value includes rows that may not appear in `.rows` because they are hidden.

        This value is an np.int64 scalar when this stripe is CAT. The value is an array
        when this stripe is multiple-response (MR) because each subvar of an MR
        dimension has a distinct table margin.

        Values are np.int64 when the cube-result is unweighted.
        """
        # TODO: see TODO in `.table_base_unpruned`
        return self._base_stripe.table_margin

    @lazyproperty
    def _ordered_rows(self):
        """Sequence of base-rows in order specified by user.

        Default ordering is "payload order", the order dimension elements appear in the
        cube result. This order may be overridden by an order transform.
        """
        # --- display_order is like (2, 1, 0, 3); each int item is the offset of a row
        # --- in the base-rows collection.
        rows = self._base_stripe.rows
        return tuple(rows[row_idx] for row_idx in self._rows_dimension.display_order)

    @lazyproperty
    def _table_margin(self):
        """Needed by inserted rows."""
        return self._base_stripe.table_margin


class _StripeInsertionHelper(object):
    """Base class for ordering partitions."""

    def __init__(self, rows_dimension, ordered_rows, table_margin):
        self._rows_dimension = rows_dimension
        self._ordered_rows = ordered_rows
        self._table_margin = table_margin

    @classmethod
    def iter_interleaved_rows(cls, rows_dimension, ordered_rows, table_margin):
        """Generate rows with subtotals inserted in correct position."""
        return cls(rows_dimension, ordered_rows, table_margin)._iter_interleaved_rows()

    def _iter_interleaved_rows(self):
        """Generate all row vectors with inserted rows interleaved at right spot."""
        # --- organize inserted-rows by anchor ---
        inserted_rows_by_anchor = collections.defaultdict(list)
        for inserted_row in self._inserted_rows:
            inserted_rows_by_anchor[inserted_row.anchor].append(inserted_row)

        # --- subtotals inserted at top ---
        for inserted_row in inserted_rows_by_anchor["top"]:
            yield inserted_row

        # --- body rows with subtotals anchored to specific body positions ---
        for row in self._ordered_rows:
            yield row
            for inserted_row in inserted_rows_by_anchor[row.element_id]:
                yield inserted_row

        # --- subtotals appended at bottom ---
        for inserted_row in inserted_rows_by_anchor["bottom"]:
            yield inserted_row

    @lazyproperty
    def _inserted_rows(self):
        """Sequence of _StripeInsertedRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # ---an aggregate rows-dimension is not summable---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            _StripeInsertedRow(subtotal, self._ordered_rows, self._table_margin)
            for subtotal in self._rows_dimension.subtotals
        )


# ===BASE STRIPES===


class _BaseBaseStripe(object):
    """Base class for all stripe objects."""

    def __init__(self, rows_dimension, unweighted_counts):
        self._rows_dimension = rows_dimension
        self._unweighted_counts = unweighted_counts

    @classmethod
    def factory(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return a base-slice object of appropriate type based on parameters."""
        counts = cube.counts
        unweighted_counts = cube.unweighted_counts

        # ---for cubes with means, create one of the means-stripe types---
        if cube.has_means:
            if rows_dimension.dimension_type == DT.MR:
                return _MeansWithMrStripe(rows_dimension, counts, unweighted_counts)
            return _MeansStripe(rows_dimension, counts, unweighted_counts)

        if ca_as_0th:
            return _CatStripe(
                rows_dimension, counts[slice_idx], unweighted_counts[slice_idx]
            )

        if rows_dimension.dimension_type == DT.MR:
            return _MrStripe(rows_dimension, counts, unweighted_counts)

        return _CatStripe(rows_dimension, counts, unweighted_counts)

    @lazyproperty
    def rows(self):
        """Sequence of rows in this stripe.

        The sequence includes a row for each valid element in the rows-dimension, in the
        order those elements are defined in the dimension (which is also the order in
        which that dimension's values appear in the cube result).
        """
        raise NotImplementedError(
            "must be implemented by each subclass"
        )  # pragma: no cover

    @lazyproperty
    def table_base(self):
        """np.int64 count of actual respondents asked this question."""
        return np.sum(self._unweighted_counts)

    @lazyproperty
    def _row_elements(self):
        """cr.cube.dimension._ValidElements instance for rows-dimension."""
        return self._rows_dimension.valid_elements


class _CatStripe(_BaseBaseStripe):
    """Special case of CAT x CAT, where the 2nd CAT doesn't exist.

    Values are treated as rows, while there's only a single column (vector).
    """

    def __init__(self, rows_dimension, counts, unweighted_counts):
        super(_CatStripe, self).__init__(rows_dimension, unweighted_counts)
        self._counts = counts

    @lazyproperty
    def rows(self):
        """tuple of _CatStripeRow for each valid element in rows-dimension."""
        table_margin = self.table_margin
        return tuple(
            _CatStripeRow(element, count, unweighted_count, table_margin)
            for (element, count, unweighted_count) in zip(
                self._row_elements, self._counts, self._unweighted_counts
            )
        )

    @lazyproperty
    def table_margin(self):
        """np.int64 total weighted count of responses to this question.

        Needed by inserted rows in later transformation step.
        """
        return np.sum(self._counts)


class _MeansStripe(_BaseBaseStripe):
    """A 1D calculator for a strand containing mean first-order measure."""

    def __init__(self, rows_dimension, means, unweighted_counts):
        super(_MeansStripe, self).__init__(rows_dimension, unweighted_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        """Sequence of _MeansStripeRow for each valid element in rows-dimension."""
        return tuple(
            _MeansStripeRow(element, unweighted_count, mean)
            for element, unweighted_count, mean in zip(
                self._row_elements, self._unweighted_counts, self._means
            )
        )

    @lazyproperty
    def table_margin(self):
        # TODO: explain in docstring why this is unweighted for means instead of _counts
        return np.sum(self._unweighted_counts)


class _MeansWithMrStripe(_BaseBaseStripe):
    """Means behavior differs when dimension is MR."""

    def __init__(self, rows_dimension, means, unweighted_counts):
        super(_MeansWithMrStripe, self).__init__(rows_dimension, unweighted_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        """Sequence of _MeansMrStripeRow for each valid element in rows-dimension."""
        return tuple(
            _MeansMrStripeRow(element, unweighted_counts, means)
            for element, unweighted_counts, means in zip(
                self._row_elements, self._unweighted_counts, self._means
            )
        )

    @lazyproperty
    def table_margin(self):
        # TODO: explain in docstring why this is unweighted for means instead of _counts
        return np.sum(self._unweighted_counts)


class _MrStripe(_BaseBaseStripe):
    """Special case of 1D MR slice (stripe)."""

    def __init__(self, rows_dimension, counts, unweighted_counts):
        super(_MrStripe, self).__init__(rows_dimension, unweighted_counts)
        self._counts = counts

    @lazyproperty
    def rows(self):
        """Sequence of _MrStripeRow for each valid element in rows-dimension."""
        return tuple(
            _MrStripeRow(element, counts, unweighted_counts, table_margin)
            for (element, counts, unweighted_counts, table_margin) in zip(
                self._row_elements,
                self._counts,
                self._unweighted_counts,
                self.table_margin,
            )
        )

    @lazyproperty
    def table_base(self):
        """1D ndarray of unweighted selected + unselected count for each row of stripe.

        Note the base (how many respondents were asked the question) can vary from row
        to row because not all choices are necessarily shown to all respondents.
        """
        return np.sum(self._unweighted_counts, axis=1)

    @lazyproperty
    def table_margin(self):
        """1D ndarray of weighted selected + unselected count for each row of stripe.

        Note this can vary from row to row because not all choices are necessarily shown
        to all respondents.
        """
        return np.sum(self._counts, axis=1)


# ===STRIPE ROWS===


class _StripeInsertedRow(object):
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
        """str or int anchor value for this inserted.

        Value can be "top", "bottom" or an int element-id indicating the position of
        this inserted-row with respect to the base-rows.
        """
        return self._subtotal.anchor

    @lazyproperty
    def base(self):
        """np.int64 sum of (unweighted) sample population of each addend row."""
        return sum(row.base for row in self._addend_rows)

    @lazyproperty
    def count(self):
        """np.float64 subtotal of weighted count of this answer across addend rows."""
        return sum(row.count for row in self._addend_rows)

    @lazyproperty
    def fill(self):
        """An inserted row can have no element-fill-color transform."""
        return None

    @lazyproperty
    def hidden(self):
        """True if subtotal is pruned. Unconditionally False for stripe subtotal row."""
        return False

    @lazyproperty
    def is_inserted(self):
        """True when this row is an inserted row.

        Unconditionally True for _StripeInsertedRow.
        """
        return True

    @lazyproperty
    def label(self):
        """str heading for this row."""
        return self._subtotal.label

    @lazyproperty
    def mean(self):
        """Means cannot be subtotaled (so far at least)."""
        return np.nan

    @lazyproperty
    def numeric_value(self):
        """Numeric value (or np.nan) assigned to the element of this row.

        This is unconditionally np.nan for a subtotal row.
        """
        return np.nan

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def table_proportions(self):
        """float 0.0->1.0 portion of weighted respondents who chose this response."""
        return self.count / self.table_margin

    @lazyproperty
    def unweighted_count(self):
        """int64 sum of unweighted-count of each addend row."""
        return sum(row.unweighted_count for row in self._addend_rows)

    @lazyproperty
    def _addend_rows(self):
        """Sequence of _BaseStripeRow subclass that contribute to this subtotal."""
        return tuple(
            row
            for row in self._base_rows
            if row.element_id in self._subtotal.addend_ids
        )


class _BaseStripeRow(object):
    """Base class for all stripe-row objects.

    Provides attributes drawn from the row element.
    """

    def __init__(self, element):
        self._element = element

    @lazyproperty
    def element_id(self):
        """int identifier of category or subvariable this row represents."""
        return self._element.element_id

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
    def is_inserted(self):
        return False

    @lazyproperty
    def label(self):
        return self._element.label

    @lazyproperty
    def mean(self):
        """Default mean value is `np.nan`, consistent with an inserted row.

        It would be very unusual for this property to be accessed on a cube with no mean
        values, but this prevents raising an AttributeError and provides a reasonable
        value.
        """
        return np.nan

    @lazyproperty
    def numeric_value(self):
        """Numeric value (or np.nan) assigned to the element of this row.

        This allows a categorical value to participate in certain quantitative
        representations. For example, very-x to not-at-all-x could be assigned the
        values from 5 to 1, allowing a mean numeric sentiment like "3.5 on scale of 5"
        to be calculated.

        This value is np.nan when no numeric value has been assigned or numeric value is
        not applicable (such as for a numeric variable).
        """
        return self._element.numeric_value

    @lazyproperty
    def unweighted_count(self):
        """np.int64 unweighted-count for this row."""
        raise NotImplementedError(
            "`.unweighted_count` must be implemented by each subclass"
        )  # pragma: no cover


class _CatStripeRow(_BaseStripeRow):
    """Stripe-row for a categorical rows-dimension."""

    def __init__(self, element, count, unweighted_count, table_margin):
        super(_CatStripeRow, self).__init__(element)
        # ---count is a numpy numeric (np.int64 or np.float64)---
        self._count = count
        # ---unweighted_count is np.int64---
        self._unweighted_count = unweighted_count
        # ---table_margin is np.int64 or np.float64---
        self._table_margin = table_margin

    @lazyproperty
    def base(self):
        """np.int64 count of actual respondents who were offered this response."""
        # TODO: this doesn't seem right. The unweighted-count is the number of actual
        # respondents who *selected* this response, not all those who were asked the
        # question (as it should be).
        return self._unweighted_count

    @lazyproperty
    def count(self):
        """Weighted count for this row."""
        return self._count

    @lazyproperty
    def pruned(self):
        """True if this question was answered by exactly zero respondents."""
        return self._unweighted_count == 0 or np.isnan(self._unweighted_count)

    @lazyproperty
    def table_proportions(self):
        """float between 0.0 and 1.0 quantifying this row's share of stripe total.

        This is the proportion of *weighted* respondents who selected this response to
        the *weighted total* of respondents asked this question.
        """
        return self.count / self._table_margin

    @lazyproperty
    def unweighted_count(self):
        """np.int64 unweighted count for this row."""
        return self._unweighted_count


class _MeansStripeRow(_BaseStripeRow):
    """Stripe-row for a non-MR 1D cube-result with means."""

    def __init__(self, element, unweighted_count, mean):
        super(_MeansStripeRow, self).__init__(element)
        # ---unweighted_count is np.int64---
        self._unweighted_count = unweighted_count
        # ---mean is np.float64 or np.nan---
        self._mean = mean

    @lazyproperty
    def base(self):
        """np.int64 base, just unweighted count for a non-MR row."""
        return self._unweighted_count

    @lazyproperty
    def count(self):
        """It's a programming error if this is ever called."""
        raise NotImplementedError(
            "Mean measure has no unweighted count."
        )  # pragma: no cover

    @lazyproperty
    def mean(self):
        """Stripe of means."""
        return self._mean

    @lazyproperty
    def pruned(self):
        return self._unweighted_count == 0 or np.isnan(self._unweighted_count)

    @lazyproperty
    def unweighted_count(self):
        """np.int64 unweighted-count for this row."""
        return self._unweighted_count


class _MeansMrStripeRow(_BaseStripeRow):
    """Stripe-row for a univariate MR cube-response with means."""

    def __init__(self, element, unweighted_counts, means):
        super(_MeansMrStripeRow, self).__init__(element)
        # ---unweighted_counts is a [selected, not] int array like [336 5501]---
        self._unweighted_counts = unweighted_counts
        # ---means is a [selected, not] float array like [2.578 1.639]---
        self._means = means

    @lazyproperty
    def base(self):
        """np.int64 unweighted count of selected."""
        return self._unweighted_counts[0]

    @lazyproperty
    def mean(self):
        """float mean of selected."""
        return self._means[0]

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def table_base(self):
        return self.base


class _MrStripeRow(_BaseStripeRow):
    """Stripe-row for use in MR stripe."""

    def __init__(self, element, counts, unweighted_counts, table_margin):
        super(_MrStripeRow, self).__init__(element)
        # --- counts is a float [selected, not-selected] array like [42.0 63.5] ---
        self._counts = counts
        # --- unweighted_counts is [selected, not-selected] int array like [607 143]---
        self._unweighted_counts = unweighted_counts
        # --- table_margin is a numpy numeric scalar ---
        self._table_margin = table_margin

    @lazyproperty
    def count(self):
        """np.float/int64 number of weighted respondents who selected this response."""
        return self._counts[0]

    @lazyproperty
    def pruned(self):
        """True if this question was asked of exactly zero actual respondents."""
        return self.table_base == 0

    @lazyproperty
    def table_base(self):
        """np.int64 sample size for MR question, how many actual folks were asked."""
        return np.sum(self._both_unweighted_counts)

    @lazyproperty
    def table_margin(self):
        """np.float64 weighted sample size, how many "weighted" folks were asked."""
        return np.sum(self._both_counts)

    @lazyproperty
    def table_proportions(self):
        """np.float64 proportion of all weighted respondents selecteing this answer."""
        return self.count / self._table_margin

    @lazyproperty
    def unweighted_count(self):
        """np.int64 unweighted count of selected responses."""
        return self._unweighted_counts[0]

    @lazyproperty
    def _both_counts(self):
        """np.float64 count of weighted respondents offered this possible response."""
        return self._counts

    @lazyproperty
    def _both_unweighted_counts(self):
        """np.int64 count of actual respondents offered this possible response."""
        return self._unweighted_counts
