# encoding: utf-8

"""A stripe is a 1D data partition, roughly a one-dimension version of matrix.

Each strand object is based on a stripe, which is where most of its values come from.
Unlike a matrix, a stripe does not represent a crosstab and does not have "body" values.
It does however have summary values of some of the same types, like counts and
proportions. It does *not* have pvals or zscores.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class TransformedStripe(object):
    """Stripe reflecting application of all transforms."""

    def __init__(self, rows_dimension, base_stripe):
        self._rows_dimension = rows_dimension
        self._base_stripe = base_stripe

    @classmethod
    def stripe(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return a TransformedStripe object constructed from this cube result."""
        base_stripe = _BaseBaseStripe.factory(
            cube, rows_dimension, ca_as_0th, slice_idx
        )
        return cls(rows_dimension, base_stripe)

    @lazyproperty
    def rows(self):
        """Sequence of post-transformation row vectors.

        All transforms are applied in this unit. `._ordered_rows` applies ordering,
        _StripeInsertionHelper creates and interleaves subtotal rows, and hidden rows
        are removed directly in this main row iterator.
        """
        return tuple(
            row
            for row in _StripeInsertionHelper.iter_interleaved_rows(
                self._rows_dimension, self._ordered_rows, self._table_margin
            )
            if not row.hidden
        )

    @lazyproperty
    def table_base_unpruned(self):
        """Hmm, weird 1D ndarray with same int value repeated for each row."""
        return self._base_stripe.table_base

    @lazyproperty
    def table_margin_unpruned(self):
        """Hmm, weird 1D ndarray with same float value repeated for each row."""
        return self._base_stripe.table_margin

    @lazyproperty
    def _ordered_rows(self):
        return tuple(np.array(self._base_stripe.rows)[self._row_order])

    @lazyproperty
    def _row_order(self):
        """Indexer value identifying rows in order, suitable for slicing an ndarray.

        This value is a 1D ndarray of int row indices, used for indexing the rows array
        to produce an ordered version.
        """
        # ---Specifying int type prevents failure when there are zero rows---
        return np.array(self._rows_dimension.display_order, dtype=int)

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
        """Generate rows with subtotals in correct position."""
        return cls(rows_dimension, ordered_rows, table_margin)._iter_interleaved_rows()

    def _iter_interleaved_rows(self):
        """Generate all row vectors with insertions interleaved at right spot."""
        # ---subtotals inserted at top---
        for row in self._all_inserted_rows:
            if row.anchor == "top":
                yield row

        # ---body rows with subtotals anchored to specific body positions---
        for idx, row in enumerate(self._ordered_rows):
            yield row
            for inserted_row in self._iter_inserted_rows_anchored_at(idx):
                yield inserted_row

        # ---subtotals appended at bottom---
        for row in self._all_inserted_rows:
            if row.anchor == "bottom":
                yield row

    @lazyproperty
    def _all_inserted_rows(self):
        """Sequence of _StripeInsertionRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # ---an aggregate rows-dimension is not summable---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            _StripeInsertionRow(subtotal, self._ordered_rows, self._table_margin)
            for subtotal in self._rows_dimension.subtotals
        )

    def _iter_inserted_rows_anchored_at(self, anchor):
        """Generate all inserted row vectors with matching `anchor`."""
        return (row for row in self._all_inserted_rows if row.anchor == anchor)


# ===BASE STRIPES===


class _BaseBaseStripe(object):
    """Base class for all stripe objects."""

    def __init__(self, rows_dimension, base_counts):
        self._rows_dimension = rows_dimension
        self._base_counts = base_counts

    @classmethod
    def factory(cls, cube, rows_dimension, ca_as_0th, slice_idx):
        """Return a base-slice object of appropriate type based on parameters."""
        counts = cube.counts
        base_counts = cube.base_counts

        # ---for cubes with means, create one of the means-stripe types---
        if cube.has_means:
            if rows_dimension.dimension_type == DT.MR:
                return _MeansWithMrStripe(rows_dimension, counts, base_counts)
            return _MeansStripe(rows_dimension, counts, base_counts)

        if ca_as_0th:
            return _CatStripe(rows_dimension, counts[slice_idx], base_counts[slice_idx])

        if rows_dimension.dimension_type == DT.MR:
            return _MrStripe(rows_dimension, counts, base_counts)

        return _CatStripe(rows_dimension, counts, base_counts)

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def _row_elements(self):
        return self._rows_dimension.valid_elements


class _CatStripe(_BaseBaseStripe):
    """Special case of CAT x CAT, where the 2nd CAT doesn't exist.

    Values are treated as rows, while there's only a single column (vector).
    """

    def __init__(self, rows_dimension, counts, base_counts):
        super(_CatStripe, self).__init__(rows_dimension, base_counts)
        self._counts = counts

    @lazyproperty
    def rows(self):
        table_margin = self.table_margin
        return tuple(
            _CatStripeRow(element, count, base_count, table_margin)
            for (element, count, base_count) in zip(
                self._row_elements, self._counts, self._base_counts
            )
        )

    @lazyproperty
    def table_margin(self):
        """Needed by inserted rows in later transformation step."""
        return np.sum(self._counts)


class _MeansStripe(_BaseBaseStripe):
    """A 1D calculator for a strand containing mean first-order measure."""

    def __init__(self, rows_dimension, means, base_counts):
        super(_MeansStripe, self).__init__(rows_dimension, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansStripeRow(element, base_count, mean)
            for element, base_count, mean in zip(
                self._rows_dimension.valid_elements, self._base_counts, self._means
            )
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._base_counts)


class _MeansWithMrStripe(_BaseBaseStripe):
    """Means behavior differs when dimension is MR."""

    def __init__(self, rows_dimension, means, base_counts):
        super(_MeansWithMrStripe, self).__init__(rows_dimension, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansMrStripeRow(element, base_counts, means)
            for element, base_counts, means in zip(
                self._rows_dimension.valid_elements, self._base_counts, self._means
            )
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._base_counts)


class _MrStripe(_BaseBaseStripe):
    """Special case of 1-D MR slice (stripe)."""

    def __init__(self, rows_dimension, counts, base_counts):
        super(_MrStripe, self).__init__(rows_dimension, base_counts)
        self._counts = counts

    @lazyproperty
    def rows(self):
        return tuple(
            _MrStripeRow(element, counts, base_counts, table_margin)
            for (element, counts, base_counts, table_margin) in zip(
                self._row_elements, self._counts, self._base_counts, self.table_margin
            )
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=1)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=1)


# ===STRIPE ROWS===


class _StripeInsertionRow(object):
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
        return sum(row.base for row in self._addend_rows)

    @lazyproperty
    def count(self):
        return sum(row.count for row in self._addend_rows)

    @lazyproperty
    def fill(self):
        """An insertion row can have no element-fill-color transform."""
        return None

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
        return self.value / self.table_margin

    @lazyproperty
    def value(self):
        return sum(row.value for row in self._addend_rows)

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


class _CatStripeRow(_BaseStripeRow):
    """Stripe-row for a categorical rows-dimension."""

    def __init__(self, element, count, base_count, table_margin):
        super(_CatStripeRow, self).__init__(element)
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
    def count(self):
        """Weighted count for this row."""
        return self._count

    @lazyproperty
    def pruned(self):
        """True if this row is "empty"."""
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def table_proportions(self):
        """float between 0.0 and 1.0 quantifying this row's share of stripe total."""
        return self.value / self._table_margin

    @lazyproperty
    def value(self):
        """Weighted count for this row."""
        return self._count


class _MeansStripeRow(_BaseStripeRow):
    """Stripe-row for a non-MR 1D cube-result with means."""

    def __init__(self, element, base_count, mean):
        super(_MeansStripeRow, self).__init__(element)
        # ---base_count is np.int64---
        self._base_count = base_count
        # ---mean is np.float64 or np.nan---
        self._mean = mean

    @lazyproperty
    def base(self):
        return self._base_count

    @lazyproperty
    def count(self):
        """It's a programming error if this is ever called."""
        raise NotImplementedError(
            "Mean measure has no unweighted count."
        )  # pragma: no cover

    @lazyproperty
    def mean(self):
        return self._mean

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)


class _MeansMrStripeRow(_BaseStripeRow):
    """Stripe-row for a univariate MR cube-response with means."""

    def __init__(self, element, base_counts, means):
        super(_MeansMrStripeRow, self).__init__(element)
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


class _MrStripeRow(_BaseStripeRow):
    """Stripe-row for use in MR stripe."""

    def __init__(self, element, counts, base_counts, table_margin):
        super(_MrStripeRow, self).__init__(element)
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
    def count(self):
        return self._count

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
        return self._count

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
