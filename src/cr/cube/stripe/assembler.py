# encoding: utf-8

"""The `StripeAssembler` object provides the external interface for this module.

Its name derives from its role to "assemble" a finished 1D array ("stripe") for a
particular measure from the base measure values and inserted subtotals, to reorder the
rows according to the dimension *order* transforms, and to hide rows that are either
hidden by the user or "pruned" because they contain no observations.
"""

from __future__ import division

import numpy as np

from cr.cube.collator import ExplicitOrderCollator, PayloadOrderCollator
from cr.cube.enums import COLLATION_METHOD as CM
from cr.cube.stripe.measure import StripeMeasures
from cr.cube.util import lazyproperty


class StripeAssembler(object):
    """Provides measures, marginals, and totals for a (1D) strand cube-slice.

    An assembled stripe measure is a 1D ndarray reflecting all ordering, insertion, and
    hiding transforms applied to the dimension. An assembled margin is often a scalar.

    `cube` is the `cr.cube.Cube` object containing the data for this matrix.

    `rows_dimension` is the Dimension object describing the stripe.

    `slice_idx` is an int offset indicating which portion of `cube` data to use for this
    stripe.
    """

    def __init__(self, cube, rows_dimension, ca_as_0th, slice_idx):
        self._cube = cube
        self._rows_dimension = rows_dimension
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each inserted row in this stripe.

        Provided index values correspond rows after any insertion of subtotals,
        re-ordering, and hiding/pruning.
        """
        return tuple(i for i, row_idx in enumerate(self._row_order) if row_idx < 0)

    @lazyproperty
    def means(self):
        """1D np.float64 ndarray of mean for each row.

        Raises ValueError when the cube-result does not include a means cube-measure.
        """
        return self._assemble_vector(self._measures.means.blocks)

    @lazyproperty
    def row_count(self):
        """int count of rows in this stripe.

        This count includes inserted rows but not rows that have been hidden/pruned.
        """
        return len(self._row_order)

    @lazyproperty
    def row_labels(self):
        """1D str ndarray of row name for each matrix row.

        These are suitable for use as row headings; labels for subtotal rows appear in
        the sequence and labels are ordered to correspond with their respective data
        row.
        """
        return np.array(
            [e.label for e in self._rows_dimension.valid_elements]
            + [s.label for s in self._rows_dimension.subtotals]
        )[self._row_order]

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
        return tuple(
            (element_fills[idx] if idx > -1 else None) for idx in self._row_order
        )

    @lazyproperty
    def scale_mean(self):
        """Optional float mean of row numeric-values (scale).

        This value is `None` when no row-elements have numeric-value. The numeric value
        (aka. "scaled-count") for a row is its count multiplied by the numeric-value of
        its element. For example, if 100 women responded "Very Likely" and the
        numeric-value of the "Very Likely" response (element) was 4, then the
        scaled-count for that row would be 400. The scale mean is the average of those
        scale values over the total count of responses. The count of a row lacking a
        numeric value does not contribute to either the numerator or denominator of that
        computation.
        """
        return self._measures.scaled_counts.scale_mean

    @lazyproperty
    def scale_median(self):
        """Optional float/int median of scaled weighted-counts.

        This value is `None` when no rows have a numeric-value assigned. The median is
        equal to a count of 1 multiplied by one of the row numeric-values, so the value
        is equal to one of the assigned numeric-values (and retains its int/float type).
        """
        return self._measures.scaled_counts.scale_median

    @lazyproperty
    def scale_stddev(self):
        """Optional np.float64 standard-deviation of scaled weighted-counts.

        This value is `None` when no rows have a numeric-value assigned. The value has
        the same units as the assigned numeric values and indicates the dispersion of
        the scaled-count distribution from its mean (scale-mean).
        """
        return self._measures.scaled_counts.scale_stddev

    @lazyproperty
    def scale_stderr(self):
        """Optional np.float64 standard-error of scaled weighted counts.

        This value is `None` when no rows have a numeric-value assigned.
        """
        return self._measures.scaled_counts.scale_stderr

    @lazyproperty
    def share_sum(self):
        """1D np.float64 ndarray of share of sum for each row.

        Raises ValueError when the cube-result does not include a sum cube-measure.
        """
        return self._assemble_vector(self._measures.share_sum.blocks)

    @lazyproperty
    def stddev(self):
        """1D np.float64 ndarray of stddev for each row.

        Raises ValueError when the cube-result does not include a stddev cube-measure.
        """
        return self._assemble_vector(self._measures.stddev.blocks)

    @lazyproperty
    def sums(self):
        """1D np.float64 ndarray of sum for each row.

        Raises ValueError when the cube-result does not include a sum cube-measure.
        """
        return self._assemble_vector(self._measures.sums.blocks)

    @lazyproperty
    def table_base_range(self):
        """[min, max] np.float64 ndarray range of (total) unweighted-N for this stripe.

        A non-MR stripe will have a single base, represented by min and max being the
        same value. Each row of an MR stripe has a distinct base, which is reduced to a
        range in that case.
        """
        return self._measures.unweighted_bases.table_base_range

    @lazyproperty
    def table_margin_range(self):
        """[min, max] np.float64 ndarray range of (total) unweighted-N for this stripe.

        A non-MR stripe will have a single margin, represented by min and max being the
        same value. Each row of an MR stripe has a distinct margin, which is reduced to
        a min/max range in that case.
        """
        return self._measures.weighted_bases.table_margin_range

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
        """1D np.float64 ndarray of fraction of weighted-N contributed by each row."""
        return self._assemble_vector(self._measures.table_proportions.blocks)

    @lazyproperty
    def unweighted_bases(self):
        """1D np.float64 ndarray of (unweighted) table-proportion denominator per row."""
        return self._assemble_vector(self._measures.unweighted_bases.blocks)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.float64 ndarray of unweighted count for each row of stripe."""
        return self._assemble_vector(self._measures.unweighted_counts.blocks)

    @lazyproperty
    def weighted_bases(self):
        """1D np.float64 ndarray of weighted table-proportion denominator per row."""
        return self._assemble_vector(self._measures.weighted_bases.blocks)

    @lazyproperty
    def weighted_counts(self):
        """1D np.float64 ndarray of weighted count for each row of stripe."""
        return self._assemble_vector(self._measures.weighted_counts.blocks)

    def _assemble_vector(self, blocks):
        """Return 1D ndarray of base_vector with inserted subtotals, in order.

        `blocks` is a pair of two 1D arrays, first the base-values and then the subtotal
        values of the stripe vector. The returned array is sequenced in the computed
        row order including possibly removing hidden or pruned values.
        """
        return np.concatenate(blocks)[self._row_order]

    @lazyproperty
    def _measures(self):
        """StripeMeasures collection object for this stripe."""
        return StripeMeasures(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )

    @lazyproperty
    def _row_order(self):
        """1D np.int64 ndarray of signed int idx for each assembled row of stripe.

        Negative values represent inserted subtotal-row locations. Indices appear in the
        order rows are to appear in the final result.
        """
        # --- specify dtype explicitly to prevent error when display-order is empty. The
        # --- default dtype is float, which cannot be used to index an array.
        return np.array(
            _BaseOrderHelper.display_order(self._rows_dimension, self._measures),
            dtype=int,
        )


# === ORDER HELPERS ===


class _BaseOrderHelper(object):
    """Base class for ordering helpers."""

    def __init__(self, rows_dimension, measures):
        self._rows_dimension = rows_dimension
        self._measures = measures

    @classmethod
    def display_order(cls, rows_dimension, measures):
        """1D np.int64 ndarray of signed int idx for each row of stripe.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        HelperCls = (
            _SortByValueHelper
            if rows_dimension.collation_method == CM.OPPOSING_ELEMENT
            else _OrderHelper
        )
        return HelperCls(rows_dimension, measures)._display_order

    @lazyproperty
    def _display_order(self):
        """tuple of signed int idx for each row of stripe.

        Negative values represent inserted-vector locations. Returned sequence reflects
        insertion, hiding, pruning, and ordering transforms specified in the
        rows-dimension.
        """
        raise NotImplementedError(
            "`%s` must implement `._display_order`" % type(self).__name__
        )  # pragma: no cover

    @lazyproperty
    def _empty_row_idxs(self):
        """tuple of int index for each row with N = 0.

        These rows are subject to pruning, depending on a user setting in the dimension.
        """
        return tuple(i for i, N in enumerate(self._measures.pruning_base) if N == 0)


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
            if self._rows_dimension.collation_method == CM.EXPLICIT_ORDER
            else PayloadOrderCollator
        )
        return CollatorCls.display_order(self._rows_dimension, self._empty_row_idxs)


class _SortByValueHelper(_BaseOrderHelper):
    """Orders rows by their values."""
