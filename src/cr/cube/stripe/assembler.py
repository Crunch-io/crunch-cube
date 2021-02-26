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
        raise NotImplementedError

    @lazyproperty
    def rows_dimension_fills(self):
        """tuple of optional RGB str like "#def032" fill color for each strand row.

        Each value reflects the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions, ordering, and hidden rows. A fill value is `None`
        when no explicit fill color is defined for that row, indicating the default fill
        color for that row should be used, probably coming from a caller-defined theme.
        """
        raise NotImplementedError

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
        """1D np.int64 ndarray of (unweighted) table-proportion denominator per row."""
        return self._assemble_vector(self._measures.unweighted_bases.blocks)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted count for each row of stripe."""
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
        )

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
