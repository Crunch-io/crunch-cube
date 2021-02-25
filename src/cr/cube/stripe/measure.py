# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

import numpy as np

from cr.cube.stripe.cubemeasure import CubeMeasures
from cr.cube.stripe.insertion import SumSubtotals
from cr.cube.util import lazyproperty


# === MEASURE COLLECTION ===


class StripeMeasures(object):
    """Intended to be a singleton for a given cube-result.

    It will give the same values if duplicated, just sacrificing some time and memory
    performance. Provides access to the variety of possible second-order measure objects
    for its cube-result. All construction and computation are lazy so only actually
    requested measures consume resources.
    """

    def __init__(self, cube, rows_dimension, ca_as_0th, slice_idx):
        self._cube = cube
        self._rows_dimension = rows_dimension
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx

    @lazyproperty
    def pruning_base(self):
        """1D np.int64 ndarray of unweighted-N for each stripe row."""
        return self._cube_measures.unweighted_cube_counts.pruning_base

    @lazyproperty
    def table_proportions(self):
        """_TableProportions measure object for this stripe."""
        return _TableProportions(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def unweighted_bases(self):
        """_UnweightedBases measure object for this stripe."""
        return _UnweightedBases(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this stripe."""
        return _UnweightedCounts(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def weighted_bases(self):
        """_WeightedBases measure object for this stripe."""
        return _WeightedBases(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def weighted_counts(self):
        """_WeightedCounts measure object for this stripe."""
        return _WeightedCounts(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def _cube_measures(self):
        """CubeMeasures collection object for this cube-result.

        This collection provides access to all cube-measure objects for the cube-result.
        The collection is provided to each measure object so it can access the cube
        measures it is based on.
        """
        return CubeMeasures(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )


# === INDIVIDUAL MEASURES ===


class _BaseSecondOrderMeasure(object):
    """Base class for all second-order measure objects."""

    def __init__(self, rows_dimension, measures, cube_measures):
        self._rows_dimension = rows_dimension
        self._measures = measures
        self._cube_measures = cube_measures

    @lazyproperty
    def base_values(self):
        """1D ndarray of measure base-value for each row.

        The base values are those that correspond 1-to-1 with the cube-result values.
        The values appear in payload order.
        """
        raise NotImplementedError(
            "`%s` must implement `.base_values`" % type(self).__name__
        )

    @lazyproperty
    def blocks(self):
        """(base_values, subtotal_values) pair comprising the "blocks" of this measure.

        Use of this default implementation assumes implementation of a `._base_values`
        and `._subtotal_values` property in the subclass. A measure which is computed
        differently can override this `.blocks` property instead of implementing those
        two components.
        """
        return (self.base_values, self.subtotal_values)

    @lazyproperty
    def subtotal_values(self):
        """1D ndarray of subtotal value for each row-subtotal."""
        raise NotImplementedError(
            "`%s` must implement `.subtotal_values`" % type(self).__name__
        )

    @lazyproperty
    def _unweighted_cube_counts(self):
        """_BaseUnweightedCubeCounts subclass instance for this measure.

        Provides cube measures associated with unweighted counts, including
        unweighted-counts and bases.
        """
        return self._cube_measures.unweighted_cube_counts

    @lazyproperty
    def _weighted_cube_counts(self):
        """_BaseWeightedCubeCounts subclass instance for this measure.

        Provides cube measures associated with weighted counts, including
        weighted-counts and table-margin.
        """
        return self._cube_measures.weighted_cube_counts


class _TableProportions(_BaseSecondOrderMeasure):
    """Provides the table-proportions measure for a stripe.

    Table-proportions is a 1D np.float64 ndarray of the proportion each row's weighted
    count contributes to the weighted-N of the table.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of table-proportion for each row of stripe."""
        weighted_counts = self._measures.weighted_counts.base_values
        table_margin = self._weighted_cube_counts.table_margin

        # --- note that table-margin can be either scalar or 1D ndarray. When it is an
        # --- array (stripe is MR), its shape is the same as the weighted_counts, so the
        # --- division works either way.

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return weighted_counts / table_margin


class _UnweightedBases(_BaseSecondOrderMeasure):
    """Provides the unweighted-bases measure for a stripe.

    unweighted-bases is a 1D np.int64 ndarray of the unweighted table-proportion
    denominator (base) for each row. This object also provides the table-base totals.
    """

    @lazyproperty
    def base_values(self):
        """1D np.int64 ndarray of unweighted table-proportion denominator per cell."""
        return self._unweighted_cube_counts.bases

    @lazyproperty
    def subtotal_values(self):
        """1D np.int64 ndarray of subtotal value for each row-subtotal."""
        # --- Background:
        # --- 1. The base is the same for all rows of a CAT stripe.
        # --- 2. An MR stripe can have no subtotals.
        # --- The strategy here is to broadcast the table-base to the size of the
        # --- subtotals array for CAT, and return an empty array for MR.

        # --- This initial subtotal-values array has the wrong values (unless it's
        # --- empty), but has the right shape and type.
        subtotal_values = SumSubtotals.subtotal_values(
            self.base_values, self._rows_dimension
        )
        # --- in the "no-subtotals" case, return that value, since it is both the right
        # --- value and the right dtype. Note this takes care of the MR stripe case.
        if subtotal_values.shape == (0,):
            return subtotal_values

        return np.broadcast_to(
            self._unweighted_cube_counts.table_base, subtotal_values.shape
        )


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.int64 ndarray of unweighted-count for each stripe base-row."""
        return self._unweighted_cube_counts.unweighted_counts

    @lazyproperty
    def subtotal_values(self):
        """1D np.int64 ndarray of sum for each row-subtotal."""
        # --- counts don't sum on an MR dimension, but an MR stripe can have no
        # --- subtotals. This just returns an empty array in that case and we don't need
        # --- to special-case MR.
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _WeightedBases(_BaseSecondOrderMeasure):
    """Provides the weighted-bases measure for a stripe.

    weighted-bases is a 1D np.float/int64 ndarray of the (weighted) table-proportion
    denominator (base) for each row.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of weighted table-proportion denominator per cell."""
        return self._weighted_cube_counts.bases

    @lazyproperty
    def subtotal_values(self):
        """1D np.float/int64 ndarray of sum for each row-subtotal."""
        # --- Background:
        # --- 1. weighted-base is the same for all rows, including subtotal rows.
        # --- 2. Only a CAT stripe can have subtotals; an MR stripe can't.
        # --- The strategy here is to broadcast the table-margin to the size of the
        # --- subtotals array for CAT, and return an empty array for MR.

        # --- This initial subtotal-values array has the wrong values (unless it's
        # --- empty), but has the right shape and type.
        subtotal_values = SumSubtotals.subtotal_values(
            self.base_values, self._rows_dimension
        )
        # --- in the "no-subtotals" case, return that value, since it is both the right
        # --- value and the right dtype. Note this takes care of the MR stripe case.
        if subtotal_values.shape == (0,):
            return subtotal_values

        return np.broadcast_to(
            self._weighted_cube_counts.table_margin, subtotal_values.shape
        )


class _WeightedCounts(_BaseSecondOrderMeasure):
    """Provides the weighted-counts measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float/int64 ndarray of weighted-count for each row."""
        return self._weighted_cube_counts.weighted_counts

    @lazyproperty
    def subtotal_values(self):
        """1D np.float/int64 ndarray of sum for each row-subtotal."""
        # --- counts don't sum on an MR dimension, but an MR stripe can have no
        # --- subtotals. This just returns an empty array in that case and we don't need
        # --- to special-case MR.
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)
