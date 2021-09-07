# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.smoothing import Smoother
from cr.cube.stripe.cubemeasure import CubeMeasures
from cr.cube.stripe.insertion import NanSubtotals, SumSubtotals
from cr.cube.util import lazyproperty


# === MEASURE COLLECTION ===


class StripeMeasures:
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
    def means(self):
        """_Means measure object for this stripe."""
        return _Means(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def population_proportions(self):
        """_PopulationPrortion measure object for this stripe."""
        return _PopulationProportions(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def population_proportion_stderrs(self):
        """_PopulationProportionStderrs measure object for this stripe."""
        return _PopulationProportionStderrs(
            self._rows_dimension, self, self._cube_measures
        )

    @lazyproperty
    def pruning_base(self):
        """1D np.float64 ndarray of unweighted-N for each stripe row."""
        return self._cube_measures.unweighted_cube_counts.pruning_base

    @lazyproperty
    def scaled_counts(self):
        """_ScaledCounts measure object for this stripe.

        Provides access to table-totals related to numeric-values/scaled-counts.
        """
        return _ScaledCounts(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def share_sum(self):
        """_ShareSum measure object for this stripe."""
        return _ShareSum(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def smoothed_means(self):
        """_MeansSmoothed measure object for this stripe.

        If smoothing is defined in the dimension transform a _MeansSmoothed object will
        be returned, otherwise fallback to unsmoothed measure object.
        """
        return _MeansSmoothed(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def stddev(self):
        """_StdDev measure object for this stripe."""
        return _StdDev(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def sums(self):
        """_Sums measure object for this stripe."""
        return _Sums(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def table_proportion_stddevs(self):
        """_TableProportionStddevs measure object for this stripe."""
        return _TableProportionStddevs(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def table_proportion_stderrs(self):
        """_TableProportionStderrs measure object for this stripe."""
        return _TableProportionStderrs(self._rows_dimension, self, self._cube_measures)

    @lazyproperty
    def table_proportion_variances(self):
        """_TableProportionVariances measure object for this stripe."""
        return _TableProportionVariances(
            self._rows_dimension, self, self._cube_measures
        )

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


class _BaseSecondOrderMeasure:
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
            f"`{type(self).__name__}` must implement `.base_values`"
        )  # pragma: no cover

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
            f"`{type(self).__name__}` must implement `.subtotal_values`"
        )  # pragma: no cover

    @lazyproperty
    def _unweighted_cube_counts(self):
        """_BaseCubeCounts for unweighted counts subclass instance for this measure.

        Provides cube measures associated with unweighted counts, including
        unweighted-counts and bases.
        """
        return self._cube_measures.unweighted_cube_counts

    @lazyproperty
    def _weighted_cube_counts(self):
        """_BaseCubeCounts for weighted counts subclass instance for this measure.

        Provides cube measures associated with weighted counts, including
        weighted-counts and table-margin.
        """
        return self._cube_measures.weighted_cube_counts


class _SmoothedMeasure(_BaseSecondOrderMeasure):
    """Mixin providing `._smoother` property for smoothed measures."""

    @lazyproperty
    def _smoother(self):
        """BaseSmoother subtype object providing smoothing as specified in spec."""
        return Smoother.factory(self._rows_dimension)


class _Means(_BaseSecondOrderMeasure):
    """Provides the means measure for a stripe.

    Relies on the presence of a means cube-measure in the cube-result.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of mean for each row."""
        return self._cube_measures.cube_means.means

    @lazyproperty
    def subtotal_values(self):
        """1D ndarray of np.nan for each row-subtotal.

        Mean values cannot be subtotaled and each subtotal value is unconditionally
        np.nan.
        """
        return NanSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _MeansSmoothed(_Means, _SmoothedMeasure):
    """Provides the smoothed means measure for a stripe.

    Relies on the presence of a means cube-measure in the cube-result.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of smoothed mean for each row."""
        return self._smoother.smooth(self._cube_measures.cube_means.means)


class _PopulationProportions(_BaseSecondOrderMeasure):
    """Provides the population-proportions measure for a stripe.

    When the variable is a categorical-date, we do not divide the population between the
    categories, so the proportion is always 1. Otherwise it is equal to the table
    proportions.

    This is used to calculate the population_counts.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of population proportion for each row."""
        proportions = self._measures.table_proportions.base_values
        return (
            np.repeat(1, proportions.shape)
            if self._rows_dimension.dimension_type == DT.CAT_DATE
            else proportions
        )

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of population proportion for each row-subtotal."""
        proportions = self._measures.table_proportions.subtotal_values
        return (
            np.repeat(1, proportions.shape)
            if self._rows_dimension.dimension_type == DT.CAT_DATE
            else proportions
        )


class _PopulationProportionStderrs(_BaseSecondOrderMeasure):
    """Provides the population-proportion-standard-errors measure for a stripe.

    When the variable is a categorical-date, we do not divide the population between the
    categories, so the standard-error is always 0. Otherwise it is equal to the table
    proportions.

    This is used to calculate the population_counts_moe.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of population proportion for each row."""
        stderrs = self._measures.table_proportion_stderrs.base_values
        return (
            np.repeat(0, stderrs.shape)
            if self._rows_dimension.dimension_type == DT.CAT_DATE
            else stderrs
        )

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of population proportion for each row-subtotal."""
        stderrs = self._measures.table_proportion_stderrs.subtotal_values
        return (
            np.repeat(0, stderrs.shape)
            if self._rows_dimension.dimension_type == DT.CAT_DATE
            else stderrs
        )


class _ScaledCounts(_BaseSecondOrderMeasure):
    """Provides access to table-totals related to numeric-values/scaled-counts."""

    @lazyproperty
    def scale_mean(self):
        """Optional float mean of scaled row-counts.

        This value is `None` when no row-elements have numeric-value. The numeric value
        (aka. "scaled-count") for a row is its count multiplied by the numeric-value of
        its element. For example, if 100 women responded "Very Likely" and the
        numeric-value of the "Very Likely" response (element) was 4, then the
        scaled-count for that row would be 400. The scale mean is the average of those
        scale values over the total count of responses. The count of a row lacking a
        numeric value does not contribute to either the numerator or denominator of that
        computation.
        """
        # --- value is None when no row-element has been assigned a numeric value ---
        if self._numeric_values.size == 0:
            return None

        # --- Also when the total count is zero, unlikely but possible and would lead to
        # --- a divide-by-zero error in this case.
        if self._total_weighted_count == 0:
            return None

        return self._total_scaled_count / self._total_weighted_count

    @lazyproperty
    def scale_median(self):
        """Optional np.float64 median of scaled weighted-counts.

        This value is `None` when no rows have a numeric-value assigned. Responses
        without a numeric value are not considered.
        """
        # --- value is None when no row-element has been assigned a numeric value ---
        if self._numeric_values.size == 0:
            return None

        # --- convert float weighted-counts to int. I'm not sure why this needs to
        # --- account for NaN values, if you know, explain it here.
        weighted_counts = np.nan_to_num(self._weighted_counts).astype("int64")

        # --- create an array with a numeric value for each individual count (response)
        # --- that can be fed directly into np.median()
        expanded_numeric_values = np.repeat(self._numeric_values, weighted_counts)

        return np.median(expanded_numeric_values)

    @lazyproperty
    def scale_stddev(self):
        """Optional np.float64 standard-deviation of scaled weighted-counts.

        This value is `None` when no row-elements have a numeric-value. The value is in
        units of scaled-counts and indicates the dispersion of the scaled-count
        distribution from its mean (scale-mean).
        """
        # --- value is None when no row-element has been assigned a numeric value
        # --- or when scale variance is not available
        if self._numeric_values.size == 0 or self._scale_variance is None:
            return None

        return np.sqrt(self._scale_variance)

    @lazyproperty
    def scale_stderr(self):
        """Optional np.float64 standard-error of scaled weighted counts.

        This value is `None` when no rows have a numeric-value assigned.
        """
        # --- value is None when no row-element has been assigned a numeric value
        # --- or when scale variance is not available
        if self._numeric_values.size == 0 or self._scale_variance is None:
            return None

        return np.sqrt(self._scale_variance / self._total_weighted_count)

    @lazyproperty
    def _has_numeric_value(self):
        """1D bool ndarray (mask) of True for each row with a defined numeric-value."""
        return ~np.isnan(np.array(self._rows_dimension.numeric_values))

    @lazyproperty
    def _numeric_values(self):
        """1D ndarray of numeric-value for each element in rows dimension that has one.

        The items in the array can be float or int. Row elements that have not been
        assigned a numeric value are skipped. Otherwise, the values appear in payload
        order.
        """
        return np.array(self._rows_dimension.numeric_values)[self._has_numeric_value]

    @lazyproperty
    def _scale_variance(self):
        """Optional np.float64 variance of scaled weighted-counts."""
        if self.scale_mean is None:
            # --- Scale mean can be None when _total_weighted_count is 0. Therefore the
            # --- scale variance has to be None to indicate that this measure is not
            # --- available.
            return None
        return (
            np.sum(
                self._weighted_counts * pow((self._numeric_values - self.scale_mean), 2)
            )
            / self._total_weighted_count
        )

    @lazyproperty
    def _total_scaled_count(self):
        """float/int total of scaled-count for rows with a numeric-value.

        The scaled-count for a row is its weighted-count multiplied by its numeric
        value (scaling-factor).
        """
        return np.sum(self._weighted_counts * self._numeric_values)

    @lazyproperty
    def _total_weighted_count(self):
        """float/int total of weighted counts for rows with a numeric-value."""
        return np.sum(self._weighted_counts)

    @lazyproperty
    def _weighted_counts(self):
        """1D float/int ndarray of weighted_count for elements with a numeric-value.

        Counts for rows that have not been assigned a numeric value are skipped.
        Otherwise, the values appear in payload order.
        """
        return self._weighted_cube_counts.counts[self._has_numeric_value]


class _ShareSum(_BaseSecondOrderMeasure):
    """Provides the share of sum measure for a stripe.

    Relies on the presence of a sum cube-measure in the cube-result.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of share of sum for each row."""
        sums = self._cube_measures.cube_sum.sums
        return sums / np.sum(sums)

    @lazyproperty
    def subtotal_values(self):
        """1D ndarray of share of sum subtotals for each row-subtotal."""
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _StdDev(_BaseSecondOrderMeasure):
    """Provides the stddev measure for a stripe.

    Relies on the presence of a stddev cube-measure in the cube-result.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of stddev for each row."""
        return self._cube_measures.cube_stddev.stddev

    @lazyproperty
    def subtotal_values(self):
        """1D ndarray of np.nan for each row-subtotal.

        StdDev values cannot be subtotaled and each subtotal value is unconditionally
        np.nan.
        """
        return NanSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _Sums(_BaseSecondOrderMeasure):
    """Provides the sum measure for a stripe.

    Relies on the presence of a sum cube-measure in the cube-result.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of sum for each row."""
        return self._cube_measures.cube_sum.sums

    @lazyproperty
    def subtotal_values(self):
        """1D ndarray of sum subtotals for each row-subtotal."""
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _TableProportionStddevs(_BaseSecondOrderMeasure):
    """Provides the table-proportion standard-deviation measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of table-prop stddev for each base row of stripe."""
        return np.sqrt(self._measures.table_proportion_variances.base_values)

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of table-prop stddev for each row subtotal."""
        return np.sqrt(self._measures.table_proportion_variances.subtotal_values)


class _TableProportionStderrs(_BaseSecondOrderMeasure):
    """Provides the table-proportion standard-error measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of table-prop stderr for each base row of stripe."""
        variances = self._measures.table_proportion_variances.base_values
        weighted_bases = self._measures.weighted_bases.base_values
        return np.sqrt(variances / weighted_bases)

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of table-prop stderr for each row subtotal."""
        variances = self._measures.table_proportion_variances.subtotal_values
        weighted_bases = self._measures.weighted_bases.subtotal_values
        return np.sqrt(variances / weighted_bases)


class _TableProportionVariances(_BaseSecondOrderMeasure):
    """Provides the table-proportion-variances measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of table-prop variance for each base row of stripe."""
        p = self._measures.table_proportions.base_values
        return p * (1 - p)

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of table-prop variance for each row subtotal."""
        p = self._measures.table_proportions.subtotal_values
        return p * (1 - p)


class _TableProportions(_BaseSecondOrderMeasure):
    """Provides the table-proportions measure for a stripe.

    Table-proportions is a 1D np.float64 ndarray of the proportion each row's weighted
    count contributes to the weighted-N of the table.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of table-proportion for each row of stripe."""
        weighted_counts = self._measures.weighted_counts.base_values
        weighted_bases = self._weighted_cube_counts.bases

        # --- note that table-margin can be either scalar or 1D ndarray. When it is an
        # --- array (stripe is MR), its shape is the same as the weighted_counts, so the
        # --- division works either way.

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return weighted_counts / weighted_bases

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of sum for each row-subtotal."""
        subtotal_values = self._measures.weighted_counts.subtotal_values
        weighted_table_base = self._weighted_cube_counts.table_base

        # --- table-base is defined when stripe is MR and the division below only
        # --- An Array stripe can have no subtotals, so the right answer is always
        # --- always np.array([]) in this case.
        if weighted_table_base is None:
            return np.array([])

        # --- do not propagate divide-by-zero warnings to stderr ---
        with np.errstate(divide="ignore", invalid="ignore"):
            return subtotal_values / weighted_table_base


class _UnweightedBases(_BaseSecondOrderMeasure):
    """Provides the unweighted-bases measure for a stripe.

    unweighted-bases is a 1D np.float64 ndarray of the unweighted table-proportion
    denominator (base) for each row. This object also provides the table-base totals.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of unweighted table-proportion denominator per cell."""
        return self._unweighted_cube_counts.bases

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of subtotal value for each row-subtotal."""
        # --- Background:
        # --- 1. The base is only defined for a CAT stripe.
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

    @lazyproperty
    def table_base_range(self):
        """[min, max] np.float64 ndarray range of (total) unweighted-N for this stripe.

        A non-MR stripe will have a single base, represented by min and max being the
        same value. Any subtotals in a non-MR stripe have that same scalar table base,
        so they do not affect the (single-valued) range.

        Each row of an MR stripe has a distinct base, which is reduced to a range in
        that case. An MR stripe can have no subtotals, so those don't come into it in
        the MR case.
        """
        bases = self._unweighted_cube_counts.bases
        return np.array([np.min(bases), np.max(bases)])


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of unweighted-count for each stripe base-row."""
        return self._unweighted_cube_counts.counts

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of sum for each row-subtotal."""
        # --- counts don't sum on an MR dimension, but an MR stripe can have no
        # --- subtotals. This just returns an empty array in that case and we don't need
        # --- to special-case MR.
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)


class _WeightedBases(_BaseSecondOrderMeasure):
    """Provides the weighted-bases measure for a stripe.

    weighted-bases is a 1D np.float64 ndarray of the (weighted) table-proportion
    denominator (base) for each row.
    """

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of weighted table-proportion denominator per cell."""
        return self._weighted_cube_counts.bases

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of sum for each row-subtotal."""
        # --- Background:
        # --- 1. The base is only defined for a CAT stripe.
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
            self._weighted_cube_counts.table_base, subtotal_values.shape
        )

    @lazyproperty
    def table_margin_range(self):
        """[min, max] np.float64 ndarray range of (total) unweighted-N for this stripe.

        A non-MR stripe will have a single margin, represented by min and max being the
        same value.

        Each row of an MR stripe has a distinct margin, which is reduced to a range in
        that case. An MR stripe can have no subtotals, so those don't enter into the
        computation in the MR case.
        """
        bases = self._weighted_cube_counts.bases
        return np.array([np.min(bases), np.max(bases)])


class _WeightedCounts(_BaseSecondOrderMeasure):
    """Provides the weighted-counts measure for a stripe."""

    @lazyproperty
    def base_values(self):
        """1D np.float64 ndarray of weighted-count for each row."""
        return self._weighted_cube_counts.counts

    @lazyproperty
    def subtotal_values(self):
        """1D np.float64 ndarray of sum for each row-subtotal."""
        # --- counts don't sum on an MR dimension, but an MR stripe can have no
        # --- subtotals. This just returns an empty array in that case and we don't need
        # --- to special-case MR.
        return SumSubtotals.subtotal_values(self.base_values, self._rows_dimension)
