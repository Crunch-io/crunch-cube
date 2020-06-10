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

from __future__ import division

import numpy as np

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.min_base_size_mask import MinBaseSizeMask
from cr.cube.measures.pairwise_significance import PairwiseSignificance
from cr.cube.matrix import TransformedMatrix
from cr.cube.scalar import MeansScalar
from cr.cube.stripe import TransformedStripe
from cr.cube.util import lazyproperty


class CubePartition(object):
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
    def cube_is_mr_by_itself(self):
        return False

    @lazyproperty
    def dimension_types(self):
        """Sequence of member of `cr.cube.enum.DIMENSION_TYPE` for each dimension.

        Items appear in rows-dimension, columns-dimension order.
        """
        return tuple(d.dimension_type for d in self._dimensions)

    @lazyproperty
    def has_means(self):
        """True if cube-result includes means values."""
        return self._cube.has_means

    @lazyproperty
    def ndim(self):
        """int count of dimensions for this partition."""
        return len(self._dimensions)

    @lazyproperty
    def population_fraction(self):
        """Returns the population fraction of the cube"""
        return self._cube.population_fraction

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
                "transforms.pairwise_indices.alpha, when defined, must be a list of 1 "
                "or 2 float values between 0.0 and 1.0 exclusive. Got %r" % value
            )

        # --- legacy float "by-itself" case ---
        if isinstance(value, float):
            if not 0.0 < value < 1.0:
                raise ValueError(
                    "alpha value, when provided, must be between 0.0 and 1.0 "
                    "exclusive. Got %r" % value
                )
            return (value, None)

        # --- sequence case ---
        for x in value[:2]:
            if not isinstance(x, float) or not 0.0 < x < 1.0:
                raise ValueError(
                    "transforms.pairwise_indices.alpha must be a list of 1 or 2 float "
                    "values between 0.0 and 1.0 exclusive. Got %r" % value
                )

        if len(value) == 1:
            return (value[0], None)

        return tuple(sorted(value[:2]))

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

    # ---interface ---------------------------------------------------

    @lazyproperty
    def base_counts(self):
        return np.array([row.base_values for row in self._matrix.rows])

    @lazyproperty
    def column_base(self):
        return np.array([column.base for column in self._matrix.columns]).T

    @lazyproperty
    def column_index(self):
        """ndarray of column index percentages.

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the corresponding variable.
        """
        return np.array([row.column_index for row in self._matrix.rows])

    @lazyproperty
    def column_labels(self):
        """Sequence of str column element names suitable for use as column headings."""
        return tuple(column.label for column in self._matrix.columns)

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def column_proportions(self):
        return np.array([col.proportions for col in self._matrix.columns]).T

    @lazyproperty
    def columns_dimension_name(self):
        """str name assigned to columns-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._columns_dimension.name

    @lazyproperty
    def columns_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` describing columns dimension."""
        return self._columns_dimension.dimension_type

    @lazyproperty
    def columns_margin(self):
        return np.array([column.margin for column in self._matrix.columns]).T

    @lazyproperty
    def columns_std_dev(self):
        """Returns the standard deviation for cell percentages
        `std_deviation = sqrt(variance)`
        """
        return np.sqrt(self._columns_variance)

    @lazyproperty
    def columns_std_err(self):
        """Returns the standard error for cell percentages
        `std_error = sqrt(variance/N)`
        """
        return np.sqrt(self._columns_variance / self.columns_margin)

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._matrix.rows])

    @lazyproperty
    def cube_is_mr_by_itself(self):
        return self._cube.is_mr_by_itself

    @lazyproperty
    def description(self):
        """str description of this slice, which it takes from its rows-dimension."""
        return self._rows_dimension.description

    @lazyproperty
    def inserted_column_idxs(self):
        return tuple(
            i for i, column in enumerate(self._matrix.columns) if column.is_inserted
        )

    @lazyproperty
    def inserted_row_idxs(self):
        return tuple(i for i, row in enumerate(self._matrix.rows) if row.is_inserted)

    @lazyproperty
    def insertions(self):
        """Returns masked array with residuals for insertions

                 0     1	 2	     3	    4	    5	    6
           0   inf   inf   inf	   inf	  inf	 -2.9	  inf
           1   inf	 inf   inf	   inf	  inf	 -4.3	  inf
           2   2.5	 1.3   3.3	 -0.70	-7.25	 -6.52	 2.25
           3   inf	 inf   inf	   inf	  inf	 -2.51	  inf
           4  -1.16	 2.20  5.84	  1.78	-8.48	 -5.92	 0.93
           5   inf   inf   inf	   inf	  inf	  9.70	  inf

           Only the insertions residuals are showed in a inf masked array
        """
        inserted_rows = self.inserted_row_idxs
        inserted_cols = self.inserted_column_idxs
        if not inserted_cols and not inserted_cols:
            return []
        mask = np.zeros(self.pvals.shape)
        mask[inserted_rows, :] = 1
        mask[:, inserted_cols] = 1
        masked_pvals = np.ma.masked_array(self.pvals, np.logical_not(mask)).filled(
            np.inf
        )
        masked_zscores = np.ma.masked_array(self.zscore, np.logical_not(mask)).filled(
            np.inf
        )
        return np.stack([masked_pvals, masked_zscores])

    @lazyproperty
    def is_empty(self):
        return any(s == 0 for s in self.shape)

    @lazyproperty
    def means(self):
        return np.array([row.means for row in self._matrix.rows])

    @lazyproperty
    def min_base_size_mask(self):
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def name(self):
        """str name assigned to this slice.

        A slice takes the name of its rows-dimension.
        """
        return self.rows_dimension_name

    @lazyproperty
    def overlaps_tstats(self):
        return self._matrix.overlaps_tstats

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
        return PairwiseSignificance.pairwise_indices(
            self, self._alpha, self._only_larger
        )

    @lazyproperty
    def pairwise_indices_alt(self):
        """2D ndarray of tuple of int column-idxs meeting alternate threshold.

        This value is None if no alternate threshold has been defined.
        """
        return (
            None
            if self._alpha_alt is None
            else PairwiseSignificance.pairwise_indices(
                self, self._alpha_alt, self._only_larger
            )
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
            for column_idx in range(len(self._matrix.columns))
        )

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def pvals(self):
        return np.array([row.pvals for row in self._matrix.rows])

    @lazyproperty
    def residual_test_stats(self):
        """Exposes pvals and zscore (with HS) stacked together

        Public method used as cube_method for the SOA API
        """
        return np.stack([self.pvals, self.zscore])

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._matrix.rows])

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._matrix.rows)

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._matrix.rows])

    @lazyproperty
    def rows_dimension_description(self):
        """str description assigned to rows-dimension.

        Reflects the resolved dimension-description transform cascade.
        """
        return self._rows_dimension.description

    @lazyproperty
    def rows_dimension_fills(self):
        """sequence of RGB str like "#def032" fill colors for row elements.

        The values reflect the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions and hidden rows.
        """
        return tuple(row.fill for row in self._matrix.rows)

    @lazyproperty
    def rows_dimension_name(self):
        """str name assigned to rows-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._rows_dimension.name

    @lazyproperty
    def rows_dimension_numeric(self):
        return self._rows_dimension_numeric

    @lazyproperty
    def rows_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` specifying type of rows dimension."""
        return self._rows_dimension.dimension_type

    @lazyproperty
    def rows_margin(self):
        return np.array([row.margin for row in self._matrix.rows])

    @lazyproperty
    def scale_mean_pairwise_indices(self):
        """Sequence of column-idx tuples indicating pairwise-t result of scale-means.

        The calculation is based on the mean of the scale (category numeric-values) for
        each column. The length of the array is that of the columns-dimension.
        """
        return tuple(
            PairwiseSignificance.scale_mean_pairwise_indices(
                self, self._alpha, self._only_larger
            ).tolist()
        )

    @lazyproperty
    def scale_mean_pairwise_indices_alt(self):
        """Sequence of column-idx tuples indicating pairwise-t result of scale-means.

        Same calculation as `.scale_mean_pairwise_indices` using the `._alpha_alt`
        value. None when no secondary alpha value was specified. The length of the
        sequence is that of the columns-dimension.
        """
        if self._alpha_alt is None:
            return None

        return tuple(
            PairwiseSignificance.scale_mean_pairwise_indices(
                self, self._alpha_alt, self._only_larger
            ).tolist()
        )

    @lazyproperty
    def scale_means_column(self):
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None

        inner = np.nansum(self._columns_dimension_numeric * self.counts, axis=1)
        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        denominator = np.sum(self.counts[:, not_a_nan_index], axis=1)
        return inner / denominator

    @lazyproperty
    def scale_means_columns_margin(self):
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None

        columns_margin = self.columns_margin
        if len(columns_margin.shape) > 1:
            # Hack for MR, where column margin is a table. Figure how to
            # fix with subclassing
            columns_margin = columns_margin[0]

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        return np.nansum(self._columns_dimension_numeric * columns_margin) / np.sum(
            columns_margin[not_a_nan_index]
        )

    @lazyproperty
    def scale_means_row(self):
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        inner = np.nansum(self._rows_dimension_numeric[:, None] * self.counts, axis=0)
        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        denominator = np.sum(self.counts[not_a_nan_index, :], axis=0)
        return inner / denominator

    @lazyproperty
    def scale_means_rows_margin(self):
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None

        rows_margin = self.rows_margin
        if len(rows_margin.shape) > 1:
            # Hack for MR, where row margin is a table. Figure how to
            # fix with subclassing
            rows_margin = rows_margin[:, 0]

        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        return np.nansum(self._rows_dimension_numeric * rows_margin) / np.sum(
            rows_margin[not_a_nan_index]
        )

    @lazyproperty
    def scale_median_column(self):
        """ -> np.int64 ndarray of the columns scale median

        The median is calculated using the standard algebra applied to the numeric
        values repeated for each related counts value
        """
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None
        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        numeric_values = self._columns_dimension_numeric[not_a_nan_index]
        counts = self.counts[:, not_a_nan_index].astype("int64")
        scale_median = np.array(
            [
                self._median(np.repeat(numeric_values, counts[i, :]))
                for i in range(counts.shape[0])
            ]
        )
        return scale_median

    @lazyproperty
    def scale_median_row(self):
        """ -> np.int64 ndarray of the rows scale median

        The median is calculated using the standard algebra applied to the numeric
        values repeated for each related counts value
        """
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        numeric_values = self._rows_dimension_numeric[not_a_nan_index]
        counts = self.counts[not_a_nan_index, :].astype("int64")
        scale_median = np.array(
            [
                self._median(np.repeat(numeric_values, counts[:, i]))
                for i in range(counts.shape[1])
            ]
        )
        return scale_median

    @lazyproperty
    def scale_median_column_margin(self):
        """ -> np.int64, represents the column scale median margin"""
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None
        columns_margin = self.columns_margin
        if len(columns_margin.shape) > 1:
            columns_margin = columns_margin[0]
        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        numeric_values = self._columns_dimension_numeric[not_a_nan_index]
        counts = columns_margin[not_a_nan_index].astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def scale_median_row_margin(self):
        """ -> np.int64, represents the rows scale median margin"""
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        rows_margin = self.rows_margin
        if len(rows_margin.shape) > 1:
            rows_margin = rows_margin[:, 0]
        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        numeric_values = self._rows_dimension_numeric[not_a_nan_index]
        counts = rows_margin[not_a_nan_index].astype("int64")
        unwrapped_num_values = np.repeat(numeric_values, counts)
        return (
            np.median(unwrapped_num_values) if unwrapped_num_values.size != 0 else None
        )

    @lazyproperty
    def scale_std_dev_column(self):
        """ -> 1D np.ndarray of the standard deviation column of scales"""
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None
        return np.sqrt(self.var_scale_means_column)

    @lazyproperty
    def scale_std_dev_row(self):
        """ -> 1D np.ndarray of the standard deviation row of scales"""
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        return np.sqrt(self.var_scale_means_row)

    @lazyproperty
    def scale_std_err_column(self):
        """ -> 1D np.ndarray of the standard error column of scales"""
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None
        return self.scale_std_dev_column / np.sqrt(self.rows_margin)

    @lazyproperty
    def scale_std_err_row(self):
        """ -> 1D np.ndarray of the standard error row of scales"""
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        return self.scale_std_dev_row / np.sqrt(self.columns_margin)

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def summary_pairwise_indices(self):
        return PairwiseSignificance(
            self, self._alpha, self._only_larger
        ).summary_pairwise_indices

    @lazyproperty
    def table_base(self):

        # We need to prune/order by both dimensions
        if self.dimension_types == (DT.MR, DT.MR):
            # TODO: Remove property from the assembler, when we figure out the pruning
            # by both rows and columns
            return self._matrix.table_base

        # We need to prune/order by rows
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_base for row in self._matrix.rows])

        # We need to prune/order by columns
        if self.dimension_types[1] == DT.MR:
            return np.array([column.table_base for column in self._matrix.columns])

        # No pruning or reordering since single value
        return self._matrix.table_base_unpruned

    @lazyproperty
    def table_base_unpruned(self):
        return self._matrix.table_base_unpruned

    @lazyproperty
    def table_margin(self):

        # We need to prune/order by both dimensions
        if self.dimension_types == (DT.MR, DT.MR):
            # TODO: Remove property from the assembler, when we figure out the pruning
            # by both rows and columns
            return self._matrix.table_margin

        # We need to prune/order by rows
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_margin for row in self._matrix.rows])

        # We need to prune/order by columns
        if self.dimension_types[1] == DT.MR:
            return np.array([column.table_margin for column in self._matrix.columns])

        # No pruning or reordering since single value
        return self._matrix.table_margin_unpruned

    @lazyproperty
    def table_margin_unpruned(self):
        return self._matrix.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Provides differentiated name for each stacked table of a 3D cube."""
        if self._cube.ndim < 3:
            return None

        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label

        if self._cube.is_mr_by_itself:
            return title

        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        return np.array([row.table_proportions for row in self._matrix.rows])

    @lazyproperty
    def table_std_dev(self):
        return np.array([row.table_std_dev for row in self._matrix.rows])

    @lazyproperty
    def table_std_err(self):
        return np.array([row.table_std_err for row in self._matrix.rows])

    @lazyproperty
    def var_scale_means_column(self):
        """ -> 1D np.ndarray of the column variance values for scales

        Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is the
              mean of the data.
        """
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        col_dim_numeric = self._columns_dimension_numeric[not_a_nan_index]

        numerator = self.counts[:, not_a_nan_index] * pow(
            np.broadcast_to(col_dim_numeric, self.counts[:, not_a_nan_index].shape)
            - self.scale_means_column.reshape(-1, 1),
            2,
        )
        denominator = np.sum(self.counts[:, not_a_nan_index], axis=1)
        return np.nansum(numerator, axis=1) / denominator

    @lazyproperty
    def var_scale_means_row(self):
        """ -> 1D np.ndarray of the row variance values for scales

        Note: the variance for scale is defined as sum((Yi−Y~)2/(N)), where Y~ is the
              mean of the data.
        """
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None

        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        row_dim_numeric = self._rows_dimension_numeric[not_a_nan_index]
        numerator = (
            self.counts[not_a_nan_index, :]
            * pow(
                np.broadcast_to(
                    row_dim_numeric, self.counts[not_a_nan_index, :].T.shape
                )
                - self.scale_means_row.reshape(-1, 1),
                2,
            ).T
        )
        denominator = np.sum(self.counts[not_a_nan_index, :], axis=0)
        return np.nansum(numerator, axis=0) / denominator

    @lazyproperty
    def zscore(self):
        return np.array([row.zscore for row in self._matrix.rows])

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _columns_dimension_numeric(self):
        return np.array([column.numeric for column in self._matrix.columns])

    @lazyproperty
    def _columns_variance(self):
        """Returns the variance for cell percentages
        `variance = p * (1-p)`
        """
        return (
            self.counts / self.columns_margin * (1 - self.counts / self.columns_margin)
        )

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
    def _matrix(self):
        """The TransformedMatrix object for this slice."""
        return TransformedMatrix.matrix(self._cube, self._dimensions, self._slice_idx)

    def _median(self, values):
        return np.median(values) if values.size != 0 else np.nan

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows_dimension_numeric(self):
        return np.array([row.numeric for row in self._matrix.rows])

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

    @lazyproperty
    def base_counts(self):
        return tuple(row.base for row in self._stripe.rows)

    @lazyproperty
    def bases(self):
        """Sequence of weighted base for each row."""
        return tuple(np.broadcast_to(self.table_margin, self.shape))

    @lazyproperty
    def counts(self):
        return tuple(row.count for row in self._stripe.rows)

    @lazyproperty
    def inserted_row_idxs(self):
        """tuple of int index of each inserted row in this strand.

        Suitable for use in applying different formatting (e.g. Bold) to inserted rows.
        Provided index values correspond to measure values as-delivered by this strand,
        after any re-ordering specified in a transform.
        """
        return tuple(i for i, row in enumerate(self._stripe.rows) if row.is_inserted)

    @lazyproperty
    def is_empty(self):
        return any(s == 0 for s in self.shape)

    @lazyproperty
    def means(self):
        return tuple(row.mean for row in self._stripe.rows)

    @lazyproperty
    def min_base_size_mask(self):
        mask = self.table_base < self._mask_size
        shape = self.shape
        return (
            mask
            if self.table_base.shape == shape
            else np.logical_or(np.zeros(shape, dtype=bool), mask)
        )

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def population_counts(self):
        return tuple(
            self._table_proportions_as_array
            * self._population
            * self._cube.population_fraction
        )

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._stripe.rows])

    @lazyproperty
    def row_count(self):
        return len(self._stripe.rows)

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._stripe.rows)

    @lazyproperty
    def rows_dimension_fills(self):
        """sequence of RGB str like "#def032" fill colors for row elements.

        The values reflect the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions and hidden rows.
        """
        return tuple(row.fill for row in self._stripe.rows)

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
        return np.array([row.count for row in self._stripe.rows])

    @lazyproperty
    def scale_mean(self):
        """float mean of numeric-value applied to elements, or None.

        This value is `None` when no row-elements have a numeric-value assigned.
        """
        # ---return None when no row-element has been assigned a numeric value. This
        # ---avoids a division-by-zero error.
        if np.all(np.isnan(self._numeric_values)):
            return None

        # ---produce operands with rows without numeric values removed. Notably, this
        # ---excludes subtotal rows.
        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask]

        # ---calculate numerator and denominator---
        total_numeric_value = np.sum(numeric_values * counts)
        total_count = np.sum(counts)

        # ---overall scale-mean is the quotient---
        return total_numeric_value / total_count

    @lazyproperty
    def scale_median(self):
        """ -> np.int64, the median of scales

        The median is calculated using the standard algebra applied to the numeric
        values repeated for each related counts value
        """
        if np.all(np.isnan(self._numeric_values)):
            return None
        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask].astype("int64")
        unwrapped_numeric_values = np.repeat(numeric_values, counts)
        return np.median(unwrapped_numeric_values)

    @lazyproperty
    def scale_std_dev(self):
        """ -> np.float64, the standard deviation of scales"""
        if np.all(np.isnan(self._numeric_values)):
            return None
        return np.sqrt(self.var_scale_mean)

    @lazyproperty
    def scale_std_err(self):
        """ -> np.float64, the standard error of scales"""
        if np.all(np.isnan(self._numeric_values)):
            return None
        counts = self._counts_as_array[self._numeric_values_mask]
        return np.sqrt(self.var_scale_mean / np.sum(counts))

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
    def table_base(self):
        """1D, single-element ndarray (like [3770])."""
        # For MR strands, table base is also a strand, since subvars never collapse.
        # We need to keep the ordering and hiding as in rows dimension. All this
        # information is already accessible in the underlying rows property
        # of the `_stripe`.
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_base for row in self._stripe.rows])

        # TODO: shouldn't this just be the regular value for a strand? Maybe change to
        # that if exporter always knows when it's getting this from a strand. The
        # ndarray "wrapper" seems like unnecessary baggage when we know it will always
        # be a scalar.
        return self._stripe.table_base_unpruned

    @lazyproperty
    def table_base_unpruned(self):
        return self._stripe.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        # For MR strands, table base is also a strand, since subvars never collapse.
        # We need to keep the ordering and hiding as in rows dimension. All this
        # information is already accessible in the underlying rows property
        # of the `_stripe`.
        if self.dimension_types[0] == DT.MR:
            return np.array([row.table_margin for row in self._stripe.rows])

        return self._stripe.table_margin_unpruned
        # return self._stripe.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self._stripe.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Only for CA-as-0th case, provides differentiated names for stacked tables."""
        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return tuple(self._table_proportions_as_array * 100)

    @lazyproperty
    def table_proportions(self):
        return tuple(row.table_proportions for row in self._stripe.rows)

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
        """Sequence of base count for each row, before weighting.

        When the rows dimension is multiple-response, each value is different,
        reflecting the base for that individual subvariable. In all other cases, the
        table base is repeated for each row.
        """
        return tuple(np.broadcast_to(self.table_base, self.shape))

    @lazyproperty
    def var_scale_mean(self):
        if np.all(np.isnan(self._numeric_values)):
            return None

        numeric_values = self._numeric_values[self._numeric_values_mask]
        counts = self._counts_as_array[self._numeric_values_mask]

        return np.nansum(counts * pow((numeric_values - self.scale_mean), 2)) / np.sum(
            counts
        )

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _counts_as_array(self):
        """1D ndarray of count for each row."""
        return np.array([row.count for row in self._stripe.rows_including_hidden])

    @lazyproperty
    def _dimensions(self):
        """tuple of (row,) Dimension object."""
        return (self._rows_dimension,)

    @lazyproperty
    def _numeric_values(self):
        """Array of numeric-value for each element in rows dimension.

        The items in the array can be numeric or np.nan, which appears for an inserted
        row (subtotal) or where the row-element has been assigned no numeric value.
        """
        return np.array(
            [row.numeric_value for row in self._stripe.rows_including_hidden]
        )

    @lazyproperty
    def _numeric_values_mask(self):
        """ -> np.ndarray, boolean elements for each element in rows dimension."

        This array contains True or False according to the nan in the numeric_values
        array
        """
        is_a_number_mask = ~np.isnan(self._numeric_values)
        return is_a_number_mask

    @lazyproperty
    def _rows_dimension(self):
        """Dimension object for the single dimension of this strand."""
        return self._cube.dimensions[-1].apply_transforms(self._row_transforms_dict)

    @lazyproperty
    def _row_transforms_dict(self):
        """Transforms dict for the single (rows) dimension of this strand."""
        return self._transforms_dict.get("rows_dimension", {})

    @lazyproperty
    def _stripe(self):
        """The post-transforms 1D data-partition for this strand."""
        return TransformedStripe.stripe(
            self._cube, self._rows_dimension, self._ca_as_0th, self._slice_idx
        )

    @lazyproperty
    def _table_proportions_as_array(self):
        return np.array([row.table_proportions for row in self._stripe.rows])


class _Nub(CubePartition):
    """0D slice."""

    @lazyproperty
    def base_count(self):
        return self._cube.base_counts

    @lazyproperty
    def is_empty(self):
        return False if self.base_count else True

    @lazyproperty
    def means(self):
        return self._scalar.means

    @lazyproperty
    def table_base(self):
        return self._scalar.table_base

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _dimensions(self):
        return ()

    @lazyproperty
    def _scalar(self):
        """The pre-transforms data-array for this slice."""
        return MeansScalar(self._cube.counts, self._cube.base_counts)
