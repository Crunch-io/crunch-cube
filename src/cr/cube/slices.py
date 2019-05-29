# encoding: utf-8

from __future__ import division

import numpy as np

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.frozen_min_base_size_mask import MinBaseSizeMask
from cr.cube.measures.new_pairwise_significance import NewPairwiseSignificance
from cr.cube.matrix import (
    MatrixFactory,
    MatrixWithHidden,
    MatrixWithInsertions,
    MeansScalar,
    OrderedMatrix,
    StripeFactory,
    StripeWithInsertions,
)
from cr.cube.util import lazyproperty
from cr.cube.vector import InsertionColumn, InsertionRow


class CubeSection(object):
    """A slice, a strand, or a nub drawn from a cube-response.

    These are 2, 1, or 0 dimensions of a cube, respectively.
    """

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
        return FrozenSlice(cube, slice_idx, transforms, population, mask_size)


class FrozenSlice(object):
    """Main point of interaction with the outer world."""

    def __init__(self, cube, slice_idx, transforms, population, mask_size):
        self._cube = cube
        self._slice_idx = slice_idx
        self._transforms_arg = transforms
        self._population = population
        self._mask_size = mask_size

    # ---interface ---------------------------------------------------

    @lazyproperty
    def base_counts(self):
        return np.array([row.base_values for row in self._assembler.rows])

    @lazyproperty
    def column_base(self):
        return np.array([column.base for column in self._assembler.columns]).T

    @lazyproperty
    def column_index(self):
        """ndarray of column index percentages.

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the corresponding variable.
        """
        return np.array([row.column_index for row in self._assembler.rows])

    @lazyproperty
    def column_labels(self):
        """Sequence of str column element names suitable for use as column headings."""
        return tuple(column.label for column in self._assembler.columns)

    @lazyproperty
    def column_margin(self):
        return np.array([column.margin for column in self._assembler.columns]).T

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def column_proportions(self):
        return np.array([col.proportions for col in self._assembler.columns]).T

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
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])

    @lazyproperty
    def dimension_types(self):
        """Sequence of member of `cr.cube.enum.DIMENSION_TYPE` for each dimension.

        Items appear in rows-dimension, columns-dimension order.
        """
        return tuple(dimension.dimension_type for dimension in self.dimensions)

    @lazyproperty
    def dimensions(self):
        """tuple containing (rows_dimension, columns_dimension) for this slice.

        Both items are `cr.cube.dimension.Dimension` objects.
        """
        # TODO: I question whether the dimensions should be published. Whatever folks
        # might need to know, like types or whatever, should be available as individual
        # properties. The dimensions are kind of an internal, especially since they
        # won't necessarily match the returned data-matrix in terms of element-order and
        # presence.
        return self._dimensions

    @lazyproperty
    def insertion_columns_idxs(self):
        return tuple(
            i for i, column in enumerate(self._assembler.columns) if column.is_insertion
        )

    @lazyproperty
    def insertion_rows_idxs(self):
        return tuple(
            i for i, row in enumerate(self._assembler.rows) if row.is_insertion
        )

    @lazyproperty
    def means(self):
        return np.array([row.means for row in self._assembler.rows])

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
    def ndim(self):
        """int count of dimensions for this slice, unconditionally 2."""
        return self._matrix.ndim

    @lazyproperty
    def pairwise_indices(self):
        alpha = self._transforms_dict.get("pairwise_indices", {}).get("alpha", 0.05)
        only_larger = self._transforms_dict.get("pairwise_indices", {}).get(
            "only_larger", True
        )
        return NewPairwiseSignificance(
            self, alpha=alpha, only_larger=only_larger
        ).pairwise_indices

    @lazyproperty
    def pairwise_significance_tests(self):
        """tuple of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are columns in the slice. Each
        significance test contains `p_vals` and `t_stats` (ndarrays that represent
        probability values and statistical scores).
        """
        return tuple(
            NewPairwiseSignificance(self).values[column_idx]
            for column_idx in range(len(self._assembler.columns))
        )

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def pvals(self):
        return np.array([row.pvals for row in self._assembler.rows])

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._assembler.rows])

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._assembler.rows)

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

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
        return tuple(row.fill for row in self._assembler.rows)

    @lazyproperty
    def rows_dimension_name(self):
        """str name assigned to rows-dimension.

        Reflects the resolved dimension-name transform cascade.
        """
        return self._rows_dimension.name

    @lazyproperty
    def rows_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` specifying type of rows dimension."""
        return self._rows_dimension.dimension_type

    @lazyproperty
    def scale_means_column(self):
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None

        inner = np.nansum(self._columns_dimension_numeric * self.counts, axis=1)
        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        denominator = np.sum(self.counts[:, not_a_nan_index], axis=1)
        return inner / denominator

    @lazyproperty
    def scale_means_column_margin(self):
        if np.all(np.isnan(self._columns_dimension_numeric)):
            return None

        column_margin = self.column_margin
        if len(column_margin.shape) > 1:
            # Hack for MR, where column margin is a table. Figure how to
            # fix with subclassing
            column_margin = column_margin[0]

        not_a_nan_index = ~np.isnan(self._columns_dimension_numeric)
        return np.nansum(self._columns_dimension_numeric * column_margin) / np.sum(
            column_margin[not_a_nan_index]
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
    def scale_means_row_margin(self):
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None

        row_margin = self.row_margin
        if len(row_margin.shape) > 1:
            # Hack for MR, where row margin is a table. Figure how to
            # fix with subclassing
            row_margin = row_margin[:, 0]

        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        return np.nansum(self._rows_dimension_numeric * row_margin) / np.sum(
            row_margin[not_a_nan_index]
        )

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def summary_pairwise_indices(self):
        alpha = self._transforms_dict.get("pairwise_indices", {}).get("alpha", 0.05)
        only_larger = self._transforms_dict.get("pairwise_indices", {}).get(
            "only_larger", True
        )
        return NewPairwiseSignificance(
            self, alpha=alpha, only_larger=only_larger
        ).summary_pairwise_indices

    @lazyproperty
    def table_base(self):
        return self._assembler.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self._assembler.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        return self._assembler.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self._assembler.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Provides differentiated name for each stacked table of a 3D cube."""
        if self._cube.ndim < 3:
            return None

        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        return np.array([row.table_proportions for row in self._assembler.rows])

    @lazyproperty
    def zscore(self):
        return np.array([row.zscore for row in self._assembler.rows])

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _assembler(self):
        return _Assembler(
            self._matrix,
            _OrderTransform(self._dimensions),
            _MatrixInsertions(
                self._rows_dimension, self._columns_dimension, self._matrix
            ),
            self._prune,
        )

    @lazyproperty
    def _columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _columns_dimension_numeric(self):
        return np.array([column.numeric for column in self._assembler.columns])

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
        """The pre-transforms matrix for this slice."""
        return MatrixFactory.matrix(
            self.dimensions,
            self._cube.counts,
            self._cube.base_counts,
            self._cube.counts_with_missings,
            self._cube,
            self._slice_idx,
        )

    @lazyproperty
    def _prune(self):
        """True if any of dimensions has pruning."""
        # TODO: Implement separarte pruning for rows and columns
        return any(dimension.prune for dimension in self.dimensions)

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows_dimension_numeric(self):
        return np.array([row.numeric for row in self._assembler.rows])

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

    @lazyproperty
    def _transforms_dict(self):
        """dict containing all transforms for this slice, provided as `transforms` arg.

        This value is an empty dict (`{}`) when no transforms were specified on
        construction.
        """
        return self._transforms_arg if self._transforms_arg is not None else {}


class _Strand(object):
    """1D slice."""

    def __init__(self, cube, transforms, population, ca_as_0th, slice_idx, mask_size):
        self._cube = cube
        self._transforms_arg = transforms
        self._population = population
        self._ca_as_0th = ca_as_0th
        # TODO: see if we really need this.
        self._slice_idx = slice_idx
        self._mask_size = mask_size

    @lazyproperty
    def base_counts(self):
        return np.array([row.base_values for row in self._assembler.rows])

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])

    @lazyproperty
    def dimension_types(self):
        """Sequence of `cr.cube.enum.DIMENSION_TYPE` member for each dimension.

        Length one in this case, containing only rows-dimension type.
        """
        # TODO: remove need for this in exporter, at least for 1D case.
        return (self._rows_dimension.dimension_type,)

    @lazyproperty
    def dimensions(self):
        """tuple of (row,) Dimension object."""
        # TODO: I question whether the dimensions should be published. Whatever folks
        # might need to know, like types or whatever, should be available as individual
        # properties. The dimensions are kind of an internal, especially since they
        # won't necessarily match the returned data-matrix in terms of element-order and
        # presence.
        return (self._rows_dimension,)

    @lazyproperty
    def insertion_columns_idxs(self):
        # TODO: pretty sure the need for this should come out of exporter.
        return ()

    @lazyproperty
    def insertion_rows_idxs(self):
        # TODO: add integration-test coverage for this.
        return tuple(
            i for i, row in enumerate(self._assembler.rows) if row.is_insertion
        )

    @lazyproperty
    def means(self):
        return np.array([row.means for row in self._assembler.rows])

    @lazyproperty
    def min_base_size_mask(self):
        # TODO: add integration test that exercises this.
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def ndim(self):
        """int count of dimensions for this strand, unconditionally 1."""

        # The value of the `ndim` on the `self._stipe` is also unconditionally 1
        return self._stripe.ndim

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._assembler.rows])

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._assembler.rows)

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])

    @lazyproperty
    def rows_dimension_fills(self):
        """sequence of RGB str like "#def032" fill colors for row elements.

        The values reflect the resolved element-fill transform cascade. The length and
        ordering of the sequence correspond to the rows in the slice, including
        accounting for insertions and hidden rows.
        """
        return tuple(row.fill for row in self._assembler.rows)

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
    def scale_means_row(self):
        if np.all(np.isnan(self._rows_dimension_numeric)):
            return None
        inner = np.nansum(self._rows_dimension_numeric[:, None] * self.counts, axis=0)
        not_a_nan_index = ~np.isnan(self._rows_dimension_numeric)
        denominator = np.sum(self.counts[not_a_nan_index, :], axis=0)
        return inner / denominator

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def table_base(self):
        """1D, single-element ndarray (like [3770])."""
        # TODO: shouldn't this just be the regular value for a strand? Maybe change to
        # that if exporter always knows when it's getting this from a strand. The
        # ndarray "wrapper" seems like unnecessary baggage when we know it will always
        # be a scalar.
        return self._assembler.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self._assembler.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        return self._assembler.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self._assembler.table_margin_unpruned

    @lazyproperty
    def table_name(self):
        """Only for CA-as-0th case, provides differentiated names for stacked tables."""
        if not self._ca_as_0th:
            return None

        title = self._cube.name
        table_name = self._cube.dimensions[0].valid_elements[self._slice_idx].label
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        return np.array([row.table_proportions for row in self._assembler.rows])

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _assembler(self):
        return _StrandAssembler(
            self._stripe,
            _OrderTransform((self._rows_dimension,)),
            _StrandInsertions(self._rows_dimension, self._stripe),
            self._rows_dimension.prune,
        )

    @lazyproperty
    def _rows_dimension(self):
        """Dimension object for the single dimension of this strand."""
        return self._cube.dimensions[-1].apply_transforms(self._row_transforms_dict)

    @lazyproperty
    def _rows_dimension_numeric(self):
        return np.array([row.numeric for row in self._assembler.rows])

    @lazyproperty
    def _row_transforms_dict(self):
        """Transforms dict for the single (rows) dimension of this strand."""
        transforms_dict = {} if self._transforms_arg is None else self._transforms_arg
        return transforms_dict.get("rows_dimension", {})

    @lazyproperty
    def _stripe(self):
        """The pre-transforms 1D matrix for this strand."""
        return StripeFactory.stripe(
            self._cube,
            self._rows_dimension,
            self._cube.counts,
            self._cube.base_counts,
            self._ca_as_0th,
            self._slice_idx,
        )


class _Nub(object):
    """0D slice."""

    def __init__(self, cube):
        self._cube = cube

    @lazyproperty
    def means(self):
        return self._scalar.means

    @lazyproperty
    def ndim(self):
        """int count of dimensions, unconditionally 0 for a Nub."""

        # The ndim on the underlying _scalar is also unconditionally 0
        return self._scalar.ndim

    @lazyproperty
    def table_base(self):
        return self._scalar.table_base

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _scalar(self):
        """The pre-transforms data-array for this slice."""
        return MeansScalar(self._cube.counts, self._cube.base_counts)


class _Assembler(object):
    """In charge of performing all the transforms sequentially."""

    def __init__(self, matrix, ordering, insertions, prune):
        self._matrix = matrix
        self._ordering = ordering
        self._insertions = insertions
        self._prune = prune

    @lazyproperty
    def columns(self):
        return self._transformed_matrix.columns

    @lazyproperty
    def rows(self):
        return self._transformed_matrix.rows

    @lazyproperty
    def table_base(self):
        return self._transformed_matrix.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self._transformed_matrix.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        return self._transformed_matrix.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self._transformed_matrix.table_margin_unpruned

    @lazyproperty
    def _transformed_matrix(self):
        """Apply all transforms sequentially."""
        matrix = OrderedMatrix(self._matrix, self._ordering)
        matrix = MatrixWithInsertions(matrix, self._insertions)
        matrix = MatrixWithHidden(matrix, self._prune)
        return matrix


class _StrandAssembler(object):
    """Perform transforms on a 1D cube-section."""

    def __init__(self, stripe, ordering, insertions, prune):
        self._stripe = stripe
        self._ordering = ordering
        self._insertions = insertions
        self._prune = prune

    @lazyproperty
    def column(self):
        """Single post-transformation column vector."""
        return self._transformed_stripe.columns[0]

    @lazyproperty
    def rows(self):
        """Sequence of post-transformation row vectors."""
        return self._transformed_stripe.rows

    @lazyproperty
    def table_base(self):
        """1D, single-element ndarray with int value."""
        return self._transformed_stripe.table_base

    @lazyproperty
    def table_base_unpruned(self):
        """Hmm, weird 1D ndarray with same int value repeated for each row."""
        return self._transformed_stripe.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        """1D, single-element ndarray with float value."""
        return self._transformed_stripe.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        """Hmm, weird 1D ndarray with same float value repeated for each row."""
        return self._transformed_stripe.table_margin_unpruned

    @lazyproperty
    def _transformed_stripe(self):
        """Apply all transforms sequentially."""
        stripe = OrderedMatrix(self._stripe, self._ordering)
        stripe = StripeWithInsertions(stripe, self._insertions)
        stripe = MatrixWithHidden(stripe, self._prune)
        return stripe


class _MatrixInsertions(object):
    """Represents subtotal rows and columns inserted into a slice."""

    def __init__(self, rows_dimension, columns_dimension, matrix):
        self._rows_dimension = rows_dimension
        self._columns_dimension = columns_dimension
        self._matrix = matrix

    @lazyproperty
    def all_inserted_columns(self):
        """Sequence of _InsertionColumn objects representing subtotal columns.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order. All subtotals defined on the column
        dimension appear in the sequence.
        """
        # ---an aggregate columns-dimension is not summable---
        if self._columns_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        return tuple(
            InsertionColumn(self._matrix, subtotal)
            for subtotal in self._columns_dimension.subtotals
        )

    @lazyproperty
    def all_inserted_rows(self):
        """Sequence of _InsertionRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # ---an aggregate rows-dimension is not summable---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            InsertionRow(self._matrix, subtotal)
            for subtotal in self._rows_dimension.subtotals
        )

    @lazyproperty
    def columns_inserted_at_left(self):
        """Sequence of InsertionColumn vectors that appear before any body columns."""
        return tuple(
            column for column in self.all_inserted_columns if column.anchor == "top"
        )

    @lazyproperty
    def columns_inserted_in_body(self):
        """Sequence of InsertionColumn vectors anchored to a table body column."""
        return tuple(
            column
            for column in self.all_inserted_columns
            if column.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def columns_inserted_at_right(self):
        """Sequence of InsertionColumn vectors appended as the last table columns."""
        return tuple(
            column for column in self.all_inserted_columns if column.anchor == "bottom"
        )

    @lazyproperty
    def rows_inserted_at_top(self):
        """Sequence of InsertionRow vectors that appear before any other table rows."""
        return tuple(row for row in self.all_inserted_rows if row.anchor == "top")

    @lazyproperty
    def rows_inserted_in_body(self):
        """Sequence of InsertionRow vectors that are anchored to a particular element.

        These are in no particular order and must be associated with their element using
        the element-id available on their `.anchor` property.
        """
        return tuple(
            row for row in self.all_inserted_rows if row.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def rows_inserted_at_bottom(self):
        """Sequence of InsertionRow vectors that appear after any other table rows."""
        return tuple(row for row in self.all_inserted_rows if row.anchor == "bottom")


class _StrandInsertions(object):
    """Represents subtotals inserted into rows-dimension of univariate."""

    def __init__(self, rows_dimension, stripe):
        self._rows_dimension = rows_dimension
        self._stripe = stripe

    @lazyproperty
    def all_inserted_rows(self):
        """Sequence of all InsertionRow vectors for this strand.

        These appear in no particular order.
        """
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            InsertionRow(self._stripe, subtotal)
            for subtotal in self._rows_dimension.subtotals
        )

    @lazyproperty
    def rows_inserted_at_top(self):
        """Sequence of InsertionRow vectors that appear before any other table rows."""
        return tuple(row for row in self.all_inserted_rows if row.anchor == "top")

    @lazyproperty
    def rows_inserted_in_body(self):
        """Sequence of InsertionRow vectors that are anchored to a particular element.

        These are in no particular order and must be associated with their element using
        the element-id available on their `.anchor` property.
        """
        return tuple(
            row for row in self.all_inserted_rows if row.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def rows_inserted_at_bottom(self):
        """Sequence of InsertionRow vectors that appear after any other table rows."""
        return tuple(row for row in self.all_inserted_rows if row.anchor == "bottom")


class _OrderTransform(object):
    """Creates ordering indexes for rows and columns based on element ids."""

    def __init__(self, dimensions):
        self._dimensions = dimensions

    @lazyproperty
    def column_order(self):
        """Indexer value identifying columns in order, suitable for slicing an ndarray.

        This value is `slice(None)` when there is no columns dimension. Otherwise it is
        a 1D ndarray of int column indices, suitable for indexing the columns array to
        produce an ordered version.
        """
        # ---if there's no column dimension, there can be no reordering for it---
        if len(self._dimensions) < 2:
            return slice(None)

        # ---Specifying int type prevents failure when there are zero columns. The
        # ---default type for ndarray is float, which is not valid for indexing.
        return np.array(self._columns_dimension.display_order, dtype=int)

    @lazyproperty
    def row_order(self):
        return np.array(self._rows_dimension.display_order, dtype=int)

    @lazyproperty
    def _columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]
