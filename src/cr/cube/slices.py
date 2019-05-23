# encoding: utf-8

from __future__ import division

import numpy as np

from cr.cube.dimension import NewDimension
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.frozen_min_base_size_mask import MinBaseSizeMask
from cr.cube.measures.new_pairwise_significance import NewPairwiseSignificance
from cr.cube.matrix import (
    _CatMatrix1D,
    _CatXCatMeansMatrix,
    MatrixFactory,
    MatrixWithHidden,
    MatrixWithInsertions,
    _MeansMatrix0D,
    _MeansMatrix1D,
    _MrMatrix1D,
    _MrWithMeansMatrix1D,
    OrderedMatrix,
    PrunedMatrix,
)
from cr.cube.util import lazyproperty
from cr.cube.vector import _InsertionColumn, _InsertionRow


class FrozenSlice(object):
    """Main point of interaction with the outer world."""

    def __init__(
        self,
        cube,
        slice_idx=0,
        transforms=None,
        population=None,
        ca_as_0th=None,
        mask_size=0,
    ):
        self._cube = cube
        self._slice_idx = slice_idx
        self._transforms_dict = {} if transforms is None else transforms
        self._population = population
        self._ca_as_0th = ca_as_0th
        self._mask_size = mask_size

    # ---interface ---------------------------------------------------

    @lazyproperty
    def _weighted(self):
        return self._cube.is_weighted

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

        The empty string ("") for a 0D or 1D slice (until we get to all slices being
        2D). Reflects the resolved dimension-name transform cascade.
        """
        if len(self.dimensions) < 2:
            return ""
        return self.dimensions[1].name

    @lazyproperty
    def columns_dimension_type(self):
        """Member of `cr.cube.enum.DIMENSION_TYPE` describing columns dimension.

        This value is None for a slice with fewer than two dimensions.
        """
        if len(self.dimensions) < 2:
            return None
        return self.dimensions[1].dimension_type

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])

    @lazyproperty
    def dimension_types(self):
        """Sequence of member of `cr.cube.enum.DIMENSION_TYPE` for each dimension.

        Items appear in rows-dimension, columns-dimension order, although there will be
        fewer than two for a slice with less than two dimensions.
        """
        return tuple(dimension.dimension_type for dimension in self.dimensions)

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
        if type(self._assembler._matrix) is _MeansMatrix0D:
            return self._assembler._matrix.means
        return np.array([row.means for row in self._assembler.rows])

    @lazyproperty
    def min_base_size_mask(self):
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def names(self):
        return self._matrix.names

    @lazyproperty
    def ndim(self):
        _1D_matrix_types = (
            _MrWithMeansMatrix1D,
            _MeansMatrix1D,
            _CatMatrix1D,
            _MrMatrix1D,
        )
        if isinstance(self._matrix, _CatXCatMeansMatrix):
            return 2
        if isinstance(self._matrix, _1D_matrix_types):
            return 1
        elif isinstance(self._matrix, _MeansMatrix0D):
            return 0
        return 2

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

        The empty string ("") for a 0D slice (until we get to all slices being 2D).
        Reflects the resolved dimension-description transform cascade.
        """
        if len(self.dimensions) == 0:
            return ""
        return self.dimensions[0].description

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

        The empty string ("") for a 0D slice (until we get to all slices being 2D).
        Reflects the resolved dimension-name transform cascade.
        """
        if len(self.dimensions) == 0:
            return ""
        return self.dimensions[0].name

    @lazyproperty
    def rows_dimension_type(self):
        """Member of DIMENSION_TYPE enum describing type of rows dimension, or None.

        This value is None for a 0D slice (until we get to all slices being 2D).
        """
        if len(self.dimensions) == 0:
            return None
        return self.dimensions[0].dimension_type

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
        if self._cube.ndim < 3 and not self._ca_as_0th:
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
        return _Assembler(self._matrix, self._transforms)

    @lazyproperty
    def _columns_dimension_numeric(self):
        return np.array([column.numeric for column in self._assembler.columns])

    @lazyproperty
    def dimensions(self):
        """tuple of (row,) or (row, col) Dimension objects, depending on 1D or 2D."""
        # TODO: pretty messy while we're shimming in NewDimensions, should clean up
        # pretty naturally after FrozenSlice has its own loader.

        dimensions = self._cube.dimensions[-2:]

        # ---special-case for 0D mean cube---
        if not dimensions:
            return dimensions

        if self._ca_as_0th:
            # Represent CA slice as 1-D rather than 2-D
            dimensions = (dimensions[-1],)

        rows_dimension = NewDimension(
            dimensions[0], self._transforms_dict.get("rows_dimension", {})
        )

        if len(dimensions) == 1:
            return (rows_dimension,)

        columns_dimension = NewDimension(
            dimensions[1], self._transforms_dict.get("columns_dimension", {})
        )

        return (rows_dimension, columns_dimension)

    @lazyproperty
    def _pruning(self):
        """True if any of dimensions has pruning."""
        # TODO: Implement separarte pruning for rows and columns
        return any(dimension.prune for dimension in self.dimensions)

    @lazyproperty
    def _rows_dimension_numeric(self):
        return np.array([row.numeric for row in self._assembler.rows])

    @lazyproperty
    def _matrix(self):
        """The pre-transforms matrix for this slice."""
        cube = self._cube
        base_counts = self._cube.base_counts
        counts_with_missings = self._cube.counts_with_missings
        counts = self._cube.counts
        return MatrixFactory.matrix(
            self.dimensions,
            counts,
            base_counts,
            counts_with_missings,
            cube,
            self._slice_idx,
            self._ca_as_0th,
        )

    @lazyproperty
    def _transforms(self):
        return _Transforms(self._matrix, self.dimensions, self._pruning)


class _Assembler(object):
    """In charge of performing all the transforms sequentially."""

    def __init__(self, matrix, transforms):
        self._matrix = matrix
        self._transforms = transforms

    @lazyproperty
    def columns(self):
        return self.matrix.columns

    @lazyproperty
    def matrix(self):
        """Apply all transforms sequentially."""
        matrix = OrderedMatrix(self._matrix, self._transforms)
        matrix = MatrixWithInsertions(matrix, self._transforms)
        matrix = MatrixWithHidden(matrix, self._transforms)
        matrix = PrunedMatrix(matrix, self._transforms)
        return matrix

    @lazyproperty
    def rows(self):
        return self.matrix.rows

    @lazyproperty
    def table_base(self):
        return self.matrix.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self.matrix.table_base_unpruned

    @lazyproperty
    def table_margin(self):
        return self.matrix.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self.matrix.table_margin_unpruned


class _Insertions(object):
    """Represents slice's insertions (inserted rows and columns).

    It generates the inserted rows and columns directly from the matrix, based on the
    subtotals.
    """

    def __init__(self, dimensions, matrix):
        self._dimensions = dimensions
        self._matrix = matrix

    @lazyproperty
    def bottom_columns(self):
        return tuple(
            columns for columns in self._inserted_columns if columns.anchor == "bottom"
        )

    @lazyproperty
    def bottom_rows(self):
        return tuple(row for row in self._rows if row.anchor == "bottom")

    @lazyproperty
    def columns(self):
        return tuple(
            column
            for column in self._inserted_columns
            if column.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def rows(self):
        return tuple(row for row in self._rows if row.anchor not in ("top", "bottom"))

    @lazyproperty
    def top_columns(self):
        return tuple(
            columns for columns in self._inserted_columns if columns.anchor == "top"
        )

    @lazyproperty
    def top_rows(self):
        return tuple(row for row in self._rows if row.anchor == "top")

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _inserted_columns(self):
        """Sequence of _InsertionColumn objects representing subtotal columns."""
        # ---a 1D slice (strand) can have no inserted columns---
        if len(self._dimensions) < 2:
            return ()
        # ---an aggregate columns-dimension is not summable---
        if self._column_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        return tuple(
            _InsertionColumn(self._matrix, subtotal)
            for subtotal in self._column_dimension.subtotals
        )

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows(self):
        if self._row_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            _InsertionRow(self._matrix, subtotal)
            for subtotal in self._row_dimension.subtotals
        )


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
        return np.array(self._columns_dimension.valid_display_order, dtype=int)

    @lazyproperty
    def row_order(self):
        return np.array(self._rows_dimension.valid_display_order, dtype=int)

    @lazyproperty
    def _columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _rows_dimension(self):
        return self._dimensions[0]


class _Transforms(object):
    """Container for the transforms."""

    def __init__(self, matrix, dimensions, pruning=None):
        self._matrix = matrix
        self._dimensions = dimensions
        self._pruning = pruning

    @lazyproperty
    def insertions(self):
        return _Insertions(self._dimensions, self._matrix)

    @lazyproperty
    def ordering(self):
        return _OrderTransform(self._dimensions)

    @lazyproperty
    def pruning(self):
        return self._pruning
