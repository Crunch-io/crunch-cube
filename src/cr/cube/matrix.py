# encoding: utf-8

"""The matrix objects used by frozen-slice.

A matrix object has rows and (usually) columns.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats.contingency import expected_freq

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty
from cr.cube.vector import (
    AssembledVector,
    CategoricalVector,
    CatXMrVector,
    InsertionColumn,
    InsertionRow,
    MeansVector,
    MeansWithMrVector,
    MultipleResponseVector,
    OrderedVector,
    VectorAfterHiding,
)


class TransformedMatrix(object):
    """Matrix reflection application of all transforms."""

    def __init__(self, base_matrix):
        self._base_matrix = base_matrix

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
        matrix = _OrderedMatrix(self._base_matrix)
        matrix = _MatrixWithInsertions(matrix)
        matrix = _MatrixWithHidden(matrix)
        return matrix


# === TRANSFORMATION-MATRIX OBJECTS ===


class _OrderedMatrix(object):
    """Matrix reflecting result of element-ordering transforms."""

    def __init__(self, base_matrix):
        self._base_matrix = base_matrix

    @lazyproperty
    def columns(self):
        return tuple(
            OrderedVector(column, self._row_order)
            for column in tuple(np.array(self._base_matrix.columns)[self._column_order])
        )

    @lazyproperty
    def columns_dimension(self):
        return self._base_matrix.columns_dimension

    @lazyproperty
    def rows(self):
        return tuple(
            OrderedVector(row, self._column_order)
            for row in tuple(np.array(self._base_matrix.rows)[self._row_order])
        )

    @lazyproperty
    def rows_dimension(self):
        return self._base_matrix.rows_dimension

    @lazyproperty
    def table_base(self):
        return self._base_matrix.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_matrix.table_margin

    @lazyproperty
    def _column_order(self):
        """Indexer value identifying columns in order, suitable for slicing an ndarray.

        This value is a 1D ndarray of int column indices, suitable for indexing the
        columns array to produce an ordered version.
        """
        # ---Specifying int type prevents failure when there are zero columns. The
        # ---default type for ndarray is float, which is not valid for indexing.
        return np.array(self.columns_dimension.display_order, dtype=int)

    @lazyproperty
    def _row_order(self):
        """Indexer value identifying rows in order, suitable for slicing an ndarray.

        This value is a 1D ndarray of int row indices, suitable for indexing the rows
        array to produce an ordered version.
        """
        # ---Specifying int type prevents failure when there are zero rows---
        return np.array(self.rows_dimension.display_order, dtype=int)


class _MatrixWithHidden(object):
    """Matrix with hidden vectors removed.

    A vector can be hidden explicitly by the user, or it can be automatically hidden
    when it is empty and the prune option for the dimension is selected.
    """

    # ---Note that hiding a vector requires not just removing that vector, but also
    # ---the element the removed vector contributes to each of the *opposing* vectors.
    # ---For example, hiding a row is removing that row-vector from `.rows`, but also
    # ---removing an element from each column-vector in `.columns`.

    def __init__(self, base_matrix):
        self._base_matrix = base_matrix

    @lazyproperty
    def columns(self):
        return tuple(
            VectorAfterHiding(column, self._base_matrix.rows)
            for column in self._base_matrix.columns
            if not column.hidden
        )

    @lazyproperty
    def rows(self):
        return tuple(
            VectorAfterHiding(row, self._base_matrix.columns)
            for row in self._base_matrix.rows
            if not row.hidden
        )

    @lazyproperty
    def table_base(self):
        margin = self._base_matrix.table_base
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]

    @lazyproperty
    def table_base_unpruned(self):
        return self._base_matrix.table_base

    @lazyproperty
    def table_margin(self):
        margin = self._base_matrix.table_margin
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]

    @lazyproperty
    def table_margin_unpruned(self):
        return self._base_matrix.table_margin


class _MatrixWithInsertions(object):
    """Represents slice with both normal and inserted bits."""

    def __init__(self, base_matrix):
        self._base_matrix = base_matrix

    @lazyproperty
    def columns(self):
        """Sequence of column vectors including inserted columns.

        Each column vector also includes any new elements introduced by inserted rows.
        """
        return tuple(self._iter_columns())

    @lazyproperty
    def rows(self):
        """Sequence of row vectors including inserted rows.

        Each row vector also reflects any new elements introduced by inserted columns.
        """
        return tuple(self._iter_rows())

    @lazyproperty
    def table_base(self):
        return self._base_matrix.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_matrix.table_margin

    @lazyproperty
    def _all_inserted_columns(self):
        """Sequence of _InsertionColumn objects representing subtotal columns.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order. All subtotals defined on the column
        dimension appear in the sequence.
        """
        # ---an aggregate columns-dimension is not summable---
        if self._columns_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        return tuple(
            InsertionColumn(self._base_matrix, subtotal)
            for subtotal in self._columns_dimension.subtotals
        )

    @lazyproperty
    def _all_inserted_rows(self):
        """Sequence of _InsertionRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # ---an aggregate rows-dimension is not summable---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            InsertionRow(self._base_matrix, subtotal)
            for subtotal in self._rows_dimension.subtotals
        )

    @lazyproperty
    def _columns_dimension(self):
        return self._base_matrix.columns_dimension

    @lazyproperty
    def _columns_inserted_at_left(self):
        """Sequence of InsertionColumn vectors that appear before any body columns."""
        return tuple(
            column for column in self._all_inserted_columns if column.anchor == "top"
        )

    @lazyproperty
    def _columns_inserted_at_right(self):
        """Sequence of InsertionColumn vectors appended as the last table columns."""
        return tuple(
            column for column in self._all_inserted_columns if column.anchor == "bottom"
        )

    def _iter_columns(self):
        """Generate all column vectors with insertions interleaved at right spot."""
        opposing_insertions = self._all_inserted_rows

        # ---subtotals inserted at top---
        for column in self._columns_inserted_at_left:
            yield AssembledVector(column, opposing_insertions)

        # ---body columns with subtotals anchored to specific body positions---
        for idx, column in enumerate(self._base_matrix.columns):
            yield AssembledVector(column, opposing_insertions)
            for inserted_column in self._iter_inserted_columns_anchored_at(idx):
                yield AssembledVector(inserted_column, opposing_insertions)

        # ---subtotals appended at bottom---
        for column in self._columns_inserted_at_right:
            yield AssembledVector(column, opposing_insertions)

    def _iter_inserted_rows_anchored_at(self, anchor):
        """Generate all inserted row vectors with matching `anchor`."""
        return (row for row in self._all_inserted_rows if row.anchor == anchor)

    def _iter_rows(self):
        """Generate all row vectors with insertions interleaved at right spot."""
        opposing_insertions = self._all_inserted_columns

        # ---subtotals inserted at top---
        for row in self._rows_inserted_at_top:
            yield AssembledVector(row, opposing_insertions)

        # ---body rows with subtotals anchored to specific body positions---
        for idx, row in enumerate(self._base_matrix.rows):
            yield AssembledVector(row, opposing_insertions)
            for inserted_row in self._iter_inserted_rows_anchored_at(idx):
                yield AssembledVector(inserted_row, opposing_insertions)

        # ---subtotals appended at bottom---
        for row in self._rows_inserted_at_bottom:
            yield AssembledVector(row, opposing_insertions)

    def _iter_inserted_columns_anchored_at(self, anchor):
        """Generate all inserted column vectors with matching `anchor`."""
        return (
            column for column in self._all_inserted_columns if column.anchor == anchor
        )

    @lazyproperty
    def _rows_dimension(self):
        return self._base_matrix.rows_dimension

    @lazyproperty
    def _rows_inserted_at_bottom(self):
        """Sequence of InsertionRow vectors that appear after any other table rows."""
        return tuple(row for row in self._all_inserted_rows if row.anchor == "bottom")

    @lazyproperty
    def _rows_inserted_at_top(self):
        """Sequence of InsertionRow vectors that appear before any other table rows."""
        return tuple(row for row in self._all_inserted_rows if row.anchor == "top")


# === BASE-MATRIX OBJECTS ===


class MatrixFactory(object):
    """Encapsulates creation of the right raw (pre-transforms) matrix object."""

    @classmethod
    def matrix(
        cls, dimensions, counts, base_counts, counts_with_missings, cube, slice_idx
    ):
        """Return a matrix object of appropriate type based on parameters."""

        # For cubes with means, create one of the means-matrix types
        if cube.has_means:
            return cls._create_means_matrix(
                counts, base_counts, cube, dimensions, slice_idx
            )

        dimension_types = cube.dimension_types[-2:]
        if cube.ndim > 2:
            base_counts = base_counts[slice_idx]
            counts = counts[slice_idx]
            counts_with_missings = counts_with_missings[slice_idx]
            if cube.dimension_types[0] == DT.MR:
                base_counts = base_counts[0]
                counts = counts[0]
                counts_with_missings = counts_with_missings[0]
        if dimension_types == (DT.MR, DT.MR):
            return _MrXMrMatrix(dimensions, counts, base_counts, counts_with_missings)
        elif dimension_types[0] == DT.MR:
            return _MrXCatMatrix(dimensions, counts, base_counts, counts_with_missings)
        elif dimension_types[1] == DT.MR:
            return _CatXMrMatrix(dimensions, counts, base_counts, counts_with_missings)
        return _CatXCatMatrix(dimensions, counts, base_counts, counts_with_missings)

    @classmethod
    def _create_means_matrix(cls, counts, base_counts, cube, dimensions, slice_idx):
        if cube.ndim == 3:
            base_counts = base_counts[slice_idx]
            counts = counts[slice_idx]
        if dimensions[0].dimension_type == DT.MR:
            return _MrXCatMeansMatrix(dimensions, counts, base_counts)
        return _CatXCatMeansMatrix(dimensions, counts, base_counts)


class _BaseMatrix(object):
    """Base class for all matrix (2D secondary-analyzer) objects."""

    def __init__(self, dimensions, counts, base_counts):
        self._dimensions = dimensions
        self._counts = counts
        self._base_counts = base_counts

    @lazyproperty
    def columns_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def ndim(self):
        """int count of dimensions in this matrix, unconditionally 2.

        A matrix is by definition two-dimensional.
        """
        return 2

    @lazyproperty
    def rows_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_elements(self):
        return self.columns_dimension.valid_elements

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T, self._base_counts.T, self._column_elements, self._zscores.T
        )

    @lazyproperty
    def _column_proportions(self):
        return np.array([col.proportions for col in self.columns]).T

    @lazyproperty
    def _row_elements(self):
        return self.rows_dimension.valid_elements

    @lazyproperty
    def _valid_rows_idxs(self):
        """ndarray-style index for only valid rows (out of missing and not-missing)."""
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)


class _CatXCatMatrix(_BaseMatrix):
    """Deals with CAT x CAT data.

    Delegates most functionality to vectors (rows or columns), but calculates some
    values by itself (like table_margin).

    This class (or its inheritants) must be instantiated as a starting point when
    dealing with slices. Other classes that represents various stages of
    transformations, need to repro a portion of this class' API (like iterating over
    rows or columns).
    """

    def __init__(self, dimensions, counts, base_counts, counts_with_missings=None):
        super(_CatXCatMatrix, self).__init__(dimensions, counts, base_counts)
        self._all_counts = counts_with_missings

    @lazyproperty
    def columns(self):
        return tuple(
            CategoricalVector(counts, base_counts, element, self.table_margin, zscore)
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            CategoricalVector(
                counts, base_counts, element, self.table_margin, zscore, column_index
            )
            for (
                counts,
                base_counts,
                element,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For CAT x CAT slice it needs to
        sum across axis 1, because that's the columns CAT dimension, that gets
        collapsed. The total is calculated by totaling the unconditional row margin.
        Please note that the total _doesn't_ include missings for the row dimension.
        """
        dim_sum = np.sum(self._all_counts, axis=1)[self._valid_rows_idxs]
        return dim_sum[:, None] / np.sum(dim_sum)

    @lazyproperty
    def _column_index(self):
        return self._column_proportions / self._baseline * 100

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self._zscores,
            self._column_index,
        )

    @staticmethod
    def _scalar_type_std_res(counts, total, colsum, rowsum):
        """Return ndarray containing standard residuals for category values.

        The shape of the return value is the same as that of *counts*.
        """
        expected_counts = expected_freq(counts)
        residuals = counts - expected_counts
        variance = (
            np.outer(rowsum, colsum)
            * np.outer(total - rowsum, total - colsum)
            / total ** 3
        )
        return residuals / np.sqrt(variance)

    @lazyproperty
    def _zscores(self):
        return self._scalar_type_std_res(
            self._counts,
            self.table_margin,
            np.sum(self._counts, axis=0),
            np.sum(self._counts, axis=1),
        )


class _CatXCatMeansMatrix(_CatXCatMatrix):
    """Cat-x-cat matrix for means measure."""

    def __init__(self, dimensions, means, base_counts):
        super(_CatXCatMeansMatrix, self).__init__(dimensions, None, base_counts)
        self._means = means

    @lazyproperty
    def columns(self):
        return tuple(
            MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self.columns_dimension.valid_elements,
                self._base_counts.T,
                self._means.T,
            )
        )

    @lazyproperty
    def rows(self):
        return tuple(
            MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self.rows_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _MatrixWithMR(_CatXCatMatrix):
    """ ... """

    @staticmethod
    def _array_type_std_res(counts, total, colsum, rowsum):
        expected_counts = rowsum * colsum / total
        # TODO: this line occasionally raises overflow warnings in the tests.
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)


class _MrXCatMatrix(_MatrixWithMR):
    """Represents MR x CAT slices.

    It's similar to CAT x CAT, other than the way it handles columns. For
    columns - which correspond to the MR dimension - it needs to handle the indexing
    of selected/not-selected correctly.
    """

    @lazyproperty
    def columns(self):
        """Use bother selected and not-selected counts."""
        return tuple(
            MultipleResponseVector(
                counts, base_counts, element, self.table_margin, zscore
            )
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(
            CatXMrVector(
                counts, base_counts, element, table_margin, zscore, column_index
            )
            for (
                counts,
                base_counts,
                element,
                table_margin,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(1, 2))

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 2))

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For MR x CAT slice it needs to
        sum across axis 2, because that's the CAT dimension, that gets collapsed.
        Then it needs to select only the selected counts of
        the MR (hence the `[:, 0]` index). The total needs include missings of the
        2nd dimension, but not of the first (hence the [:, 0:2] index, which only
        includes the selected and not-selected of the MR dimension).
        """
        dim_sum = np.sum(self._all_counts, axis=2)[:, 0][self._valid_rows_idxs]
        total = np.sum(self._all_counts[self._valid_rows_idxs][:, 0:2], axis=(1, 2))
        return (dim_sum / total)[:, None]

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._column_index,
        )

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, 0, :],
            self.table_margin[:, None],
            np.sum(self._counts, axis=1),
            np.sum(self._counts[:, 0, :], axis=1)[:, None],
        )


class _MrXCatMeansMatrix(_MrXCatMatrix):
    """ ... """

    def __init__(self, dimensions, means, base_counts):
        counts = np.empty(means.shape)
        super(_MrXCatMeansMatrix, self).__init__(dimensions, counts, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            MeansWithMrVector(element, base_counts, means[0])
            for element, base_counts, means in zip(
                self.rows_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _CatXMrMatrix(_MatrixWithMR):
    """Handles CAT x MR slices.

    Needs to handle correctly the indexing for the selected/not-selected for rows
    (which correspond to the MR dimension).
    """

    @lazyproperty
    def columns(self):
        return tuple(
            CatXMrVector(counts.T, base_counts.T, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            MultipleResponseVector(
                counts.T,
                base_counts.T,
                element,
                self.table_margin,
                zscore,
                column_index,
            )
            for (
                counts,
                base_counts,
                element,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(0, 2))

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(0, 2))

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For CAT x MR slice it needs to sum
        across axis 2, because that's the MR_CAT dimension, that gets collapsed
        (but the MR subvars don't get collapsed). Then it needs to calculate the total,
        which is easily obtained by summing across the CAT dimension (hence `axis=0`).
        """
        dim_sum = np.sum(self._all_counts, axis=2)[self._valid_rows_idxs]
        return dim_sum / np.sum(dim_sum, axis=0)

    @lazyproperty
    def _column_generator(self):
        return zip(
            # self._counts.T[0],
            np.array([self._counts.T[0].T, self._counts.T[1].T]).T,
            # self._base_counts.T[0],
            np.array([self._base_counts.T[0].T, self._base_counts.T[1].T]).T,
            self._column_elements,
            self.table_margin,
        )

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, :, 0],
            self.table_margin,
            np.sum(self._counts[:, :, 0], axis=0),
            np.sum(self._counts, axis=2),
        )


class _MrXMrMatrix(_MatrixWithMR):
    """Represents MR x MR slices.

    Needs to properly index both rows and columns (selected/not-selected).
    """

    @lazyproperty
    def columns(self):
        return tuple(
            MultipleResponseVector(counts, base_counts, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            MultipleResponseVector(
                counts[0].T,
                base_counts[0].T,
                element,
                table_margin,
                zscore,
                column_index,
            )
            for (
                counts,
                base_counts,
                element,
                table_margin,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(1, 3))

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 3))

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For MR x MR slice it needs to sum
        across axis 3, because that's the MR_CAT dimension, that gets collapsed
        (but the MR subvars don't get collapsed). Then it needs to calculate the total,
        which is obtained by summing across both MR_CAT dimensions. However, please
        note, that in calculating the unconditional total, missing elements need to be
        included for the column dimension, while they need to be _excluded_ for the row
        dimension. Hence the `[:, 0:2]` indexing for the first MR_CAT, but not the 2nd.
        """
        dim_sum = np.sum(self._all_counts[:, 0:2], axis=3)[self._valid_rows_idxs][:, 0]
        total = np.sum(self._all_counts[:, 0:2], axis=(1, 3))[self._valid_rows_idxs]
        return dim_sum / total

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T[0],
            self._base_counts.T[0],
            self._column_elements,
            self.table_margin.T,
        )

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._column_index,
        )

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, 0, :, 0],
            self.table_margin,
            np.sum(self._counts, axis=1)[:, :, 0],
            np.sum(self._counts, axis=3)[:, 0, :],
        )
