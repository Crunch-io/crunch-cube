# encoding: utf-8

"""A matrix is the 2D cube-data partition used by a slice.

A matrix object has rows and columns.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import sys

import numpy as np
from scipy.stats import norm
from scipy.stats.contingency import expected_freq

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class TransformedMatrix(object):
    """Matrix reflecting application of ordering, insertion, and hiding transforms."""

    def __init__(self, unordered_matrix):
        self._unordered_matrix = unordered_matrix

    @classmethod
    def matrix(cls, cube, dimensions, slice_idx):
        """Return a TransformedMatrix object constructed from this cube result."""
        return cls(_BaseBaseMatrix.factory(cube, dimensions, slice_idx))

    @lazyproperty
    def columns(self):
        return tuple(
            _VectorAfterHiding(column, self._assembled_rows)
            for column in self._assembled_columns
            if not column.hidden
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _VectorAfterHiding(row, self._assembled_columns)
            for row in self._assembled_rows
            if not row.hidden
        )

    @lazyproperty
    def table_base(self):
        return self.table_base_unpruned[
            np.ix_(self._visible_rows_mask, self._visible_cols_mask)
        ]

    @lazyproperty
    def table_base_unpruned(self):
        return self._unordered_matrix.table_base

    @lazyproperty
    def table_margin(self):
        return self.table_margin_unpruned[
            np.ix_(self._visible_rows_mask, self._visible_cols_mask)
        ]

    @lazyproperty
    def table_margin_unpruned(self):
        return self._unordered_matrix.table_margin

    @lazyproperty
    def _assembled_columns(self):
        """Sequence of column vectors including inserted columns.

        Each column vector also includes any new elements introduced by inserted rows.
        """
        opposing_insertions = self._inserted_rows

        return tuple(
            _AssembledVector(column, opposing_insertions, 0 if idx < 0 else idx)
            for _, idx, column in sorted(
                itertools.chain(
                    (column.ordering for column in self._inserted_columns),
                    (column.ordering for column in self._base_columns),
                )
            )
        )

    @lazyproperty
    def _assembled_rows(self):
        """Sequence of row vectors including inserted rows.

        Each row vector also reflects any new elements introduced by inserted columns.
        """
        opposing_insertions = self._inserted_columns

        return tuple(
            _AssembledVector(row, opposing_insertions, 0 if idx < 0 else idx)
            for _, idx, row in sorted(
                itertools.chain(
                    (row.ordering for row in self._inserted_rows),
                    (row.ordering for row in self._base_rows),
                )
            )
        )

    @lazyproperty
    def _base_columns(self):
        """columns after ordering but prior to insertions"""
        return tuple(
            _OrderedVector(column, self._row_order, idx)
            for idx, column in enumerate(
                tuple(np.array(self._unordered_matrix.columns)[self._column_order])
            )
        )

    @lazyproperty
    def _base_rows(self):
        """rows after ordering but prior to insertions"""
        return tuple(
            _OrderedVector(row, self._column_order, idx)
            for idx, row in enumerate(
                tuple(np.array(self._unordered_matrix.rows)[self._row_order])
            )
        )

    @lazyproperty
    def _column_order(self):
        """ -> 1D ndarray of int col idx specifying order of unordered-array columns."""
        # ---Specifying int type prevents failure when there are zero columns. The
        # ---default type for ndarray is float, which is not valid for indexing.
        return np.array(self._columns_dimension.display_order, dtype=int)

    @lazyproperty
    def _columns_dimension(self):
        return self._unordered_matrix.columns_dimension

    @lazyproperty
    def _inserted_columns(self):
        """ -> tuple of _InsertionColumn objects representing subtotal columns.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order. All subtotals defined on the column
        dimension appear in the sequence.
        """
        # ---an aggregate columns-dimension is not summable---
        if self._columns_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        subtotals = self._columns_dimension.subtotals
        # ---insertions are indexed using their *negative* idx, i.e. their "distance"
        # ---from the end of the subtotals sequence. This insures their ordering tuple
        # ---sorts before all base-columns with the same position while still providing
        # ---an idx that works for indexed access (if required).
        neg_idxs = range(-len(subtotals), 0)  # ---like [-3, -2, -1]---
        table_margin = self._unordered_matrix.table_margin
        base_rows = self._base_rows
        base_cols = self._base_columns

        return tuple(
            _InsertionColumn(subtotal, neg_idx, table_margin, base_rows, base_cols)
            for subtotal, neg_idx in zip(subtotals, neg_idxs)
        )

    @lazyproperty
    def _inserted_rows(self):
        """ -> tuple of _InsertionRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # ---an aggregate rows-dimension is not summable---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        subtotals = self._rows_dimension.subtotals
        neg_idxs = range(-len(subtotals), 0)  # ---like [-3, -2, -1]---
        table_margin = self._unordered_matrix.table_margin
        base_rows = self._base_rows
        base_cols = self._base_columns

        return tuple(
            _InsertionRow(subtotal, neg_idx, table_margin, base_rows, base_cols)
            for subtotal, neg_idx in zip(subtotals, neg_idxs)
        )

    @lazyproperty
    def _row_order(self):
        """ -> 1D ndarray of int row idx specifying order of unordered-array rows."""
        # ---Specifying int type prevents failure when there are zero rows---
        return np.array(self._rows_dimension.display_order, dtype=int)

    @lazyproperty
    def _rows_dimension(self):
        return self._unordered_matrix.rows_dimension

    @lazyproperty
    def _visible_cols_mask(self):
        return [not col.hidden for col in self._assembled_columns]

    @lazyproperty
    def _visible_rows_mask(self):
        return [not row.hidden for row in self._assembled_rows]


# === BASE-MATRIX OBJECTS ===


class _BaseBaseMatrix(object):
    """Base class for all matrix (2D secondary-analyzer) objects."""

    def __init__(self, dimensions, counts, base_counts):
        self._dimensions = dimensions
        self._counts = counts
        self._base_counts = base_counts

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return a base-matrix object of appropriate type for `cube`."""
        counts = cube.counts
        base_counts = cube.base_counts
        counts_with_missings = cube.counts_with_missings
        dimension_types = cube.dimension_types[-2:]

        # For cubes with means, create one of the means-matrix types
        if cube.has_means:
            if cube.ndim == 3:
                base_counts = base_counts[slice_idx]
                counts = counts[slice_idx]
            if dimension_types == (DT.MR, DT.MR):
                # TODO: Potentially address this case, which didn't arise yet
                raise NotImplementedError(
                    "MR x MR with means is not implemented."
                )  # pragma: no cover
            if dimension_types[1] == DT.MR:
                return _CatXMrMeansMatrix(dimensions, counts, base_counts)
            if dimensions[0].dimension_type == DT.MR:
                return _MrXCatMeansMatrix(dimensions, counts, base_counts)
            return _CatXCatMeansMatrix(dimensions, counts, base_counts)

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

    @lazyproperty
    def columns_dimension(self):
        return self._dimensions[1]

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


class _CatXCatMatrix(_BaseBaseMatrix):
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
            _CategoricalVector(
                counts,
                base_counts,
                element,
                self.table_margin,
                zscore,
                opposite_margins=self._row_margins,
            )
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _CategoricalVector(
                counts,
                base_counts,
                element,
                self.table_margin,
                zscore,
                column_index,
                opposite_margins=self._column_margins,
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
    def _column_margins(self):
        return np.sum(self._counts, axis=0)

    @lazyproperty
    def _row_margins(self):
        return np.sum(self._counts, axis=1)

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
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            # If the matrix is "defective", in the sense that it doesn't have at least
            # two rows and two columns that are "full" of data, don't calculate zscores.
            return np.zeros(counts.shape)

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
            _MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self.columns_dimension.valid_elements,
                self._base_counts.T,
                self._means.T,
            )
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self.rows_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _MatrixWithMR(_CatXCatMatrix):
    """ ... """

    @staticmethod
    def _array_type_std_res(counts, total, colsum, rowsum):
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            # If the matrix is "defective", in the sense that it doesn't have at least
            # two rows and two columns that are "full" of data, don't calculate zscores.
            return np.zeros(counts.shape)

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
            _MultipleResponseVector(
                counts, base_counts, element, self.table_margin, zscore
            )
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(
            _CatXMrVector(
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
        counts = np.zeros(means.shape)
        super(_MrXCatMeansMatrix, self).__init__(dimensions, counts, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansWithMrVector(element, base_counts, means[0])
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
            _CatXMrVector(counts.T, base_counts.T, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _MultipleResponseVector(
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


class _CatXMrMeansMatrix(_CatXMrMatrix):
    def __init__(self, dimensions, means, base_counts):
        counts = np.zeros(means.shape)
        super(_CatXMrMeansMatrix, self).__init__(dimensions, counts, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        """rows of CAT x MR matrix."""

        return tuple(
            # We must inflate the means with [:, 0], because the values are oriented
            # like columns (0th is selected while 1st is other)
            _MeansWithMrVector(element, base_counts, means[:, 0])
            for element, base_counts, means in zip(
                self.rows_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _MrXMrMatrix(_MatrixWithMR):
    """Represents MR x MR slices.

    Needs to properly index both rows and columns (selected/not-selected).
    """

    @lazyproperty
    def columns(self):
        return tuple(
            _MultipleResponseVector(counts, base_counts, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _MultipleResponseVector(
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


# ===INSERTION (SUBTOTAL) VECTORS===


class _BaseMatrixInsertionVector(object):
    """Base class for matrix insertion vectors.

    There are some differences that arise when there are rows *and* columns, which
    entails the complication of insertion *intersections*.
    """

    def __init__(self, subtotal, neg_idx, table_margin, base_rows, base_columns):
        self._subtotal = subtotal
        # ---the *negative* idx of this vector among its peer insertions---
        self._neg_idx = neg_idx
        self._table_margin = table_margin
        self._base_rows = base_rows
        self._base_columns = base_columns

    @lazyproperty
    def addend_idxs(self):
        return np.array(self._subtotal.addend_idxs)

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def base(self):
        return np.sum(np.array([vec.base for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def base_values(self):
        return np.sum(
            np.array([row.base_values for row in self._addend_vectors]), axis=0
        )

    @lazyproperty
    def column_index(self):
        # TODO: Calculate insertion column index for real. Check with Mike
        return np.array([np.nan] * len(self.values))

    @lazyproperty
    def fill(self):
        """Unconditionally `None` for an insertion vector.

        A `fill` value is normally a str RGB value like "#da09fc", specifying the color
        to use for a chart category or series representing this element. The value
        reflects the resolved element-fill transform cascade. Since an insertion cannot
        (currently) have a fill-transform, the default value of `None` (indicating "use
        default color") is unconditionally returned.
        """
        return None

    @lazyproperty
    def hidden(self):
        """True if vector is pruned.

        Insertions can never be hidden explicitly (for now). They can also almost never
        be pruned, except in the case when all of the opposite vectors are also pruned
        (thus leaving no elements for this insertion vector).
        """
        return self.pruned

    @lazyproperty
    def is_insertion(self):
        return True

    @lazyproperty
    def label(self):
        return self._subtotal.label

    @lazyproperty
    def margin(self):
        return np.sum(np.array([vec.margin for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def means(self):
        """ndarray of NaN values, of the same shape as values.
        Insertions are not defined for means, this is just a placeholder.
        """
        return np.full(self.values.shape, np.nan)

    @lazyproperty
    def numeric(self):
        return np.nan

    @lazyproperty
    def ordering(self):
        """ -> (position, index, self) tuple used for interleaving with base vectors.

        This value allows the interleaving of insertions with base vectors to be reduced
        to a sorting operation.

        The int position value is roughly equivalent to the notion of "anchor". It is
        0 for anchor=="top", sys.maxsize for anchor=="bottom", and int(anchor) + 1
        otherwise. The +1 ensures insertions appear *after* the vector they are anchored
        to.

        The `index` value is the *negative* index of this subtotal in its collection
        (i.e. the "distance from the end" of this insertion). This ensures that an
        insertion will always sort *prior* to a base vector with the same position while
        preserving the payload order of the insertion when two or more are anchored to
        the same vector.
        """
        return (self._anchor_n, self._neg_idx, self)

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)

    @lazyproperty
    def zscore(self):
        return self._zscore

    @lazyproperty
    def _anchor_n(self):
        """Anchor expressed as an int.

        An anchor to a column is incremented by 1 to ensure it applies *after* the
        anchor column rather than before. See `.ordering` for more.
        """
        anchor = self.anchor
        return (
            0
            if anchor == "top"
            else sys.maxsize
            if anchor == "bottom"
            else int(self.anchor) + 1
        )

    @lazyproperty
    def _zscore(self):
        # TODO: remove this if statement - temporary hack until MR zscore implementation
        if self.opposite_margins is None:
            return tuple([np.nan] * len(self.values))
        variance = (
            self.opposite_margins
            * self.margin
            * (
                (self.table_margin - self.opposite_margins)
                * (self.table_margin - self.margin)
            )
            / self.table_margin ** 3
        )
        return self._residuals / np.sqrt(variance)


class _InsertionColumn(_BaseMatrixInsertionVector):
    """Represents an inserted (subtotal) column."""

    @lazyproperty
    def opposite_margins(self):
        return self._addend_vectors[0].opposite_margins

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        Insertions can almost never be pruned, except in the case when all of the
        opposite vectors are also pruned (thus leaving no elements for this
        insertion vector).
        """
        return self._subtotal.prune and not np.any(
            np.array([row.base for row in self._base_rows])
        )

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            column
            for i, column in enumerate(self._base_columns)
            if i in self._subtotal.addend_idxs
        )

    @lazyproperty
    def _residuals(self):
        return self.values - self._expected_counts

    @lazyproperty
    def _expected_counts(self):
        return self.opposite_margins * self.margin / self.table_margin


class _InsertionRow(_BaseMatrixInsertionVector):
    """Represents an inserted (subtotal) row."""

    @lazyproperty
    def opposite_margins(self):
        return self._addend_vectors[0].opposite_margins

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        Insertions can almost never be pruned, except in the case when all of the
        opposite vectors are also pruned (thus leaving no elements for this
        insertion vector).
        """
        return self._subtotal.prune and not np.any(
            np.array([column.base for column in self._base_columns])
        )

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._base_rows)
            if i in self._subtotal.addend_idxs
        )

    @lazyproperty
    def _residuals(self):
        return self.values - self._expected_counts

    @lazyproperty
    def _expected_counts(self):
        return self.opposite_margins * self.margin / self.table_margin


# ===TRANSFORMATION VECTORS===


class _BaseTransformationVector(object):
    """Base class for most transformation vectors."""

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._base_vector.fill

    @lazyproperty
    def hidden(self):
        return self._base_vector.hidden

    @lazyproperty
    def is_insertion(self):
        return self._base_vector.is_insertion

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def means(self):
        return self._base_vector.means

    @lazyproperty
    def numeric(self):
        return self._base_vector.numeric

    @lazyproperty
    def opposite_margins(self):
        return self._base_vector.opposite_margins

    @lazyproperty
    def table_base(self):
        return self._base_vector.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_vector.table_margin


class _AssembledVector(_BaseTransformationVector):
    """Vector with base, as well as inserted, elements (of the opposite dimension)."""

    def __init__(self, base_vector, opposite_inserted_vectors, vector_idx=0):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors
        self._vector_idx = vector_idx

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        base_values = self._base_vector.base_values

        def fsubtot(subtotal):
            return np.sum(base_values[subtotal.addend_idxs])

        return self._apply_interleaved(base_values, fsubtot)

    @lazyproperty
    def column_index(self):
        def fsubtot(subtotal):
            # TODO: Replace with real column index values from insertions vectors. This
            # should be something like:
            #   col_ind = (ins1.prop + ins2.prop) / (ins1.baseline + ins2.baseline)
            # ask @mike to confirm
            return np.nan

        return self._apply_interleaved(self._base_vector.column_index, fsubtot)

    @lazyproperty
    def means(self):
        # ---just np.nan for insertions for now---
        return self._apply_interleaved(
            self._base_vector.means, fsubtot=lambda _: np.nan
        )

    @lazyproperty
    def proportions(self):
        return self.values / self.margin

    @lazyproperty
    def pvals(self):
        return 2 * (1 - norm.cdf(np.abs(self.zscore)))

    @lazyproperty
    def table_proportions(self):
        return self.values / self._base_vector.table_margin

    @lazyproperty
    def values(self):
        values = self._base_vector.values

        def fsubtot(subtotal):
            return np.sum(values[subtotal.addend_idxs])

        return self._apply_interleaved(values, fsubtot)

    @lazyproperty
    def zscore(self):
        def fsubtot(subtotal):
            if self.is_insertion:
                opposite_margin = np.sum(self.opposite_margins[subtotal.addend_idxs])
                variance = (
                    opposite_margin
                    * self.margin
                    * (
                        (self.table_margin - opposite_margin)
                        * (self.table_margin - self.margin)
                    )
                    / self.table_margin ** 3
                )
                expected_count = opposite_margin * self.margin / self.table_margin
                cell_value = np.sum(self._base_vector.values[subtotal.addend_idxs])
                residuals = cell_value - expected_count
                zscore = residuals / np.sqrt(variance)

            else:
                zscore = subtotal.zscore[self._vector_idx]

            return zscore

        return self._apply_interleaved(self._base_vector.zscore, fsubtot)

    def _apply_interleaved(self, base_values, fsubtot):
        """ -> 1D array of result of applying fbase or fsubtot to each interleaved item.

        `base_values` is the "unassembled" vector measure values.

        `fsubtot(subtot)` :: subtotal -> intersection_value

        Takes care of the details of getting vector "cells" interleaved in the right
        order, you just provide the "unassembled" values and a function to apply to each
        subtotal to get its value.
        """
        subtotals = self._opposite_inserted_vectors

        return np.array(
            tuple(
                fsubtot(subtotals[idx]) if idx < 0 else base_values[idx]
                for idx in self._interleaved_idxs
            )
        )

    @lazyproperty
    def _interleaved_idxs(self):
        """ -> tuple of int: idx for base and inserted values, in display order.

        Inserted value indicies are negative, to distinguish them from base vector
        indices. The indexes are interleaved simply by sorting their orderings. An
        ordering is a (position, idx) pair.
        """

        def iter_insertion_orderings():
            """Generate (int: position, int: idx) for each opposing insertion.

            The position for an insertion is an int representation of its anchor and its
            idx is the *negative* offset of its position in the opposing insertions
            sequence (like -3, -2, -1 for a sequence of length 3). The negative idx
            works just as well as the normal one for accessing the subtotal but insures
            that an insertion at the same position as a base row always sorts *before*
            the base row.

            The `position` int for a subtotal is 0 for anchor "top", sys.maxsize for
            anchor "bottom", and int(anchor) + 1 for all others. The +1 ensures
            a subtotal appears *after* the vector it is anchored to.
            """
            for v in self._opposite_inserted_vectors:
                pos, idx, _ = v.ordering
                yield pos, idx

        def iter_base_value_orderings():
            """Generate (int: position, int: idx) for each base value.

            The position of a base value is simply it's index in the vector.
            """
            for idx in range(len(self._base_vector.values)):
                yield idx, idx

        return tuple(
            idx
            for pos, idx in sorted(
                itertools.chain(iter_insertion_orderings(), iter_base_value_orderings())
            )
        )


class _VectorAfterHiding(_BaseTransformationVector):
    """Reflects a row or column with hidden elements removed."""

    def __init__(self, base_vector, opposite_vectors):
        self._base_vector = base_vector
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def base(self):
        if not isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base
        return self._base_vector.base[self._visible_element_idxs]

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self._visible_element_idxs]

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index[self._visible_element_idxs]

    @lazyproperty
    def margin(self):
        if not isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin
        return self._base_vector.margin[self._visible_element_idxs]

    @lazyproperty
    def means(self):
        return self._base_vector.means[self._visible_element_idxs]

    @lazyproperty
    def proportions(self):
        return self._base_vector.proportions[self._visible_element_idxs]

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals[self._visible_element_idxs]

    @lazyproperty
    def table_proportions(self):
        return self._base_vector.table_proportions[self._visible_element_idxs]

    @lazyproperty
    def values(self):
        return self._base_vector.values[self._visible_element_idxs]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore[self._visible_element_idxs]

    @lazyproperty
    def _visible_element_idxs(self):
        """An 1D ndarray of int idxs of non-hidden values, suitable for indexing.

        This value is derived from the opposing vectors collection, based on the hidden
        status of its elements.
        """
        return np.array(
            [
                idx
                for idx, opposite_vector in enumerate(self._opposite_vectors)
                if not opposite_vector.hidden
            ],
            dtype=int,
        )


class _OrderedVector(_BaseTransformationVector):
    """Rearranges its "cells" to the order of its opposing-vectors.

    For example, the cells in a row appear in the order of the column vectors.
    """

    def __init__(self, base_vector, opposing_order, index):
        self._base_vector = base_vector
        self._opposing_order_arg = opposing_order
        self._index = index

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self._opposing_order]

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def ordering(self):
        """ -> (position, index, self) tuple used for interleaving with insertions.

        This value allows the interleaving of base vectors with insertions to be reduced
        to a sorting operation.

        The position and index of a base vector are both its index within its ordered
        collection.
        """
        return (self._index, self._index, self)

    @lazyproperty
    def values(self):
        return self._base_vector.values[self._opposing_order]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore

    @lazyproperty
    def _opposing_order(self):
        return (
            slice(None)
            if self._opposing_order_arg is None
            else self._opposing_order_arg
        )


# ===OPERAND VECTORS===


class _BaseVector(object):
    """Base class for all vector objects.

    A vector represents a row or column of data in the overall data matrix. It composes
    the element that corresponds to the row or column and so knows the name, element_id,
    numeric value, etc. for the row or column.
    """

    def __init__(self, element, base_counts):
        self._element = element
        self._base_counts = base_counts

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts)

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
    def numeric(self):
        return self._element.numeric_value

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def zscore(self):
        variance = (
            self.opposite_margins
            * self.margin
            * (
                (self.table_margin - self.opposite_margins)
                * (self.table_margin - self.margin)
            )
            / self.table_margin ** 3
        )

        return self._residuals / np.sqrt(variance)

    @lazyproperty
    def _residuals(self):
        return self.values - self._expected_counts

    @lazyproperty
    def _expected_counts(self):
        return self.opposite_margins * self.margin / self.table_margin


class _CategoricalVector(_BaseVector):
    """Main staple of all measures.

    Some of the measures it can calculate by itself, others it needs to receive at
    construction time (like table margin and zscores).
    """

    def __init__(
        self,
        counts,
        base_counts,
        element,
        table_margin,
        zscore=None,
        column_index=None,
        opposite_margins=None,
    ):
        super(_CategoricalVector, self).__init__(element, base_counts)
        self._counts = counts
        self._table_margin = table_margin
        self._zscore = zscore
        self._column_index = column_index
        self.opposite_margins = opposite_margins

    @lazyproperty
    def base_values(self):
        return self._base_counts

    @lazyproperty
    def column_index(self):
        return self._column_index

    @lazyproperty
    def margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def values(self):
        return self._counts


class _CatXMrVector(_CategoricalVector):
    """Used for categorical dimension when opposing dimension is multiple-response."""

    def __init__(
        self, counts, base_counts, label, table_margin, zscore=None, column_index=None
    ):
        super(_CatXMrVector, self).__init__(
            counts[0], base_counts[0], label, table_margin, zscore, column_index
        )
        self._all_bases = base_counts
        self._all_counts = counts

    @lazyproperty
    def pruned(self):
        return self.table_base == 0

    @lazyproperty
    def table_base(self):
        return np.sum(self._all_bases)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._all_counts)

    @lazyproperty
    def zscore(self):
        # TODO: Implement the real zscore calc for MR
        return self._zscore


class _MeansVector(_BaseVector):
    """Used on a non-MR dimension when cube-result contains means."""

    def __init__(self, element, base_counts, means):
        super(_MeansVector, self).__init__(element, base_counts)
        self._means = means

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def values(self):
        return self._means


class _MeansWithMrVector(_MeansVector):
    """MR vector with means for use in a matrix."""

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts[0])


class _MultipleResponseVector(_CategoricalVector):
    """Handles MR vectors (either rows or columns)

    Needs to handle selected and not-selected properly. Consequently, it calculates the
    right margin (for itself), but receives table margin on construction time (from the
    slice).
    """

    @lazyproperty
    def base(self):
        counts = zip(self._selected_unweighted, self._not_selected_unweighted)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def base_values(self):
        return self._base_counts[0, :]

    @lazyproperty
    def margin(self):
        counts = zip(self._selected, self._not_selected)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def pruned(self):
        return np.all(self.base == 0) or np.all(np.isnan(self.base))

    @lazyproperty
    def values(self):
        return self._selected

    @lazyproperty
    def _not_selected(self):
        return self._counts[1, :]

    @lazyproperty
    def _not_selected_unweighted(self):
        return self._base_counts[1, :]

    @lazyproperty
    def _selected(self):
        return self._counts[0, :]

    @lazyproperty
    def _selected_unweighted(self):
        return self._base_counts[0, :]

    @lazyproperty
    def zscore(self):
        # TODO: Implement the real zscore calc for MR
        return self._zscore
