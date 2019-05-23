# encoding: utf-8

"""The matrix objects used by frozen-slice.

A matrix object has rows and (usually) columns.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple
import numpy as np
from scipy.stats.contingency import expected_freq

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty
from cr.cube.vector import (
    AssembledInsertionVector,
    AssembledVector,
    BaseVector,
    CategoricalVector,
    CatXMrVector,
    HiddenVector,
    MeansVector,
    MeansWithMrVector,
    MultipleResponseVector,
    OrderedVector,
    PrunedVector,
)


class _BaseTransformedMatrix(object):
    """Base class for late-stage matrices that transform a base matrix."""

    def __init__(self, base_slice, transforms):
        self._base_slice = base_slice
        self._transforms = transforms

    @lazyproperty
    def table_base(self):
        return self._base_slice.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_slice.table_margin


class OrderedMatrix(_BaseTransformedMatrix):
    """Result of the ordering transform.

    In charge of indexing rows and columns properly.
    """

    @lazyproperty
    def columns(self):
        return tuple(
            OrderedVector(column, self._ordering.row_order)
            for column in tuple(
                np.array(self._base_slice.columns)[self._ordering.column_order]
            )
        )

    @lazyproperty
    def rows(self):
        return tuple(
            OrderedVector(row, self._ordering.column_order)
            for row in tuple(np.array(self._base_slice.rows)[self._ordering.row_order])
        )

    @lazyproperty
    def _ordering(self):
        return self._transforms.ordering


class PrunedMatrix(_BaseTransformedMatrix):
    """Matrix with rows or columns pruned.

    While the rows and/or columns need to be pruned, each one of the remaining
    vectors also needs to be pruned based on the opposite dimension's base.
    """

    @lazyproperty
    def columns(self):
        if not self._applied:
            return self._base_slice.columns

        return tuple(
            PrunedVector(column, self._base_slice.rows)
            for column in self._base_slice.columns
            if not column.pruned
        )

    @lazyproperty
    def rows(self):
        if not self._applied:
            return self._base_slice.rows

        return tuple(
            PrunedVector(row, self._base_slice.columns)
            for row in self._base_slice.rows
            if not row.pruned
        )

    @lazyproperty
    def table_base(self):
        if not self._applied:
            return self._base_slice.table_base

        margin = self._base_slice.table_base
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]

    @lazyproperty
    def table_base_unpruned(self):
        return self._base_slice.table_base

    @lazyproperty
    def table_margin(self):
        if not self._applied:
            return self._base_slice.table_margin

        margin = self._base_slice.table_margin
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]

    @lazyproperty
    def table_margin_unpruned(self):
        return self._base_slice.table_margin

    @lazyproperty
    def _applied(self):
        return self._transforms._pruning


class MatrixWithHidden(_BaseTransformedMatrix):
    @lazyproperty
    def columns(self):
        return tuple(
            HiddenVector(column, self._base_slice.rows)
            for column in self._base_slice.columns
            if not column.hidden
        )

    @lazyproperty
    def rows(self):
        return tuple(
            HiddenVector(row, self._base_slice.columns)
            for row in self._base_slice.rows
            if not row.hidden
        )


class MatrixWithInsertions(_BaseTransformedMatrix):
    """Represents slice with both normal and inserted bits."""

    @lazyproperty
    def columns(self):
        return tuple(
            self._top_columns + self._interleaved_columns + self._bottom_columns
        )

    @lazyproperty
    def rows(self):
        return tuple(self._top_rows + self._interleaved_rows + self._bottom_rows)

    @lazyproperty
    def _assembled_columns(self):
        return tuple(
            AssembledVector(column, self._insertion_rows)
            for column in self._base_slice.columns
        )

    @lazyproperty
    def _assembled_insertion_columns(self):
        return tuple(
            AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.columns
        )

    @lazyproperty
    def _assembled_insertion_rows(self):
        return tuple(
            AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.rows
        )

    @lazyproperty
    def _assembled_rows(self):
        return tuple(
            AssembledVector(row, self._insertion_columns)
            for row in self._base_slice.rows
        )

    @lazyproperty
    def _bottom_columns(self):
        return tuple(
            AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.bottom_columns
        )

    @lazyproperty
    def _bottom_rows(self):
        return tuple(
            AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.bottom_rows
        )

    @lazyproperty
    def _insertion_columns(self):
        return self._insertions._inserted_columns

    @lazyproperty
    def _insertion_rows(self):
        return self._insertions._rows

    @lazyproperty
    def _insertions(self):
        return self._transforms.insertions

    @lazyproperty
    def _interleaved_columns(self):
        columns = []
        for i in range(len(self._base_slice.columns)):
            columns.append(self._assembled_columns[i])
            for insertion_column in self._assembled_insertion_columns:
                if i == insertion_column.anchor:
                    columns.append(insertion_column)
        return tuple(columns)

    @lazyproperty
    def _interleaved_rows(self):
        rows = []
        for i in range(len(self._base_slice.rows)):
            rows.append(self._assembled_rows[i])
            for insertion_row in self._assembled_insertion_rows:
                if i == insertion_row.anchor:
                    rows.append(insertion_row)
        return tuple(rows)

    @lazyproperty
    def _top_rows(self):
        return tuple(
            AssembledVector(row, self._insertion_columns)
            for row in self._insertions.top_rows
        )

    @lazyproperty
    def _top_columns(self):
        return tuple(
            AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.top_columns
        )


# === pre-transform Matrix objects ===


# ---Used to represent the non-existent dimension in case of 1D vectors (that need to be
# ---accessed as slices, to support cr.exporter).
_PlaceholderElement = namedtuple("_PlaceholderElement", "label, is_hidden")


class MatrixFactory(object):
    """Encapsulates creation of the right raw (pre-transforms) matrix object."""

    @classmethod
    def matrix(
        cls,
        dimensions,
        counts,
        base_counts,
        counts_with_missings,
        cube,
        slice_idx,
        ca_as_0th,
    ):
        """Return a matrix object of appropriate type based on parameters."""

        # For cubes with means, cerate one of means-matrix types
        if cube.has_means:
            return cls._create_means_matrix(
                counts, base_counts, cube, dimensions, slice_idx
            )

        dimension_types = cube.dimension_types[-2:]
        if cube.ndim > 2 or ca_as_0th:
            base_counts = base_counts[slice_idx]
            counts = counts[slice_idx]
            counts_with_missings = counts_with_missings[slice_idx]
            if cube.dimension_types[0] == DT.MR:
                base_counts = base_counts[0]
                counts = counts[0]
                counts_with_missings = counts_with_missings[0]
            elif ca_as_0th:
                table_name = "%s: %s" % (
                    cube.dimensions[-2:][0].name,
                    cube.dimensions[-2:][0].valid_elements[slice_idx].label,
                )
                return _CaCatMatrix1D(dimensions, counts, base_counts, table_name)
        elif cube.ndim < 2:
            if dimension_types[0] == DT.MR:
                return _MrMatrix1D(dimensions, counts, base_counts)
            return _CatMatrix1D(dimensions, counts, base_counts)
        if dimension_types == (DT.MR, DT.MR):
            return _MrXMrMatrix(dimensions, counts, base_counts, counts_with_missings)
        elif dimension_types[0] == DT.MR:
            return _MrXCatMatrix(dimensions, counts, base_counts, counts_with_missings)
        elif dimension_types[1] == DT.MR:
            return _CatXMrMatrix(dimensions, counts, base_counts, counts_with_missings)
        return _CatXCatMatrix(dimensions, counts, base_counts, counts_with_missings)

    @classmethod
    def _create_means_matrix(cls, counts, base_counts, cube, dimensions, slice_idx):
        if cube.ndim == 0:
            return _MeansMatrix0D(counts, base_counts)
        elif cube.ndim == 1:
            if dimensions[0].dimension_type == DT.MR:
                return _MrWithMeansMatrix1D(dimensions[0], counts, base_counts)
            return _MeansMatrix1D(dimensions[0], counts, base_counts)
        elif cube.ndim >= 2:
            if cube.ndim == 3:
                base_counts = base_counts[slice_idx]
                counts = counts[slice_idx]
            if dimensions[0].dimension_type == DT.MR:
                return _MrXCatMeansMatrix(dimensions, counts, base_counts)
            return _CatXCatMeansMatrix(dimensions, counts, base_counts)


class _MeansMatrix0D(object):
    """Represents slices with means (and no counts)."""

    # TODO: We might need to have 2 of these, one for 0-D, and one for 1-D mean cubes
    def __init__(self, means, base_counts):
        self._means = means
        self._base_counts = base_counts

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def table_base(self):
        # TODO: Check why we expect mean instead of the real base in this case.
        return self.means

    @lazyproperty
    def table_margin(self):
        return np.sum(self._base_counts)


class _MeansMatrix1D(_MeansMatrix0D):
    def __init__(self, dimension, means, base_counts):
        super(_MeansMatrix1D, self).__init__(means, base_counts)
        self._dimension = dimension

    @lazyproperty
    def columns(self):
        """A single vector that is used only for pruning Means slices."""
        return (
            BaseVector(_PlaceholderElement("Means Summary", False), self._base_counts),
        )

    @lazyproperty
    def rows(self):
        """Rows for Means slice, that enable iteration over labels.

        These vectors are not used for any computations. `means` is used for that,
        directly. However, for the wirng of the exporter, these mean slices need to
        support some additional API, such as labels. And for that, they need to
        support row iteration.
        """
        return tuple(
            MeansVector(element, base_counts, np.array([means]))
            for element, base_counts, means in zip(
                self._dimension.valid_elements, self._base_counts, self._means
            )
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)


class _MrWithMeansMatrix1D(_MeansMatrix1D):
    @lazyproperty
    def rows(self):
        return tuple(
            MeansWithMrVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._dimension.valid_elements, self._base_counts, self._means
            )
        )


class _CatXCatMatrix(object):
    """Deals with CAT x CAT data.

    Delegates most functionality to vectors (rows or columns), but calculates some
    values by itself (like table_margin).

    This class (or its inheritants) must be instantiated as a starting point when
    dealing with slices. Other classes that represents various stages of
    transformations, need to repro a portion of this class' API (like iterating over
    rows or columns).
    """

    def __init__(self, dimensions, counts, base_counts, counts_with_missings=None):
        self._dimensions = dimensions
        self._counts = counts
        self._base_counts = base_counts
        self._all_counts = counts_with_missings

    @lazyproperty
    def columns(self):
        return tuple(
            CategoricalVector(counts, base_counts, element, self.table_margin, zscore)
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def names(self):
        return tuple([dimension.name for dimension in self._dimensions])

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
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _column_elements(self):
        return self._column_dimension.valid_elements

    @lazyproperty
    def _column_index(self):
        # TODO: This is a hack to make it work. It should be addressed properly with
        # passing `counts_with_missings` in all the right places in the factory.
        # Also - subclass for proper functionality in various MR cases.
        if self._all_counts is None:
            return self._column_proportions

        return self._column_proportions / self._baseline * 100

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T, self._base_counts.T, self._column_elements, self._zscores.T
        )

    @lazyproperty
    def _column_proportions(self):
        return np.array([col.proportions for col in self.columns]).T

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _row_elements(self):
        return self._row_dimension.valid_elements

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
    def _valid_rows_idxs(self):
        """ndarray-style index for only valid rows (out of missing and not-missing)."""
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)

    @lazyproperty
    def _zscores(self):
        return self._scalar_type_std_res(
            self._counts,
            self.table_margin,
            np.sum(self._counts, axis=0),
            np.sum(self._counts, axis=1),
        )


class _CatXCatMeansMatrix(_CatXCatMatrix):
    def __init__(self, dimensions, means, base_counts):
        super(_CatXCatMeansMatrix, self).__init__(dimensions, None, base_counts)
        self._means = means

    @lazyproperty
    def columns(self):
        return tuple(
            MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._column_dimension.valid_elements,
                self._base_counts.T,
                self._means.T,
            )
        )

    @lazyproperty
    def rows(self):
        return tuple(
            MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._row_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _CatMatrix1D(_CatXCatMatrix):
    """Special case of CAT x CAT, where the 2nd CAT doesn't exist.

    Values are treated as rows, while there's only a single column (vector).
    """

    @lazyproperty
    def columns(self):
        return tuple(
            [
                CategoricalVector(
                    self._counts,
                    self._base_counts,
                    _PlaceholderElement("Summary", False),
                    self.table_margin,
                )
            ]
        )

    @lazyproperty
    def _zscores(self):
        # TODO: Fix with real zscores
        return tuple([np.nan for _ in self._counts])


class _CaCatMatrix1D(_CatMatrix1D):
    def __init__(self, dimensions, counts, base_counts, table_name):
        super(_CaCatMatrix1D, self).__init__(dimensions, counts, base_counts)
        self._table_name = table_name


class _MatrixWithMR(_CatXCatMatrix):
    @staticmethod
    def _array_type_std_res(counts, total, colsum, rowsum):
        expected_counts = rowsum * colsum / total
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
    def __init__(self, dimensions, means, base_counts):
        counts = np.empty(means.shape)
        super(_MrXCatMeansMatrix, self).__init__(dimensions, counts, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            MeansWithMrVector(element, base_counts, means[0])
            for element, base_counts, means in zip(
                self._row_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _MrMatrix1D(_MrXCatMatrix):
    """Special case of 1-D MR slice (vector)."""

    @lazyproperty
    def columns(self):
        return tuple(
            [
                MultipleResponseVector(
                    self._counts.T,
                    self._base_counts.T,
                    _PlaceholderElement("Summary", False),
                    self.table_margin,
                )
            ]
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=1)

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def _zscores(self):
        return np.array([np.nan] * self._base_counts.shape[0])


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

    Needs to properly index both rows and columns (selected/not-selected.
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
