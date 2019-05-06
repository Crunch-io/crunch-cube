# encoding: utf-8

from __future__ import division

import numpy as np
from scipy.stats.contingency import expected_freq
from scipy.stats import norm

from cr.cube.util import lazyproperty
from cr.cube.enum import DIMENSION_TYPE as DT


class _MeansSlice(object):
    def __init__(self, means, base_counts):
        self._means = means
        self._base_counts = base_counts

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def table_margin(self):
        return np.sum(self._base_counts)


class _CatXCatSlice(object):
    def __init__(self, dimensions, counts, base_counts):
        self._dimensions = dimensions
        self._counts = counts
        self._base_counts = base_counts

    @lazyproperty
    def names(self):
        return tuple([dimension.name for dimension in self._dimensions])

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _row_elements(self):
        return self._row_dimension.valid_elements

    @lazyproperty
    def _column_elements(self):
        return self._column_dimension.valid_elements

    @lazyproperty
    def _row_generator(self):
        return zip(self._counts, self._base_counts, self._row_elements, self._zscores)

    @lazyproperty
    def _column_generator(self):
        return zip(self._counts.T, self._base_counts.T, self._column_elements)

    @lazyproperty
    def rows(self):
        return tuple(
            _CategoricalVector(
                counts, base_counts, element.label, self.table_margin, zscore
            )
            for counts, base_counts, element, zscore in self._row_generator
        )

    @lazyproperty
    def columns(self):
        return tuple(
            _CategoricalVector(counts, base_counts, element.label, self.table_margin)
            for counts, base_counts, element in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def _zscores(self):
        total = self.table_margin
        column_margin = np.sum(self._counts, axis=0)
        row_margin = np.sum(self._counts, axis=1)
        expected_counts = expected_freq(self._counts)
        residuals = self._counts - expected_counts
        variance = (
            np.outer(row_margin, column_margin)
            * np.outer(total - row_margin, total - column_margin)
            / total ** 3
        )
        return residuals / np.sqrt(variance)


class _1DCatSlice(_CatXCatSlice):
    @lazyproperty
    def _zscores(self):
        # TODO: Fix with real zscores
        return tuple([np.nan for _ in self._counts])

    @lazyproperty
    def columns(self):
        return tuple(
            [
                _CategoricalVector(
                    self._counts, self._base_counts, "Summary", self.table_margin
                )
            ]
        )


class _MrXCatSlice(_CatXCatSlice):
    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(
            _CategoricalVector(counts[0], base_counts[0], element.label, table_margin)
            for counts, base_counts, element, table_margin in self._row_generator
        )

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts, self._base_counts, self._row_elements, self.table_margin
        )

    @lazyproperty
    def columns(self):
        """Use bother selected and not-selected counts."""
        return tuple(
            _MultipleResponseVector(
                counts, base_counts, element.label, self.table_margin
            )
            for counts, base_counts, element in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(1, 2))


class _1DMrSlice(_MrXCatSlice):
    @lazyproperty
    def columns(self):
        return tuple(
            [
                _MultipleResponseVector(
                    self._counts.T, self._base_counts.T, "Summary", self.table_margin
                )
            ]
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=1)


class _CatXMrSlice(_CatXCatSlice):
    @lazyproperty
    def _zscores(self):
        # TODO: Fix with real zscores
        return tuple([np.nan for _ in self._counts])

    @lazyproperty
    def rows(self):
        return tuple(
            _MultipleResponseVector(
                counts.T, base_counts.T, element.label, self.table_margin
            )
            for counts, base_counts, element, zscore in self._row_generator
            # TODO: Actually use zscores
        )

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T[0],
            self._base_counts.T[0],
            self._column_elements,
            self.table_margin,
        )

    @lazyproperty
    def columns(self):
        return tuple(
            _CategoricalVector(counts, base_counts, element.label, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(0, 2))


class _MrXMrSlice(_CatXCatSlice):
    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts, self._base_counts, self._row_elements, self.table_margin
        )

    @lazyproperty
    def rows(self):
        # return tuple(_MultipleResponseVector(counts[0].T) for counts in self._counts)
        return tuple(
            _MultipleResponseVector(
                counts[0].T, base_counts[0].T, element.label, table_margin
            )
            for counts, base_counts, element, table_margin in self._row_generator
        )

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T[0],
            self._base_counts.T[0],
            self._column_elements,
            self.table_margin.T,
        )

    @lazyproperty
    def columns(self):
        # return tuple(_MultipleResponseVector(counts) for counts in self._counts.T[0])
        return tuple(
            _MultipleResponseVector(counts, base_counts, element.label, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 3))


class _CategoricalVector(object):
    def __init__(self, counts, base_counts, label, table_margin, zscore=None):
        self._counts = counts
        self._base_counts = base_counts
        self._label = label
        self._table_margin = table_margin
        self._zscore = zscore

    @lazyproperty
    def pvals(self):
        return 2 * (1 - norm.cdf(np.abs(self._zscore)))

    @lazyproperty
    def zscore(self):
        return self._zscore

    @lazyproperty
    def label(self):
        return self._label

    @lazyproperty
    def values(self):
        if not isinstance(self._counts, np.ndarray):
            return np.array([self._counts])
        return self._counts

    @lazyproperty
    def base_values(self):
        if not isinstance(self._base_counts, np.ndarray):
            return np.array([self._base_counts])
        return self._base_counts

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def proportions(self):
        return self.values / self.margin
        # return self.values / self.base

    @lazyproperty
    def table_proportions(self):
        return self.values / self.table_margin

    @lazyproperty
    def pruned(self):
        # return self.margin == 0
        return self.base == 0 or np.isnan(self.base)


class _MultipleResponseVector(_CategoricalVector):
    @lazyproperty
    def values(self):
        return self._selected

    @lazyproperty
    def base_values(self):
        return self._base_counts[0, :]

    @lazyproperty
    def base(self):
        counts = zip(self._selected_unweighted, self._not_selected_unweighted)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def margin(self):
        counts = zip(self._selected, self._not_selected)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def pruned(self):
        # return np.all(self.margin == 0)
        return np.all(self.base == 0) or np.all(np.isnan(self.base))

    @lazyproperty
    def _selected(self):
        return self._counts[0, :]

    @lazyproperty
    def _not_selected(self):
        return self._counts[1, :]

    @lazyproperty
    def _selected_unweighted(self):
        return self._base_counts[0, :]

    @lazyproperty
    def _not_selected_unweighted(self):
        return self._base_counts[1, :]


class Insertions(object):
    def __init__(self, dimensions, slice_):
        self._dimensions = dimensions
        self._slice = slice_

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def row_anchors(self):
        return tuple(row.anchor for row in self.rows)

    @lazyproperty
    def column_anchors(self):
        return tuple(column.anchor for column in self.columns)

    @lazyproperty
    def _rows(self):
        if self._row_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            InsertionRow(self._slice, subtotal)
            for subtotal in self._row_dimension._subtotals
        )

    @lazyproperty
    def _columns(self):
        if len(self._dimensions) < 2 or self._column_dimension.dimension_type in (
            DT.MR,
            DT.CA,
        ):
            return tuple()

        return tuple(
            InsertionColumn(self._slice, subtotal)
            for subtotal in self._column_dimension._subtotals
        )

    @lazyproperty
    def top_rows(self):
        return tuple(row for row in self._rows if row.anchor == "top")

    @lazyproperty
    def bottom_rows(self):
        return tuple(row for row in self._rows if row.anchor == "bottom")

    @lazyproperty
    def top_columns(self):
        return tuple(columns for columns in self._columns if columns.anchor == "top")

    @lazyproperty
    def bottom_columns(self):
        return tuple(columns for columns in self._columns if columns.anchor == "bottom")

    @lazyproperty
    def rows(self):
        return tuple(row for row in self._rows if row.anchor not in ("top", "bottom"))

    @lazyproperty
    def columns(self):
        return tuple(
            column for column in self._columns if column.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def intersections(self):
        intersections = []
        for row in self._rows:
            row_col_intersections = []
            row_idx = (
                0
                if row.anchor == "top"
                else -1
                if row.anchor == "bottom"
                else row.anchor
            )
            for col in self._columns:
                col_idx = (
                    0
                    if col.anchor == "top"
                    else -1
                    if col.anchor == "bottom"
                    else col.anchor + 1
                )
                intersection_value = row.values[col_idx] + col.values[row_idx]
                row_col_intersections.append(intersection_value)
            intersections.append(row_col_intersections)

        return np.array(intersections)


class _InsertionVector(object):
    def __init__(self, slice_, subtotal):
        self._slice = slice_
        self._subtotal = subtotal

    @lazyproperty
    def table_margin(self):
        return self._slice.table_margin

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def addend_idxs(self):
        return np.array(self._subtotal.addend_idxs)

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)

    @lazyproperty
    def base_values(self):
        return np.sum(
            np.array([row.base_values for row in self._addend_vectors]), axis=0
        )

    @lazyproperty
    def margin(self):
        return np.sum(np.array([vec.margin for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def base(self):
        return np.sum(np.array([vec.base for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin
        # return self.values / self.base

    @lazyproperty
    def table_proportions(self):
        return self.values / self.table_proportions

    @lazyproperty
    def pruned(self):
        """Insertions are never pruned."""
        return False


class InsertionRow(_InsertionVector):
    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._slice.rows)
            if i in self._subtotal.addend_idxs
        )


class InsertionColumn(_InsertionVector):
    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._slice.columns)
            if i in self._subtotal.addend_idxs
        )


class Assembler(object):
    def __init__(self, slice_, transforms):
        self._slice = slice_
        self._transforms = transforms

    @lazyproperty
    def slice(self):
        """Apply all transforms sequentially."""
        slice_ = self._slice

        if self._transforms.ordering:
            slice_ = OrderedSlice(slice_, self._transforms.ordering)
        if self._transforms.insertions:
            slice_ = SliceWithInsertions(slice_, self._transforms.insertions)
        if self._transforms.pruning:
            slice_ = PrunedSlice(slice_)

        return slice_

    @lazyproperty
    def rows(self):
        return self.slice.rows

    @lazyproperty
    def columns(self):
        return self.slice.columns

    @lazyproperty
    def table_margin(self):
        return self.slice.table_margin

    @lazyproperty
    def table_base(self):
        # return np.sum([row.base for row in self._slice.rows])
        return self.slice.table_base


class SliceWithInsertions(object):
    def __init__(self, slice_, insertions):
        self._slice = slice_
        self._insertions = insertions

    @lazyproperty
    def table_margin(self):
        return self._slice.table_margin

    @lazyproperty
    def table_base(self):
        return self._slice.table_base

    @lazyproperty
    def rows(self):
        return tuple(self._top_rows + self._interleaved_rows + self._bottom_rows)

    @lazyproperty
    def columns(self):
        return tuple(
            self._top_columns + self._interleaved_columns + self._bottom_columns
        )

    @lazyproperty
    def _insertion_columns(self):
        return self._insertions._columns

    @lazyproperty
    def _insertion_rows(self):
        return self._insertions._rows

    @lazyproperty
    def _assembled_rows(self):
        return tuple(
            _AssembledVector(row, self._insertion_columns) for row in self._slice.rows
        )

    @lazyproperty
    def _bottom_rows(self):
        return tuple(
            _AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.bottom_rows
        )

    @lazyproperty
    def _top_rows(self):
        return tuple(
            _AssembledVector(row, self._insertion_columns)
            for row in self._insertions.top_rows
        )

    @lazyproperty
    def _top_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.top_columns
        )

    @lazyproperty
    def _bottom_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.bottom_columns
        )

    @lazyproperty
    def _assembled_columns(self):
        return tuple(
            _AssembledVector(column, self._insertion_rows)
            for column in self._slice.columns
        )

    @lazyproperty
    def _assembled_insertion_rows(self):
        return tuple(
            _AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.rows
        )

    @lazyproperty
    def _interleaved_rows(self):
        rows = []
        for i in range(len(self._slice.rows)):
            rows.append(self._assembled_rows[i])
            for insertion_row in self._assembled_insertion_rows:
                if i == insertion_row.anchor:
                    rows.append(insertion_row)
        return tuple(rows)

    @lazyproperty
    def _assembled_insertion_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.columns
        )

    @lazyproperty
    def _interleaved_columns(self):
        columns = []
        for i in range(len(self._slice.columns)):
            columns.append(self._assembled_columns[i])
            for insertion_column in self._assembled_insertion_columns:
                if i == insertion_column.anchor:
                    columns.append(insertion_column)
        return tuple(columns)


class _AssembledVector(object):
    def __init__(self, base_vector, opposite_inserted_vectors):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors

    @lazyproperty
    def pvals(self):
        return (
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_pvals
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def _interleaved_pvals(self):
        pvals = []
        for i, value in enumerate(self._base_vector.pvals):
            pvals.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    pvals.append(np.nan)
        return tuple(pvals)

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def pruned(self):
        return self._base_vector.pruned

    @lazyproperty
    def proportions(self):
        # return self.values / self.base
        return self.values / self.margin

    @lazyproperty
    def table_proportions(self):
        return self.values / self._base_vector.table_margin

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
        )

    @lazyproperty
    def base_values(self):
        # TODO: Do for real
        return np.array(
            self._top_base_values
            + self._interleaved_base_values
            + self._bottom_base_values
        )

    @lazyproperty
    def _column_anchors(self):
        return tuple(col.anchor for col in self._opposite_inserted_vectors)

    @lazyproperty
    def _top_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _top_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _bottom_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _bottom_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _interleaved_values(self):
        values = []
        for i in range(len(self._base_vector.values)):
            values.append(self._base_vector.values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.values[inserted_vector.addend_idxs]
                    )
                    values.append(insertion_value)
        return tuple(values)

    @lazyproperty
    def _interleaved_base_values(self):
        base_values = []
        for i in range(len(self._base_vector.base_values)):
            base_values.append(self._base_vector.base_values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.base_values[inserted_vector.addend_idxs]
                    )
                    base_values.append(insertion_value)
        return tuple(base_values)


class _AssembledInsertionVector(_AssembledVector):
    @lazyproperty
    def anchor(self):
        return self._base_vector.anchor


class Calculator(object):
    def __init__(self, assembler):
        self._assembler = assembler

    @lazyproperty
    def pvals(self):
        return np.array([row.pvals for row in self._assembler.rows])

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

    @lazyproperty
    def column_proportions(self):
        return np.array([col.proportions for col in self._assembler.columns]).T

    @lazyproperty
    def table_proportions(self):
        return np.array([row.table_proportions for row in self._assembler.rows])

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])

    @lazyproperty
    def column_margin(self):
        return np.array([column.margin for column in self._assembler.columns]).T

    @lazyproperty
    def table_margin(self):
        return self._assembler.table_margin

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._assembler.rows])

    @lazyproperty
    def column_base(self):
        return np.array([column.base for column in self._assembler.columns]).T

    @lazyproperty
    def table_base(self):
        return self._assembler.table_base

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])

    @lazyproperty
    def base_counts(self):
        return np.array([row.base_values for row in self._assembler.rows])

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._assembler.rows)

    @lazyproperty
    def column_labels(self):
        return tuple(column.label for column in self._assembler.columns)

    @lazyproperty
    def zscore(self):
        return np.array([row.zscore for row in self._assembler.rows])
        # return self._scalar_type_std_res(
        #     self.counts, self.table_margin, self.column_margin, self.row_margin
        # )


class FrozenSlice(object):
    def __init__(
        self,
        cube,
        slice_idx=0,
        use_insertions=False,
        reordered_ids=None,
        pruning=False,
        weighted=True,
        population=None,
        ca_as_0th=None,
    ):
        self._cube = cube
        self._slice_idx = slice_idx
        self._use_insertions = use_insertions
        self._reordered_ids = reordered_ids
        self._pruning = pruning
        self._weighted = weighted
        self._population = population
        self._ca_as_0th = ca_as_0th

    # API ----------------------------------------------------------------------------

    @lazyproperty
    def names(self):
        return self._slice.names

    @lazyproperty
    def means(self):
        return self._slice.means

    @lazyproperty
    def pvals(self):
        return self._calculator.pvals

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def zscore(self):
        return self._calculator.zscore

    @lazyproperty
    def row_proportions(self):
        return self._calculator.row_proportions

    @lazyproperty
    def column_proportions(self):
        return self._calculator.column_proportions

    @lazyproperty
    def table_proportions(self):
        return self._calculator.table_proportions

    @lazyproperty
    def counts(self):
        return self._calculator.counts

    @lazyproperty
    def base_counts(self):
        return self._calculator.base_counts

    @lazyproperty
    def row_margin(self):
        return self._calculator.row_margin

    @lazyproperty
    def column_margin(self):
        return self._calculator.column_margin

    @lazyproperty
    def table_margin(self):
        return self._calculator.table_margin

    @lazyproperty
    def row_base(self):
        return self._calculator.row_base

    @lazyproperty
    def column_base(self):
        return self._calculator.column_base

    @lazyproperty
    def table_base(self):
        return self._calculator.table_base

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def row_labels(self):
        return self._calculator.row_labels

    @lazyproperty
    def column_labels(self):
        return self._calculator.column_labels

    # Properties ---------------------------------------------------------------------

    @lazyproperty
    def _ordering(self):
        return (
            OrderTransform(self._cube.dimensions, self._reordered_ids)
            if self._reordered_ids
            else None
        )

    @lazyproperty
    def _transforms(self):
        return Transforms(self._ordering, self._pruning, self._insertions)

    @lazyproperty
    def _assembler(self):
        return Assembler(self._slice, self._transforms)

    @lazyproperty
    def _calculator(self):
        return Calculator(self._assembler)

    @lazyproperty
    def _dimensions(self):
        dimensions = self._cube.dimensions[-2:]
        if self._ca_as_0th:
            return tuple([dimensions[-1]])
        return dimensions

    @lazyproperty
    def _insertions(self):
        return (
            Insertions(self._dimensions, self._slice) if self._use_insertions else None
        )

    @lazyproperty
    def _slice(self):
        dimensions = self._cube.dimensions[-2:]
        base_counts = self._cube._apply_missings(
            # self._cube._measure(False).raw_cube_array
            self._cube._measures.unweighted_counts.raw_cube_array
        )
        counts = self._cube._apply_missings(
            self._cube._measure(self._weighted).raw_cube_array
        )
        type_ = self._cube.dim_types[-2:]
        if self._cube.ndim == 0 and self._cube.has_means:
            return _MeansSlice(counts, base_counts)
        if self._cube.ndim == 1 and self._cube.has_means:
            return _MeansSlice(counts, base_counts)
        if self._cube.ndim > 2 or self._ca_as_0th:
            base_counts = base_counts[self._slice_idx]
            counts = counts[self._slice_idx]
            if self._cube.dim_types[0] == DT.MR:
                base_counts = base_counts[0]
                counts = counts[0]
            elif self._ca_as_0th:
                return _1DCatSlice(self._dimensions, counts, base_counts)
        elif self._cube.ndim < 2:
            if type_[0] == DT.MR:
                return _1DMrSlice(dimensions, counts, base_counts)
            return _1DCatSlice(dimensions, counts, base_counts)
        if type_ == (DT.MR, DT.MR):
            return _MrXMrSlice(dimensions, counts, base_counts)
        elif type_[0] == DT.MR:
            return _MrXCatSlice(dimensions, counts, base_counts)
        elif type_[1] == DT.MR:
            return _CatXMrSlice(dimensions, counts, base_counts)
        return _CatXCatSlice(dimensions, counts, base_counts)


class OrderTransform(object):
    def __init__(self, dimensions, ordered_ids):
        self._dimensions = dimensions
        self._ordered_ids = ordered_ids

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _row_ids(self):
        return tuple(el.element_id for el in self._row_dimension.valid_elements)

    @lazyproperty
    def _column_ids(self):
        return tuple(el.element_id for el in self._column_dimension.valid_elements)

    @lazyproperty
    def _ordered_row_ids(self):
        return self._ordered_ids[0]

    @lazyproperty
    def _ordered_column_ids(self):
        ids = self._ordered_ids[1]
        return ids if ids is not None else None

    @lazyproperty
    def row_order(self):
        return np.array(
            [self._row_ids.index(ordered_id) for ordered_id in self._ordered_row_ids]
        )

    @lazyproperty
    def column_order(self):
        if self._ordered_column_ids is None:
            return None
        return np.array(
            [
                self._column_ids.index(ordered_id)
                for ordered_id in self._ordered_column_ids
            ]
        )


class OrderedVector(object):
    def __init__(self, vector, order):
        self._vector = vector
        self._order = order

    @lazyproperty
    def order(self):
        return self._order if self._order is not None else slice(None)

    @lazyproperty
    def values(self):
        return self._vector.values[self.order]

    @lazyproperty
    def base_values(self):
        return self._vector.base_values[self.order]


class OrderedSlice(object):
    def __init__(self, slice_, ordering):
        self._slice = slice_
        self._ordering = ordering

    @lazyproperty
    def rows(self):
        return tuple(
            OrderedVector(row, self._ordering.column_order)
            for row in tuple(np.array(self._slice.rows)[self._ordering.row_order])
        )


class Transforms(object):
    def __init__(self, ordering=None, pruning=None, insertions=None):
        self._ordering = ordering
        self._pruning = pruning
        self._insertions = insertions

    @lazyproperty
    def ordering(self):
        return self._ordering

    @lazyproperty
    def pruning(self):
        return self._pruning

    @lazyproperty
    def insertions(self):
        return self._insertions


class PrunedVector(object):
    def __init__(self, vector, opposite_vectors):
        self._vector = vector
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def pvals(self):
        return np.array(
            [
                pvals
                for pvals, opposite_vector in zip(
                    self._vector.pvals, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def values(self):
        return np.array(
            [
                value
                for value, opposite_vector in zip(
                    self._vector.values, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def base_values(self):
        return np.array(
            [
                value
                for value, opposite_vector in zip(
                    self._vector.base_values, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def proportions(self):
        return np.array(
            [
                proportion
                for proportion, opposite_vector in zip(
                    self._vector.proportions, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def table_proportions(self):
        return np.array(
            [
                proportion
                for proportion, opposite_vector in zip(
                    self._vector.table_proportions, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def margin(self):
        return self._vector.margin


class PrunedSlice(object):
    def __init__(self, slice_):
        self._slice = slice_

    @lazyproperty
    def rows(self):
        return tuple(
            PrunedVector(row, self._slice.columns)
            for row in self._slice.rows
            if not row.pruned
        )

    @lazyproperty
    def columns(self):
        return tuple(
            PrunedVector(column, self._slice.rows)
            for column in self._slice.columns
            if not column.pruned
        )

    @lazyproperty
    def table_margin(self):
        margin = self._slice.table_margin
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]
        # pruned_margin = margin[margin != 0]
        # return pruned_margin if margin.ndim == 1 else pruned_margin[:, None]
