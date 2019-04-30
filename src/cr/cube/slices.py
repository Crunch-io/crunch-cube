# encoding: utf-8

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty
from cr.cube.enum import DIMENSION_TYPE as DT


class _CatXCatSlice(object):
    def __init__(self, raw_counts):
        self._raw_counts = raw_counts

    @lazyproperty
    def rows(self):
        return tuple(_CategoricalVector(counts) for counts in self._raw_counts)

    @lazyproperty
    def columns(self):
        return tuple(_CategoricalVector(counts) for counts in self._raw_counts.T)


class _MrXCatSlice(_CatXCatSlice):
    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(_CategoricalVector(counts[0]) for counts in self._raw_counts)

    @lazyproperty
    def columns(self):
        """Use bother selected and not-selected counts."""
        return tuple(_MultipleResponseVector(counts) for counts in self._raw_counts.T)


class _CatXMrSlice(_CatXCatSlice):
    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(_MultipleResponseVector(counts.T) for counts in self._raw_counts)

    @lazyproperty
    def columns(self):
        """Use bother selected and not-selected counts."""
        return tuple(_CategoricalVector(counts) for counts in self._raw_counts.T[0])


class _MrXMrSlice(_CatXCatSlice):
    @lazyproperty
    def rows(self):
        return tuple(
            _MultipleResponseVector(counts[0].T) for counts in self._raw_counts
        )

    @lazyproperty
    def columns(self):
        return tuple(
            _MultipleResponseVector(counts) for counts in self._raw_counts.T[0]
        )


class _CategoricalVector(object):
    def __init__(self, raw_counts):
        self._raw_counts = raw_counts

    @lazyproperty
    def values(self):
        return self._raw_counts

    @lazyproperty
    def margin(self):
        return np.sum(self._raw_counts)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin


class _MultipleResponseVector(_CategoricalVector):
    @lazyproperty
    def values(self):
        return self._selected

    @lazyproperty
    def margin(self):
        counts = zip(self._selected, self._not_selected)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def _selected(self):
        return self._raw_counts[0, :]

    @lazyproperty
    def _not_selected(self):
        return self._raw_counts[1, :]


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
        return tuple(
            InsertionRow(self._slice, subtotal)
            for subtotal in self._row_dimension._subtotals
        )

    @lazyproperty
    def _columns(self):
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
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def addend_idxs(self):
        return np.array(self._subtotal.addend_idxs)

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)

    @lazyproperty
    def margin(self):
        return np.sum(np.array([vec.margin for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin


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

        return slice_

    @lazyproperty
    def rows(self):
        return self.slice.rows

    @lazyproperty
    def columns(self):
        return self.slice.columns


class SliceWithInsertions(object):
    def __init__(self, slice_, insertions):
        self._slice = slice_
        self._insertions = insertions

    @lazyproperty
    def rows(self):
        return tuple(
            self._insertions.top_rows
            + self._interleaved_rows
            + self._insertions.bottom_rows
        )

    @lazyproperty
    def columns(self):
        return tuple(
            self._insertions.top_columns
            + self._interleaved_columns
            + self._insertions.bottom_columns
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
    def _assembled_columns(self):
        return tuple(
            _AssembledVector(column, self._insertion_rows)
            for column in self._slice.columns
        )

    @lazyproperty
    def _assembled_insertion_rows(self):
        return tuple(
            _AssembledVector(row, self._insertion_columns)
            for row in self._insertions.rows
        )

    @lazyproperty
    def _interleaved_rows(self):
        rows = []
        for i in range(len(self._slice.rows)):
            rows.append(self._assembled_rows[i])
            if i in self._insertions.row_anchors:
                insertion_idx = self._insertions.row_anchors.index(i)
                rows.append(self._assembled_insertion_rows[insertion_idx])
        return tuple(rows)

    @lazyproperty
    def _assembled_insertion_columns(self):
        return tuple(
            _AssembledVector(column, self._insertion_rows)
            for column in self._insertions.columns
        )

    @lazyproperty
    def _interleaved_columns(self):
        columns = []
        for i in range(len(self._slice.columns)):
            columns.append(self._assembled_columns[i])
            if i in self._insertions.column_anchors:
                insertion_idx = self._insertions.column_anchors.index(i)
                columns.append(self._assembled_insertion_columns[insertion_idx])
        return tuple(columns)


class _AssembledVector(object):
    def __init__(self, base_vector, opposite_inserted_vectors):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def proportions(self):
        return self.values / self.margin

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
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
    def _bottom_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _interleaved_values(self):
        values = []
        for i in range(len(self._base_vector.values)):
            values.append(self._base_vector.values[i])
            if i in self._column_anchors:
                insertion_column = self._opposite_inserted_vectors[
                    self._column_anchors.index(i)
                ]
                insertion_value = np.sum(
                    self._base_vector.values[insertion_column.addend_idxs]
                )
                values.append(insertion_value)
        return tuple(values)


class Calculator(object):
    def __init__(self, assembler):
        self._assembler = assembler

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

    @lazyproperty
    def column_proportions(self):
        return np.array([col.proportions for col in self._assembler.columns]).T

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])


class FrozenSlice(object):
    def __init__(self, cube, slice_idx=0, use_insertions=False, reordered_ids=None):
        self._cube = cube
        self._slice_idx = slice_idx
        self._use_insertions = use_insertions
        self._reordered_ids = reordered_ids

    # API ----------------------------------------------------------------------------

    @lazyproperty
    def row_proportions(self):
        return self._calculator.row_proportions

    @lazyproperty
    def column_proportions(self):
        return self._calculator.column_proportions

    @lazyproperty
    def counts(self):
        return self._calculator.counts

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
        return Transforms(self._ordering, self._insertions)

    @lazyproperty
    def _assembler(self):
        return Assembler(self._slice, self._transforms)

    @lazyproperty
    def _calculator(self):
        return Calculator(self._assembler)

    @lazyproperty
    def _dimensions(self):
        return self._cube.dimensions

    @lazyproperty
    def _insertions(self):
        return (
            Insertions(self._dimensions, self._slice) if self._use_insertions else None
        )

    @lazyproperty
    def _slice(self):
        raw_counts = self._cube._apply_missings(
            self._cube._measure(False).raw_cube_array
        )
        type_ = self._cube.dim_types
        if self._cube.ndim > 2:
            raw_counts = raw_counts[self._slice_idx]
        if type_ == (DT.MR, DT.CAT):
            return _MrXCatSlice(raw_counts)
        if type_ == (DT.CAT, DT.MR):
            return _CatXMrSlice(raw_counts)
        if type_ == (DT.MR, DT.MR):
            return _MrXMrSlice(raw_counts)
        return _CatXCatSlice(raw_counts)


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
    def __init__(self, ordering, insertions):
        self._ordering = ordering
        self._insertions = insertions

    @lazyproperty
    def ordering(self):
        return self._ordering

    @lazyproperty
    def insertions(self):
        return self._insertions
