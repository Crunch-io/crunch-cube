# encoding: utf-8

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty


class _CatXCatSlice(object):
    def __init__(self, raw_counts):
        self._raw_counts = raw_counts

    @lazyproperty
    def rows(self):
        return tuple(_CategoricalVector(counts) for counts in self._raw_counts)

    @lazyproperty
    def columns(self):
        return tuple(_CategoricalVector(counts) for counts in self._raw_counts.T)


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
        # return self._raw_counts / self.margin
        return self.values / self.margin


class Insertions(object):
    def __init__(self, dimensions, slice_):
        self._dimensions = dimensions
        self._slice = slice_

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
    def rows(self):
        return tuple(row for row in self._rows if row.anchor not in ("top", "bottom"))


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
        return np.sum(self.values)

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
    def __init__(self, slice_, insertions):
        self._slice = slice_
        self._insertions = insertions

    @lazyproperty
    def rows(self):
        insertions = self._insertions

        # No insertions - don't assemble
        if not insertions or not insertions.rows:
            return self._slice.rows

        return tuple(
            insertions.top_rows + self._interleaved_rows + insertions.bottom_rows
        )

    @lazyproperty
    def _insertion_columns(self):
        return self._insertions._columns

    @lazyproperty
    def _interleaved_rows(self):
        rows = []
        for i in range(len(self._slice.rows)):
            if i in self._insertions.row_anchors:
                insertion_idx = self._insertions.row_anchors.index(i)
                rows.append(
                    _AssembledRow(self._slice.rows[i].values, self._insertion_columns)
                )
                rows.append(
                    _AssembledRow(
                        self._insertions.rows[insertion_idx].values,
                        self._insertion_columns,
                    )
                )
            else:
                rows.append(
                    _AssembledRow(self._slice.rows[i].values, self._insertion_columns)
                )
        return tuple(rows)


class _AssembledRow(_CategoricalVector):
    def __init__(self, raw_counts, inserted_columns):
        super(_AssembledRow, self).__init__(raw_counts)
        self._inserted_columns = inserted_columns

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
        )

    @lazyproperty
    def _column_anchors(self):
        return tuple(col.anchor for col in self._inserted_columns)

    @lazyproperty
    def _top_values(self):
        return tuple(
            np.sum(self._raw_counts[col.addend_idxs])
            for col in self._inserted_columns
            if col.anchor == "top"
        )

    @lazyproperty
    def _bottom_values(self):
        return tuple(
            np.sum(self._raw_counts[col.addend_idxs])
            for col in self._inserted_columns
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _interleaved_values(self):
        values = []
        for i in range(len(self._raw_counts)):
            if i in self._column_anchors:
                insertion_column = self._inserted_columns[self._column_anchors.index(i)]
                insertion_value = np.sum(self._raw_counts[insertion_column.addend_idxs])
                values.append(self._raw_counts[i])
                values.append(insertion_value)
            else:
                values.append(self._raw_counts[i])
        return tuple(values)


class Calculator(object):
    def __init__(self, assembler):
        self._assembler = assembler

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])


class FrozenSlice(object):
    def __init__(self, cube, slice_idx=0, use_insertions=False):
        self._cube = cube
        self._slice_idx = slice_idx
        self._use_insertions = use_insertions

    # API ----------------------------------------------------------------------------

    @lazyproperty
    def row_proportions(self):
        return self._calculator.row_proportions

    # Properties ---------------------------------------------------------------------

    @lazyproperty
    def _assembler(self):
        return Assembler(self._slice, self._insertions)

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
        if self._cube.ndim > 2:
            raw_counts = raw_counts[self._slice_idx]
        return _CatXCatSlice(raw_counts)
