# encoding: utf-8

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty


class _CatXCatSlice:
    def __init__(self, raw_counts):
        self._raw_counts = raw_counts

    @lazyproperty
    def rows(self):
        return tuple(_CatRow(counts) for counts in self._raw_counts)


class _CatRow:
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
        return self._raw_counts / self.margin


class Insertions:
    def __init__(self, dimensions, slice_):
        self._dimensions = dimensions
        self._slice = slice_

    @lazyproperty
    def anchors(self):
        return tuple(row.anchor for row in self.rows)

    @lazyproperty
    def rows(self):
        return tuple(
            InsertionRow(self._slice, subtotal)
            for subtotal in self._dimensions[0]._subtotals
        )


class InsertionRow:
    def __init__(self, slice_, subtotal):
        self._slice = slice_
        self._subtotal = subtotal

    @lazyproperty
    def _addend_rows(self):
        return tuple(
            row
            for i, row in enumerate(self._slice.rows)
            if i in self._subtotal.addend_idxs
        )

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_rows]), axis=0)

    @lazyproperty
    def margin(self):
        return np.sum(self.values)

    @lazyproperty
    def proportions(self):
        return self.values / self.margin


class Assembler:
    def __init__(self, slice_, insertions):
        self._slice = slice_
        self._insertions = insertions

    @lazyproperty
    def rows(self):
        # No insertions - don't assemble
        if not self._insertions or not self._insertions.rows:
            return self._slice.rows

        # Assemble
        rows = []
        for i in range(len(self._slice.rows)):
            if i in self._insertions.anchors:
                insertion_idx = self._insertions.anchors.index(i)
                if i == 0:
                    rows.append(self._insertions.rows[insertion_idx])
                    rows.append(self._slice.rows[i])
                else:
                    rows.append(self._slice.rows[i])
                    rows.append(self._insertions.rows[insertion_idx])
            else:
                rows.append(self._slice.rows[i])

        return tuple(rows)


class Calculator:
    def __init__(self, assembler):
        self._assembler = assembler

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])
