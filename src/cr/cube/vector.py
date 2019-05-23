# encoding: utf-8

"""Vector objects used by frozen-slice.

A vector represents a row or column of data in the overall data matrix. It composes the
element that corresponds to the row or column and so knows the name, element_id, numeric
value, etc. for the row or column.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import norm

from cr.cube.util import lazyproperty


class BaseVector(object):
    """Base class for all vector objects, although is used directly in some cases."""

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
        return self._element.is_hidden

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


class _BaseInsertionVector(object):
    """Represents constituent vectors of the `Insertions` class.

    Needs to repro the API of the more basic vectors - because of
    composition (and not inheritance)
    """

    def __init__(self, slice_, subtotal):
        self._slice = slice_
        self._subtotal = subtotal

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
        """Insertion cannot be hidden."""
        return False

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
        return np.array([np.nan])

    @lazyproperty
    def numeric(self):
        return np.nan

    @lazyproperty
    def table_margin(self):
        return self._slice.table_margin

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)


class CategoricalVector(BaseVector):
    """Main staple of all measures.

    Some of the measures it can calculate by itself, others it needs to receive at
    construction time (like table margin and zscores).
    """

    def __init__(
        self, counts, base_counts, element, table_margin, zscore=None, column_index=None
    ):
        super(CategoricalVector, self).__init__(element, base_counts)
        self._counts = counts
        self._table_margin = table_margin
        self._zscore = zscore
        self._column_index = column_index

    @lazyproperty
    def base_values(self):
        if not isinstance(self._base_counts, np.ndarray):
            return np.array([self._base_counts])
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
        # return self.values / self.base

    @lazyproperty
    def pvals(self):
        return 2 * (1 - norm.cdf(np.abs(self._zscore)))

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def values(self):
        if not isinstance(self._counts, np.ndarray):
            return np.array([self._counts])
        return self._counts

    @lazyproperty
    def zscore(self):
        return self._zscore


class CatXMrVector(CategoricalVector):
    def __init__(
        self, counts, base_counts, label, table_margin, zscore=None, column_index=None
    ):
        super(CatXMrVector, self).__init__(
            counts[0], base_counts[0], label, table_margin, zscore, column_index
        )
        self._pruning_bases = base_counts

    @lazyproperty
    def pruned(self):
        return np.sum(self._pruning_bases) == 0


class _InsertionColumn(_BaseInsertionVector):
    """Represents an inserted (subtotal) column."""

    @lazyproperty
    def pruned(self):
        return not np.any(np.array([row.base for row in self._slice.rows]))

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            column
            for i, column in enumerate(self._slice.columns)
            if i in self._subtotal.addend_idxs
        )


class _InsertionRow(_BaseInsertionVector):
    """Represents an inserted (subtotal) row."""

    @lazyproperty
    def pruned(self):
        return not np.any(np.array([column.base for column in self._slice.columns]))

    @lazyproperty
    def pvals(self):
        return np.array([np.nan] * len(self._slice.columns))

    @lazyproperty
    def zscore(self):
        return np.array([np.nan] * len(self._slice.columns))

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._slice.rows)
            if i in self._subtotal.addend_idxs
        )


class MeansVector(BaseVector):
    def __init__(self, element, base_counts, means):
        super(MeansVector, self).__init__(element, base_counts)
        self._means = means

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def values(self):
        return self._means


class MeansWithMrVector(MeansVector):
    """This is a row of a 1-D MR with Means.

    This vector is special in the sense that it doesn't provide us with the normal base
    (which is selected + not-selected for a 1-D MR *without* means). Instead, it
    calculates the base as *just* the selected, which is the correct base for the 1-D MR
    with means.
    """

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts[0])


class MultipleResponseVector(CategoricalVector):
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


class _TransformedVector(object):
    # @lazyproperty
    # def counts(self):
    #     return self._base_vector.counts

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
    def pruned(self):
        return self._base_vector.pruned

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def means(self):
        return self._base_vector.means

    @lazyproperty
    def numeric(self):
        return self._base_vector.numeric

    # @lazyproperty
    # def table_base(self):
    #     return self._base_vector.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_vector.table_margin

    # @lazyproperty
    # def values(self):
    #     return self.counts


class AssembledVector(_TransformedVector):
    """Vector with base, as well as inserted, elements (of the opposite dimension)."""

    def __init__(self, base_vector, opposite_inserted_vectors):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        # TODO: Do for real
        return np.array(
            self._top_base_values
            + self._interleaved_base_values
            + self._bottom_base_values
        )

    @lazyproperty
    def column_index(self):
        # return self._base_vector.column_index
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_column_index
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def proportions(self):
        # return self.values / self.base
        return self.values / self.margin

    @lazyproperty
    def pruned(self):
        return self._base_vector.pruned

    @lazyproperty
    def pvals(self):
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_pvals
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def table_proportions(self):
        return self.values / self._base_vector.table_margin

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
        )

    @lazyproperty
    def zscore(self):
        return np.array(
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_zscore
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def _bottom_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _bottom_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

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

    @lazyproperty
    def _interleaved_column_index(self):
        # TODO: Replace with real column index values from insertions vectors. This
        # should be something like:
        #   col_ind = (ins1.prop + ins2.prop) / (ins1.baseline + ins2.baseline)
        # ask @mike to confirm
        column_index = []
        for i, value in enumerate(self._base_vector.column_index):
            column_index.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    column_index.append(np.nan)
        return tuple(column_index)

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
    def _interleaved_zscore(self):
        zscore = []
        for i, value in enumerate(self._base_vector.zscore):
            zscore.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    zscore.append(np.nan)
        return tuple(zscore)

    @lazyproperty
    def _top_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _top_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )


class AssembledInsertionVector(AssembledVector):
    """Inserted row or col, but with elements from opposite dimension insertions.

    Needs to be subclassed from _AssembledVector, because it needs to provide the
    anchor, in order to know where it (itself) gets inserted.
    """

    @lazyproperty
    def anchor(self):
        return self._base_vector.anchor


class _BasePrunedOrHiddenVector(_TransformedVector):
    def __init__(self, base_vector, opposite_vectors):
        self._base_vector = base_vector
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def base(self):
        if not isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base
        return self._base_vector.base[self._valid_elements_idxs]

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self._valid_elements_idxs]

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index[self._valid_elements_idxs]

    @lazyproperty
    def margin(self):
        if not isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin
        return self._base_vector.margin[self._valid_elements_idxs]

    @lazyproperty
    def means(self):
        return self._base_vector.means[self._valid_elements_idxs]

    @lazyproperty
    def proportions(self):
        return self._base_vector.proportions[self._valid_elements_idxs]

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals[self._valid_elements_idxs]

    @lazyproperty
    def table_proportions(self):
        return self._base_vector.table_proportions[self._valid_elements_idxs]

    @lazyproperty
    def values(self):
        return self._base_vector.values[self._valid_elements_idxs]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore[self._valid_elements_idxs]

    @lazyproperty
    def _valid_elements_idxs(self):
        raise NotImplementedError("must be implemented by each subclass")


class HiddenVector(_BasePrunedOrHiddenVector):
    """Vector with elements from the opposide dimensions hidden."""

    @lazyproperty
    def _valid_elements_idxs(self):
        """An 1D ndarray of int idxs of visible values, suitable for array indexing."""
        return np.array(
            [
                index
                for index, opposite_vector in enumerate(self._opposite_vectors)
                if not opposite_vector.hidden
            ],
            dtype=int,
        )


class OrderedVector(_TransformedVector):
    """In charge of indexing elements properly, after ordering transform."""

    def __init__(self, vector, order):
        self._base_vector = vector
        self._order = order

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self.order]

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def order(self):
        return self._order if self._order is not None else slice(None)

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals

    @lazyproperty
    def values(self):
        return self._base_vector.values[self.order]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore


class PrunedVector(_BasePrunedOrHiddenVector):
    """Vector with elements from the opposide dimensions pruned."""

    @lazyproperty
    def _valid_elements_idxs(self):
        """An 1D ndarray of int idxs of unpruned values, suitable for array indexing."""
        return np.array(
            [
                index
                for index, opposite_vector in enumerate(self._opposite_vectors)
                if not opposite_vector.pruned
            ],
            dtype=int,
        )
