# encoding: utf-8

"""Objects related to ordering the elements in a dimension.

There are several different collation (ordering) methods, each of which has a distinct
collator class here. Each has the single public (class)method `.display_order()` which
returns a tuple of signed int indices. A positive index refers to a base (non-inserted)
element and a negative index refers to an inserted subtotal. Both positive and negative
indices work for accessing the specified vector from the payload-order collection of
base vectors and inserted vectors, respectively.
"""

import collections
import sys

import numpy as np

from cr.cube.util import lazyproperty


class _BaseCollator(object):
    """Base class for all collator objects, providing shared properties.

    `empty_idxs` is a tuple of int element-index for each element vector with a zero
    unweighted N (N = 0).
    """

    def __init__(self, dimension, empty_idxs):
        self._dimension = dimension
        self._empty_idxs = tuple(empty_idxs) if empty_idxs else ()

    @lazyproperty
    def _element_ids(self):
        """Sequence of int element-id for each category or subvar in dimension.

        Element-ids appear in the order there were defined in the cube-result.
        """
        return self._dimension.element_ids

    @lazyproperty
    def _element_subvar_ids(self):
        """something to do with being able to use str ids (aliases) for subvars.

        Added by Slobodan, ask him for details. I expect the "element_ids": field in a
        explicit-order sort transform can be a list of strs instead of a list of ints.
        """
        return self._dimension.element_subvar_ids

    @lazyproperty
    def _hidden_idxs(self):
        """frozenset of int element-idx of each vector for which to suppress display."""
        empty_idxs = self._empty_idxs if self._dimension.prune else ()
        return frozenset(empty_idxs + self._dimension.hidden_idxs)

    @lazyproperty
    def _order_spec(self):
        """_OrderSpec object specifying ordering details."""
        return self._dimension.order_spec

    @lazyproperty
    def _subtotals(self):
        """Sequence of _Subtotal object for each inserted subtotal in dimension."""
        return self._dimension.subtotals


class _BaseAnchoredCollator(_BaseCollator):
    """Base class for collators that respect insertion anchors.

    The payload-order and explicit-order collators both respect the anchors on
    insertions while sort-by-value collators override those anchors. The two anchored
    collators share all their behaviors except how they order their base-elements.
    """

    @classmethod
    def display_order(cls, dimension, empty_idxs):
        """Return sequence of int element-idx specifying ordering of dimension elements.

        The returned indices are "signed", with positive indices applying to base
        vectors and negative indices applying to inserted vectors. Both work for
        indexing in their respective unordered collections.

        `empty_idxs` identifies vectors with N=0, which may be "pruned", depending on
        a user setting in the dimension.
        """
        return cls(dimension, empty_idxs)._display_order

    @lazyproperty
    def _display_order(self):
        """tuple of int element-idx for each element in assembly order.

        An assembled vector contains both base and inserted cells. The returned
        element-indices are signed; positive indices are base-elements and negative
        indices refer to inserted subtotals.
        """
        hidden_idxs = self._hidden_idxs
        return tuple(
            idx
            for _, idx in sorted(
                self._base_element_orderings + self._insertion_orderings
            )
            if idx not in hidden_idxs
        )

    @lazyproperty
    def _base_element_orderings(self):
        """tuple of (int: position, int: idx) for each base-vector value.

        The position of a base value is it's index in the ordered base vector.
        """
        return tuple(
            (position, idx) for position, idx, _ in self._element_order_descriptors
        )

    @lazyproperty
    def _element_order_descriptors(self):
        """tuple of (position, idx, element_id) triple for each element in dimension."""
        raise NotImplementedError(
            "`%s` must implement `._element_order_descriptors`" % type(self).__name__
        )

    @lazyproperty
    def _element_positions_by_id(self):
        """dict of int base-element position keyed by that element's id.

        Allows O(1) lookup of base-element position by element-idx for purposes of
        positioning an inserted subtotal after its anchor element.
        """
        return {
            element_id: position
            for position, _, element_id in self._element_order_descriptors
        }

    @lazyproperty
    def _insertion_orderings(self):
        """tuple of (int: position, int: idx) for each inserted-vector value.

        The position for an insertion is an int representation of its anchor and its
        idx is the *negative* offset of its position in the opposing insertions
        sequence (like -3, -2, -1 for a sequence of length 3). The negative idx
        works just as well as the normal one for accessing the subtotal but insures
        that an insertion at the same position as a base row always sorts *before*
        the base row.

        The `position` int for a subtotal is 0 for anchor "top", sys.maxsize for
        anchor "bottom", and int(anchor) + 1 for all others. The +1 ensures
        a subtotal appears *after* the vector it is anchored to and can be interpreted
        as the index of the base-element it should appear *before*.

        Multiple insertions having the same anchor appear in payload order within that
        group. The strictly increasing insertion index values (-3 < -2 < -1) ensure
        insertions with the same anchor appear in payload order after that anchor.
        """
        subtotals = self._subtotals
        n_subtotals = len(subtotals)
        neg_idxs = tuple(i - n_subtotals for i in range(n_subtotals))
        return tuple(
            (self._insertion_position(subtotal), neg_idx)
            for subtotal, neg_idx in zip(subtotals, neg_idxs)
        )

    def _insertion_position(self, subtotal):
        """Subtotal position expressed as int index among base-vector indices.

        The return value represents the payload-order base-vector idx *before which* the
        subtotal should appear (even though subtotals appear *after* the row they are
        anchored to.

        A subtotal with position `0` appears at the top, one with an anchor of `3`
        appears *before* the base row at offset 3; `sys.maxsize` is used as the position
        for a "bottom" anchored subtotal.

        To make this work, the position of a subtotal is idx+1 of the base row it
        is anchored to (for subtotals anchored to a row, not "top" or "bottom").
        Combining this +1 characteristic with placing subtotals before rows with
        idx=insertion_position produces the right positioning and also allows top and
        bottom anchors to work while representing the position as a single non-negative
        int.
        """
        anchor = subtotal.anchor

        # --- "top" and "bottom" have fixed position mappings ---
        if anchor == "top":
            return 0
        if anchor == "bottom":
            return sys.maxsize

        # --- otherwise look up anchor-element position by id and add 1, defaulting to
        # --- bottom if target anchor element not found ---
        element_id = int(anchor)
        return (
            self._element_positions_by_id[element_id] + 1
            if element_id in self._element_positions_by_id
            else sys.maxsize
        )


class ExplicitOrderCollator(_BaseAnchoredCollator):
    """Orders elements in the sequence specified in order transform."""

    @lazyproperty
    def _element_order_descriptors(self):
        """tuple of (position, idx, element_id) triple for each element in dimension.

        The `position` of an element is it's index in the ordered collection. The
        returned triples appear in position order, but the position can be accessed
        directly as the first item of each triple. The `idx` value in each tuple is the
        index of that element as it appears in the cube-result (unordered).

        An explicit-order transform on a dimension looks like::

            "transforms": {
                "(rows|columns)_dimension": {
                    "order": {
                        "type": "explicit",
                        "element_ids": [3, 1, 5]
                    }
                }
            }

        This descriptor collection represents all valid elements of the dimension, even
        if they are not mentioned in the `"element_ids":` list of the explicit-order
        transform.

        The algorithm is tolerant of mismatches between element-ids specified in the
        transform and those present on the dimension:

        * When a element-id appears in the ordering array but not in the dimension, that
          element-id is ignored. This is important because an element (e.g. category)
          can be removed from a variable by the user after the transform is saved to an
          analysis.

        * When the ordering array does not include all element-ids present on the
          dimension, those present appear first, in the specified order, and the
          remaining elements appear after, in payload order.
        """

        def iter_element_order_descriptors():
            """Generate (idx, element_id) pair for each base-vector value.

            The (idx, id) pairs are generated in position order. The position of an
            element is it's index in the ordered element sequence.
            """
            # --- OrderedDict mapping element-id to payload-order, like {15:0, 12:1,..}.
            # --- This gives us payload-idx lookup along with duplicate and leftover
            # --- tracking.

            # TODO: this has been crudely hacked to accommodate both subvar aliases or
            # int element-ids. All that detail should be pushed down and encapsulated in
            # some code or object that works out the type of the ids and allows this
            # algorithm to be uniform.
            remaining_element_idxs_by_id = collections.OrderedDict(
                (id_, (idx, subvar_id))
                for idx, (id_, subvar_id) in enumerate(
                    zip(self._element_ids, self._element_subvar_ids)
                )
            )
            remaining_element_idxs_by_subvar_id = collections.OrderedDict(
                (subvar_id, (idx, id_))
                for idx, (subvar_id, id_) in enumerate(
                    zip(self._element_subvar_ids, self._element_ids)
                )
            )

            # --- yield (idx, id) pair for each element mentioned by id in transform,
            # --- in the order mentioned. Remove each from remaining as we go to track
            # --- dups and leftovers.
            for element_id in self._order_spec.element_ids:
                # --- An element-id appearing in transform but not in dimension is
                # --- ignored. Also, an element-id that appears more than oncce in
                # --- order-array is only used on first encounter.

                # TODO: this duplication of the same logic for two cases is part of the
                # same crude hack as above.

                if element_id in remaining_element_idxs_by_subvar_id:
                    (idx, id_) = remaining_element_idxs_by_subvar_id.pop(element_id)
                    if id_ in remaining_element_idxs_by_id:
                        remaining_element_idxs_by_id.pop(id_)
                    yield idx, id_

                if element_id in remaining_element_idxs_by_id:
                    (idx, subvar_id) = remaining_element_idxs_by_id.pop(element_id)
                    if subvar_id in remaining_element_idxs_by_subvar_id:
                        remaining_element_idxs_by_subvar_id.pop(subvar_id)
                    yield idx, element_id

            # --- any remaining elements are generated in the order they originally
            # --- appeared in the cube-result.
            for element_id, (idx, _) in remaining_element_idxs_by_id.items():
                yield idx, element_id

        return tuple(
            (position, idx, element_id)
            for position, (idx, element_id) in enumerate(
                iter_element_order_descriptors()
            )
        )


class PayloadOrderCollator(_BaseAnchoredCollator):
    """Leaves elements in the order they appeared in the cube payload.

    Insertion anchors are respected and each insertion-index is interleaved according to
    the anchor specified in its insertion transform.
    """

    @lazyproperty
    def _element_order_descriptors(self):
        """tuple of (position, idx, element_id) triple for each element in dimension.

        In payload-order, the position of an element is simply it's index in the
        sequence of element-ids; the result looks like:

            ((0, 0, {id}), (1, 1, {id}), ..., (n, n, {id}))

        where `n` is `len(element_ids) - 1`.
        """
        return tuple(
            (idx, idx, element_id) for idx, element_id in enumerate(self._element_ids)
        )


class SortByValueCollator(_BaseCollator):
    """Produces an idx ordering based on values in a vector.

    In general, the anchors used to position inserted subtotals lose their meaning when
    the dimension is sorted by-value. In sort-by-value cases, subtotals are grouped at
    the top (when sort direction is descending (default)) or the bottom (when direction
    is ascending), while also being sorted by the specified value.

    `element_values`, `subtotal_values`, and `empty_idxs` must each be a sequence; a 1D
    numpy array works fine but so does a list or tuple. The items in `element_values`
    must correspond directly the the (valid) elements of `dimension`, both in number and
    sequence (payload order). The items in `subtotal_values` must also correspond
    directly in number and sequence with the subtotals defined on `dimension`.
    """

    def __init__(self, dimension, element_values, subtotal_values, empty_idxs):
        super(SortByValueCollator, self).__init__(dimension, empty_idxs)
        self._element_values = element_values
        self._subtotal_values = subtotal_values

    @classmethod
    def display_order(cls, dimension, element_values, subtotal_values, empty_idxs):
        """Return sequence of int element-idxs ordered by sort on `element_values`.

        The returned tuple contains a signed-int value for each vector and subtotal of
        `dimension` that is not hidden, sorted (primarily) by the element value.
        """
        return cls(
            dimension, element_values, subtotal_values, empty_idxs
        )._display_order

    @property
    def _display_order(self):
        """tuple of int element-idx specifying ordering of dimension elements.

        The element-indices are signed; positive indices are base-elements and negative
        indices refer to inserted subtotals.

        Subtotal elements all appear at the top when the sort direction is descending
        and all appear at the bottom when sort-direction is ascending. Top-anchored
        "fixed-from-sort" elements appear after any top subtotals, followed by
        non-fixed base-elements, bottom-anchored base-elements, and finally
        bottom-subtotals (only when sort-direction is ascending).

        Subtotal elements appear in value-sorted order, respecting the sort-direction
        specified in the request. Fixed base elements appear in the order mentioned
        in the `"fixed": [...]` array of the order transform. Base elements appear in
        value-sorted order within their grouping.
        """
        hidden_idxs = self._hidden_idxs
        return tuple(
            idx
            for idx in (
                self._top_subtotal_idxs
                + self._top_fixed_idxs
                + self._body_idxs
                + self._bottom_fixed_idxs
                + self._bottom_subtotal_idxs
            )
            if idx not in hidden_idxs
        )

    @lazyproperty
    def _body_idxs(self):
        """tuple of int element-idx for each non-anchored dimension element.

        These values appear in sorted order. The sequence is determined by the
        `._target_values` property defined in the subclass and the "top" and "bottom"
        anchored elements specified in the `"order": {}` dict.
        """
        fixed_idxs = frozenset(self._top_fixed_idxs + self._bottom_fixed_idxs)
        sorted_value_idx_pairs = sorted(
            (
                (value, idx)
                for idx, value in enumerate(self._element_values)
                if idx not in fixed_idxs
            ),
            reverse=self._descending,
        )
        return tuple(idx for _, idx in sorted_value_idx_pairs)

    @lazyproperty
    def _bottom_fixed_idxs(self):
        """Tuple of (positive) idx of each fixed base element anchored to bottom.

        The items appear in the order specified in the "bottom" fix-grouping of the
        transform; they are not subject to sorting-by-value.
        """
        return tuple(self._iter_fixed_idxs(self._order_spec.bottom_fixed_ids))

    @lazyproperty
    def _bottom_subtotal_idxs(self):
        """Tuple of negative idx of each subtotal vector in order it appears on bottom.

        Subtotal vectors all appear as a sorted group at the top of the table when the
        sort-direction is descending (the default). Otherwise all subtotal vectors
        appear at the bottom. In either case, they are ordered by the value of the
        specified measure in each subtotal, except that any NaN values drop to the end
        of the subtotal group.
        """
        return () if self._descending else self._subtotal_idxs

    @lazyproperty
    def _descending(self):
        """True if collation direction is larger-to-smaller, False otherwise.

        Descending is the default direction because it is so much more common than
        ascending in survey analysis.
        """
        return self._order_spec.descending

    def _iter_fixed_idxs(self, fixed_element_ids):
        """Generate the element-idx of each element represented by `fixed_element_ids`.

        Any element-id specified in the `fixed_element_ids` that is not present in the
        dimension is ignored. This is important because an element (e.g. category) can
        be removed after the analysis
        """
        element_idxs_by_id = {id_: idx for idx, id_ in enumerate(self._element_ids)}
        for fixed_element_id in fixed_element_ids:
            if fixed_element_id not in element_idxs_by_id:
                continue
            yield element_idxs_by_id[fixed_element_id]

    @lazyproperty
    def _subtotal_idxs(self):
        """tuple of int (negative) element-idx for each subtotal of this dimension.

        These values appear in sorted order. The values are determined by the
        `._subtotal_values` property defined in the subclass and the sort direction
        is derived from the `"order": {}` dict.
        """
        subtotal_values = self._subtotal_values
        n_values = len(subtotal_values)
        # --- `keys` looks like [(75.36, -3), (18.17, -2), (23.46, -1)], providing a
        # --- sequence that can be sorted and then harvested for ordered idxs.
        keys, nans = [], []
        for i, val in enumerate(subtotal_values):
            neg_idx = i - n_values
            group = nans if np.isnan(val) else keys
            group.append((val, neg_idx))

        return tuple(idx for _, idx in (sorted(keys, reverse=self._descending) + nans))

    @lazyproperty
    def _top_fixed_idxs(self):
        """Tuple of (positive) payload-order idx for each top-anchored element.

        The items appear in the order specified in the "top" fix-grouping of the
        transform; they are not subject to sorting-by-value.
        """
        return tuple(self._iter_fixed_idxs(self._order_spec.top_fixed_ids))

    @lazyproperty
    def _top_subtotal_idxs(self):
        """Tuple of negative idx of each subtotal vector in the order it appears on top.

        Subtotal vectors all appear as a sorted group at the top of the table when the
        sort-direction is descending. When sort-direction is ascending, all subtotal
        vectors appear at the bottom and this tuple will be empty.
        """
        return self._subtotal_idxs if self._descending else ()
