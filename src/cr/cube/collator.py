# encoding: utf-8

"""Objects related to ordering the elements in a dimension.

There are several different collation (ordering) methods, each of which has a distinct
collator class here. Each has the single public (class)method `.display_order()` which
returns a tuple of signed int indices. A positive index refers to a base (non-inserted)
element and a negative index refers to an inserted subtotal. Both positive and negative
indices work for accessing the specified vector from the payload-order collection of
base vectors and inserted vectors, respectively.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from cr.cube.util import lazyproperty


class _BaseCollator(object):
    """Base class for all collator objects, providing shared properties."""

    def __init__(self, dimension):
        self._dimension = dimension

    @lazyproperty
    def _element_ids(self):
        """Sequence of int element-id for each category or subvar in dimension.

        Element-ids appear in the order there were defined in the cube-result.
        """
        return self._dimension.element_ids

    @lazyproperty
    def _subtotals(self):
        """Sequence of _Subtotal object for each inserted subtotal in dimension."""
        raise NotImplementedError


class _BaseAnchoredCollator(_BaseCollator):
    """Base class for collators that respect insertion anchors.

    The payload-order and explicit-order collators both respect the anchors on
    insertions while sort-by-value collators override those anchors. The two anchored
    collators share all their behaviors except how they order their base-elements.
    """

    @classmethod
    def display_order(cls, dimension):
        """ -> sequence of int element-idx specifying ordering of dimension elements.

        The returned indices are "signed", with positive indices applying to base
        vectors and negative indices applying to inserted vectors. Both work for
        indexing in their respective unordered collections.
        """
        return cls(dimension)._display_order

    @property
    def _display_order(self):
        """tuple of int element-idx for each element in assembly order.

        An assembled vector contains both base and inserted cells. The returned
        element-indices are signed; positive indices are base-elements and negative
        indices refer to inserted subtotals.
        """
        return tuple(
            idx
            for _, idx in sorted(
                self._base_element_orderings + self._insertion_orderings
            )
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
            "`._element_order_descriptors` must be implemented by each subclass"
        )

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
        raise NotImplementedError


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
