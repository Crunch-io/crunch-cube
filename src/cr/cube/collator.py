# encoding: utf-8

"""Objects related to ordering the elements in a dimension.

There are several different collation (ordering) methods, each of which has a distinct
collator class here. Each has the single public (class)method `.display_order()` which
returns a tuple of signed int indices. A positive index refers to a base (non-inserted)
element and a negative index refers to an inserted subtotal. Both positive and negative
indices work for accessing the specified vector from the payload-order collection of
base vectors and inserted vectors, respectively.
"""


class _BaseCollator(object):
    """Base class for all collator objects, providing shared properties."""

    def __init__(self, dimension, empty_idxs):
        self._dimension = dimension
        self._empty_idxs = tuple(empty_idxs) if empty_idxs else ()


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
        """
        return cls(dimension, empty_idxs)._display_order

    @property
    def _display_order(self):
        """tuple of int element-idx for each element in assembly order.

        An assembled vector contains both base and inserted cells. The returned
        element-indices are signed; positive indices are base-elements and negative
        indices refer to inserted subtotals.
        """
        raise NotImplementedError


class PayloadOrderCollator(_BaseAnchoredCollator):
    """Leaves elements in the order they appeared in the cube payload.

    Insertion anchors are respected and each insertion-index is interleaved according to
    the anchor specified in its insertion transform.
    """
