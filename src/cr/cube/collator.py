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

import collections
import sys

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
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
    def _order_dict(self):
        """dict transforms payload section specifying ordering."""
        return self._dimension.order_dict

    @lazyproperty
    def _subtotals(self):
        """Sequence of _Subtotal object for each inserted subtotal in dimension."""
        # --- elements of an aggregate/array dimension cannot meaningfully be summed, so
        # --- an array dimension cannot have subtotals
        if self._dimension.dimension_type in (DT.MR, DT.CA_SUBVAR):
            return tuple()
        return self._dimension.subtotals


# === CONVENTIONAL (ANCHOR-PRESERVING) COLLATORS ===


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

        # --- otherwise look up anchor-element position by id and add 1 ---
        element_positions_by_id = self._element_positions_by_id
        element_id = int(anchor)

        # --- default to bottom if target anchor element not found ---
        if element_id not in element_positions_by_id:
            return sys.maxsize

        return element_positions_by_id[element_id] + 1


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
            """Generate (idx, element_id) triple for each base-vector value.

            The (idx, id) pairs are generated in position order. The position of an
            element is it's index in the ordered element sequence.
            """
            # --- OrderedDict mapping element-id to payload-order, like {15:0, 12:1,..}.
            # --- This gives us payload-idx lookup along with duplicate and leftover
            # --- tracking.
            remaining_element_idxs_by_id = collections.OrderedDict(
                (id_, idx) for idx, id_ in enumerate(self._element_ids)
            )

            # --- yield (idx, id) pair for each element mentioned by id in transform,
            # --- in the order mentioned. Remove each from remaining as we go to track
            # --- dups and leftovers.
            for element_id in self._order_dict.get("element_ids", []):
                # --- An element-id appearing in transform but not in dimension is
                # --- ignored. Also, an element-id that appears more than oncce in
                # --- order-array is only used on first encounter.
                if element_id not in remaining_element_idxs_by_id:
                    continue
                yield remaining_element_idxs_by_id.pop(element_id), element_id

            # --- any remaining elements are generated in the order they originally
            # --- appeared in the cube-result.
            for element_id, idx in remaining_element_idxs_by_id.items():
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


# === SORT-BY-VALUE COLLATORS ===


class _BaseSortByValueCollator(_BaseCollator):
    """Base class for Collators that perform sorting of values.

    In general, the anchors used to position inserted subtotals lose their meaning when
    the dimension is sorted by-value. In sort-by-value cases, subtotals are grouped at
    the top (when sort direction is descending (default)) or the bottom (when direction
    is ascending), while also being sorted within the group of subtotal by the specified
    value.
    """

    @property
    def _display_order(self):
        """tuple of int element-idx specifying ordering of dimension elements.

        The element-indices are signed; positive indices are base-elements and negative
        indices refer to inserted subtotals.

        Subtotal elements all appear at the top when the sort direction is descending
        and all appear at the bottom when sort-direction is ascending. Top-anchored
        "excluded-from-sort" elements appear after any top subtotals, followed by
        non-excluded base-elements, bottom-anchored base-elements, and finally
        bottom-subtotals (only when sort-direction is ascending).

        Subtotal elements appear in value-sorted order, respecting the sort-direction
        specified in the request. Excluded base elements appear in the order mentioned
        in the `"exclude": [...]` array of the order transform. Base elements appear in
        value-sorted order within their grouping.
        """
        return (
            self._top_subtotal_idxs
            + self._top_exclusion_idxs
            + self._body_idxs
            + self._bottom_exclusion_idxs
            + self._bottom_subtotal_idxs
        )

    @lazyproperty
    def _body_idxs(self):
        """tuple of int element-idx for each non-anchored dimension element.

        These values appear in sorted order. The sequence is determined by the
        `._target_values` property defined in the subclass and the "top" and "bottom"
        anchored elements specified in the `"order": {}` dict.
        """
        excluded_idxs = frozenset(
            self._top_exclusion_idxs + self._bottom_exclusion_idxs
        )
        sorted_value_idx_pairs = sorted(
            (
                (value, idx)
                for idx, value in enumerate(self._element_values)
                if idx not in excluded_idxs
            ),
            reverse=self._descending,
        )
        return tuple(idx for _, idx in sorted_value_idx_pairs)

    @lazyproperty
    def _bottom_exclusion_idxs(self):
        """Tuple of (positive) idx of each excluded base element anchored to bottom.

        The items appear in the order specified in the "bottom" exclude-grouping of the
        transform; they are not subject to sorting-by-value.
        """
        return tuple(self._iter_exclusion_idxs("bottom"))

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
        return self._order_dict.get("direction", "descending") != "ascending"

    @lazyproperty
    def _element_idxs_by_id(self):
        """dict mapping element-id to payload-order element-idx."""
        return {id_: idx for idx, id_ in enumerate(self._element_ids)}

    @lazyproperty
    def _element_values(self):
        """tuple of the measure value for each dimension element, in payload order."""
        raise NotImplementedError(
            "`._element_values must be implemented by each subclass"
        )

    def _iter_exclusion_idxs(self, top_or_bottom):
        """Generate the element-idx of each exclusion in the `top_or_bottom` group.

        `top_or_bottom` must be one of "top" or "bottom". Any element-id specified in
        the exclusion-group that is not present in the dimension is ignored. This is
        important because an element (e.g. category) can be removed after the analysis
        is saved and may no longer be present at export time.
        """
        element_idxs_by_id = self._element_idxs_by_id
        for element_id in self._order_dict.get("exclude", {}).get(top_or_bottom, []):
            if element_id not in element_idxs_by_id:
                continue
            yield element_idxs_by_id[element_id]

    @lazyproperty
    def _measure_propname(self):
        """ -> str base-vector property name corresponding to `measure`."""
        # --- the `"measure":` field is required. If this statement raises KeyError
        # --- (in either of the two lookups), it indicates a validation gap. There is no
        # --- default measure.
        return {"unweighted_count": "unweighted_counts", "count": "counts"}[
            self._order_dict["measure"]
        ]

    @lazyproperty
    def _subtotal_idxs(self):
        """tuple of int (negative) element-idx for each subtotal of this dimension.

        These values appear in sorted order. The sequence is determined by the
        `._subtotal_values` property defined in the subclass and sort direction
        specified in the `"order": {}` dict.
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
    def _subtotal_values(self):
        """tuple of the measure value for each inserted subtotal, in payload order."""
        raise NotImplementedError(
            "`._subtotal_values must be implemented by each subclass"
        )

    @lazyproperty
    def _top_exclusion_idxs(self):
        """Tuple of (positive) payload-order idx for each top-anchored element.

        The items appear in the order specified in the "top" exclude-grouping of the
        transform; they are not subject to sorting-by-value.
        """
        return tuple(self._iter_exclusion_idxs("top"))

    @lazyproperty
    def _top_subtotal_idxs(self):
        """Tuple of negative idx of each subtotal vector in the order it appears on top.

        Subtotal vectors all appear as a sorted group at the top of the table when the
        sort-direction is descending. When sort-direction is ascending, all subtotal
        vectors appear at the bottom.
        """
        return self._subtotal_idxs if self._descending else ()


class MarginalCollator(_BaseSortByValueCollator):
    """Orders elements in the sequence of specified marginal values.

    `vectors` is a sequence of vector objects to be interrogated for their marginal
    value. A typical example is `cr.cube.matrix._CategoricalVector` with properties like
    `.base` and `.margin` (unweighted and weighted-N respectively).
    """

    def __init__(self, dimension, vectors, inserted_vectors):
        super(MarginalCollator, self).__init__(dimension)
        self._vectors = vectors
        self._inserted_vectors = inserted_vectors

    @classmethod
    def display_order(cls, dimension, vectors, inserted_vectors):
        """ -> sequence of int element-idx, reflecting sort-by-marginal transform.

        This value is an exhaustive collection of (valid) element offsets, sorted by the
        value of their margin value.

        An sort-by-marginal transform on a dimension looks like::

            "transforms": {
                "(rows|columns)_dimension": {
                    "order": {
                        "type": "marginal",
                        "marginal": "base",  # --- unweighted-N ---
                        "exclude": {"bottom": [999]},
                    }
                }
            }

        """
        return cls(dimension, vectors, inserted_vectors)._display_order

    @lazyproperty
    def _element_values(self):
        """Sequence of marginal values for the provided vectors.

        These would be array-values like rows-margin, columns-base, and
        table_proportions.

        Can possibly include NaN values.
        """
        propname = self._marginal_propname
        return tuple(getattr(v, propname) for v in self._vectors)

    @lazyproperty
    def _marginal_propname(self):
        """ -> str property name corresponding to the specified marginal."""
        # --- the `"marginal":` field is required. If this statement raises KeyError, it
        # --- indicates a validation gap. There is no default marginal.
        return {"unweighted_N": "base", "weighted_N": "margin"}[
            self._order_dict["marginal"]
        ]

    @lazyproperty
    def _subtotal_values(self):
        """tuple of the measure value for each inserted subtotal, in payload order."""
        marginal_propname = self._marginal_propname
        return tuple(getattr(v, marginal_propname) for v in self._inserted_vectors)


class OpposingElementCollator(_BaseSortByValueCollator):
    """Orders elements by the values of an opposing base vector.

    This would be like "order rows in descending order by value of 'Strongly Agree'
    column. An opposing-element ordering is only available on a matrix, because only
    a matrix dimension has an opposing dimension.
    """

    def __init__(self, dimension, opposing_vectors):
        super(OpposingElementCollator, self).__init__(dimension)
        self._opposing_vectors = opposing_vectors

    @classmethod
    def display_order(cls, dimension, opposing_vectors):
        """ -> sequence of int element-idx specifying ordering of dimension elements."""
        return cls(dimension, opposing_vectors)._display_order

    @lazyproperty
    def _element_values(self):
        """tuple of meaure values in the specified opposing vector, in payload order.

        Can possibly include NaN values.
        """
        return tuple(getattr(self._opposing_vector, self._measure_propname))

    @lazyproperty
    def _opposing_vector(self):
        """Base-vector object providing key-values for the sort."""
        key_element_id = self._order_dict["element_id"]
        for vector in self._opposing_vectors:
            if vector.element_id == key_element_id:
                return vector
        raise ValueError(
            "sort-by-value key element-id %r not found in opposing dimension"
            % key_element_id
        )

    @lazyproperty
    def _subtotal_values(self):
        """tuple of the measure value of each inserted subtotal, in payload order."""
        measure_propname = self._measure_propname

        def subtotal_value(subtotal):
            values = getattr(self._opposing_vector, measure_propname)
            return sum(values[idx] for idx in subtotal.addend_idxs)

        return tuple(subtotal_value(s) for s in self._subtotals)


class OpposingSubtotalCollator(_BaseSortByValueCollator):
    """Orders elements by value of an opposing subtotal vector.

    This would be like "order rows in descending order by value of 'Top 3' subtotal
    column". An opposing-subtotal ordering is only available on a matrix, because only
    a matrix dimension can have an opposing dimension.
    """

    def __init__(self, dimension, opposing_inserted_vectors):
        super(OpposingSubtotalCollator, self).__init__(dimension)
        self._opposing_inserted_vectors = opposing_inserted_vectors

    @classmethod
    def display_order(cls, dimension, opposing_inserted_vectors):
        """ -> sequence of int element-idx specifying ordering of dimension elements.

        The returned indices are "signed", with positive indices applying to base
        vectors and negative indices applying to inserted vectors. Both work for
        indexing in their respective unordered collections.
        """
        return cls(dimension, opposing_inserted_vectors)._display_order

    @lazyproperty
    def _opposing_subtotal(self):
        """Insertion-vector object providing values by which to sort."""
        key_insertion_id = self._order_dict["insertion_id"]
        for v in self._opposing_inserted_vectors:
            if v.insertion_id == key_insertion_id:
                return v
        raise ValueError(
            "sort-by-subtotal key insertion-id %r not found in opposing insertions"
            % key_insertion_id
        )

    @lazyproperty
    def _subtotal_values(self):
        """tuple of the measure value for each inserted subtotal, in payload order."""
        measure_propname = self._measure_propname

        # TODO: this calculation needs to be more sophisticated than just `sum()` for
        # certain measures. Probably just np.nan in most of those cases, but in general
        # the function should be looked up from a source shared with the
        # vector-assembler, as long as all subtotaling functions can share a signature.
        def subtotal_value(subtotal):
            values = getattr(self._opposing_subtotal, measure_propname)
            return sum(values[idx] for idx in subtotal.addend_idxs)

        return tuple(subtotal_value(s) for s in self._subtotals)