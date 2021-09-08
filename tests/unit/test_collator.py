# encoding: utf-8

"""Unit test suite for `cr.cube.collator` module."""

import sys

import numpy as np
import pytest

from cr.cube.collator import (
    _BaseAnchoredCollator,
    _BaseCollator,
    ExplicitOrderCollator,
    PayloadOrderCollator,
    SortByValueCollator,
)
from cr.cube.dimension import Dimension, _Element, _OrderSpec, _Subtotal, _ValidElements

from ..unitutil import (
    ANY,
    call,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_BaseCollator:
    """Unit-test suite for `cr.cube.collator._BaseCollator` object."""

    def it_provides_access_to_the_dimension_element_ids_to_help(self, dimension_):
        dimension_.element_ids = (42, 24, 1, 6)
        assert _BaseCollator(dimension_, None)._element_ids == (42, 24, 1, 6)

    @pytest.mark.parametrize(
        "empty_idxs, prune, hidden_idxs, expected_value",
        (
            ((), False, (), ()),
            ((), False, (4, 5, 6), (4, 5, 6)),
            ((1, 2, 3), False, (4, 5, 6), (4, 5, 6)),
            ((), True, (), ()),
            ((), True, (4, 5, 6), (4, 5, 6)),
            ((1, 2, 3), True, (4, 5, 6), (1, 2, 3, 4, 5, 6)),
        ),
    )
    def it_knows_the_hidden_idxs_to_help(
        self, dimension_, empty_idxs, prune, hidden_idxs, expected_value
    ):
        dimension_.hidden_idxs = hidden_idxs
        dimension_.prune = prune
        collator = _BaseCollator(dimension_, empty_idxs)

        assert collator._hidden_idxs == frozenset(expected_value)

    def it_provides_access_to_the_order_spec_to_help(self, request, dimension_):
        order_spec_ = instance_mock(request, _OrderSpec)
        dimension_.order_spec = order_spec_
        assert _BaseCollator(dimension_, None)._order_spec is order_spec_

    def it_provides_access_to_the_dimension_subtotals_to_help(self, dimension_):
        assert _BaseCollator(dimension_, None)._subtotals is dimension_.subtotals

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseAnchoredCollator:
    """Unit-test suite for `cr.cube.collator._BaseAnchoredCollator` object."""

    def it_provides_an_interface_classmethod(self, request, dimension_):
        _init_ = initializer_mock(request, _BaseAnchoredCollator)
        _display_order_ = property_mock(
            request, _BaseAnchoredCollator, "_display_order"
        )
        _display_order_.return_value = (-3, 0, 1, -2, 2, 3, -1)

        display_order = _BaseAnchoredCollator.display_order(dimension_, empty_idxs=(2,))

        _init_.assert_called_once_with(ANY, dimension_, (2,))
        assert display_order == (-3, 0, 1, -2, 2, 3, -1)

    @pytest.mark.parametrize(
        "base_order, insertion_order, derived_order, hidden_idxs, expected_value",
        (
            # --- base-values but no insertions nor derived ---
            (((0, 0, 0), (1, 0, 1), (2, 0, 2)), (), (), (), (0, 1, 2)),
            # --- insertions but no base-values nor derived (not expected) ---
            (
                (),
                ((sys.maxsize, 0, -3), (0, 1, -2), (sys.maxsize, 0, -1)),
                (),
                (1,),
                (-2, -3, -1),
            ),
            # --- both base-values and insertions but no derived ---
            (
                ((0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)),
                ((-1, 0, -4), (1, 1, -3), (2, -1, -2), (sys.maxsize, 0, -1)),
                (),
                (1, 3),
                (-4, 0, -3, -2, 2, -1),
            ),
            # --- base-values and derived elements (but no insertions) ---
            (
                ((0, 0, 0), (2, 0, 2), (3, 0, 3)),
                (),
                ((-1, 0, 4), (2, 1, 1)),
                (),
                (4, 0, 2, 1, 3),
            ),
        ),
    )
    def it_computes_the_display_order_to_help(
        self,
        request,
        base_order,
        insertion_order,
        derived_order,
        hidden_idxs,
        expected_value,
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_hidden_idxs",
            return_value=hidden_idxs,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_insertion_orderings",
            return_value=insertion_order,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_base_element_orderings",
            return_value=base_order,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_derived_element_orderings",
            return_value=derived_order,
        )

        assert _BaseAnchoredCollator(None, None)._display_order == expected_value

    @pytest.mark.parametrize(
        "element_order_descriptors, expected_value",
        (
            # --- no elements in dimension (not an expected case) ---
            ((), ()),
            # --- 1 element ---
            (((0, 0, 6),), ((0, 0, 0),)),
            # --- 3 elements in dimension ---
            (((0, 2, 18), (1, 0, 12), (2, 1, 5)), ((0, 0, 2), (1, 0, 0), (2, 0, 1))),
        ),
    )
    def it_computes_the_base_element_orderings_to_help(
        self, request, element_order_descriptors, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_order_descriptors",
            return_value=element_order_descriptors,
        )
        collator = _BaseAnchoredCollator(None, None)

        assert collator._base_element_orderings == expected_value

    def it_has_empty_derived_element_orderings(self):
        collator = _BaseAnchoredCollator(None, None)
        assert collator._derived_element_orderings == tuple()

    def it_raises_on_element_order_descriptors_access(self):
        """Error message identifies the non-implementing subclass."""
        with pytest.raises(NotImplementedError) as e:
            _BaseAnchoredCollator(None, None)._element_order_descriptors
        assert str(e.value) == (
            "`_BaseAnchoredCollator` must implement `._element_order_descriptors`"
        )

    @pytest.mark.parametrize(
        "element_order_descriptors, expected_value",
        (
            # --- no elements in dimension (not an expected case) ---
            ((), {}),
            # --- 1 element ---
            (((0, 0, 6),), {6: 0}),
            # --- 3 elements in dimension ---
            (((0, 2, 18), (1, 0, 12), (2, 1, 5)), {18: 0, 12: 1, 5: 2}),
        ),
    )
    def it_maps_the_element_ids_to_their_position_to_help(
        self, request, element_order_descriptors, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_order_descriptors",
            return_value=element_order_descriptors,
        )
        collator = _BaseAnchoredCollator(None, None)

        assert collator._element_positions_by_id == expected_value

    @pytest.mark.parametrize(
        "insertion_positions, expected_value",
        (
            # --- no inserted-subtotals in dimension ---
            ((), ()),
            # --- 1 insertion ---
            (((6, 1),), ((6, 1, -1),)),
            # --- 3 insertions ---
            (
                ((7, 1), (sys.maxsize, 0), (0, 1)),
                ((7, 1, -3), (sys.maxsize, 0, -2), (0, 1, -1)),
            ),
        ),
    )
    def it_computes_the_insertion_orderings_to_help(
        self, request, insertion_positions, expected_value
    ):
        subtotals_ = tuple(
            instance_mock(request, _Subtotal) for _ in insertion_positions
        )
        property_mock(
            request, _BaseAnchoredCollator, "_subtotals", return_value=subtotals_
        )
        _insertion_position_ = method_mock(
            request,
            _BaseAnchoredCollator,
            "_insertion_position",
            side_effect=iter(insertion_positions),
        )
        collator = _BaseAnchoredCollator(None, None)

        insertion_orderings = collator._insertion_orderings

        assert _insertion_position_.call_args_list == [
            call(collator, s) for s in subtotals_
        ]
        assert insertion_orderings == expected_value

    @pytest.mark.parametrize(
        "anchor, expected_value",
        (
            ("top", (-1, 0)),
            ("bottom", (sys.maxsize, 0)),
            (666, (sys.maxsize, 0)),
            (1, (0, 1)),
            (2, (1, 1)),
            (3, (2, 1)),
        ),
    )
    def it_can_compute_the_insertion_position_for_a_subtotal_to_help(
        self, request, anchor, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_positions_by_id",
            return_value={1: 0, 2: 1, 3: 2},
        )
        subtotal_ = instance_mock(request, _Subtotal, anchor=anchor)
        collator = _BaseAnchoredCollator(None, None)

        assert collator._insertion_position(subtotal_) == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class DescribeExplicitOrderCollator:
    """Unit-test suite for `cr.cube.collator.ExplicitOrderCollator` object."""

    @pytest.mark.parametrize(
        "ids, are_derived, element_positions, expected_value",
        (
            # --- 0 derived elements
            (
                ("s1", "s2"),
                (False, False),
                (),
                (),
            ),
            # --- 2 derived elements, 1 non-derived
            (
                ("s1", "s2", "s3"),
                (True, False, True),
                ((2, 0), (-1, 0)),
                ((2, 0, 0), (-1, 0, 2)),
            ),
        ),
    )
    def it_computes_the_derived_element_orderings_to_help(
        self, request, ids, are_derived, element_positions, expected_value
    ):
        derived_elements_ = tuple(
            instance_mock(request, _Element, element_id=id_, derived=is_derived)
            for id_, is_derived in zip(ids, are_derived)
        )
        property_mock(
            request, _BaseAnchoredCollator, "_elements", return_value=derived_elements_
        )
        _derived_element_position_ = method_mock(
            request,
            ExplicitOrderCollator,
            "_derived_element_position",
            side_effect=iter(element_positions),
        )
        collator = ExplicitOrderCollator(None, None)

        derived_orderings = collator._derived_element_orderings

        assert _derived_element_position_.call_args_list == [
            call(collator, s.element_id) for s in derived_elements_ if s.derived
        ]
        assert derived_orderings == expected_value

    @pytest.mark.parametrize(
        "element_id, anchor, expected_value",
        (
            (1, "top", (-1, 0)),
            (2, "bottom", (sys.maxsize, 0)),
            (3, {"alias": "a", "position": "before"}, (5, -1)),
            (4, {"alias": "c", "position": "after"}, (2, 1)),
            (2, None, (sys.maxsize, 0)),
        ),
    )
    def it_computes_the_derived_element_position_to_help(
        self, request, element_id, anchor, expected_value
    ):
        element_ = instance_mock(request, _Element, anchor=anchor)
        get_by_id_ = method_mock(
            request, _ValidElements, "get_by_id", return_value=element_
        )
        valid_elements_ = _ValidElements(None, None)
        property_mock(
            request, ExplicitOrderCollator, "_elements", return_value=valid_elements_
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_positions_by_id",
            return_value={"a": 5, "b": 1, "c": 2},
        )
        collator = ExplicitOrderCollator(None, None)

        actual_value = collator._derived_element_position(element_id)

        assert actual_value == expected_value
        get_by_id_.assert_called_once_with(valid_elements_, element_id)

    @pytest.mark.parametrize(
        "element_ids, element_id_order, expected_value",
        (
            ((), [], ()),
            (
                (1, 2, 3),
                [2, 3, 1],
                ((0, 1, 2), (1, 2, 3), (2, 0, 1)),
            ),
            # --- element not mentioned in transform appears at end in payload order ---
            ((1, 2, 3), [], ((0, 0, 1), (1, 1, 2), (2, 2, 3))),
            ((1, 2, 3), [2], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            # --- id appearing in transform but not dimension is ignored ---
            ((), [3, 2, 1], ()),
            (
                (1, 2, 3),
                [2, 4, 1],
                ((0, 1, 2), (1, 0, 1), (2, 2, 3)),
            ),
            # --- id repeated in transform is only used on first encounter ---
            (
                (1, 2, 3),
                [2, 2, 2],
                ((0, 1, 2), (1, 0, 1), (2, 2, 3)),
            ),
            (
                (1, 2, 3),
                [3, 1, 3],
                ((0, 2, 3), (1, 0, 1), (2, 1, 2)),
            ),
        ),
    )
    def it_computes_the_element_order_descriptors_to_help(
        self, request, element_ids, element_id_order, expected_value
    ):
        valid_elements_ = [
            instance_mock(request, _Element, element_id=id, derived=False)
            for id in element_ids
        ]
        dimension_ = instance_mock(request, Dimension, valid_elements=valid_elements_)
        order_spec_ = instance_mock(request, _OrderSpec, element_ids=element_id_order)
        property_mock(
            request,
            ExplicitOrderCollator,
            "_order_spec",
            return_value=order_spec_,
        )
        collator = ExplicitOrderCollator(dimension_, None)

        assert collator._element_order_descriptors == expected_value


class DescribePayloadOrderCollator:
    """Unit-test suite for `cr.cube.collator.PayloadOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, expected_value",
        (
            # --- no (valid) elements in dimension (not an expected case) ---
            ((), ()),
            # --- 1 element ---
            ((302,), ((0, 0, 302),)),
            # --- 3 elements in dimension ---
            ((47, 103, 18), ((0, 0, 47), (1, 1, 103), (2, 2, 18))),
        ),
    )
    def it_computes_the_element_order_descriptors_to_help(
        self, request, element_ids, expected_value
    ):
        property_mock(
            request, PayloadOrderCollator, "_element_ids", return_value=element_ids
        )
        collator = PayloadOrderCollator(None, None)

        assert collator._element_order_descriptors == expected_value


class DescribeSortByValueCollator:
    """Unit-test suite for `cr.cube.collator.SortByValueCollator` object.

    SortByValueCollator computes element ordering for sort-by-value order transforms. It
    is used for all sort-by-value cases, performing the sort based on a base-vector
    provided to it on construction. This base vector can be a "body" vector, a subtotal,
    or a marginal.
    """

    def it_provides_an_interface_classmethod(self, request):
        dimension_ = instance_mock(request, Dimension)
        _init_ = initializer_mock(request, SortByValueCollator)
        property_mock(
            request,
            SortByValueCollator,
            "_display_order",
            return_value=(-3, -1, -2, 1, 0, 2, 3),
        )
        element_values = [2, 3, 1, 0]
        subtotal_values = [9, 7, 8]
        empty_idxs = [3]

        display_order = SortByValueCollator.display_order(
            dimension_, element_values, subtotal_values, empty_idxs
        )

        _init_.assert_called_once_with(
            ANY, dimension_, element_values, subtotal_values, empty_idxs
        )
        assert display_order == (-3, -1, -2, 1, 0, 2, 3)

    def it_computes_the_display_order_to_help(
        self, request, _top_fixed_idxs_prop_, _bottom_fixed_idxs_prop_
    ):
        property_mock(
            request, SortByValueCollator, "_top_subtotal_idxs", return_value=(-2, -3)
        )
        _top_fixed_idxs_prop_.return_value = (3,)
        property_mock(
            request, SortByValueCollator, "_body_idxs", return_value=(2, 1, 5)
        )
        _bottom_fixed_idxs_prop_.return_value = (0, 4)
        property_mock(
            request, SortByValueCollator, "_bottom_subtotal_idxs", return_value=()
        )
        property_mock(request, SortByValueCollator, "_hidden_idxs", return_value=(5,))
        collator = SortByValueCollator(None, None, None, None)

        assert collator._display_order

    @pytest.mark.parametrize(
        "top_fixed_idxs, bottom_fixed_idxs, descending, element_values, expected_value",
        (
            # --- ascending sort ---
            ((), (), False, (8.0, 2.0, 4.0, 1.0), (3, 1, 2, 0)),
            # --- ascending with fixed on top (which therefore do not appear ---
            ((2, 1), (), False, (8.0, 2.0, 4.0, 1.0), (3, 0)),
            # --- descending sort ---
            ((), (), True, (8.0, 2.0, 4.0, 1.0), (0, 2, 1, 3)),
            # --- descending with fixed on bottom ---
            ((), (1, 0), True, (8.0, 2.0, 4.0, 1.0), (2, 3)),
            # --- descending with both kinds of fixed idxs ---
            ((0,), (3,), True, (8.0, 2.0, 4.0, 1.0), (2, 1)),
            # --- ascending sort with nans (at end in order) ---
            ((), (), False, (8.0, np.nan, 2.0, np.nan), (2, 0, 1, 3)),
            # --- descending sort with nans (at end in order) ---
            ((), (), True, (8.0, np.nan, 2.0, np.nan), (0, 2, 1, 3)),
        ),
    )
    def it_computes_the_sorted_body_idxs_to_help(
        self,
        _top_fixed_idxs_prop_,
        _bottom_fixed_idxs_prop_,
        _descending_prop_,
        top_fixed_idxs,
        bottom_fixed_idxs,
        descending,
        element_values,
        expected_value,
    ):
        """Body-idxs are for elements that are not subtotals and not fixed."""
        _top_fixed_idxs_prop_.return_value = top_fixed_idxs
        _bottom_fixed_idxs_prop_.return_value = bottom_fixed_idxs
        _descending_prop_.return_value = descending
        collator = SortByValueCollator(None, element_values, None, None)

        assert collator._body_idxs == expected_value

    def it_computes_the_bottom_exclusion_idxs_to_help(
        self, _iter_fixed_idxs_, _order_spec_prop_, order_spec_
    ):
        _iter_fixed_idxs_.return_value = (n for n in (4, 0, 5, 2))
        _order_spec_prop_.return_value = order_spec_
        order_spec_.bottom_fixed_ids = (0, 1, 2)
        collator = SortByValueCollator(None, None, None, None)

        bottom_exclusion_idxs = collator._bottom_fixed_idxs

        _iter_fixed_idxs_.assert_called_once_with(collator, (0, 1, 2))
        assert bottom_exclusion_idxs == (4, 0, 5, 2)

    @pytest.mark.parametrize(
        "descending, subtotal_idxs, expected_value",
        (
            # --- ascending sort, all subtotals at bottom ---
            (False, (-3, -1, -2), (-3, -1, -2)),
            # --- descending sort, no subtotals at bottom ---
            (True, (-3, -1, -2), ()),
        ),
    )
    def it_computes_the_bottom_subtotal_idxs_to_help(
        self,
        _descending_prop_,
        _subtotal_idxs_prop_,
        descending,
        subtotal_idxs,
        expected_value,
    ):
        _descending_prop_.return_value = descending
        _subtotal_idxs_prop_.return_value = subtotal_idxs
        collator = SortByValueCollator(None, None, None, None)

        assert collator._bottom_subtotal_idxs == expected_value

    @pytest.mark.parametrize(
        "fixed_element_ids, expected_value",
        (((), ()), ((1, 3), (0, 2)), ((1, 3, 7), (0, 2)), ((4, 2), (3, 1))),
    )
    def it_can_iterate_the_fixed_idxs_to_help(
        self, request, fixed_element_ids, expected_value
    ):
        property_mock(
            request, SortByValueCollator, "_element_ids", return_value=(1, 2, 3, 4, 5)
        )
        collator = SortByValueCollator(None, None, None, None)

        assert tuple(collator._iter_fixed_idxs(fixed_element_ids)) == expected_value

    @pytest.mark.parametrize(
        "descending, expected_value",
        (
            (True, True),
            (False, False),
        ),
    )
    def it_knows_whether_the_sort_direction_is_descending_to_help(
        self, _order_spec_prop_, order_spec_, descending, expected_value
    ):
        _order_spec_prop_.return_value = order_spec_
        order_spec_.descending = descending
        collator = SortByValueCollator(None, None, None, None)

        assert collator._descending == expected_value

    @pytest.mark.parametrize(
        "value, expected",
        ((0, False), (1.1, False), ("a", False), (np.nan, True)),
    )
    def it_can_compute_is_nan_to_help(self, value, expected):
        assert SortByValueCollator._is_nan(value) == expected

    @pytest.mark.parametrize(
        "subtotal_values, descending, expected_value",
        (
            ((1, 2, 3), True, (-1, -2, -3)),
            ((1, 2, 3), False, (-3, -2, -1)),
            # --- NaN values fall to end of sequence in payload order ---
            ((1, np.nan, 2, np.nan, 3), True, (-1, -3, -5, -4, -2)),
            # --- regardless of collation-order ---
            ((1, np.nan, 2, np.nan, 3), False, (-5, -3, -1, -4, -2)),
        ),
    )
    def it_computes_the_subtotal_idxs_to_help(
        self, _descending_prop_, descending, subtotal_values, expected_value
    ):
        _descending_prop_.return_value = descending
        collator = SortByValueCollator(None, None, subtotal_values, None)

        assert collator._subtotal_idxs == expected_value

    def it_computes_the_top_fixed_idxs_to_help(
        self, _order_spec_prop_, order_spec_, _iter_fixed_idxs_
    ):
        _order_spec_prop_.return_value = order_spec_
        order_spec_.top_fixed_ids = (4, 2)
        _iter_fixed_idxs_.return_value = (i for i in (1, 2, 4))
        collator = SortByValueCollator(None, None, None, None)

        top_exclusion_idxs = collator._top_fixed_idxs

        _iter_fixed_idxs_.assert_called_once_with(collator, (4, 2))
        assert top_exclusion_idxs == (1, 2, 4)

    @pytest.mark.parametrize(
        "subtotal_idxs, descending, expected_value",
        (((-3, -1, -2), True, (-3, -1, -2)), ((-1, -2, -3), False, ())),
    )
    def it_computes_the_top_subtotal_idxs_to_help(
        self,
        _subtotal_idxs_prop_,
        subtotal_idxs,
        _descending_prop_,
        descending,
        expected_value,
    ):
        _subtotal_idxs_prop_.return_value = subtotal_idxs
        _descending_prop_.return_value = descending
        collator = SortByValueCollator(None, None, None, None)

        assert collator._top_subtotal_idxs == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _bottom_fixed_idxs_prop_(self, request):
        return property_mock(request, SortByValueCollator, "_bottom_fixed_idxs")

    @pytest.fixture
    def _descending_prop_(self, request):
        return property_mock(request, SortByValueCollator, "_descending")

    @pytest.fixture
    def _iter_fixed_idxs_(self, request):
        return method_mock(request, SortByValueCollator, "_iter_fixed_idxs")

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, SortByValueCollator, "_order_spec")

    @pytest.fixture
    def _subtotal_idxs_prop_(self, request):
        return property_mock(request, SortByValueCollator, "_subtotal_idxs")

    @pytest.fixture
    def _top_fixed_idxs_prop_(self, request):
        return property_mock(request, SortByValueCollator, "_top_fixed_idxs")
