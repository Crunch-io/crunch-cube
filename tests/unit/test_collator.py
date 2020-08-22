# encoding: utf-8

"""Unit test suite for `cr.cube.collator` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import pytest

from cr.cube.collator import (
    _BaseCollator,
    _BaseAnchoredCollator,
    _BaseSortByValueCollator,
    ExplicitOrderCollator,
    MarginalCollator,
    PayloadOrderCollator,
)
from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.enums import DIMENSION_TYPE as DT

from ..unitutil import (
    ANY,
    call,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_BaseCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseCollator` object."""

    def it_provides_access_to_the_dimension_element_ids_to_help(self, dimension_):
        dimension_.element_ids = (42, 24, 1, 6)
        collator = _BaseCollator(dimension_)

        element_ids = collator._element_ids

        assert element_ids == (42, 24, 1, 6)

    def it_provides_access_to_the_order_transform_dict_to_help(self, dimension_):
        dimension_.order_dict = {"order": "dict"}
        collator = _BaseCollator(dimension_)

        order_dict = collator._order_dict

        assert order_dict == {"order": "dict"}

    @pytest.mark.parametrize(
        "subtotals, dimension_type, expected_value",
        (
            ((), DT.CAT, ()),
            ((), DT.MR, ()),
            (("sub", "tot", "als"), DT.CAT, ("sub", "tot", "als")),
            (("sub", "tot", "als"), DT.CA_SUBVAR, ()),
            (("sub", "tot", "als"), DT.MR, ()),
        ),
    )
    def it_provides_access_to_the_dimension_subtotals_to_help(
        self, dimension_, subtotals, dimension_type, expected_value
    ):
        dimension_.dimension_type = dimension_type
        dimension_.subtotals = subtotals
        collator = _BaseCollator(dimension_)

        subtotals = collator._subtotals

        assert subtotals == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseAnchoredCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseAnchoredCollator` object."""

    def it_provides_an_interface_classmethod(self, request, dimension_):
        _init_ = initializer_mock(request, _BaseAnchoredCollator)
        _display_order_ = property_mock(
            request, _BaseAnchoredCollator, "_display_order"
        )
        _display_order_.return_value = (-3, 0, 1, -2, 2, 3, -1)

        display_order = _BaseAnchoredCollator.display_order(dimension_)

        _init_.assert_called_once_with(ANY, dimension_)
        assert display_order == (-3, 0, 1, -2, 2, 3, -1)

    @pytest.mark.parametrize(
        "base_orderings, insertion_orderings, expected_value",
        (
            # --- base-values but no insertions ---
            (((0, 0), (1, 1), (2, 2)), (), (0, 1, 2)),
            # --- insertions but no base-values (not expected) ---
            ((), ((sys.maxsize, -3), (0, -2), (sys.maxsize, -1)), (-2, -3, -1)),
            # --- both base-values and insertions ---
            (
                ((0, 0), (1, 1), (2, 2), (3, 3)),
                ((0, -3), (2, -2), (sys.maxsize, -1)),
                (-3, 0, 1, -2, 2, 3, -1),
            ),
        ),
    )
    def it_computes_the_display_order_to_help(
        self, request, base_orderings, insertion_orderings, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_insertion_orderings",
            return_value=insertion_orderings,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_base_element_orderings",
            return_value=base_orderings,
        )
        collator = _BaseAnchoredCollator(None)

        display_order = collator._display_order

        assert display_order == expected_value

    @pytest.mark.parametrize(
        "element_order_descriptors, expected_value",
        (
            # --- no elements in dimension (not an expected case) ---
            ((), ()),
            # --- 1 element ---
            (((0, 0, 6),), ((0, 0),)),
            # --- 3 elements in dimension ---
            (((0, 2, 18), (1, 0, 12), (2, 1, 5)), ((0, 2), (1, 0), (2, 1))),
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
        collator = _BaseAnchoredCollator(None)

        base_element_orderings = collator._base_element_orderings

        assert base_element_orderings == expected_value

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
        collator = _BaseAnchoredCollator(None)

        element_positions_by_id = collator._element_positions_by_id

        assert element_positions_by_id == expected_value

    @pytest.mark.parametrize(
        "insertion_positions, expected_value",
        (
            # --- no inserted-subtotals in dimension ---
            ((), ()),
            # --- 1 insertion ---
            ((6,), ((6, -1),)),
            # --- 3 insertions ---
            ((7, sys.maxsize, 0), ((7, -3), (sys.maxsize, -2), (0, -1))),
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
        collator = _BaseAnchoredCollator(None)

        insertion_orderings = collator._insertion_orderings

        assert _insertion_position_.call_args_list == [
            call(collator, s) for s in subtotals_
        ]
        assert insertion_orderings == expected_value

    @pytest.mark.parametrize(
        "anchor, expected_value",
        (
            ("top", 0),
            ("bottom", sys.maxsize),
            (666, sys.maxsize),
            (1, 1),
            (2, 2),
            (3, 3),
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
        collator = _BaseAnchoredCollator(None)

        insertion_position = collator._insertion_position(subtotal_)

        assert insertion_position == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseSortByValueCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseSortByValueCollator` object."""

    def it_computes_the_display_order_to_help(
        self, request, _top_exclusion_idxs_prop_, _bottom_exclusion_idxs_prop_
    ):
        _top_exclusion_idxs_prop_.return_value = (3,)
        _bottom_exclusion_idxs_prop_.return_value = (7, 8)
        property_mock(
            request, _BaseSortByValueCollator, "_top_subtotal_idxs", return_value=(1, 2)
        )
        property_mock(
            request, _BaseSortByValueCollator, "_body_idxs", return_value=(4, 5, 6)
        )
        property_mock(
            request,
            _BaseSortByValueCollator,
            "_bottom_subtotal_idxs",
            return_value=(9,),
        )
        collator = _BaseSortByValueCollator(None)

        display_order = collator._display_order

        assert display_order == (1, 2, 3, 4, 5, 6, 7, 8, 9)

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({"direction": "ascending"}, False),
            ({"direction": "descending"}, True),
            ({}, True),
            ({"direction": None}, True),
            ({"direction": 42}, True),
            ({"direction": False}, True),
            ({"direction": "foobar"}, True),
        ),
    )
    def it_knows_whether_the_sort_direction_is_descending(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        """Otherwise, sort direction is ascending."""
        _order_dict_prop_.return_value = order_dict
        collator = _BaseSortByValueCollator(None)

        descending = collator._descending

        assert descending == expected_value

    @pytest.mark.parametrize(
        "descending, subtotal_idxs, expected_value",
        (
            # --- ascending sort, no subtotals at top ---
            (False, (-3, -1, -2), ()),
            # --- descending sort, all subtotals at top ---
            (True, (-3, -1, -2), (-3, -1, -2)),
        ),
    )
    def it_knows_the_top_subtotal_idxs_to_help(
        self,
        _descending_prop_,
        _subtotal_idxs_prop_,
        descending,
        subtotal_idxs,
        expected_value,
    ):
        _descending_prop_.return_value = descending
        _subtotal_idxs_prop_.return_value = subtotal_idxs
        collator = _BaseSortByValueCollator(None)

        top_subtotal_idxs = collator._top_subtotal_idxs

        assert top_subtotal_idxs == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _bottom_exclusion_idxs_prop_(self, request):
        return property_mock(
            request, _BaseSortByValueCollator, "_bottom_exclusion_idxs"
        )

    @pytest.fixture
    def _descending_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_descending")

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_order_dict")

    @pytest.fixture
    def _subtotal_idxs_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_subtotal_idxs")

    @pytest.fixture
    def _top_exclusion_idxs_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_top_exclusion_idxs")


class DescribeExplicitOrderCollator(object):
    """Unit-test suite for `cr.cube.collator.ExplicitOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, element_id_order, expected_value",
        (
            ((), [], ()),
            ((1, 2, 3), [2, 3, 1], ((0, 1, 2), (1, 2, 3), (2, 0, 1))),
            # --- element not mentioned in transform appears at end in payload order ---
            ((1, 2, 3), [], ((0, 0, 1), (1, 1, 2), (2, 2, 3))),
            ((1, 2, 3), [2], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            # --- id appearing in transform but not dimension is ignored ---
            ((), [3, 2, 1], ()),
            ((1, 2, 3), [2, 4, 1], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            # --- id repeated in transform is only used on first encounter ---
            ((1, 2, 3), [2, 2, 2], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            ((1, 2, 3), [3, 1, 3], ((0, 2, 3), (1, 0, 1), (2, 1, 2))),
        ),
    )
    def it_computes_the_element_order_descriptors_to_help(
        self, request, element_ids, element_id_order, expected_value
    ):
        property_mock(
            request, ExplicitOrderCollator, "_element_ids", return_value=element_ids
        )
        property_mock(
            request,
            ExplicitOrderCollator,
            "_order_dict",
            return_value={"element_ids": element_id_order},
        )
        collator = ExplicitOrderCollator(None)

        element_order_descriptors = collator._element_order_descriptors

        assert element_order_descriptors == expected_value


class DescribeMarginalCollator(object):
    """Unit-test suite for `cr.cube.collator.MarginalCollator` object.

    MarginalCollator computes element ordering for sort-by-marginal order transforms.
    Margins it can sort by include weighted-N (margin), unweighted-N (base), and "All"
    (percent).
    """

    def it_provides_an_interface_classmethod(self, request, dimension_):
        _init_ = initializer_mock(request, MarginalCollator)
        _display_order_ = property_mock(request, MarginalCollator, "_display_order")
        _display_order_.return_value = (-3, -1, -2, 1, 0, 2, 3)

        display_order = MarginalCollator.display_order(
            dimension_, ("vec", "tors"), ("inserted", "vectors")
        )

        _init_.assert_called_once_with(
            ANY, dimension_, ("vec", "tors"), ("inserted", "vectors")
        )
        assert display_order == (-3, -1, -2, 1, 0, 2, 3)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class DescribePayloadOrderCollator(object):
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
        collator = PayloadOrderCollator(None)

        element_order_descriptors = collator._element_order_descriptors

        assert element_order_descriptors == expected_value
