# encoding: utf-8

"""Partial-integration test suite for `cr.cube.collator` module."""

import pytest

from cr.cube.collator import (
    ExplicitOrderCollator,
    PayloadOrderCollator,
    SortByValueCollator,
)
from cr.cube.dimension import _Subtotals, Dimension, Element, _OrderSpec, _Subtotal
from cr.cube.enums import DIMENSION_TYPE as DT, ORDER_FORMAT

from ..unitutil import instance_mock, property_mock


class TestExplicitOrderCollator:
    """Partial-integration test suite for `ExplicitOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, order, anchors, expected_value",
        (
            (
                (1, 2, 3),
                [2, 3, 1],
                ("bottom", 3, 3, "top"),
                (-1, 1, 2, -3, -2, 0, -4),
            ),
            ((9, 3, 7), [7], (), (2, 0, 1)),
        ),
    )
    def test_it_knows_the_display_order_for_a_dimension(
        self, request, element_ids, order, anchors, expected_value
    ):
        subtotals_ = [instance_mock(request, _Subtotal, anchor=a) for a in anchors]
        order_spec_ = instance_mock(request, _OrderSpec, element_ids=order)
        elements_ = [
            instance_mock(request, Element, element_id=id_, derived=False)
            for id_ in element_ids
        ]
        dimension_ = instance_mock(
            request,
            Dimension,
            valid_elements=elements_,
            subtotals=subtotals_,
            order_spec=order_spec_,
        )

        assert (
            ExplicitOrderCollator.display_order(
                dimension_, (), ORDER_FORMAT.SIGNED_INDEXES
            )
            == expected_value
        )


class TestPayloadOrderCollator:
    """Partial-integration test suite for `PayloadOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, anchors, expected_value",
        (
            ((1, 2, 3), ("bottom", 3, 3, "top"), (-1, 0, 1, 2, -3, -2, -4)),
            ((9, 3, 7), (), (0, 1, 2)),
            ((), ("bottom", 3, 3, "top"), (-1, -4, -3, -2)),
        ),
    )
    def test_it_knows_the_display_order_for_a_dimension(
        self, request, element_ids, anchors, expected_value
    ):
        subtotals_ = [instance_mock(request, _Subtotal, anchor=a) for a in anchors]
        elements_ = [
            instance_mock(request, Element, element_id=id_, derived=False)
            for id_ in element_ids
        ]
        dimension_ = instance_mock(
            request,
            Dimension,
            element_ids=element_ids,
            valid_elements=elements_,
            subtotals=subtotals_,
        )

        display_order = PayloadOrderCollator.display_order(
            dimension_, (), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert display_order == expected_value

    @pytest.mark.parametrize(
        "element_ids, anchors, expected_value",
        (
            ((1, 2, 3), ("bottom", 3, 3, "top"), ("ins_4", 0, 1, 2, "ins_2", "ins_3")),
            ((9, 3, 7), (), (0, 1, 2)),
            ((), ("bottom", 3, 3, "top"), ("ins_4", "ins_2", "ins_3")),
        ),
    )
    def test_it_knows_the_payload_order_for_a_dimension(
        self, request, element_ids, anchors, expected_value
    ):
        subtotals_ = [
            instance_mock(request, _Subtotal, anchor=a, insertion_id=i)
            for i, a in enumerate(anchors)
        ]
        _Subtotals_ = instance_mock(request, _Subtotals)
        _Subtotals_.insertion_ids = (1, 2, 3, 4)
        property_mock(
            request,
            PayloadOrderCollator,
            "_subtotals_bogus_ids",
            return_value=("ins_1", "ins_2", "ins_3", "ins_4"),
        )
        elements_ = [
            instance_mock(request, Element, element_id=id_, derived=False)
            for id_ in element_ids
        ]
        dimension_ = instance_mock(
            request,
            Dimension,
            element_ids=element_ids,
            valid_elements=elements_,
            subtotals=_Subtotals_,
            subtotals_in_payload_order=subtotals_,
        )

        payload_order = PayloadOrderCollator(
            dimension_, (), ORDER_FORMAT.SIGNED_INDEXES
        ).payload_order
        assert payload_order == expected_value


class TestSortByValueCollator:
    """Partial-integration test suite for `SortByValueCollator` object."""

    @pytest.mark.parametrize(
        "order, xtop, xbot, element_vals, empty_idxs, expected_value",
        (
            # --- descending: subtots at top, then body ---
            ("D", [], [], (10, 30, 20, 40), (), (-2, -1, 3, 1, 2, 0)),
            ("D", [1], [], (10, 30, 20, 40), (), (-2, -1, 0, 3, 1, 2)),
            ("D", [], [4], (10, 30, 20, 40), (), (-2, -1, 1, 2, 0, 3)),
            ("D", [4], [2], (10, 30, 20, 40), (), (-2, -1, 3, 2, 0, 1)),
            ("D", [], [], (10, 30, 20, 40), (1,), (-2, -1, 3, 2, 0)),
            ("D", [3], [2], (10, 30, 20, 40), (0, 3), (-2, -1, 2, 1)),
            ("D", [3], [2], (10, 30, 20, 40), (2, 3), (-2, -1, 0, 1)),
            # --- ascending: body first, all subtots at bottom ---
            ("A", [], [], (10, 30, 20, 40), (), (0, 2, 1, 3, -1, -2)),
            ("A", [2], [], (10, 30, 20, 40), (), (1, 0, 2, 3, -1, -2)),
            ("A", [], [3], (10, 30, 20, 40), (), (0, 1, 3, 2, -1, -2)),
            ("A", [4], [1], (10, 30, 20, 40), (), (3, 2, 1, 0, -1, -2)),
            ("A", [], [], (10, 30, 20, 40), (2,), (0, 1, 3, -1, -2)),
            ("A", [4], [1], (10, 30, 20, 40), (1, 2), (3, 0, -1, -2)),
            ("A", [4], [1], (10, 30, 20, 40), (0, 3), (2, 1, -1, -2)),
        ),
    )
    def test_it_knows_the_display_order_for_a_dimension(
        self, order, xtop, xbot, element_vals, empty_idxs, expected_value
    ):
        subtot_vals = [60, 40]
        dimension = Dimension(
            dimension_dict={
                "type": {
                    "categories": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
                    "class": "categorical",
                }
            },
            dimension_type=DT.CAT,
            dimension_transforms={
                "order": {
                    "direction": "ascending" if order == "A" else "descending",
                    "fixed": {"top": xtop, "bottom": xbot},
                },
                "prune": True,
            },
        )

        display_order = SortByValueCollator.display_order(
            dimension,
            element_vals,
            subtot_vals,
            empty_idxs,
            ORDER_FORMAT.SIGNED_INDEXES,
        )

        assert display_order == expected_value

    @pytest.mark.parametrize(
        "order, xtop, xbot, element_vals, empty_idxs, expected_value",
        (
            # --- descending: subtots at top, then body ---
            ("D", [], [], (10, 30, 20, 40), (), ("ins_1", "ins_2", 3, 1, 2, 0)),
            ("D", [1], [], (10, 30, 20, 40), (), ("ins_1", "ins_2", 0, 3, 1, 2)),
            ("D", [], [4], (10, 30, 20, 40), (), ("ins_1", "ins_2", 1, 2, 0, 3)),
            ("D", [4], [2], (10, 30, 20, 40), (), ("ins_1", "ins_2", 3, 2, 0, 1)),
            ("D", [], [], (10, 30, 20, 40), (1,), ("ins_1", "ins_2", 3, 2, 0)),
            ("D", [3], [2], (10, 30, 20, 40), (0, 3), ("ins_1", "ins_2", 2, 1)),
            ("D", [3], [2], (10, 30, 20, 40), (2, 3), ("ins_1", "ins_2", 0, 1)),
            # --- ascending: body first, all subtots at bottom ---
            ("A", [], [], (10, 30, 20, 40), (), (0, 2, 1, 3, "ins_2", "ins_1")),
            ("A", [2], [], (10, 30, 20, 40), (), (1, 0, 2, 3, "ins_2", "ins_1")),
            ("A", [], [3], (10, 30, 20, 40), (), (0, 1, 3, 2, "ins_2", "ins_1")),
            ("A", [4], [1], (10, 30, 20, 40), (), (3, 2, 1, 0, "ins_2", "ins_1")),
            ("A", [], [], (10, 30, 20, 40), (2,), (0, 1, 3, "ins_2", "ins_1")),
            ("A", [4], [1], (10, 30, 20, 40), (1, 2), (3, 0, "ins_2", "ins_1")),
            ("A", [4], [1], (10, 30, 20, 40), (0, 3), (2, 1, "ins_2", "ins_1")),
        ),
    )
    def test_it_knows_the_display_order_for_a_dimension_in_bogus_id_fmt(
        self, request, order, xtop, xbot, element_vals, empty_idxs, expected_value
    ):
        subtot_vals = [60, 40]
        dimension = Dimension(
            dimension_dict={
                "type": {
                    "categories": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
                    "class": "categorical",
                }
            },
            dimension_type=DT.CAT,
            dimension_transforms={
                "order": {
                    "direction": "ascending" if order == "A" else "descending",
                    "fixed": {"top": xtop, "bottom": xbot},
                },
                "prune": True,
            },
        )
        property_mock(
            request,
            SortByValueCollator,
            "_subtotals_bogus_ids",
            return_value=("ins_1", "ins_2"),
        )
        display_order = SortByValueCollator.display_order(
            dimension,
            element_vals,
            subtot_vals,
            empty_idxs,
            ORDER_FORMAT.BOGUS_IDS,
        )

        assert display_order == expected_value
