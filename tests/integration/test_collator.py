# encoding: utf-8

"""Partial-integration test suite for `cr.cube.collator` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.collator import (
    ExplicitOrderCollator,
    MarginalCollator,
    PayloadOrderCollator,
)
from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.matrix import _BaseInsertedVector, _CategoricalVector

from ..unitutil import instance_mock


class DescribeExplicitOrderCollator(object):
    """Partial-integration test suite for `ExplicitOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, order, anchors, expected_value",
        (
            ((1, 2, 3), [2, 3, 1], ("bottom", 3, 3, "top"), (-1, 1, 2, -3, -2, 0, -4)),
            ((9, 3, 7), [7], (), (2, 0, 1)),
        ),
    )
    def it_knows_the_display_order_for_a_dimension(
        self, request, element_ids, order, anchors, expected_value
    ):
        subtotals_ = [instance_mock(request, _Subtotal, anchor=a) for a in anchors]
        dimension_ = instance_mock(
            request,
            Dimension,
            element_ids=element_ids,
            subtotals=subtotals_,
            order_dict={"element_ids": order},
        )

        display_order = ExplicitOrderCollator.display_order(dimension_)

        assert display_order == expected_value


class DescribeMarginalCollator(object):
    """Partial-integration test suite for `MarginalCollator` object."""

    @pytest.mark.parametrize(
        "direction, vectors, inserted_vectors, expected_value",
        (
            # --- descending: subtots at top, then body ---
            (
                "descending",
                ((3, 30.0, ""), (1, 10.0, "xbot"), (2, 20.0, "")),
                (("top", 22.4), (2, 37.8), ("bottom", 12.7)),
                (-2, -3, -1, 0, 2, 1),
            ),
            # --- ascending: body first, all subtots at bottom ---
            (
                "ascending",
                ((3, 30.0, ""), (1, 10.0, "xbot"), (2, 20.0, ""), (4, 40.0, "xtop")),
                (("top", 22.4), (2, 37.8), ("bottom", 12.7)),
                (3, 2, 0, 1, -1, -3, -2),
            ),
        ),
    )
    def it_knows_the_display_order_for_a_dimension(
        self, request, direction, vectors, inserted_vectors, expected_value
    ):
        vectors_ = [
            instance_mock(request, _CategoricalVector, margin=margin)
            for _, margin, _ in vectors
        ]
        inserted_vectors_ = [
            instance_mock(request, _BaseInsertedVector, anchor=anchor, margin=margin)
            for anchor, margin in inserted_vectors
        ]
        dimension_ = instance_mock(
            request,
            Dimension,
            element_ids=tuple(id_ for id_, _, _ in vectors),
            order_dict={
                "direction": direction,
                "marginal": "weighted_N",
                "exclude": {
                    "top": [id_ for id_, _, exclude in vectors if exclude == "xtop"],
                    "bottom": [id_ for id_, _, exclude in vectors if exclude == "xbot"],
                },
            },
        )

        display_order = MarginalCollator.display_order(
            dimension_, vectors_, inserted_vectors_
        )

        assert display_order == expected_value


class DescribePayloadOrderCollator(object):
    """Partial-integration test suite for `PayloadOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, anchors, expected_value",
        (
            ((9, 3, 7), ("bottom", 3, 3, "top"), (-1, 0, 1, -3, -2, 2, -4)),
            ((9, 3, 7), (), (0, 1, 2)),
            ((), ("bottom", 3, 3, "top"), (-1, -4, -3, -2)),
        ),
    )
    def it_knows_the_display_order_for_a_dimension(
        self, request, element_ids, anchors, expected_value
    ):
        subtotals_ = [instance_mock(request, _Subtotal, anchor=a) for a in anchors]
        dimension_ = instance_mock(
            request, Dimension, element_ids=element_ids, subtotals=subtotals_
        )

        display_order = PayloadOrderCollator.display_order(dimension_)

        assert display_order == expected_value
