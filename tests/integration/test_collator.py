# encoding: utf-8

"""Partial-integration test suite for `cr.cube.collator` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.collator import PayloadOrderCollator
from cr.cube.dimension import Dimension, _Subtotal

from ..unitutil import instance_mock


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
