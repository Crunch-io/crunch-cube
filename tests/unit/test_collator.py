# encoding: utf-8

"""Unit test suite for `cr.cube.collator` module."""

import sys

import pytest

from cr.cube.collator import _BaseAnchoredCollator
from cr.cube.dimension import Dimension

from ..unitutil import ANY, initializer_mock, instance_mock, property_mock


class Describe_BaseAnchoredCollator(object):
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
        "base_orderings, insertion_orderings, hidden_idxs, expected_value",
        (
            # --- base-values but no insertions ---
            (((0, 0), (1, 1), (2, 2)), (), (), (0, 1, 2)),
            # --- insertions but no base-values (not expected) ---
            ((), ((sys.maxsize, -3), (0, -2), (sys.maxsize, -1)), (1,), (-2, -3, -1)),
            # --- both base-values and insertions ---
            (
                ((0, 0), (1, 1), (2, 2), (3, 3)),
                ((0, -3), (2, -2), (sys.maxsize, -1)),
                (1, 3),
                (-3, 0, -2, 2, -1),
            ),
        ),
    )
    def it_computes_the_display_order_to_help(
        self, request, base_orderings, insertion_orderings, hidden_idxs, expected_value
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
            return_value=insertion_orderings,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_base_element_orderings",
            return_value=base_orderings,
        )

        assert _BaseAnchoredCollator(None, None)._display_order == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)
