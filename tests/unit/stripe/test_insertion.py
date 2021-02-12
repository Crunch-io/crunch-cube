# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.insertion` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.stripe.insertion import _BaseSubtotals, SumSubtotals

from ...unitutil import (
    ANY,
    call,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_BaseSubtotals(object):
    """Unit test suite for `cr.cube.stripe._BaseSubtotals` object."""

    def it_provides_a_subtotal_values_interface_method(self, request, rows_dimension_):
        base_values = [1, 2, 3]
        _init_ = initializer_mock(request, _BaseSubtotals)
        property_mock(
            request, _BaseSubtotals, "_subtotal_values", return_value=np.array([3, 5])
        )

        subtotal_values = _BaseSubtotals.subtotal_values(base_values, rows_dimension_)

        _init_.assert_called_once_with(ANY, base_values, rows_dimension_)
        assert subtotal_values.tolist() == [3, 5]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)


class DescribeSumSubtotals(object):
    """Unit test suite for `cr.cube.stripe.SumSubtotals` object."""

    def it_computes_the_subtotal_values(self, request):
        row_subtotals_ = tuple(instance_mock(request, _Subtotal) for _ in range(3))
        property_mock(
            request, SumSubtotals, "_row_subtotals", return_value=row_subtotals_
        )
        _subtotal_value_ = method_mock(
            request, SumSubtotals, "_subtotal_value", side_effect=iter((9, 8, 7, 6, 5))
        )
        sum_subtotals = SumSubtotals(None, None)

        subtotal_values = sum_subtotals._subtotal_values

        assert _subtotal_value_.call_args_list == [
            call(sum_subtotals, row_subtotals_[0]),
            call(sum_subtotals, row_subtotals_[1]),
            call(sum_subtotals, row_subtotals_[2]),
        ]
        assert subtotal_values.tolist() == [9, 8, 7]

    def but_it_returns_an_empty_array_when_there_are_no_subtotals(self, request):
        """The dtype of the empty array is the same as that of the base-values."""
        property_mock(request, SumSubtotals, "_row_subtotals", return_value=())
        property_mock(request, SumSubtotals, "_dtype", return_value=np.int64)
        sum_subtotals = SumSubtotals(None, None)

        subtotal_values = sum_subtotals._subtotal_values

        assert subtotal_values.tolist() == []
        assert subtotal_values.dtype == int
