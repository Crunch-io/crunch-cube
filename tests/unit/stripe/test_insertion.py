# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.insertion` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension
from cr.cube.stripe.insertion import _BaseSubtotals

from ...unitutil import ANY, initializer_mock, instance_mock, property_mock


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
