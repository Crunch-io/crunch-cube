# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.insertion` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.stripe.insertion import (
    _BaseSubtotals,
    NanSubtotals,
    NegativeTermSubtotals,
    PositiveTermSubtotals,
    SumSubtotals,
)

from ...unitutil import ANY, initializer_mock, instance_mock, property_mock


class Test_BaseSubtotals:
    """Unit test suite for `cr.cube.stripe._BaseSubtotals` object."""

    def test_it_provides_a_subtotal_values_interface_method(
        self, request, rows_dimension_
    ):
        base_values = [1, 2, 3]
        _init_ = initializer_mock(request, _BaseSubtotals)
        property_mock(
            request, _BaseSubtotals, "_subtotal_values", return_value=np.array([3, 5])
        )

        subtotal_values = _BaseSubtotals.subtotal_values(base_values, rows_dimension_)

        _init_.assert_called_once_with(ANY, base_values, rows_dimension_)
        assert subtotal_values.tolist() == [3, 5]

    def test_it_provides_access_to_the_row_subtotal_objects(
        self, request, rows_dimension_
    ):
        row_subtotals_ = tuple(instance_mock(request, _Subtotal) for _ in range(3))
        rows_dimension_.subtotals = row_subtotals_
        subtotals = _BaseSubtotals(None, rows_dimension_)

        assert subtotals._row_subtotals is row_subtotals_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)


class TestNanSubtotals:
    """Unit test suite for `cr.cube.stripe.NanSubtotals` object."""

    def test_it_computes_the_subtotal_values(self, request):
        property_mock(
            request,
            NanSubtotals,
            "_row_subtotals",
            return_value=tuple(instance_mock(request, _Subtotal) for _ in range(3)),
        )
        nan_subtotals = NanSubtotals(None, None)

        assert nan_subtotals._subtotal_values == pytest.approx(
            [np.nan, np.nan, np.nan], nan_ok=True
        )


class TestNegativeTermSubtotals:
    """Unit test suite for `cr.cube.matrix.NegativeTermSubtotals` object."""

    @pytest.mark.parametrize(
        ("addend_idxs", "subtrahend_idxs", "expected"),
        (
            ([0, 1], [2, 3], 12),
            ([1], [], 0),
            ([], [1], 2),
        ),
    )
    def test_it_computes_the_subtotal_values(
        self, request, addend_idxs, subtrahend_idxs, expected
    ):
        property_mock(
            request,
            _Subtotal,
            "addend_idxs",
            return_value=np.array(addend_idxs, dtype=int),
        )
        property_mock(
            request,
            _Subtotal,
            "subtrahend_idxs",
            return_value=np.array(subtrahend_idxs, dtype=int),
        )

        subtotal_value = NegativeTermSubtotals(
            np.array([1, 2, 4, 8]), None
        )._subtotal_value(_Subtotal(None, None))

        np.testing.assert_equal(subtotal_value, expected)


class TestPositiveTermSubtotals:
    """Unit test suite for `cr.cube.matrix.PositiveTermSubtotals` object."""

    @pytest.mark.parametrize(
        ("addend_idxs", "subtrahend_idxs", "expected"),
        (
            ([0, 1], [2, 3], 3),
            ([1], [], 2),
            ([], [1], 0),
        ),
    )
    def test_it_computes_the_subtotal_values(
        self, request, addend_idxs, subtrahend_idxs, expected
    ):
        property_mock(
            request,
            _Subtotal,
            "addend_idxs",
            return_value=np.array(addend_idxs, dtype=int),
        )
        property_mock(
            request,
            _Subtotal,
            "subtrahend_idxs",
            return_value=np.array(subtrahend_idxs, dtype=int),
        )

        subtotal_value = PositiveTermSubtotals(
            np.array([1, 2, 4, 8]), None
        )._subtotal_value(_Subtotal(None, None))

        np.testing.assert_equal(subtotal_value, expected)


class TestSumSubtotals:
    """Unit test suite for `cr.cube.matrix.SumSubtotals` object."""

    @pytest.mark.parametrize(
        ("addend_idxs", "subtrahend_idxs", "expected"),
        (
            ([0, 1], [2, 3], -9),
            ([1], [], 2),
            ([], [1], -2),
        ),
    )
    def test_it_calculates_subtotal_value_correctly(
        self, request, addend_idxs, subtrahend_idxs, expected
    ):
        property_mock(
            request,
            _Subtotal,
            "addend_idxs",
            return_value=np.array(addend_idxs, dtype=int),
        )
        property_mock(
            request,
            _Subtotal,
            "subtrahend_idxs",
            return_value=np.array(subtrahend_idxs, dtype=int),
        )

        subtotal_value = SumSubtotals(np.array([1, 2, 4, 8]), None)._subtotal_value(
            _Subtotal(None, None)
        )

        np.testing.assert_equal(subtotal_value, expected)
