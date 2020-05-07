# encoding: utf-8

"""Unit test suite for `cr.cube.stripe` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe import (
    _BaseBaseStripe,
    _BaseStripeRow,
    _StripeInsertedRow,
    _StripeInsertionHelper,
    TransformedStripe,
)

from ..unitutil import (
    ANY,
    class_mock,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class DescribeTransformedStripe(object):
    """Unit test suite for `cr.cube.stripe.TransformedStripe` object."""

    def it_provides_a_constructor_classmethod(
        self, request, cube_, dimension_, base_stripe_, stripe_
    ):
        _BaseBaseStripe_ = class_mock(request, "cr.cube.stripe._BaseBaseStripe")
        _BaseBaseStripe_.factory.return_value = base_stripe_
        _init_ = initializer_mock(request, TransformedStripe)

        stripe = TransformedStripe.stripe(cube_, dimension_, True, 42)

        _BaseBaseStripe_.factory.assert_called_once_with(cube_, dimension_, True, 42)
        _init_.assert_called_once_with(ANY, dimension_, base_stripe_)
        assert isinstance(stripe, TransformedStripe)

    def it_provides_access_to_its_rows(self, request):
        rows_ = tuple(
            instance_mock(
                request, _BaseStripeRow, hidden=bool(i % 2), name="row[%d]" % i
            )
            for i in range(4)
        )
        property_mock(
            request, TransformedStripe, "rows_including_hidden", return_value=rows_
        )
        stripe = TransformedStripe(None, None)

        assert stripe.rows == (rows_[0], rows_[2])

    def it_provides_access_to_its_rows_including_hidden_ones(self, request, dimension_):
        rows_ = tuple(
            instance_mock(
                request, _BaseStripeRow, name="row[%d]" % i, hidden=bool(i % 2)
            )
            for i in range(4)
        )
        property_mock(
            request,
            TransformedStripe,
            "_ordered_rows",
            return_value=("ordered", "rows"),
        )
        property_mock(request, TransformedStripe, "_table_margin", return_value=42)
        _StripeInsertionHelper_ = class_mock(
            request, "cr.cube.stripe._StripeInsertionHelper"
        )
        _StripeInsertionHelper_.iter_interleaved_rows.return_value = iter(rows_)
        stripe = TransformedStripe(dimension_, None)

        rows = stripe.rows_including_hidden

        _StripeInsertionHelper_.iter_interleaved_rows.assert_called_once_with(
            dimension_, ("ordered", "rows"), 42
        )
        assert rows == (rows_[0], rows_[1], rows_[2], rows_[3])

    @pytest.mark.parametrize(
        "display_order", ((), (0,), (1, 0), (0, 1, 2), (2, 1, 0, 3))
    )
    def it_provides_access_to_the_ordered_base_rows_to_help(
        self, request, display_order, dimension_, base_stripe_
    ):
        dimension_.display_order = display_order
        rows_ = tuple(
            instance_mock(request, _BaseStripeRow, name="row[%d]" % i)
            for i in range(len(display_order))
        )
        base_stripe_.rows = rows_
        stripe = TransformedStripe(dimension_, base_stripe_)

        ordered_rows = stripe._ordered_rows

        assert ordered_rows == tuple(rows_[idx] for idx in display_order)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def base_stripe_(self, request):
        return instance_mock(request, _BaseBaseStripe)

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def stripe_(self, request):
        return instance_mock(request, TransformedStripe)


class Describe_StripeInsertionHelper(object):
    """Unit test suite for `cr.cube.stripe._StripeInsertionHelper` function-object."""

    def it_provides_an_interface_classmethod(self, request, dimension_):
        _init_ = initializer_mock(request, _StripeInsertionHelper)
        _iter_interleaved_rows_ = method_mock(
            request,
            _StripeInsertionHelper,
            "_iter_interleaved_rows",
            return_value=iter(("interleaved", "rows")),
        )

        interleaved_rows = _StripeInsertionHelper.iter_interleaved_rows(
            dimension_, ("ordered", "rows"), 42
        )

        _init_.assert_called_once_with(ANY, dimension_, ("ordered", "rows"), 42)
        _iter_interleaved_rows_.assert_called_once_with(ANY)
        assert next(interleaved_rows) == "interleaved"
        assert next(interleaved_rows) == "rows"
        assert next(interleaved_rows, None) is None

    def it_generates_the_interleaved_rows_in_order_to_help(self, request):
        subtotal_rows_ = (
            instance_mock(request, _StripeInsertedRow, name="subt_0", anchor=0),
            instance_mock(request, _StripeInsertedRow, name="subt_1", anchor="bottom"),
            instance_mock(request, _StripeInsertedRow, name="subt_2", anchor=1),
            instance_mock(request, _StripeInsertedRow, name="subt_3", anchor="top"),
            instance_mock(request, _StripeInsertedRow, name="subt_4", anchor="top"),
        )
        property_mock(
            request,
            _StripeInsertionHelper,
            "_inserted_rows",
            return_value=subtotal_rows_,
        )
        ordered_rows_ = (
            instance_mock(request, _BaseStripeRow, name="row_0"),
            instance_mock(request, _BaseStripeRow, name="row_1"),
        )
        insertion_helper = _StripeInsertionHelper(None, ordered_rows_, None)

        interleaved_rows = insertion_helper._iter_interleaved_rows()

        assert next(interleaved_rows) == subtotal_rows_[3]
        assert next(interleaved_rows) == subtotal_rows_[4]
        assert next(interleaved_rows) == ordered_rows_[0]
        assert next(interleaved_rows) == subtotal_rows_[0]
        assert next(interleaved_rows) == ordered_rows_[1]
        assert next(interleaved_rows) == subtotal_rows_[2]
        assert next(interleaved_rows) == subtotal_rows_[1]
        assert next(interleaved_rows, None) is None

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)
