# encoding: utf-8

"""Unit test suite for `cr.cube.stripe` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Element, _Subtotal
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.stripe import (
    _BaseBaseStripe,
    _BaseStripeRow,
    _MeansStripeRow,
    _StripeInsertedRow,
    _StripeInsertionHelper,
    TransformedStripe,
)

from ..unitutil import (
    ANY,
    call,
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
            instance_mock(request, _StripeInsertedRow, name="subt_0", anchor=1),
            instance_mock(request, _StripeInsertedRow, name="subt_1", anchor="bottom"),
            instance_mock(request, _StripeInsertedRow, name="subt_2", anchor=2),
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
            instance_mock(request, _BaseStripeRow, name="row_0", element_id=1),
            instance_mock(request, _BaseStripeRow, name="row_1", element_id=2),
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

    @pytest.mark.parametrize(
        ("dimension_type", "expected_len"), ((DT.MR, 0), (DT.CA, 0), (DT.CAT, 3))
    )
    def it_assembles_the_inserted_rows_to_help(
        self, request, dimension_type, expected_len, dimension_
    ):
        subtotal_rows_ = (
            instance_mock(request, _StripeInsertedRow, name="subtotal_0"),
            instance_mock(request, _StripeInsertedRow, name="subtotal_1"),
            instance_mock(request, _StripeInsertedRow, name="subtotal_2"),
        )
        _StripeInsertedRow_ = class_mock(
            request, "cr.cube.stripe._StripeInsertedRow", side_effect=subtotal_rows_
        )
        dimension_.dimension_type = dimension_type
        dimension_.subtotals = subtotal_rows_
        insertion_helper = _StripeInsertionHelper(dimension_, ("ordered", "rows"), 42)

        inserted_rows = insertion_helper._inserted_rows

        assert _StripeInsertedRow_.call_args_list == [
            call(row, ("ordered", "rows"), 42) for row in subtotal_rows_[:expected_len]
        ]
        assert inserted_rows == subtotal_rows_[:expected_len]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_StripeInsertedRow(object):
    """Unit test suite for `cr.cube.stripe._StripeInsertedRow` object."""

    @pytest.mark.parametrize("anchor_value", ("top", "bottom", 1, 36))
    def it_knows_its_anchor_location(self, anchor_value, subtotal_):
        subtotal_.anchor = anchor_value
        inserted_row = _StripeInsertedRow(subtotal_, None, None)

        anchor = inserted_row.anchor

        assert anchor == anchor_value

    def it_knows_it_is_inserted(self):
        assert _StripeInsertedRow(None, None, None).is_inserted is True

    def it_knows_its_unweighted_count(self, request):
        addend_rows_ = tuple(
            instance_mock(request, _BaseStripeRow, unweighted_count=i + 1)
            for i in range(3)
        )
        property_mock(
            request, _StripeInsertedRow, "_addend_rows", return_value=addend_rows_
        )
        inserted_row = _StripeInsertedRow(None, None, None)

        unweighted_count = inserted_row.unweighted_count

        assert unweighted_count == 6  # --- 1+2+3 ---

    def it_gathers_the_addend_rows_to_help(self, request, subtotal_):
        base_rows_ = tuple(
            instance_mock(
                request, _BaseStripeRow, name="base_rows_[%d]" % i, element_id=i + 1
            )
            for i in range(4)
        )
        subtotal_.addend_ids = (1, 3)
        inserted_row = _StripeInsertedRow(subtotal_, base_rows_, None)

        addend_rows = inserted_row._addend_rows

        assert addend_rows == (base_rows_[0], base_rows_[2])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)


class Describe_BaseStripeRow(object):
    """Unit test suite for `cr.cube.stripe._BaseStripeRow` object."""

    def it_knows_its_element_id(self, element_):
        element_.element_id = 42
        row = _BaseStripeRow(element_)

        element_id = row.element_id

        assert element_id == 42

    def it_knows_it_is_not_inserted(self):
        assert _BaseStripeRow(None).is_inserted is False

    # fixture components ---------------------------------------------

    @pytest.fixture
    def element_(self, request):
        return instance_mock(request, _Element)


class Describe_MeansStripeRow(object):
    """Unit test suite for `cr.cube.stripe.Describe_MeansStripeRow` object."""

    def it_knows_its_unweighted_count(self):
        assert _MeansStripeRow(None, 42, None).unweighted_count == 42
