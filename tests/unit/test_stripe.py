# encoding: utf-8

"""Unit test suite for `cr.cube.stripe` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe import _BaseBaseStripe, TransformedStripe

from ..unitutil import ANY, class_mock, initializer_mock, instance_mock


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
