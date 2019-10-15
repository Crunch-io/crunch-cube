# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cubepart import _Slice, _Strand, _Nub
from cr.cube.matrix import TransformedMatrix, _VectorAfterHiding
from ..unitutil import instance_mock, property_mock


class Describe_Slice(object):
    """Unit test suite for `cr.cube.cubepart._Slice` object."""

    def it_knows_its_data_status(self, _slice_prop_, slice_is_empty_fixture):
        slice_shape, expected_value = slice_is_empty_fixture
        _slice_prop_.return_value = slice_shape
        slice_ = _Slice(None, None, None, None, None)

        is_empty = slice_.is_empty

        assert is_empty is expected_value

    def it_knows_the_row_proportions(self, request, _matrix_prop_, matrix_):
        _matrix_prop_.return_value = matrix_
        matrix_.rows = (
            instance_mock(request, _VectorAfterHiding, proportions=(0.1, 0.2, 0.3)),
            instance_mock(request, _VectorAfterHiding, proportions=(0.4, 0.5, 0.6)),
        )
        slice_ = _Slice(None, None, None, None, None)

        row_proportions = slice_.row_proportions

        np.testing.assert_almost_equal(
            row_proportions, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

    def it_knows_the_rows_margin(self, request, _matrix_prop_, matrix_):
        _matrix_prop_.return_value = matrix_
        matrix_.rows = (
            instance_mock(request, _VectorAfterHiding, margin=(1, 2)),
            instance_mock(request, _VectorAfterHiding, margin=(3, 4)),
        )
        slice_ = _Slice(None, None, None, None, None)

        rows_margin = slice_.rows_margin

        np.testing.assert_almost_equal(rows_margin, [[1, 2], [3, 4]])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _matrix_prop_(self, request):
        return property_mock(request, _Slice, "_matrix")

    @pytest.fixture
    def matrix_(self, request):
        return instance_mock(request, TransformedMatrix)

    @pytest.fixture
    def _slice_prop_(self, request):
        return property_mock(request, _Slice, "shape")

    # fixtures ---------------------------------------------

    @pytest.fixture(
        params=[((1,), False), ((0,), True), ((7, 6), False), ((0, 0), True)]
    )
    def slice_is_empty_fixture(self, request):
        slice_shape, expected_value = request.param
        return slice_shape, expected_value


class Describe_Strand(object):
    """Unit test suite for `cr.cube.cubepart._Strand` object."""

    def it_knows_its_data_status(self, _strand_prop_, strand_is_empty_fixture):
        strand_shape, expected_value = strand_is_empty_fixture
        _strand_prop_.return_value = strand_shape
        strand_ = _Strand(None, None, None, None, None, None)

        is_empty = strand_.is_empty

        assert is_empty is expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _strand_prop_(self, request):
        return property_mock(request, _Strand, "_shape")

    # fixtures ---------------------------------------------

    @pytest.fixture(
        params=[((1,), False), ((0,), True), ((7, 6), False), ((0, 0), True)]
    )
    def strand_is_empty_fixture(self, request):
        slice_shape, expected_value = request.param
        return slice_shape, expected_value


class Describe_Nub(object):
    """Unit test suite for `cr.cube.cubepart._Nub` object."""

    def it_knows_its_data_status(self, _nub_prop_, nub_is_empty_fixture):
        base_count, expected_value = nub_is_empty_fixture
        _nub_prop_.return_value = base_count
        nub_ = _Nub(None)

        is_empty = nub_.is_empty

        assert is_empty == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _nub_prop_(self, request):
        return property_mock(request, _Nub, "base_count")

    # fixtures ---------------------------------------------

    @pytest.fixture(params=[(None, True), (45.4, False)])
    def nub_is_empty_fixture(self, request):
        base_count, expected_value = request.param
        return base_count, expected_value
