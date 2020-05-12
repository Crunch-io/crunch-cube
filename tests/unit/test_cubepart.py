# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import CubePartition, _Slice, _Strand, _Nub
from cr.cube.dimension import Dimension
from cr.cube.matrix import TransformedMatrix, _VectorAfterHiding
from cr.cube.stripe import _BaseStripeRow, TransformedStripe

from ..unitutil import class_mock, instance_mock, property_mock


class DescribeCubePartition(object):
    """Unit test suite for `cr.cube.cubepart.CubePartition` object."""

    def it_constructs_a_slice_with_its_factory_for_a_2D_cube(self, request, cube_):
        cube_.ndim = 2
        _Slice_ = class_mock(request, "cr.cube.cubepart._Slice")

        slice_ = CubePartition.factory(cube_, 42, {"trans": "forms"}, 1000, False, 10)

        _Slice_.assert_called_once_with(cube_, 42, {"trans": "forms"}, 1000, 10)
        assert slice_ is _Slice_.return_value

    def but_it_constructs_a_strand_for_a_2D_cube_when_ca_as_0th(self, request, cube_):
        cube_.ndim = 2
        _Strand_ = class_mock(request, "cr.cube.cubepart._Strand")

        strand = CubePartition.factory(cube_, 0, ca_as_0th=True)

        _Strand_.assert_called_once_with(cube_, None, None, True, 0, 0)
        assert strand is _Strand_.return_value

    def and_it_constructs_a_strand_for_a_1D_cube(self, request, cube_):
        cube_.ndim = 1
        _Strand_ = class_mock(request, "cr.cube.cubepart._Strand")

        strand = CubePartition.factory(cube_, 42, {"trans": "forms"}, 1000, False, 10)

        _Strand_.assert_called_once_with(cube_, {"trans": "forms"}, 1000, False, 42, 10)
        assert strand is _Strand_.return_value

    def and_it_constructs_a_nub_for_a_0D_cube(self, request, cube_):
        cube_.ndim = 0
        _Nub_ = class_mock(request, "cr.cube.cubepart._Nub")

        nub = CubePartition.factory(cube_)

        _Nub_.assert_called_once_with(cube_)
        assert nub is _Nub_.return_value

    def it_knows_the_index_of_its_cube_in_the_cube_set(self, cube_):
        cube_.cube_index = 42
        cube_partition = CubePartition(cube_)

        cube_index = cube_partition.cube_index

        assert cube_index == 42

    def it_knows_if_cube_is_mr_by_itself(self):
        # --- default of False is overridden by subclasses when appropriate ---
        assert CubePartition(None).cube_is_mr_by_itself is False

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)


class Describe_Slice(object):
    """Unit test suite for `cr.cube.cubepart._Slice` object."""

    @pytest.mark.parametrize(
        ("shape", "expected_value"),
        (((4, 2), False), ((4, 0), True), ((0, 2), True), ((0, 0), True)),
    )
    def it_knows_whether_it_is_empty(self, shape, expected_value, shape_prop_):
        shape_prop_.return_value = shape
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
    def matrix_(self, request):
        return instance_mock(request, TransformedMatrix)

    @pytest.fixture
    def _matrix_prop_(self, request):
        return property_mock(request, _Slice, "_matrix")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Slice, "shape")


class Describe_Strand(object):
    """Unit test suite for `cr.cube.cubepart._Strand` object."""

    def it_knows_which_of_its_rows_are_inserted(self, request, _stripe_prop_, stripe_):
        stripe_.rows = tuple(
            instance_mock(request, _BaseStripeRow, is_inserted=bool(i % 2))
            for i in range(7)
        )
        _stripe_prop_.return_value = stripe_
        strand = _Strand(None, None, None, None, None, None)

        inserted_row_idxs = strand.inserted_row_idxs

        assert inserted_row_idxs == (1, 3, 5)

    @pytest.mark.parametrize(("shape", "expected_value"), (((1,), False), ((0,), True)))
    def it_knows_whether_it_is_empty(self, shape, expected_value, shape_prop_):
        shape_prop_.return_value = shape
        strand = _Strand(None, None, None, None, None, None)

        is_empty = strand.is_empty

        assert is_empty is expected_value

    def it_knows_its_title(self, cube_):
        cube_.title = "Unmarried"
        strand_ = _Strand(cube_, None, None, None, None, None)

        title = strand_.title

        assert title == "Unmarried"

    def it_constructs_its_underlying_stripe_to_help(
        self, request, cube_, _rows_dimension_prop_, dimension_, stripe_
    ):
        TransformedStripe_ = class_mock(request, "cr.cube.cubepart.TransformedStripe")
        TransformedStripe_.stripe.return_value = stripe_
        _rows_dimension_prop_.return_value = dimension_
        strand = _Strand(cube_, None, None, False, 42, None)

        stripe = strand._stripe

        TransformedStripe_.stripe.assert_called_once_with(cube_, dimension_, False, 42)
        assert stripe is stripe_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _Strand, "_rows_dimension")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Strand, "shape")

    @pytest.fixture
    def stripe_(self, request):
        return instance_mock(request, TransformedStripe)

    @pytest.fixture
    def _stripe_prop_(self, request):
        return property_mock(request, _Strand, "_stripe")


class Describe_Nub(object):
    """Unit test suite for `cr.cube.cubepart._Nub` object."""

    def it_knows_its_data_status(self, _nub_prop_, nub_is_empty_fixture):
        base_count, expected_value = nub_is_empty_fixture
        _nub_prop_.return_value = base_count
        nub_ = _Nub(None)

        is_empty = nub_.is_empty

        assert is_empty == expected_value

    def it_knows_if_cube_is_mr_by_itself(self):
        nub_ = _Nub(None)

        assert nub_.cube_is_mr_by_itself is False

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _nub_prop_(self, request):
        return property_mock(request, _Nub, "base_count")

    # fixtures ---------------------------------------------

    @pytest.fixture(params=[(None, True), (45.4, False)])
    def nub_is_empty_fixture(self, request):
        base_count, expected_value = request.param
        return base_count, expected_value
