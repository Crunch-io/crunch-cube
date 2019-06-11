# encoding: utf-8

"""Unit test suite for cr.cube.legacy_min_base_size module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube_slice import CubeSlice
from cr.cube.legacy_min_base_size_mask import MinBaseSizeMask
from cr.cube.enum import DIMENSION_TYPE as DT

from ...unitutil import instance_mock, method_mock, property_mock


class DescribeMinBaseSizeMask:
    """Unit-test suite for `cr.cube.min_base_size_mask.MinBaseSizeMask` object."""

    def it_provides_access_to_column_direction_mask(
        self, _margin, _get_shape, _ndim, column_mask_fixture
    ):
        size, shape, margin, expected_mask = column_mask_fixture
        _margin.return_value = margin
        _get_shape.return_value = shape
        _ndim.return_value = len(shape)
        row_mask = MinBaseSizeMask(CubeSlice(None, None), size).column_mask
        np.testing.assert_array_equal(row_mask, expected_mask)

    def it_provides_access_to_row_direction_mask(
        self, _margin, _get_shape, _ndim, row_mask_fixture
    ):
        size, shape, margin, expected_mask = row_mask_fixture
        _margin.return_value = margin
        _get_shape.return_value = shape
        _ndim.return_value = len(shape)
        row_mask = MinBaseSizeMask(CubeSlice(None, None), size).row_mask
        np.testing.assert_array_equal(row_mask, expected_mask)

    def it_provides_access_to_table_direction_mask(
        self, _margin, _get_shape, _ndim, _dim_types, table_mask_fixture
    ):
        size, shape, dim_types, margin, expected_mask = table_mask_fixture
        _margin.return_value = margin
        _get_shape.return_value = shape
        _dim_types.return_value = dim_types
        _ndim.return_value = len(shape)
        table_mask = MinBaseSizeMask(CubeSlice(None, None), size).table_mask
        np.testing.assert_array_equal(table_mask, expected_mask)

    def it_retains_single_element_dimension_in_shape(
        self, _ndim, _get_shape, shape_fixture
    ):
        slice_shape, ndim, expected_mask_shape = shape_fixture
        min_base_size = MinBaseSizeMask(CubeSlice(None, None), None)
        _ndim.return_value = ndim
        _get_shape.return_value = slice_shape
        assert min_base_size._shape == expected_mask_shape

    def it_sets_slice_on_construction(self, slice_):
        size = 50
        min_base_size = MinBaseSizeMask(slice_, size)
        assert min_base_size._slice is slice_
        assert min_base_size._size is size

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            # Margin is just a single row - broadcast it across shape
            (30, (2, 3), [10, 20, 30], [[True, True, False], [True, True, False]]),
            # Margin is 2D table (as in CAT x MR), use that shape (don't broadcast)
            (
                40,
                (2, 3),
                [[10, 20, 40], [30, 50, 60]],
                [[True, True, False], [True, False, False]],
            ),
        ]
    )
    def column_mask_fixture(self, request):
        size, shape, margin, expected = request.param
        margin = np.array(margin)
        expected = np.array(expected)
        return size, shape, margin, expected

    @pytest.fixture(
        params=[
            # Margin is just a single column - broadcast it across shape
            (30, (3, 2), [10, 20, 30], [[True, True], [True, True], [False, False]]),
            # Margin is 2D table (as in CAT x MR), use that shape (don't broadcast)
            (
                40,
                (2, 3),
                [[10, 20, 40], [30, 50, 60]],
                [[True, True, False], [True, False, False]],
            ),
        ]
    )
    def row_mask_fixture(self, request):
        size, shape, margin, expected = request.param
        margin = np.array(margin)
        expected = np.array(expected)
        return size, shape, margin, expected

    @pytest.fixture(params=[((2, 3), 2, (2, 3)), ((2,), 2, (2, 1))])
    def shape_fixture(self, request):
        slice_shape, ndim, expected = request.param
        return slice_shape, ndim, expected

    @pytest.fixture(
        params=[
            (
                30,
                (3, 2),
                (DT.CAT, DT.CAT),
                10,
                [[True, True], [True, True], [True, True]],
            ),
            (
                30,
                (3, 2),
                (DT.CAT, DT.CAT),
                40,
                [[False, False], [False, False], [False, False]],
            ),
            (
                40,
                (2, 3),
                (DT.CAT, DT.CAT),
                [[10, 20, 40], [30, 50, 60]],
                [[True, True, False], [True, False, False]],
            ),
            (
                40,
                (2, 3),
                (DT.CAT, DT.MR),
                [10, 20, 40],
                [[True, True, False], [True, True, False]],
            ),
            (
                40,
                (2, 3),
                (DT.MR, DT.CAT),
                [10, 40],
                [[True, True, True], [False, False, False]],
            ),
        ]
    )
    def table_mask_fixture(self, request):
        size, shape, dim_types, margin, expected = request.param
        margin = np.array(margin)
        expected = np.array(expected)
        return size, shape, dim_types, margin, expected

    # fixture components ---------------------------------------------

    @pytest.fixture
    def slice_(self, request):
        return instance_mock(request, CubeSlice)

    @pytest.fixture
    def _margin(self, request):
        return method_mock(request, CubeSlice, "margin")

    @pytest.fixture
    def _get_shape(self, request):
        return method_mock(request, CubeSlice, "get_shape")

    @pytest.fixture
    def _dim_types(self, request):
        return property_mock(request, CubeSlice, "dim_types")

    @pytest.fixture
    def _ndim(self, request):
        return property_mock(request, CubeSlice, "ndim")
