# encoding: utf-8

"""Unit tests for the index functionality."""

from __future__ import absolute_import, division, print_function, unicode_literals

# pylint: disable=missing-docstring, invalid-name, redefined-outer-name

import pytest
import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.cube_slice import CubeSlice
from cr.cube.enum import DIMENSION_TYPE as DT

from ..unitutil import instance_mock


class DescribeIndexTable(object):
    def test_index_performs_correct_division(self, index_fixture, cube_):
        proportions, axis, base, expected = index_fixture
        cube_.dim_types = (DT.CAT, DT.CAT)
        cube_.ndim = 2
        cube_.mr_dim_ind = None
        cube_.proportions.return_value = np.array(proportions)
        cube_.as_array.return_value = np.array(proportions)
        cube_slice = CubeSlice(cube_, 0)

        actual = cube_slice.index_table(axis, base)

        np.testing.assert_almost_equal(actual, expected)

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, CrunchCube)

    @pytest.fixture(
        params=[
            (
                [[0.714285714286, 0.285714285714], [0.625, 0.375]],
                1,
                [0.6, 0.4],
                [[119.047619047619, 71.4285714285714], [104.16666666666667, 93.75]],
            ),
            (
                [[0.5, 0.4], [0.5, 0.6]],
                0,
                [0.6, 0.4],
                [[83.3333333333333, 66.6666666666667], [125, 150]],
            ),
        ]
    )
    def index_fixture(self, request):
        proportions, axis, base, expected = request.param
        # ---fixture only returns case data; combining that data into test-fixturing
        #    happens up in the test itself to give better context when reading test---
        return proportions, axis, np.array(base), expected
