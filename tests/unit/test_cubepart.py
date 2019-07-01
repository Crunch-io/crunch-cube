# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.matrix import _CatXCatMatrix, TransformedMatrix
from ..unitutil import instance_mock, property_mock


class DescribeIntegrated_Slice(object):
    """Partial-integration test suite for `cr.cube.cubepart._Slice` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_knows_the_row_proportions(self, row_proportions_fixture, _assembler_prop_):
        slice_, insertions, expected = row_proportions_fixture
        _assembler_prop_.return_value = TransformedMatrix(
            slice_, None, insertions, None
        )
        slice_ = _Slice(None, None, None, None, None)

        row_proportions = slice_.row_proportions

        np.testing.assert_almost_equal(row_proportions, expected)

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_knows_the_rows_margin(self, row_margin_fixture, _assembler_prop_):
        slice_, insertions, expected = row_margin_fixture
        _assembler_prop_.return_value = TransformedMatrix(
            slice_, None, insertions, None
        )
        slice_ = _Slice(None, None)

        margin = slice_.row_margin

        np.testing.assert_almost_equal(margin, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                [[1, 2], [3, 4]],
                [],
                [[0.33333333, 0.66666667], [0.42857143, 0.57142857]],
            ),
            (
                [[1, 2], [3, 4]],
                [(0, (0,))],
                [
                    [0.33333333, 0.66666667],
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                ],
            ),
            (
                [[1, 2], [3, 4]],
                [(1, (0,))],
                [
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                    [0.33333333, 0.66666667],
                ],
            ),
            (
                [[1, 2], [3, 4], [5, 6]],
                [("top", (1,)), (2, (1, 2))],
                [
                    [0.42857143, 0.57142857],
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                    [0.45454545, 0.54545455],
                    [0.44444444, 0.55555556],
                ],
            ),
        ]
    )
    def row_proportions_fixture(self, request, subtotals_prop_, dimension_):
        counts, row_subtotals, expected_row_proportions = request.param
        dimensions = (Dimension(None, None), dimension_)
        slice_ = _CatXCatMatrix(dimensions, counts, None)
        subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        return slice_, np.array(expected_row_proportions)

    @pytest.fixture(
        params=[
            ([[1, 2], [3, 4]], [], [3, 7]),
            ([[1, 2], [3, 4]], [(0, (0,))], [3, 3, 7]),
            ([[1, 2], [3, 4]], [(1, (0,))], [3, 7, 3]),
            ([[1, 2], [3, 4], [5, 6]], [("top", (1,)), (2, (1, 2))], [7, 3, 7, 11, 18]),
        ]
    )
    def row_margin_fixture(self, request, _subtotals_prop_, dimension_):
        counts, row_subtotals, expected_row_margin = request.param
        slice_ = _CatXCatMatrix(counts)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        return slice_, np.array(expected_row_margin)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assembler_prop_(self, request):
        return property_mock(request, _Slice, "_assembler")

    @pytest.fixture
    def subtotals_prop_(self, request):
        return property_mock(request, Dimension, "subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, subtotals=tuple())
