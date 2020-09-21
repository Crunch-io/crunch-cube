# encoding: utf-8

"""Unit test suite for the  cr.cube.measures.scale_means module."""

from __future__ import division

import numpy as np
import pytest

from cr.cube.measures.scale_means import ScaleMeans

from ..unitutil import Mock, property_mock

# pylint: disable=missing-docstring, redefined-outer-name, protected-access

COLS_DIM_VALUES = np.array([0, 1, 2])
ROWS_DIM_VALUES = np.array([3, 4, 5, 6])
ROW_MARGIN = np.array([0, 1, 2])
COL_MARGIN = np.array([3, 4, 5, 6])


def margin(axis):
    return ROW_MARGIN if axis else COL_MARGIN


@pytest.mark.parametrize(
    "values, margin, axis, ndim, expected",
    (
        (
            [None, ROWS_DIM_VALUES],
            margin,
            0,
            2,
            np.sum(COL_MARGIN * ROWS_DIM_VALUES) / np.sum(COL_MARGIN),
        ),
        (
            [COLS_DIM_VALUES, None],
            margin,
            1,
            2,
            np.sum(ROW_MARGIN * COLS_DIM_VALUES) / np.sum(ROW_MARGIN),
        ),
        (
            [None, ROWS_DIM_VALUES],
            margin,
            0,
            2,
            np.sum(COL_MARGIN * ROWS_DIM_VALUES) / np.sum(COL_MARGIN),
        ),
        ([None, None], margin, 0, 2, None),
    ),
)
def test_scale_means_marginal(values, margin, axis, ndim, expected, values_prop_):
    values_prop_.return_value = values
    slice_ = Mock()
    slice_.ndim = ndim
    slice_.margin = margin
    scale_means = ScaleMeans(slice_)
    assert scale_means.margin(axis) == expected


@pytest.mark.parametrize(
    "dim_ind, ndim, shape, expected ",
    (
        (0, 2, (Mock(), Mock()), True),
        (1, 2, (Mock(), Mock()), False),
        (0, 2, (Mock(),), False),
        (0, 1, (Mock(),), False),
    ),
)
def test_inflate(dim_ind, ndim, shape, expected):
    slice_ = Mock()
    slice_.ndim = ndim
    slice_.get_shape.return_value = shape
    scale_means = ScaleMeans(slice_)
    assert scale_means._inflate(dim_ind) == expected


@pytest.mark.parametrize(
    "dimensions, axis, expected",
    (
        ([Mock(numeric_values=[])], 0, [slice(None)]),
        ([Mock(numeric_values=[1, 2, 3])], 0, [np.array([True, True, True])]),
        (
            [Mock(numeric_values=[1, 2, np.nan, 4])],
            0,
            [np.array([True, True, False, True])],
        ),
        ([Mock(numeric_values=[1])], 0, [np.array([True])]),
        (
            [Mock(numeric_values=[1, 2, 3]), Mock(numeric_values=[])],
            0,
            [np.array([True, True, True])],
        ),
        ([Mock(numeric_values=[1, 2, 3]), Mock(numeric_values=[])], 1, [slice(None)]),
    ),
)
def test_valid_indices(dimensions, axis, expected):
    slice_ = Mock()
    slice_.dimensions = dimensions
    actual = ScaleMeans(slice_)._valid_indices(axis)
    np.testing.assert_equal(actual, expected)


# fixture components ---------------------------------------------


@pytest.fixture
def values_prop_(request):
    return property_mock(request, ScaleMeans, "values")
