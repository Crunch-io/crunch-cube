from __future__ import division
from mock import Mock
import numpy as np
import pytest

from cr.cube.measures.scale_means import ScaleMeans

from .unitutil import property_mock

COLS_DIM_VALUES = np.array([0, 1, 2])
ROWS_DIM_VALUES = np.array([3, 4, 5, 6])
ROW_MARGIN = np.array([0, 1, 2])
COL_MARGIN = np.array([3, 4, 5, 6])


def margin(axis):
    return ROW_MARGIN if axis else COL_MARGIN


def test_scale_means_marginal(scale_means_fixture):
    scale_means, axis, expected = scale_means_fixture
    assert scale_means.margin(axis) == expected


@pytest.fixture(params=[
    (
        [None, ROWS_DIM_VALUES], margin, 0, 2,
        np.sum(COL_MARGIN * ROWS_DIM_VALUES) / np.sum(COL_MARGIN),
    ),
    (
        [COLS_DIM_VALUES, None], margin, 1, 2,
        np.sum(ROW_MARGIN * COLS_DIM_VALUES) / np.sum(ROW_MARGIN),
    ),
    (
        [None, ROWS_DIM_VALUES], margin, 0, 2,
        np.sum(COL_MARGIN * ROWS_DIM_VALUES) / np.sum(COL_MARGIN),
    ),
    (
        [None, None], margin, 0, 2, None,
    ),
])
def scale_means_fixture(request, values_prop_):
    values, margin, axis, ndim, expected = request.param
    values_prop_.return_value = values
    slice_ = Mock()
    slice_.ndim = ndim
    slice_.margin = margin
    scale_means = ScaleMeans(slice_)
    return scale_means, axis, expected


@pytest.fixture
def values_prop_(request):
    return property_mock(request, ScaleMeans, 'values')
