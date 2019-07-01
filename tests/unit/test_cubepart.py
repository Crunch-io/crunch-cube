# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.matrix import TransformedMatrix, _VectorAfterHiding
from ..unitutil import instance_mock, property_mock


class Describe_Slice(object):
    """Unit test suite for `cr.cube.cubepart._Slice` object."""

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
