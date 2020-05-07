# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.matrix import _BaseBaseMatrix, TransformedMatrix

from ..unitutil import ANY, class_mock, initializer_mock, instance_mock


class DescribeTransformedMatrix(object):
    """Unit test suite for `cr.cube.matrix.TransformedMatrix` object."""

    def it_provides_a_constructor_classmethod(self, request, base_matrix_, cube_):
        _BaseBaseMatrix_ = class_mock(request, "cr.cube.matrix._BaseBaseMatrix")
        _BaseBaseMatrix_.factory.return_value = base_matrix_
        dimensions_ = tuple(instance_mock(request, Dimension) for _ in range(2))
        _init_ = initializer_mock(request, TransformedMatrix)

        matrix = TransformedMatrix.matrix(cube_, dimensions_, slice_idx=7)

        _BaseBaseMatrix_.factory.assert_called_once_with(cube_, dimensions_, 7)
        _init_.assert_called_once_with(ANY, base_matrix_)
        assert isinstance(matrix, TransformedMatrix)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def base_matrix_(self, request):
        return instance_mock(request, _BaseBaseMatrix)

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)
