# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.cubemeasure` module."""

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe.cubemeasure import _BaseUnweightedCubeCounts, CubeMeasures

from ...unitutil import class_mock, instance_mock


class DescribeCubeMeasures(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure.CubeMeasures` object."""

    def it_provides_access_to_the_unweighted_cube_counts_object(
        self, request, cube_, rows_dimension_
    ):
        unweighted_cube_counts_ = instance_mock(request, _BaseUnweightedCubeCounts)
        _BaseUnweightedCubeCounts_ = class_mock(
            request, "cr.cube.stripe.cubemeasure._BaseUnweightedCubeCounts"
        )
        _BaseUnweightedCubeCounts_.factory.return_value = unweighted_cube_counts_
        cube_measures = CubeMeasures(cube_, rows_dimension_, False, slice_idx=7)

        unweighted_cube_counts = cube_measures.unweighted_cube_counts

        _BaseUnweightedCubeCounts_.factory.assert_called_once_with(
            cube_, rows_dimension_, False, 7
        )
        assert unweighted_cube_counts is unweighted_cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)
