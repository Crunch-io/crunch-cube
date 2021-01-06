# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.measure` module."""

import pytest

from cr.cube.dimension import Dimension
from cr.cube.matrix.cubemeasure import CubeMeasures
from cr.cube.matrix.measure import SecondOrderMeasures, _UnweightedCounts

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeSecondOrderMeasures(object):
    """Unit test suite for `cr.cube.matrix.measure.SecondOrderMeasures` object."""

    def it_provides_access_to_unweighted_counts_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        unweighted_counts_ = instance_mock(request, _UnweightedCounts)
        _UnweightedCounts_ = class_mock(
            request,
            "cr.cube.matrix.measure._UnweightedCounts",
            return_value=unweighted_counts_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        unweighted_counts = measures.unweighted_counts

        _UnweightedCounts_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert unweighted_counts is unweighted_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def _cube_measures_prop_(self, request):
        return property_mock(request, SecondOrderMeasures, "_cube_measures")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))
