# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.measure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.matrix.cubemeasure import (
    CubeMeasures,
    _BaseUnweightedCubeCounts,
    _BaseWeightedCubeCounts,
)
from cr.cube.matrix.measure import (
    _BaseSecondOrderMeasure,
    _ColumnUnweightedBases,
    SecondOrderMeasures,
    _UnweightedCounts,
    _WeightedCounts,
)

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeSecondOrderMeasures(object):
    """Unit test suite for `cr.cube.matrix.measure.SecondOrderMeasures` object."""

    def it_provides_access_to_the_column_unweighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        column_unweighted_bases_ = instance_mock(request, _ColumnUnweightedBases)
        _ColumnUnweightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._ColumnUnweightedBases",
            return_value=column_unweighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        column_unweighted_bases = measures.column_unweighted_bases

        _ColumnUnweightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert column_unweighted_bases is column_unweighted_bases_

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

    def it_provides_access_to_weighted_counts_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        weighted_counts_ = instance_mock(request, _WeightedCounts)
        _WeightedCounts_ = class_mock(
            request,
            "cr.cube.matrix.measure._WeightedCounts",
            return_value=weighted_counts_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        weighted_counts = measures.weighted_counts

        _WeightedCounts_.assert_called_once_with(dimensions_, measures, cube_measures_)
        assert weighted_counts is weighted_counts_

    def it_provides_access_to_the_cube_measures_to_help(
        self, request, cube_, dimensions_, cube_measures_
    ):
        CubeMeasures_ = class_mock(
            request,
            "cr.cube.matrix.measure.CubeMeasures",
            return_value=cube_measures_,
        )
        measures = SecondOrderMeasures(cube_, dimensions_, slice_idx=42)

        cube_measures = measures._cube_measures

        CubeMeasures_.assert_called_once_with(cube_, dimensions_, 42)
        assert cube_measures is cube_measures_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def _cube_measures_prop_(self, request):
        return property_mock(request, SecondOrderMeasures, "_cube_measures")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))


class Describe_BaseSecondOrderMeasure(object):
    """Unit test suite for `cr.cube.matrix.measure._BaseSecondOrderMeasure` object."""

    def it_assembles_the_blocks_for_the_measure(self, request):
        property_mock(
            request, _BaseSecondOrderMeasure, "_base_values", return_value="A"
        )
        property_mock(
            request, _BaseSecondOrderMeasure, "_subtotal_columns", return_value="B"
        )
        property_mock(
            request, _BaseSecondOrderMeasure, "_subtotal_rows", return_value="C"
        )
        property_mock(
            request, _BaseSecondOrderMeasure, "_intersections", return_value="D"
        )
        measure = _BaseSecondOrderMeasure(None, None, None)

        blocks = measure.blocks

        assert blocks == [["A", "B"], ["C", "D"]]

    def it_provides_access_to_the_unweighted_cube_counts_object_to_help(
        self, request, cube_measures_
    ):
        unweighted_cube_counts_ = instance_mock(request, _BaseUnweightedCubeCounts)
        cube_measures_.unweighted_cube_counts = unweighted_cube_counts_
        measure = _BaseSecondOrderMeasure(None, None, cube_measures_)

        unweighted_cube_counts = measure._unweighted_cube_counts

        assert unweighted_cube_counts is unweighted_cube_counts_

    def it_provides_access_to_the_weighted_cube_counts_object_to_help(
        self, request, cube_measures_
    ):
        weighted_cube_counts_ = instance_mock(request, _BaseWeightedCubeCounts)
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        measure = _BaseSecondOrderMeasure(None, None, cube_measures_)

        weighted_cube_counts = measure._weighted_cube_counts

        assert weighted_cube_counts is weighted_cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)


class Describe_UnweightedCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._UnweightedCounts` object."""

    def it_computes_its_blocks_to_help(self, request, dimensions_):
        # --- these need to be in list form because the assert-called-with mechanism
        # --- uses equality, which doesn't work on numpy arrays. Normally this would be
        # --- the array itself.
        ucounts = np.arange(12).reshape(3, 4).tolist()
        unweighted_cube_counts_ = instance_mock(
            request, _BaseUnweightedCubeCounts, unweighted_counts=ucounts
        )
        property_mock(
            request,
            _UnweightedCounts,
            "_unweighted_cube_counts",
            return_value=unweighted_cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        unweighted_counts = _UnweightedCounts(dimensions_, None, None)

        blocks = unweighted_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(ucounts, dimensions_)
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))


class Describe_WeightedCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._WeightedCounts` object."""

    def it_computes_its_blocks_to_help(self, request, dimensions_):
        # --- these need to be in list form because the assert-called-with mechanism
        # --- uses equality, which doesn't work on numpy arrays. Normally this would be
        # --- the array itself.
        counts = np.arange(12).reshape(3, 4).tolist()
        weighted_cube_counts_ = instance_mock(
            request, _BaseWeightedCubeCounts, weighted_counts=counts
        )
        property_mock(
            request,
            _WeightedCounts,
            "_weighted_cube_counts",
            return_value=weighted_cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        weighted_counts = _WeightedCounts(dimensions_, None, None)

        blocks = weighted_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(counts, dimensions_)
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))
