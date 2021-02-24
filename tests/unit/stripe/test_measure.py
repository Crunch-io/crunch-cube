# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.measure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe.cubemeasure import (
    _BaseUnweightedCubeCounts,
    _BaseWeightedCubeCounts,
    CubeMeasures,
)
from cr.cube.stripe.measure import (
    _BaseSecondOrderMeasure,
    StripeMeasures,
    _UnweightedCounts,
    _WeightedCounts,
)

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeStripeMeasures(object):
    """Unit test suite for `cr.cube.stripe.measure.StripeMeasures` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("unweighted_counts", _UnweightedCounts),
            ("weighted_counts", _WeightedCounts),
        ),
    )
    def it_provides_access_to_various_measure_objects(
        self,
        request,
        rows_dimension_,
        _cube_measures_prop_,
        cube_measures_,
        measure_prop_name,
        MeasureCls,
    ):
        measure_ = instance_mock(request, MeasureCls)
        MeasureCls_ = class_mock(
            request,
            "cr.cube.stripe.measure.%s" % MeasureCls.__name__,
            return_value=measure_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = StripeMeasures(None, rows_dimension_, None, None)

        measure = getattr(measures, measure_prop_name)

        MeasureCls_.assert_called_once_with(rows_dimension_, measures, cube_measures_)
        assert measure is measure_

    def it_provides_access_to_the_pruning_base(
        self, request, _cube_measures_prop_, cube_measures_
    ):
        unweighted_cube_counts_ = instance_mock(
            request, _BaseUnweightedCubeCounts, pruning_base=np.array([0, 2, 7])
        )
        cube_measures_.unweighted_cube_counts = unweighted_cube_counts_
        _cube_measures_prop_.return_value = cube_measures_
        measures = StripeMeasures(None, None, None, None)

        assert measures.pruning_base.tolist() == [0, 2, 7]

    def it_provides_access_to_the_cube_measures_to_help(
        self, request, cube_, rows_dimension_, cube_measures_
    ):
        CubeMeasures_ = class_mock(
            request,
            "cr.cube.stripe.measure.CubeMeasures",
            return_value=cube_measures_,
        )
        measures = StripeMeasures(cube_, rows_dimension_, True, slice_idx=42)

        cube_measures = measures._cube_measures

        CubeMeasures_.assert_called_once_with(cube_, rows_dimension_, True, 42)
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
        return property_mock(request, StripeMeasures, "_cube_measures")

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseSecondOrderMeasure(object):
    """Unit test suite for `cr.cube.stripe.measure._BaseSecondOrderMeasure` object."""

    def it_gathers_the_blocks_for_the_measure(self, request):
        property_mock(request, _BaseSecondOrderMeasure, "base_values", return_value="A")
        property_mock(
            request, _BaseSecondOrderMeasure, "subtotal_values", return_value="B"
        )
        measure = _BaseSecondOrderMeasure(None, None, None)

        assert measure.blocks == ("A", "B")

    def it_provides_access_to_the_unweighted_cube_counts_object_to_help(
        self, request, cube_measures_
    ):
        unweighted_cube_counts_ = instance_mock(request, _BaseUnweightedCubeCounts)
        cube_measures_.unweighted_cube_counts = unweighted_cube_counts_
        measure = _BaseSecondOrderMeasure(None, None, cube_measures_)

        assert measure._unweighted_cube_counts is unweighted_cube_counts_

    def it_provides_access_to_the_weighted_cube_counts_object_to_help(
        self, request, cube_measures_
    ):
        weighted_cube_counts_ = instance_mock(request, _BaseWeightedCubeCounts)
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        measure = _BaseSecondOrderMeasure(None, None, cube_measures_)

        assert measure._weighted_cube_counts is weighted_cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)


class Describe_UnweightedCounts(object):
    """Unit test suite for `cr.cube.stripe.measure._UnweightedCounts` object."""

    def it_knows_its_base_values(
        self, _unweighted_cube_counts_prop_, unweighted_cube_counts_
    ):
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.unweighted_counts = np.array([1, 2, 3])
        unweighted_counts = _UnweightedCounts(None, None, None)

        assert unweighted_counts.base_values.tolist() == [1, 2, 3]

    def it_knows_its_subtotal_values(self, request):
        rows_dimension_ = instance_mock(request, Dimension)
        property_mock(request, _UnweightedCounts, "base_values", return_value=[1, 2, 3])
        SumSubtotals_ = class_mock(request, "cr.cube.stripe.measure.SumSubtotals")
        SumSubtotals_.subtotal_values.return_value = np.array([3, 5])
        unweighted_counts = _UnweightedCounts(rows_dimension_, None, None)

        subtotal_values = unweighted_counts.subtotal_values

        SumSubtotals_.subtotal_values.assert_called_once_with(
            [1, 2, 3], rows_dimension_
        )
        assert subtotal_values.tolist() == [3, 5]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)

    @pytest.fixture
    def _unweighted_cube_counts_prop_(self, request):
        return property_mock(request, _UnweightedCounts, "_unweighted_cube_counts")


class Describe_WeightedCounts(object):
    """Unit test suite for `cr.cube.stripe.measure._WeightedCounts` object."""

    def it_computes_its_base_values_to_help(
        self, _weighted_cube_counts_prop_, weighted_cube_counts_
    ):
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.weighted_counts = np.array([1, 2, 3])
        weighted_counts = _WeightedCounts(None, None, None)

        assert weighted_counts.base_values.tolist() == [1, 2, 3]

    def it_computes_its_subtotal_values_to_help(self, request):
        rows_dimension_ = instance_mock(request, Dimension)
        property_mock(
            request, _WeightedCounts, "base_values", return_value=[1.1, 2.2, 3.3]
        )
        SumSubtotals_ = class_mock(request, "cr.cube.stripe.measure.SumSubtotals")
        SumSubtotals_.subtotal_values.return_value = np.array([3.3, 5.5])
        weighted_counts = _WeightedCounts(rows_dimension_, None, None)

        subtotal_values = weighted_counts.subtotal_values

        SumSubtotals_.subtotal_values.assert_called_once_with(
            [1.1, 2.2, 3.3], rows_dimension_
        )
        assert subtotal_values.tolist() == [3.3, 5.5]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _WeightedCounts, "_weighted_cube_counts")
