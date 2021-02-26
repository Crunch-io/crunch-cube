# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.measure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe.cubemeasure import (
    _BaseUnweightedCubeCounts,
    _BaseWeightedCubeCounts,
    _CatUnweightedCubeCounts,
    _CatWeightedCubeCounts,
    CubeMeasures,
)
from cr.cube.stripe.measure import (
    _BaseSecondOrderMeasure,
    StripeMeasures,
    _TableProportionStddevs,
    _TableProportionVariances,
    _TableProportions,
    _UnweightedBases,
    _UnweightedCounts,
    _WeightedBases,
    _WeightedCounts,
)

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeStripeMeasures(object):
    """Unit test suite for `cr.cube.stripe.measure.StripeMeasures` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("table_proportion_stddevs", _TableProportionStddevs),
            ("table_proportion_variances", _TableProportionVariances),
            ("table_proportions", _TableProportions),
            ("unweighted_bases", _UnweightedBases),
            ("unweighted_counts", _UnweightedCounts),
            ("weighted_bases", _WeightedBases),
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


class Describe_TableProportionStddevs(object):
    """Unit test suite for `cr.cube.stripe.measure._TableProportionStddevs` object."""

    def it_computes_its_base_values_to_help(
        self, measures_, table_proportion_variances_
    ):
        table_proportion_variances_.base_values = np.array([0.04, 0.09, 0.16])
        measures_.table_proportion_variances = table_proportion_variances_
        table_proportion_stddevs = _TableProportionStddevs(None, measures_, None)

        assert table_proportion_stddevs.base_values == pytest.approx([0.2, 0.3, 0.4])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)

    @pytest.fixture
    def table_proportion_variances_(self, request):
        return instance_mock(request, _TableProportionVariances)


class Describe_TableProportionVariances(object):
    """Unit test suite for `cr.cube.stripe.measure._TableProportionVariances` object."""

    def it_computes_its_base_values_to_help(self, measures_, table_proportions_):
        table_proportions_.base_values = np.array([0.2, 0.3, 0.5])
        measures_.table_proportions = table_proportions_
        table_proportion_variances = _TableProportionVariances(None, measures_, None)

        assert table_proportion_variances.base_values == pytest.approx(
            [0.16, 0.21, 0.25]
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)

    @pytest.fixture
    def table_proportions_(self, request):
        return instance_mock(request, _TableProportions)


class Describe_TableProportions(object):
    """Unit test suite for `cr.cube.stripe.measure._TableProportions` object."""

    @pytest.mark.parametrize(
        "table_margin, expected_value",
        (
            (np.array([4.5, 6.7]), [0.7555556, 0.8358209]),
            (42.42, [0.08015087, 0.1320132]),
        ),
    )
    def it_computes_its_base_values_to_help(
        self,
        measures_,
        weighted_counts_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
        table_margin,
        expected_value,
    ):
        weighted_counts_.base_values = np.array([3.4, 5.6])
        measures_.weighted_counts = weighted_counts_
        weighted_cube_counts_.table_margin = table_margin
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        table_proportions = _TableProportions(None, measures_, None)

        assert table_proportions.base_values == pytest.approx(expected_value)

    @pytest.mark.parametrize(
        "table_margin, expected_value",
        (
            (np.array([4.5, 6.7]), []),
            (42.42, [0.2310231, 0.1791608]),
        ),
    )
    def it_computes_its_subtotal_values_to_help(
        self,
        measures_,
        weighted_counts_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
        table_margin,
        expected_value,
    ):
        weighted_counts_.subtotal_values = np.array([9.8, 7.6])
        measures_.weighted_counts = weighted_counts_
        weighted_cube_counts_.table_margin = table_margin
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        table_proportions = _TableProportions(None, measures_, None)

        assert table_proportions.subtotal_values == pytest.approx(expected_value)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)

    @pytest.fixture
    def weighted_counts_(self, request):
        return instance_mock(request, _WeightedCounts)

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _TableProportions, "_weighted_cube_counts")


class Describe_UnweightedBases(object):
    """Unit test suite for `cr.cube.stripe.measure._UnweightedBases` object."""

    def it_knows_its_base_values(
        self, _unweighted_cube_counts_prop_, unweighted_cube_counts_
    ):
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.bases = np.array([3, 4, 5])
        unweighted_bases = _UnweightedBases(None, None, None)

        assert unweighted_bases.base_values.tolist() == [3, 4, 5]

    @pytest.mark.parametrize(
        "subtotal_values, expected_value",
        ((np.array([]), []), (np.array([5, 3]), [42, 42])),
    )
    def it_knows_its_subtotal_values(
        self, request, _unweighted_cube_counts_prop_, subtotal_values, expected_value
    ):
        rows_dimension_ = instance_mock(request, Dimension)
        property_mock(request, _UnweightedBases, "base_values", return_value=[3, 2, 1])
        SumSubtotals_ = class_mock(request, "cr.cube.stripe.measure.SumSubtotals")
        SumSubtotals_.subtotal_values.return_value = subtotal_values
        _unweighted_cube_counts_prop_.return_value = instance_mock(
            request, _CatUnweightedCubeCounts, table_base=42
        )
        unweighted_bases = _UnweightedBases(rows_dimension_, None, None)

        subtotal_values = unweighted_bases.subtotal_values

        SumSubtotals_.subtotal_values.assert_called_once_with(
            [3, 2, 1], rows_dimension_
        )
        assert subtotal_values.tolist() == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)

    @pytest.fixture
    def _unweighted_cube_counts_prop_(self, request):
        return property_mock(request, _UnweightedBases, "_unweighted_cube_counts")


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


class Describe_WeightedBases(object):
    """Unit test suite for `cr.cube.stripe.measure._WeightedBases` object."""

    def it_knows_its_base_values(
        self, _weighted_cube_counts_prop_, weighted_cube_counts_
    ):
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.bases = np.array([3.3, 4.4, 5.5])
        weighted_bases = _WeightedBases(None, None, None)

        assert weighted_bases.base_values.tolist() == [3.3, 4.4, 5.5]

    @pytest.mark.parametrize(
        "subtotal_values, expected_value",
        ((np.array([]), []), (np.array([5.5, 3.3]), [42.24, 42.24])),
    )
    def it_knows_its_subtotal_values(
        self,
        request,
        _weighted_cube_counts_prop_,
        subtotal_values,
        expected_value,
    ):
        rows_dimension_ = instance_mock(request, Dimension)
        property_mock(
            request, _WeightedBases, "base_values", return_value=[3.3, 2.2, 1.1]
        )
        SumSubtotals_ = class_mock(request, "cr.cube.stripe.measure.SumSubtotals")
        SumSubtotals_.subtotal_values.return_value = subtotal_values
        _weighted_cube_counts_prop_.return_value = instance_mock(
            request, _CatWeightedCubeCounts, table_margin=42.24
        )
        weighted_bases = _WeightedBases(rows_dimension_, None, None)

        subtotal_values = weighted_bases.subtotal_values

        SumSubtotals_.subtotal_values.assert_called_once_with(
            [3.3, 2.2, 1.1], rows_dimension_
        )
        assert subtotal_values.tolist() == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _WeightedBases, "_weighted_cube_counts")


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
