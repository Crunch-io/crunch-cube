# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.measure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT, MARGINAL_ORIENTATION as MO
from cr.cube.matrix.cubemeasure import (
    CubeMeasures,
    _BaseUnweightedCubeCounts,
    _BaseWeightedCubeCounts,
)
from cr.cube.matrix.measure import (
    _BaseMarginal,
    _BaseSecondOrderMeasure,
    _BaseScaledCountMarginal,
    _ColumnComparableCounts,
    _ColumnProportions,
    _ColumnShareSum,
    _ColumnUnweightedBases,
    _ColumnWeightedBases,
    _RowComparableCounts,
    _RowProportions,
    _RowShareSum,
    _RowUnweightedBases,
    _RowWeightedBases,
    _ScaleMean,
    _ScaleMeanStddev,
    _ScaleMedian,
    SecondOrderMeasures,
    _Sums,
    _TableUnweightedBases,
    _TableWeightedBases,
    _TotalShareSum,
    _UnweightedCounts,
    _WeightedCounts,
    _Zscores,
)

from ...unitutil import ANY, call, class_mock, instance_mock, method_mock, property_mock


class DescribeSecondOrderMeasures(object):
    """Unit test suite for `cr.cube.matrix.measure.SecondOrderMeasures` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("column_proportions", _ColumnProportions),
            ("column_unweighted_bases", _ColumnUnweightedBases),
            ("column_weighted_bases", _ColumnWeightedBases),
            ("row_proportions", _RowProportions),
            ("row_unweighted_bases", _RowUnweightedBases),
            ("row_weighted_bases", _RowWeightedBases),
            ("table_unweighted_bases", _TableUnweightedBases),
            ("table_weighted_bases", _TableWeightedBases),
            ("weighted_counts", _WeightedCounts),
            ("unweighted_counts", _UnweightedCounts),
            ("zscores", _Zscores),
        ),
    )
    def it_provides_access_to_various_measure_objects(
        self,
        request,
        dimensions_,
        _cube_measures_prop_,
        cube_measures_,
        measure_prop_name,
        MeasureCls,
    ):
        measure_ = instance_mock(request, MeasureCls)
        MeasureCls_ = class_mock(
            request,
            "cr.cube.matrix.measure.%s" % MeasureCls.__name__,
            return_value=measure_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        measure = getattr(measures, measure_prop_name)

        MeasureCls_.assert_called_once_with(dimensions_, measures, cube_measures_)
        assert measure is measure_

    def it_provides_access_to_the_columns_pruning_base(
        self, _cube_measures_prop_, cube_measures_, unweighted_cube_counts_
    ):
        _cube_measures_prop_.return_value = cube_measures_
        cube_measures_.unweighted_cube_counts = unweighted_cube_counts_
        unweighted_cube_counts_.columns_pruning_base = np.array([8, 5, 7, 4])
        measures = SecondOrderMeasures(None, None, None)

        assert measures.columns_pruning_base.tolist() == [8, 5, 7, 4]

    def it_provides_access_to_the_rows_pruning_base(
        self, _cube_measures_prop_, cube_measures_, unweighted_cube_counts_
    ):
        _cube_measures_prop_.return_value = cube_measures_
        cube_measures_.unweighted_cube_counts = unweighted_cube_counts_
        unweighted_cube_counts_.rows_pruning_base = np.array([7, 4, 0, 2])
        measures = SecondOrderMeasures(None, None, None)

        assert measures.rows_pruning_base.tolist() == [7, 4, 0, 2]

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

    @pytest.mark.parametrize(
        "orientation, measure, MarginalCls",
        (
            ("rows", "rows_scale_median", _ScaleMedian),
            ("columns", "columns_scale_median", _ScaleMedian),
            ("rows", "rows_scale_mean", _ScaleMean),
            ("columns", "columns_scale_mean", _ScaleMean),
            ("rows", "rows_scale_mean_stddev", _ScaleMeanStddev),
            ("columns", "columns_scale_mean_stddev", _ScaleMeanStddev),
        ),
    )
    def it_provides_access_to_the_scale_marginals(
        self,
        request,
        dimensions_,
        _cube_measures_prop_,
        cube_measures_,
        orientation,
        measure,
        MarginalCls,
    ):
        marginal_ = instance_mock(request, MarginalCls)
        MarginalCls_ = class_mock(
            request,
            "cr.cube.matrix.measure.%s" % MarginalCls.__name__,
            return_value=marginal_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        marginal = getattr(measures, measure)

        MarginalCls_.assert_called_once_with(
            dimensions_, measures, cube_measures_, MO(orientation)
        )
        assert marginal is marginal_

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

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)


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

        assert measure._unweighted_cube_counts is unweighted_cube_counts_

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


class Describe_ColumnComparableCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnComparableCounts` object."""

    def it_computes_its_blocks(self, request, dimensions_, cube_measures_):
        counts = np.arange(12).reshape(3, 4).tolist()
        weighted_cube_counts_ = instance_mock(
            request, _BaseWeightedCubeCounts, weighted_counts=counts
        )
        _weighted_cube_counts = property_mock(
            request, _BaseSecondOrderMeasure, "_weighted_cube_counts"
        )
        _weighted_cube_counts.return_value = weighted_cube_counts_
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        col_comparable_counts = _ColumnComparableCounts(
            dimensions_, None, cube_measures_
        )

        blocks = col_comparable_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(counts, dimensions_, True)
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)


class Describe_ColumnProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnProportions` object."""

    def it_computes_its_blocks(self, request):
        column_comparable_counts_ = instance_mock(
            request, _ColumnComparableCounts, blocks=[[5.0, 12.0], [21.0, 32.0]]
        )
        column_weighted_bases_ = instance_mock(
            request, _ColumnWeightedBases, blocks=[[5.0, 6.0], [7.0, 8.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            column_comparable_counts=column_comparable_counts_,
            column_weighted_bases=column_weighted_bases_,
        )
        cube_measures_ = class_mock(request, "cr.cube.matrix.cubemeasure.CubeMeasures")

        column_proportions = _ColumnProportions(
            None, second_order_measures_, cube_measures_
        )

        assert column_proportions.blocks == [[1.0, 2.0], [3.0, 4.0]]


class Describe_ColumnUnweightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnUnweightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _unweighted_cube_counts_prop_, unweighted_cube_counts_
    ):
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.column_bases = np.arange(6).reshape(2, 3)
        column_unweighted_bases = _ColumnUnweightedBases(None, None, None)

        assert column_unweighted_bases._base_values.tolist() == [[0, 1, 2], [3, 4, 5]]

    def it_computes_its_intersections_block_to_help(
        self, request, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        property_mock(
            request,
            _ColumnUnweightedBases,
            "_subtotal_columns",
            return_value=np.array([[9, 6], [9, 6], [9, 6]]),
        )
        _base_values_prop_.return_value = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        SumSubtotals_.intersections.return_value = np.array([[8, 4], [3, 7]])
        column_unweighted_bases = _ColumnUnweightedBases(dimensions_, None, None)

        intersections = column_unweighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], dimensions_, diff_cols_nan=True
        )
        assert intersections.tolist() == [[9, 6], [9, 6]]

    def it_computes_its_subtotal_columns_to_help(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        _base_values_prop_.return_value = [[1, 2], [3, 4]]
        SumSubtotals_.subtotal_columns.return_value = np.array([[5, 8], [3, 7]])
        column_unweighted_bases = _ColumnUnweightedBases(dimensions_, None, None)

        subtotal_columns = column_unweighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[1, 2], [3, 4]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_columns.tolist() == [[5, 8], [3, 7]]

    def it_computes_its_subtotal_rows_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _unweighted_cube_counts_prop_,
        unweighted_cube_counts_,
    ):
        _base_values_prop_.return_value = [[4, 3], [2, 1]]
        SumSubtotals_.subtotal_rows.return_value = np.array([[8, 3], [6, 4]])
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.columns_base = np.array([4, 7])
        column_unweighted_bases = _ColumnUnweightedBases(dimensions_, None, None)

        subtotal_rows = column_unweighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[4, 3], [2, 1]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_rows.tolist() == [[4, 7], [4, 7]]

    def but_it_returns_empty_array_of_right_shape_when_there_are_no_row_subtotals(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        """Empty shape must be (0, ncols) to compose properly in `np.block()` call."""
        _base_values_prop_.return_value = [[4, 3, 2], [1, 0, 9]]
        SumSubtotals_.subtotal_rows.return_value = np.array([], dtype=int).reshape(0, 3)
        column_unweighted_bases = _ColumnUnweightedBases(dimensions_, None, None)

        subtotal_rows = column_unweighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[4, 3, 2], [1, 0, 9]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_rows.tolist() == []
        assert subtotal_rows.shape == (0, 3)
        assert subtotal_rows.dtype == int

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _ColumnUnweightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)

    @pytest.fixture
    def _unweighted_cube_counts_prop_(self, request):
        return property_mock(request, _ColumnUnweightedBases, "_unweighted_cube_counts")


class Describe_ColumnWeightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnWeightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _weighted_cube_counts_prop_, weighted_cube_counts_
    ):
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.column_bases = np.arange(6).reshape(3, 2)
        column_weighted_bases = _ColumnWeightedBases(None, None, None)

        assert column_weighted_bases._base_values.tolist() == [[0, 1], [2, 3], [4, 5]]

    def it_computes_its_intersections_block_to_help(
        self, request, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        property_mock(
            request,
            _ColumnWeightedBases,
            "_subtotal_columns",
            return_value=np.array([[9.9, 6.6], [9.9, 6.6], [9.9, 6.6]]),
        )
        _base_values_prop_.return_value = [
            [9.9, 8.8, 7.7],
            [6.6, 5.5, 4.4],
            [3.3, 2.2, 1.1],
        ]
        SumSubtotals_.intersections.return_value = np.array([[8.8, 4.4], [3.3, 7.7]])
        column_weighted_bases = _ColumnWeightedBases(dimensions_, None, None)

        intersections = column_weighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[9.9, 8.8, 7.7], [6.6, 5.5, 4.4], [3.3, 2.2, 1.1]],
            dimensions_,
            diff_cols_nan=True,
        )
        assert intersections.tolist() == [[9.9, 6.6], [9.9, 6.6]]

    def it_computes_its_subtotal_columns_to_help(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        _base_values_prop_.return_value = [[1.1, 2.2], [3.3, 4.4]]
        SumSubtotals_.subtotal_columns.return_value = np.array([[5.5, 8.8], [3.3, 7.7]])
        column_weighted_bases = _ColumnWeightedBases(dimensions_, None, None)

        subtotal_columns = column_weighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[1.1, 2.2], [3.3, 4.4]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_columns.tolist() == [[5.5, 8.8], [3.3, 7.7]]

    def it_computes_its_subtotal_rows_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
    ):
        _base_values_prop_.return_value = [[4.4, 3.3], [2.2, 1.1]]
        SumSubtotals_.subtotal_rows.return_value = np.array([[8.8, 3.3], [6.6, 4.4]])
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.columns_margin = np.array([4.4, 7.7])
        column_weighted_bases = _ColumnWeightedBases(dimensions_, None, None)

        subtotal_rows = column_weighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[4.4, 3.3], [2.2, 1.1]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_rows.tolist() == [[4.4, 7.7], [4.4, 7.7]]

    def but_it_returns_empty_array_of_right_shape_when_there_are_no_row_subtotals(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        """Empty shape must be (0, ncols) to compose properly in `np.block()` call."""
        _base_values_prop_.return_value = [[4.4, 3.3, 2.2], [1.1, 0.0, 9.9]]
        SumSubtotals_.subtotal_rows.return_value = np.array([], dtype=int).reshape(0, 3)
        column_weighted_bases = _ColumnWeightedBases(dimensions_, None, None)

        subtotal_rows = column_weighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[4.4, 3.3, 2.2], [1.1, 0.0, 9.9]], dimensions_, diff_cols_nan=True
        )
        assert subtotal_rows.tolist() == []
        assert subtotal_rows.shape == (0, 3)
        assert subtotal_rows.dtype == int

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _ColumnWeightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _ColumnWeightedBases, "_weighted_cube_counts")


class Describe_RowComparableCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._RowComparableCounts` object."""

    def it_computes_its_blocks(self, request, dimensions_, cube_measures_):
        counts = np.arange(12).reshape(3, 4).tolist()
        weighted_cube_counts_ = instance_mock(
            request, _BaseWeightedCubeCounts, weighted_counts=counts
        )
        _weighted_cube_counts = property_mock(
            request, _BaseSecondOrderMeasure, "_weighted_cube_counts"
        )
        _weighted_cube_counts.return_value = weighted_cube_counts_
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        row_comparable_counts = _RowComparableCounts(dimensions_, None, cube_measures_)

        blocks = row_comparable_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(
            counts, dimensions_, diff_rows_nan=True
        )
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)


class Describe_RowProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._RowProportions` object."""

    def it_computes_its_blocks(self, request):
        row_comparable_counts_ = instance_mock(
            request, _RowComparableCounts, blocks=[[5.0, 12.0], [21.0, 32.0]]
        )
        row_weighted_bases_ = instance_mock(
            request, _RowWeightedBases, blocks=[[5.0, 6.0], [7.0, 8.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            row_weighted_bases=row_weighted_bases_,
            row_comparable_counts=row_comparable_counts_,
        )
        cube_measures_ = class_mock(request, "cr.cube.matrix.cubemeasure.CubeMeasures")

        row_proportions = _RowProportions(None, second_order_measures_, cube_measures_)

        assert row_proportions.blocks == [[1.0, 2.0], [3.0, 4.0]]


class Describe_RowUnweightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._RowUnweightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _unweighted_cube_counts_prop_, unweighted_cube_counts_
    ):
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.row_bases = np.arange(6).reshape(2, 3)
        row_unweighted_bases = _RowUnweightedBases(None, None, None)

        assert row_unweighted_bases._base_values.tolist() == [[0, 1, 2], [3, 4, 5]]

    @pytest.mark.parametrize(
        "subtotal_rows, intersections, expected_value, expected_shape",
        (
            (
                np.array([[9, 9, 9], [6, 6, 6]]),
                np.array([[8, 4], [3, 7]]),
                [[9, 9], [6, 6]],
                (2, 2),
            ),
            (
                np.array([], dtype=int).reshape(0, 3),
                np.array([], dtype=int).reshape(0, 2),
                [],
                (0, 2),
            ),
        ),
    )
    def it_computes_its_intersections_block_to_help(
        self,
        request,
        subtotal_rows,
        intersections,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        expected_value,
        expected_shape,
    ):
        property_mock(
            request, _RowUnweightedBases, "_subtotal_rows", return_value=subtotal_rows
        )
        _base_values_prop_.return_value = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        SumSubtotals_.intersections.return_value = intersections
        row_unweighted_bases = _RowUnweightedBases(dimensions_, None, None)

        intersections = row_unweighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], dimensions_, diff_rows_nan=True
        )
        assert intersections.tolist() == expected_value
        assert intersections.shape == expected_shape
        assert intersections.dtype == int

    def it_computes_its_subtotal_columns_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _unweighted_cube_counts_prop_,
        unweighted_cube_counts_,
    ):
        _base_values_prop_.return_value = [[3, 4, 5], [1, 2, 3]]
        SumSubtotals_.subtotal_columns.return_value = np.array([[3, 8], [4, 6]])
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.rows_base = np.array([7, 4])
        row_unweighted_bases = _RowUnweightedBases(dimensions_, None, None)

        subtotal_columns = row_unweighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[3, 4, 5], [1, 2, 3]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_columns.tolist() == [[7, 7], [4, 4]]

    def but_it_returns_empty_array_of_right_shape_when_there_are_no_column_subtotals(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        """Empty shape must be (nrows, 0) to compose properly in `np.block()` call."""
        _base_values_prop_.return_value = [[2, 3, 4], [9, 0, 1]]
        SumSubtotals_.subtotal_columns.return_value = np.array([], dtype=int).reshape(
            3, 0
        )
        row_unweighted_bases = _RowUnweightedBases(dimensions_, None, None)

        subtotal_columns = row_unweighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[2, 3, 4], [9, 0, 1]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_columns.tolist() == [[], [], []]
        assert subtotal_columns.shape == (3, 0)
        assert subtotal_columns.dtype == int

    def it_computes_its_subtotal_rows_to_help(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        _base_values_prop_.return_value = [[1, 2], [3, 4]]
        SumSubtotals_.subtotal_rows.return_value = np.array([[5, 8], [3, 7]])
        row_unweighted_bases = _RowUnweightedBases(dimensions_, None, None)

        subtotal_rows = row_unweighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[1, 2], [3, 4]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_rows.tolist() == [[5, 8], [3, 7]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _RowUnweightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)

    @pytest.fixture
    def _unweighted_cube_counts_prop_(self, request):
        return property_mock(request, _RowUnweightedBases, "_unweighted_cube_counts")


class Describe_RowWeightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._RowWeightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _weighted_cube_counts_prop_, weighted_cube_counts_
    ):
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.row_bases = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        row_weighted_bases = _RowWeightedBases(None, None, None)

        assert row_weighted_bases._base_values.tolist() == [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
        ]

    @pytest.mark.parametrize(
        "subtotal_rows, intersections, expected_value, expected_shape",
        (
            (
                np.array([[9.9, 9.9, 9.9], [6.6, 6.6, 6.6]]),
                np.array([[8.8, 4.4], [3.3, 7.7]]),
                [[9.9, 9.9], [6.6, 6.6]],
                (2, 2),
            ),
            (
                np.array([], dtype=float).reshape(0, 3),
                np.array([], dtype=float).reshape(0, 2),
                [],
                (0, 2),
            ),
        ),
    )
    def it_computes_its_intersections_block_to_help(
        self,
        request,
        subtotal_rows,
        intersections,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        expected_value,
        expected_shape,
    ):
        property_mock(
            request, _RowWeightedBases, "_subtotal_rows", return_value=subtotal_rows
        )
        _base_values_prop_.return_value = [
            [9.9, 8.8, 7.7],
            [6.6, 5.5, 4.4],
            [3.3, 2.2, 1.1],
        ]
        SumSubtotals_.intersections.return_value = intersections
        row_weighted_bases = _RowWeightedBases(dimensions_, None, None)

        intersections = row_weighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[9.9, 8.8, 7.7], [6.6, 5.5, 4.4], [3.3, 2.2, 1.1]], dimensions_
        )
        assert intersections.tolist() == expected_value
        assert intersections.shape == expected_shape
        assert intersections.dtype == float

    def it_computes_its_subtotal_columns_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
    ):
        _base_values_prop_.return_value = [[3.3, 4.4, 5.5], [1.1, 2.2, 3.3]]
        SumSubtotals_.subtotal_columns.return_value = np.array([[3.3, 8.8], [4.4, 6.6]])
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.rows_margin = np.array([7.7, 4.4])
        row_weighted_bases = _RowWeightedBases(dimensions_, None, None)

        subtotal_columns = row_weighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[3.3, 4.4, 5.5], [1.1, 2.2, 3.3]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_columns.tolist() == [[7.7, 7.7], [4.4, 4.4]]

    def but_it_returns_empty_array_of_right_shape_when_there_are_no_column_subtotals(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        """Empty shape must be (nrows, 0) to compose properly in `np.block()` call."""
        _base_values_prop_.return_value = [[2.2, 3.3, 4.4], [9.9, 0.0, 1.1]]
        SumSubtotals_.subtotal_columns.return_value = np.array([], dtype=int).reshape(
            3, 0
        )
        row_weighted_bases = _RowWeightedBases(dimensions_, None, None)

        subtotal_columns = row_weighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[2.2, 3.3, 4.4], [9.9, 0.0, 1.1]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_columns.tolist() == [[], [], []]
        assert subtotal_columns.shape == (3, 0)
        assert subtotal_columns.dtype == int

    def it_computes_its_subtotal_rows_to_help(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        _base_values_prop_.return_value = [[1.1, 2.2], [3.3, 4.4]]
        SumSubtotals_.subtotal_rows.return_value = np.array([[5.5, 8.8], [3.3, 7.7]])
        row_weighted_bases = _RowWeightedBases(dimensions_, None, None)

        subtotal_rows = row_weighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[1.1, 2.2], [3.3, 4.4]], dimensions_, diff_rows_nan=True
        )
        assert subtotal_rows.tolist() == [[5.5, 8.8], [3.3, 7.7]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _RowWeightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _RowWeightedBases, "_weighted_cube_counts")


class Describe_ColumnShareSum(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnShareSum` object."""

    def it_computes_its_blocks(self, request):
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [
            [[5.0, 12.0], [21.0, 32.0]],
            [[], []],
            [[], []],
            [[], []],
        ]
        sums_blocks_ = instance_mock(request, _Sums, blocks=[[5.0, 6.0], [7.0, 8.0]])
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            sums=sums_blocks_,
        )
        cube_measures_ = class_mock(request, "cr.cube.matrix.cubemeasure.CubeMeasures")

        col_share_sum = _ColumnShareSum(None, second_order_measures_, cube_measures_)

        assert col_share_sum.blocks[0][0] == pytest.approx([0.2941176, 0.7058823])
        assert col_share_sum.blocks[0][1] == pytest.approx([0.3962264, 0.6037735])
        SumSubtotals_.blocks.assert_called_once_with(ANY, None, True, True)


class Describe_RowShareSum(object):
    """Unit test suite for `cr.cube.matrix.measure._RowShareSum` object."""

    def it_computes_its_blocks(self, request):
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [
            np.array([[[5.0, 12.0]], [[21.0, 32.0]]]),
            np.array([[[]], [[]]]),
            np.array([[[]], [[]]]),
            np.array([[[]], [[]]]),
        ]
        sums_blocks_ = instance_mock(
            request, _Sums, blocks=np.array([[[5.0, 6.0]], [[7.0, 8.0]]])
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            sums=sums_blocks_,
        )
        cube_measures_ = class_mock(request, "cr.cube.matrix.cubemeasure.CubeMeasures")

        row_share_sum = _RowShareSum(None, second_order_measures_, cube_measures_)

        assert row_share_sum.blocks[0][0] == pytest.approx(
            np.array([[0.29411765, 0.70588235]])
        )
        assert row_share_sum.blocks[0][1] == pytest.approx(
            np.array([[0.3962264, 0.6037735]])
        )
        SumSubtotals_.blocks.assert_called_once_with(ANY, None, True, True)


class Describe_TotalShareSum(object):
    """Unit test suite for `cr.cube.matrix.measure._RowShareSum` object."""

    def it_computes_its_blocks(self, request):
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [
            np.array([[[5.0, 12.0]], [[21.0, 32.0]]]),
            np.array([[[]], [[]]]),
            np.array([[[]], [[]]]),
            np.array([[[]], [[]]]),
        ]
        sums_blocks_ = instance_mock(
            request, _Sums, blocks=np.array([[[5.0, 6.0]], [[7.0, 8.0]]])
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            sums=sums_blocks_,
        )
        cube_measures_ = class_mock(request, "cr.cube.matrix.cubemeasure.CubeMeasures")

        total_share_sum = _TotalShareSum(None, second_order_measures_, cube_measures_)

        assert total_share_sum.blocks[0][0] == pytest.approx(
            np.array([[0.29411765, 0.70588235]])
        )
        assert total_share_sum.blocks[0][1] == pytest.approx(
            np.array([[0.3962264, 0.6037735]])
        )
        SumSubtotals_.blocks.assert_called_once_with(ANY, None, True, True)


class Describe_TableUnweightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._TableUnweightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _unweighted_cube_counts_prop_, unweighted_cube_counts_
    ):
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.table_bases = np.array([[3, 0, 5], [1, 4, 2]])
        table_unweighted_bases = _TableUnweightedBases(None, None, None)

        assert table_unweighted_bases._base_values.tolist() == [[3, 0, 5], [1, 4, 2]]

    @pytest.mark.parametrize(
        "intersections, table_base, expected_value, expected_shape",
        (
            # --- CAT_X_CAT with both row-subtotals and column-subtotals ---
            (
                np.array([[8, 4], [3, 7]]),
                7,
                [[7, 7], [7, 7]],
                (2, 2),
            ),
            # --- No intersections but table-base is an array (one of the MR cases) ---
            (
                np.array([], dtype=int).reshape(0, 2),
                [6, 6, 6],
                [],
                (0, 2),
            ),
            # --- No intersections and table-base is a scalar ---
            (
                np.array([], dtype=int).reshape(0, 2),
                9,
                [],
                (0, 2),
            ),
        ),
    )
    def it_computes_its_intersections_block_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _unweighted_cube_counts_prop_,
        unweighted_cube_counts_,
        intersections,
        table_base,
        expected_value,
        expected_shape,
    ):
        _base_values_prop_.return_value = [[6, 5, 4], [9, 8, 7], [3, 2, 1]]
        SumSubtotals_.intersections.return_value = intersections
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.table_base = table_base
        table_unweighted_bases = _TableUnweightedBases(dimensions_, None, None)

        intersections = table_unweighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[6, 5, 4], [9, 8, 7], [3, 2, 1]], dimensions_
        )
        assert intersections.tolist() == expected_value
        assert intersections.shape == expected_shape
        assert intersections.dtype == int

    @pytest.mark.parametrize(
        "table_base, expected_value",
        ((np.array([7, 4]), [[7, 7], [4, 4]]), (9, [[9, 9], [9, 9]])),
    )
    def it_computes_its_subtotal_columns_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _unweighted_cube_counts_prop_,
        unweighted_cube_counts_,
        table_base,
        expected_value,
    ):
        _base_values_prop_.return_value = [[5, 6, 3], [4, 1, 2]]
        SumSubtotals_.subtotal_columns.return_value = np.array([[0, 0], [0, 0]])
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.table_base = table_base
        table_unweighted_bases = _TableUnweightedBases(dimensions_, None, None)

        subtotal_columns = table_unweighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[5, 6, 3], [4, 1, 2]], dimensions_
        )
        assert subtotal_columns.tolist() == expected_value

    def but_it_returns_empty_array_of_right_shape_when_there_are_no_column_subtotals(
        self, _base_values_prop_, dimensions_, SumSubtotals_
    ):
        """Empty shape must be (nrows, 0) to compose properly in `np.block()` call."""
        _base_values_prop_.return_value = [[2, 3], [4, 9], [0, 1]]
        SumSubtotals_.subtotal_columns.return_value = np.array([], dtype=int).reshape(
            3, 0
        )
        table_unweighted_bases = _TableUnweightedBases(dimensions_, None, None)

        subtotal_columns = table_unweighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[2, 3], [4, 9], [0, 1]], dimensions_
        )
        assert subtotal_columns.tolist() == [[], [], []]
        assert subtotal_columns.shape == (3, 0)
        assert subtotal_columns.dtype == int

    @pytest.mark.parametrize(
        "subtotal_rows, table_base, expected_value, expected_shape",
        (
            # --- no subtotal-rows case (including MR_X_CAT and MR_X_MR cases) ---
            (np.array([], dtype=int).reshape(0, 2), None, [], (0, 2)),
            # --- scalar table-base (CAT_X_CAT case) ---
            (np.array([[0, 0], [0, 0]]), 8, [[8, 8], [8, 8]], (2, 2)),
            # --- vector table-base (CAT_X_MR case) ---
            (np.array([[0, 0], [0, 0]]), np.array([4, 2]), [[4, 2], [4, 2]], (2, 2)),
        ),
    )
    def it_computes_its_subtotal_rows_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _unweighted_cube_counts_prop_,
        unweighted_cube_counts_,
        subtotal_rows,
        table_base,
        expected_value,
        expected_shape,
    ):
        _base_values_prop_.return_value = [[3, 5, 6], [2, 4, 1]]
        SumSubtotals_.subtotal_rows.return_value = subtotal_rows
        _unweighted_cube_counts_prop_.return_value = unweighted_cube_counts_
        unweighted_cube_counts_.table_base = table_base
        table_unweighted_bases = _TableUnweightedBases(dimensions_, None, None)

        subtotal_rows = table_unweighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[3, 5, 6], [2, 4, 1]], dimensions_
        )
        assert subtotal_rows.tolist() == expected_value
        assert subtotal_rows.dtype == int
        assert subtotal_rows.shape == expected_shape

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _TableUnweightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def unweighted_cube_counts_(self, request):
        return instance_mock(request, _BaseUnweightedCubeCounts)

    @pytest.fixture
    def _unweighted_cube_counts_prop_(self, request):
        return property_mock(request, _TableUnweightedBases, "_unweighted_cube_counts")


class Describe_TableWeightedBases(object):
    """Unit test suite for `cr.cube.matrix.measure._TableWeightedBases` object."""

    def it_computes_its_base_values_to_help(
        self, _weighted_cube_counts_prop_, weighted_cube_counts_
    ):
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.table_bases = np.array([[3.3, 0.0, 5.5], [1.1, 4.4, 2.2]])
        table_weighted_bases = _TableWeightedBases(None, None, None)

        assert table_weighted_bases._base_values.tolist() == [
            [3.3, 0.0, 5.5],
            [1.1, 4.4, 2.2],
        ]

    @pytest.mark.parametrize(
        "intersections, table_margin, expected_value, expected_shape",
        (
            # --- CAT_X_CAT with both row-subtotals and column-subtotals ---
            (
                np.array([[8, 4], [3, 7]]),
                7,
                [[7, 7], [7, 7]],
                (2, 2),
            ),
            # --- All other cases, intersections array is empty ---
            (
                np.array([], dtype=int).reshape(0, 2),
                None,
                [],
                (0, 2),
            ),
        ),
    )
    def it_computes_its_intersections_block_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
        intersections,
        table_margin,
        expected_value,
        expected_shape,
    ):
        _base_values_prop_.return_value = [
            [6.6, 5.5, 4.4],
            [9.9, 8.8, 7.7],
            [3.3, 2.2, 1.1],
        ]
        SumSubtotals_.intersections.return_value = intersections
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.table_margin = table_margin
        table_weighted_bases = _TableWeightedBases(dimensions_, None, None)

        intersections = table_weighted_bases._intersections

        SumSubtotals_.intersections.assert_called_once_with(
            [[6.6, 5.5, 4.4], [9.9, 8.8, 7.7], [3.3, 2.2, 1.1]], dimensions_
        )
        assert intersections.tolist() == expected_value
        assert intersections.shape == expected_shape
        assert intersections.dtype == int

    @pytest.mark.parametrize(
        "subtotal_columns_, table_margin, expected_value",
        (
            # --- CAT_X_CAT case, scalar table-margin ---
            (
                np.array([[6.1, 6.2], [6.3, 6.4]]),
                9.9,
                np.array([[9.9, 9.9], [9.9, 9.9]]),
            ),
            # --- CAT_X_MR and MR_X_MR cases, subtotal-columns is (nrows, 0) array ---
            (
                np.array([[], []]),
                None,
                np.array([[], []]),
            ),
            # --- MR_X_CAT case, table-margin is a 1D array "column" ---
            (
                np.array([[6.6, 7.7], [8.8, 9.9]]),
                np.array([3.4, 5.6]),
                np.array([[3.4, 3.4], [5.6, 5.6]]),
            ),
        ),
    )
    def it_computes_its_subtotal_columns_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        subtotal_columns_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
        table_margin,
        expected_value,
    ):
        _base_values_prop_.return_value = [[5.5, 6.6, 3.3], [4.4, 1.1, 2.2]]
        SumSubtotals_.subtotal_columns.return_value = subtotal_columns_
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.table_margin = table_margin
        table_weighted_bases = _TableWeightedBases(dimensions_, None, None)

        subtotal_columns = table_weighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            [[5.5, 6.6, 3.3], [4.4, 1.1, 2.2]], dimensions_
        )
        assert subtotal_columns == pytest.approx(expected_value)

    @pytest.mark.parametrize(
        "subtotal_rows_, table_margin, expected_value",
        (
            # --- CAT_X_CAT case, scalar table-margin ---
            (
                np.array([[6.1, 6.2, 6.3], [6.4, 6.5, 6.6]]),
                9.9,
                np.array([[9.9, 9.9, 9.9], [9.9, 9.9, 9.9]]),
            ),
            # --- CAT_X_MR case, table-margin is a 1D array "row" ---
            (
                np.array([[6.1, 6.2, 6.3], [6.4, 6.5, 6.6]]),
                np.array([3.4, 5.6, 7.8]),
                np.array(
                    [
                        [3.4, 5.6, 7.8],
                        [3.4, 5.6, 7.8],
                    ]
                ),
            ),
            # --- MR_X_CAT and MR_X_MR cases, subtotal-rows is (0, ncols) array ---
            (
                np.array([]).reshape(0, 3),
                None,
                np.array([]).reshape(0, 3),
            ),
        ),
    )
    def it_computes_its_subtotal_rows_to_help(
        self,
        _base_values_prop_,
        dimensions_,
        SumSubtotals_,
        subtotal_rows_,
        _weighted_cube_counts_prop_,
        weighted_cube_counts_,
        table_margin,
        expected_value,
    ):
        _base_values_prop_.return_value = [[5.5, 6.6, 3.3], [4.4, 1.1, 2.2]]
        SumSubtotals_.subtotal_rows.return_value = subtotal_rows_
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        weighted_cube_counts_.table_margin = table_margin
        table_weighted_bases = _TableWeightedBases(dimensions_, None, None)

        subtotal_rows = table_weighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            [[5.5, 6.6, 3.3], [4.4, 1.1, 2.2]], dimensions_
        )
        assert subtotal_rows == pytest.approx(expected_value)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _TableWeightedBases, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.measure.SumSubtotals")

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _TableWeightedBases, "_weighted_cube_counts")


class Describe_UnweightedCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._UnweightedCounts` object."""

    def it_computes_its_blocks_to_help(self, request, dimensions_, cube_measures_):
        # --- these need to be in list form because the assert-called-with mechanism
        # --- uses equality, which doesn't work on numpy arrays. Normally this would be
        # --- the array itself.
        ucounts = np.arange(12).reshape(3, 4).tolist()
        unweighted_cube_counts_ = instance_mock(
            request,
            _BaseUnweightedCubeCounts,
            unweighted_counts=ucounts,
            diff_nans=False,
        )
        property_mock(
            request,
            _UnweightedCounts,
            "_unweighted_cube_counts",
            return_value=unweighted_cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        unweighted_counts = _UnweightedCounts(dimensions_, None, cube_measures_)

        blocks = unweighted_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(ucounts, dimensions_, False, False)
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)


class Describe_WeightedCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._WeightedCounts` object."""

    def it_computes_its_blocks_to_help(self, request, dimensions_, cube_measures_):
        # --- these need to be in list form because the assert-called-with mechanism
        # --- uses equality, which doesn't work on numpy arrays. Normally this would be
        # --- the array itself.
        counts = np.arange(12).reshape(3, 4).tolist()
        weighted_cube_counts_ = instance_mock(
            request, _BaseWeightedCubeCounts, weighted_counts=counts, diff_nans=False
        )
        property_mock(
            request,
            _WeightedCounts,
            "_weighted_cube_counts",
            return_value=weighted_cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        weighted_counts = _WeightedCounts(dimensions_, None, cube_measures_)

        blocks = weighted_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(counts, dimensions_, False, False)
        assert blocks == [[[1], [2]], [[3], [4]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)


class Describe_Zscores(object):
    """Unit test suite for `cr.cube.matrix.measure._Zscores` object."""

    def it_computes_zscore_subtotals_blocks(self, request, dimensions_):
        weighted_cube_counts_ = instance_mock(request, _BaseWeightedCubeCounts)
        property_mock(
            request,
            _Zscores,
            "_weighted_cube_counts",
            return_value=weighted_cube_counts_,
        )
        ZscoreSubtotals_ = class_mock(request, "cr.cube.matrix.measure.ZscoreSubtotals")
        ZscoreSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        zscores = _Zscores(dimensions_, None, None)

        blocks = zscores.blocks

        ZscoreSubtotals_.blocks.assert_called_once_with(
            weighted_cube_counts_, dimensions_
        )
        assert blocks == [[[1], [2]], [[3], [4]]]

    def but_the_subtotal_blocks_are_NaNs_when_an_MR_dimension_is_present(
        self, request, dimensions_
    ):
        weighted_cube_counts_ = instance_mock(
            request, _BaseWeightedCubeCounts, zscores=[[1, 2], [3, 4]]
        )
        property_mock(
            request,
            _Zscores,
            "_weighted_cube_counts",
            return_value=weighted_cube_counts_,
        )
        NanSubtotals_ = class_mock(request, "cr.cube.matrix.measure.NanSubtotals")
        NanSubtotals_.blocks.return_value = [[[1], [np.nan]], [[np.nan], [np.nan]]]
        dimensions_[0].dimension_type = DT.MR_SUBVAR
        zscores = _Zscores(dimensions_, None, None)

        blocks = zscores.blocks

        NanSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        assert blocks == [[[1], [np.nan]], [[np.nan], [np.nan]]]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)


# === Marginals ===


class Describe_BaseMarginal(object):
    """Unit test suite for `cr.cube.matrix.measure._BaseMarginal` object."""

    def it_is_defined_by_default(self):
        marginal = _BaseMarginal(None, None, None, None)
        assert marginal.is_defined is True

    @pytest.mark.parametrize("orientation", (MO.ROWS, MO.COLUMNS))
    def it_knows_the_orientation_of_the_marginal(self, orientation):
        marginal = _BaseMarginal(None, None, None, orientation)

        assert marginal.orientation == orientation

    @pytest.mark.parametrize(
        "orientation, array, expected",
        (
            (MO.ROWS, np.array([[0, 1, 2], [3, 4, 5]]), [3, 12]),
            (MO.COLUMNS, np.array([[0, 1, 2], [3, 4, 5]]), [3, 5, 7]),
            (MO.ROWS, np.array([]).reshape(0, 6), []),
            (MO.COLUMNS, np.array([]).reshape(6, 0), []),
        ),
    )
    def it_can_apply_along_orientation_to_help(self, orientation, array, expected):
        marginal = _BaseMarginal(None, None, None, orientation)
        result = marginal._apply_along_orientation(np.sum, array)
        assert result.tolist() == expected

    def it_gets_the_right_counts_for_rows(self, request):
        row_comparable_counts_ = instance_mock(
            request, _BaseMarginal, blocks=[["a", "b"], ["c", "d"]]
        )
        second_order_measures_ = instance_mock(
            request, SecondOrderMeasures, row_comparable_counts=row_comparable_counts_
        )
        median = _BaseMarginal(None, second_order_measures_, None, MO.ROWS)

        assert median._counts == ["a", "c"]

    def it_gets_the_right_counts_for_columns(self, request):
        column_comparable_counts_ = instance_mock(
            request, _BaseMarginal, blocks=[["a", "b"], ["c", "d"]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            column_comparable_counts=column_comparable_counts_,
        )
        median = _BaseMarginal(None, second_order_measures_, None, MO.COLUMNS)

        assert median._counts == ["a", "b"]


class Describe_BaseScaledCountMarginal(object):
    """Unit test suite for `cr.cube.matrix.measure._BaseScaledCountMarginal` object."""

    @pytest.mark.parametrize(
        "orientation, expected", ((MO.ROWS, [1]), (MO.COLUMNS, [0]))
    )
    def it_provides_opposing_numeric_values_to_help(
        self, request, orientation, expected
    ):
        row_dim_ = instance_mock(request, Dimension, numeric_values=[0])
        col_dim_ = instance_mock(request, Dimension, numeric_values=[1])
        marginal = _BaseScaledCountMarginal(
            [row_dim_, col_dim_], None, None, orientation
        )

        assert marginal._opposing_numeric_values == expected


class Describe_ScaleMean(object):
    """Unit test suite for `cr.cube.matrix.measure._ScaleMean` object."""

    def it_provides_blocks(self, request):
        property_mock(request, _ScaleMean, "is_defined", return_value=True)
        property_mock(
            request, _ScaleMean, "_proportions", return_value=["prop1", "prop2"]
        )
        property_mock(request, _ScaleMean, "_opposing_numeric_values")
        _apply_along_orientation_ = method_mock(
            request,
            _ScaleMean,
            "_apply_along_orientation",
            side_effect=("result1", "result2"),
        )
        mean = _ScaleMean(None, None, None, MO.ROWS)

        results = mean.blocks

        assert results == ["result1", "result2"]
        assert _apply_along_orientation_.call_args_list == [
            call(
                mean,
                mean._weighted_mean,
                "prop1",
                values=mean._opposing_numeric_values,
            ),
            call(
                mean,
                mean._weighted_mean,
                "prop2",
                values=mean._opposing_numeric_values,
            ),
        ]

    def but_blocks_raises_if_it_is_undefined(self, request):
        property_mock(request, _ScaleMean, "is_defined", return_value=False)

        with pytest.raises(ValueError) as e:
            _ScaleMean(None, None, None, MO.ROWS).blocks

        assert str(e.value) == (
            "rows-scale-mean is undefined if no numeric values are defined on opposing "
            "dimension."
        )

    @pytest.mark.parametrize(
        "values, expected", ((np.array([0, 1]), True), (np.array([np.nan]), False))
    )
    def it_can_tell_if_it_is_defined(self, request, values, expected):
        property_mock(
            request, _ScaleMean, "_opposing_numeric_values", return_value=values
        )
        mean = _ScaleMean(None, None, None, MO.ROWS)
        assert mean.is_defined == expected

    def it_gets_the_right_proportions_for_rows(self, request):
        row_proportions_ = instance_mock(
            request, _RowProportions, blocks=[["a", "b"], ["c", "d"]]
        )
        second_order_measures_ = instance_mock(
            request, SecondOrderMeasures, row_proportions=row_proportions_
        )

        mean = _ScaleMean(None, second_order_measures_, None, MO.ROWS)

        assert mean._proportions == ["a", "c"]

    def it_gets_the_right_counts_for_columns(self, request):
        column_proportions_ = instance_mock(
            request, _ColumnProportions, blocks=[["a", "b"], ["c", "d"]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            column_proportions=column_proportions_,
        )

        mean = _ScaleMean(None, second_order_measures_, None, MO.COLUMNS)

        assert mean._proportions == ["a", "b"]

    @pytest.mark.parametrize(
        "proportions, values, expected",
        (
            ([0.0, 0.5, 0.5], [1.0, 2.0, 3.0], 2.5),
            ([0.0, 0.5, 0.5], [1.0, 2.0, np.nan], 2.0),
            ([np.nan, np.nan, np.nan], [1.0, 2.0, 3.0], np.nan),
        ),
    )
    def it_calculates_weighted_mean(self, proportions, values, expected):
        mean = _ScaleMean._weighted_mean(np.array(proportions), np.array(values))
        assert mean == pytest.approx(expected, nan_ok=True)


class Describe_ScaleMeanStddev(object):
    """Unit test suite for `cr.cube.matrix.measure._ScaleMeanStddev` object."""

    def it_provides_blocks_for_rows(self, request):
        pass

    def but_blocks_raises_if_undefined(self, request):
        property_mock(request, _ScaleMeanStddev, "is_defined", return_value=False)

        with pytest.raises(ValueError) as e:
            _ScaleMeanStddev(None, None, None, MO.ROWS).blocks

        assert (
            str(e.value)
            == "rows-scale-mean-standard-deviation is undefined if no numeric values are "
            + "defined on opposing dimension."
        )

    def it_gets_the_right_scale_mean_for_columns(self, request):
        columns_scale_mean_ = instance_mock(request, _ScaleMean, blocks=["a", "b"])
        second_order_measures_ = instance_mock(
            request, SecondOrderMeasures, columns_scale_mean=columns_scale_mean_
        )

        stddev = _ScaleMeanStddev(None, second_order_measures_, None, MO.COLUMNS)

        assert stddev._scale_means == ["a", "b"]

    def it_gets_the_right_scale_mean_for_rows(self, request):
        rows_scale_mean_ = instance_mock(request, _ScaleMean, blocks=["a", "b"])
        second_order_measures_ = instance_mock(
            request, SecondOrderMeasures, rows_scale_mean=rows_scale_mean_
        )

        stddev = _ScaleMeanStddev(None, second_order_measures_, None, MO.ROWS)

        assert stddev._scale_means == ["a", "b"]

    @pytest.mark.parametrize(
        "counts, values, scale_mean, expected",
        (
            (
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                np.array([7.0, 8.0, 9.0]),
                np.array([10.0, 11.0]),
                np.array([1.8257418, 2.977694]),
            ),
            (
                np.array([[1.0, 2.0, 3.0, 3.5], [4.0, 5.0, 6.0, 6.5]]),
                np.array([7.0, 8.0, 9.0, np.nan]),
                np.array([10.0, 11.0]),
                np.array([1.8257418, 2.977694]),
            ),
            (
                np.array([]).reshape(0, 3),
                np.array([7.0, 8.0, 9.0]),
                np.array([]),
                np.array([]),
            ),
        ),
    )
    def it_can_calculate_stddev_for_rows(self, counts, values, scale_mean, expected):
        result = _ScaleMeanStddev._rows_weighted_mean_stddev(counts, values, scale_mean)

        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    @pytest.mark.parametrize(
        "counts, values, scale_mean, expected",
        (
            (
                np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
                np.array([7.0, 8.0, 9.0]),
                np.array([10.0, 11.0]),
                np.array([1.8257418, 2.977694]),
            ),
            (
                np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [3.5, 6.5]]),
                np.array([7.0, 8.0, 9.0, np.nan]),
                np.array([10.0, 11.0]),
                np.array([1.8257418, 2.977694]),
            ),
            (
                np.array([]).reshape(3, 0),
                np.array([7.0, 8.0, 9.0]),
                np.array([]),
                np.array([]),
            ),
        ),
    )
    def it_can_calculate_stddev_for_columns(self, counts, values, scale_mean, expected):
        result = _ScaleMeanStddev._columns_weighted_mean_stddev(
            counts, values, scale_mean
        )

        assert result.tolist() == pytest.approx(expected, nan_ok=True)


class Describe_ScaleMedian(object):
    """Unit test suite for `cr.cube.matrix.measure._ScaleMedian` object."""

    def it_provides_blocks(self, request):
        property_mock(request, _ScaleMedian, "is_defined", return_value=True)
        property_mock(request, _ScaleMedian, "_sorted_values")
        property_mock(
            request, _ScaleMedian, "_sorted_counts", return_value=["count1", "count2"]
        )
        _apply_along_orientation_ = method_mock(
            request,
            _ScaleMedian,
            "_apply_along_orientation",
            side_effect=("result1", "result2"),
        )
        median = _ScaleMedian(None, None, None, MO.ROWS)

        results = median.blocks

        assert results == ["result1", "result2"]
        assert _apply_along_orientation_.call_args_list == [
            call(
                median,
                median._weighted_median,
                "count1",
                sorted_values=median._sorted_values,
            ),
            call(
                median,
                median._weighted_median,
                "count2",
                sorted_values=median._sorted_values,
            ),
        ]

    def but_blocks_raises_if_it_is_undefined(self, request):
        property_mock(request, _ScaleMedian, "is_defined", return_value=False)

        with pytest.raises(ValueError) as e:
            _ScaleMedian(None, None, None, MO.ROWS).blocks

        assert (
            str(e.value)
            == "rows-scale-median is undefined if no numeric values are defined on opposing "
            + "dimension."
        )

    @pytest.mark.parametrize(
        "values, expected",
        (
            ([100], [0]),
            ([2.1, 42, -20], [2, 0, 1]),
            ([0, np.nan, 2], [0, 2]),
            ([25, 10, np.nan, 5, np.nan], [3, 1, 0]),
        ),
    )
    def it_gets_the_right_sort_order(self, request, values, expected):
        property_mock(
            request,
            _ScaleMedian,
            "_opposing_numeric_values",
            return_value=np.array(values),
        )
        sort_order = _ScaleMedian(None, None, None, MO.ROWS)._values_sort_order
        assert sort_order.tolist() == expected

    @pytest.mark.parametrize(
        "values, expected",
        (
            ([100], [100]),
            ([2.1, 42, -20], [-20, 2.1, 42]),
            ([0, np.nan, 2], [0, 2]),
            ([25, 10, np.nan, 5, np.nan], [5, 10, 25]),
        ),
    )
    def it_sorts_the_values_to_help(self, request, values, expected):
        property_mock(
            request,
            _ScaleMedian,
            "_opposing_numeric_values",
            return_value=np.array(values),
        )
        sort_order = _ScaleMedian(None, None, None, MO.ROWS)._sorted_values
        assert sort_order.tolist() == expected

    @pytest.mark.parametrize(
        "orientation, sort_order, base_values, subtotals, expected",
        (
            (
                MO.COLUMNS,
                np.array([2, 1]),
                np.array([[0, 1], [2, 3], [4, 5]]),
                np.array([]).reshape(3, 0),
                [[[4, 5], [2, 3]], [[], []]],
            ),
            (
                MO.COLUMNS,
                np.array([0, 2, 1]),
                np.array([[0, 1], [2, 3], [4, 5]]),
                np.array([[6], [7], [8]]),
                [[[0, 1], [4, 5], [2, 3]], [[6], [8], [7]]],
            ),
            (
                MO.ROWS,
                np.array([3, 1]),
                np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
                np.array([]).reshape(0, 3),
                [[[3, 1], [7, 5], [11, 9]], []],
            ),
            (
                MO.ROWS,
                np.array([1, 0]),
                np.array([[0, 1], [2, 3], [4, 5]]),
                np.array([[6, 7], [8, 9], [10, 11]]),
                [[[1, 0], [3, 2], [5, 4]], [[7, 6], [9, 8], [11, 10]]],
            ),
        ),
    )
    def it_sorts_the_counts_to_help(
        self, request, orientation, sort_order, base_values, subtotals, expected
    ):
        property_mock(
            request, _ScaleMedian, "_values_sort_order", return_value=sort_order
        )
        property_mock(
            request,
            _ScaleMedian,
            "_counts",
            return_value=[base_values, subtotals],
        )

        results = list(_ScaleMedian(None, None, None, orientation)._sorted_counts)

        assert results[0].tolist() == expected[0]
        assert results[1].tolist() == expected[1]

    @pytest.mark.parametrize(
        "counts, values, expected",
        (
            ([1.0, 2.0], [10.0, 20.0], 20.0),
            ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 25.0),
            ([100000000.0, 100000000.0, 200000000.0], [10.0, 20.0, 30.0], 25.0),
            ([1.9, 2.9, 4.1], [10.0, 20.0, 30.0], 20.0),
            ([0.9, 0.1], [10.0, 20.0], 10.0),
            ([1.0, np.nan, 2.0], [10.0, 20.0, 30.0], 30.0),
            ([np.nan, np.nan, np.nan], [10.0, 20.0, 30.0], np.nan),
            ([0.0, 0.0, 0.0], [10.0, 20.0, 30.0], np.nan),
        ),
    )
    def it_calculates_weighted_median(self, counts, values, expected):
        assert _ScaleMedian._weighted_median(
            np.array(counts), np.array(values)
        ) == pytest.approx(expected, nan_ok=True)
