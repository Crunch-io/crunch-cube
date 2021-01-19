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
    _ColumnWeightedBases,
    _RowUnweightedBases,
    _RowWeightedBases,
    SecondOrderMeasures,
    _TableUnweightedBases,
    _TableWeightedBases,
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

    def it_provides_access_to_the_column_weighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        column_weighted_bases_ = instance_mock(request, _ColumnWeightedBases)
        _ColumnWeightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._ColumnWeightedBases",
            return_value=column_weighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        column_weighted_bases = measures.column_weighted_bases

        _ColumnWeightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert column_weighted_bases is column_weighted_bases_

    def it_provides_access_to_the_row_unweighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        row_unweighted_bases_ = instance_mock(request, _RowUnweightedBases)
        _RowUnweightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._RowUnweightedBases",
            return_value=row_unweighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        row_unweighted_bases = measures.row_unweighted_bases

        _RowUnweightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert row_unweighted_bases is row_unweighted_bases_

    def it_provides_access_to_the_row_weighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        row_weighted_bases_ = instance_mock(request, _RowWeightedBases)
        _RowWeightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._RowWeightedBases",
            return_value=row_weighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        row_weighted_bases = measures.row_weighted_bases

        _RowWeightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert row_weighted_bases is row_weighted_bases_

    def it_provides_access_to_the_table_unweighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        table_unweighted_bases_ = instance_mock(request, _TableUnweightedBases)
        _TableUnweightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._TableUnweightedBases",
            return_value=table_unweighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        table_unweighted_bases = measures.table_unweighted_bases

        _TableUnweightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert table_unweighted_bases is table_unweighted_bases_

    def it_provides_access_to_the_table_weighted_bases_measure_object(
        self, request, dimensions_, _cube_measures_prop_, cube_measures_
    ):
        table_weighted_bases_ = instance_mock(request, _TableWeightedBases)
        _TableWeightedBases_ = class_mock(
            request,
            "cr.cube.matrix.measure._TableWeightedBases",
            return_value=table_weighted_bases_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        table_weighted_bases = measures.table_weighted_bases

        _TableWeightedBases_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert table_weighted_bases is table_weighted_bases_

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
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], dimensions_
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
            [[1, 2], [3, 4]], dimensions_
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
            [[4, 3], [2, 1]], dimensions_
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
            [[4, 3, 2], [1, 0, 9]], dimensions_
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
            [[9.9, 8.8, 7.7], [6.6, 5.5, 4.4], [3.3, 2.2, 1.1]], dimensions_
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
            [[1.1, 2.2], [3.3, 4.4]], dimensions_
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
            [[4.4, 3.3], [2.2, 1.1]], dimensions_
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
            [[4.4, 3.3, 2.2], [1.1, 0.0, 9.9]], dimensions_
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
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], dimensions_
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
            [[3, 4, 5], [1, 2, 3]], dimensions_
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
            [[2, 3, 4], [9, 0, 1]], dimensions_
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
            [[1, 2], [3, 4]], dimensions_
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
            [[3.3, 4.4, 5.5], [1.1, 2.2, 3.3]], dimensions_
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
            [[2.2, 3.3, 4.4], [9.9, 0.0, 1.1]], dimensions_
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
            [[1.1, 2.2], [3.3, 4.4]], dimensions_
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
