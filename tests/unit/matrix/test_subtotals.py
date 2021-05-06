# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.subtotals` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.matrix.cubemeasure import BaseCubeResultMatrix, _BaseWeightedCubeCounts
from cr.cube.matrix.subtotals import (
    _BaseSubtotals,
    NanSubtotals,
    SumSubtotals,
    TableStdErrSubtotals,
    ZscoreSubtotals,
)

from ...unitutil import ANY, initializer_mock, instance_mock, method_mock, property_mock


class Describe_BaseSubtotals(object):
    """Unit test suite for `cr.cube.matrix._BaseSubtotals` object."""

    def it_provides_a_blocks_interface_method(self, request, cube_result_matrix_):
        _init_ = initializer_mock(request, _BaseSubtotals)
        _blocks_ = property_mock(
            request, _BaseSubtotals, "_blocks", return_value=[[1, 2], [3, 4]]
        )

        blocks = _BaseSubtotals.blocks(cube_result_matrix_, "weighted_counts")

        _init_.assert_called_once()
        _blocks_.assert_called_once()
        assert blocks == [[1, 2], [3, 4]]

    def it_can_compute_its_blocks(self, request):
        property_mock(request, _BaseSubtotals, "_subtotal_columns")
        property_mock(request, _BaseSubtotals, "_subtotal_rows")
        property_mock(request, _BaseSubtotals, "_intersections")
        base_subtotals = _BaseSubtotals("_base_values", None)

        blocks = base_subtotals._blocks

        assert blocks == [
            ["_base_values", base_subtotals._subtotal_columns],
            [base_subtotals._subtotal_rows, base_subtotals._intersections],
        ]

    def it_provides_access_to_the_column_subtotals_to_help(self, dimension_):
        subtotals = _BaseSubtotals(None, (None, dimension_))
        assert subtotals._column_subtotals is dimension_.subtotals

    def it_assembles_its_intersections_to_help(
        self, _intersection_, _column_subtotals_prop_, _row_subtotals_prop_
    ):
        _intersection_.return_value = 10
        _column_subtotals_prop_.return_value = [1, 2, 3]
        _row_subtotals_prop_.return_value = [1, 2]
        subtotals = _BaseSubtotals(None, None)

        assert (
            subtotals._intersections.tolist()
            == np.full(
                (
                    2,  # row subtotals has len 2
                    3,  # column subtotals has len 3
                ),
                10,  # dummy fill value from `_intersection` fixture
            ).tolist()
        )

    def it_knows_how_many_columns_are_in_the_base_matrix_to_help(self):
        base_values = np.arange(12).reshape(3, 4)
        assert _BaseSubtotals(base_values, None)._ncols == 4

    def it_knows_how_many_rows_are_in_the_base_matrix_to_help(self):
        base_values = np.arange(12).reshape(3, 4)
        assert _BaseSubtotals(base_values, None)._nrows == 3

    def it_provides_access_to_the_row_subtotals_to_help(self, dimension_):
        subtotals = _BaseSubtotals(None, (dimension_, None))
        assert subtotals._row_subtotals is dimension_.subtotals

    @pytest.mark.parametrize(
        ("nrows", "n_subtotals", "expected_value"),
        (
            (3, 0, np.empty((3, 0), dtype=np.float64)),
            (3, 2, np.array([[1, 1], [2, 2], [3, 3]])),
        ),
    )
    def it_assembles_its_subtotal_columns_to_help(
        self,
        _column_subtotals_prop_,
        _nrows_prop_,
        _subtotal_column_,
        subtotal_,
        nrows,
        n_subtotals,
        expected_value,
    ):
        _column_subtotals_prop_.return_value = (subtotal_,) * n_subtotals
        _subtotal_column_.return_value = np.array([1, 2, 3])
        _nrows_prop_.return_value = nrows

        subtotal_columns = _BaseSubtotals(None, None)._subtotal_columns

        assert subtotal_columns.tolist() == expected_value.tolist()

    @pytest.mark.parametrize(
        ("ncols", "row_subtotals", "shape", "expected_value"),
        (
            (3, [], (0, 3), []),
            (3, [1, 2], (2, 3), [[1, 2, 3], [1, 2, 3]]),
        ),
    )
    def it_assembles_its_subtotal_rows_to_help(
        self,
        _row_subtotals_prop_,
        _ncols_prop_,
        _subtotal_row_,
        ncols,
        row_subtotals,
        shape,
        expected_value,
    ):
        _row_subtotals_prop_.return_value = row_subtotals
        _subtotal_row_.return_value = np.array([1, 2, 3])
        _ncols_prop_.return_value = ncols

        subtotal_rows = _BaseSubtotals(None, None)._subtotal_rows

        assert subtotal_rows.shape == shape
        assert subtotal_rows.tolist() == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _column_subtotals_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_column_subtotals")

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, BaseCubeResultMatrix)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _intersection_(self, request):
        return method_mock(request, _BaseSubtotals, "_intersection")

    @pytest.fixture
    def _ncols_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_ncols")

    @pytest.fixture
    def _nrows_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_nrows")

    @pytest.fixture
    def _row_subtotals_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_row_subtotals")

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)

    @pytest.fixture
    def _subtotal_column_(self, request):
        return method_mock(request, _BaseSubtotals, "_subtotal_column")

    @pytest.fixture
    def _subtotal_row_(self, request):
        return method_mock(request, _BaseSubtotals, "_subtotal_row")


class DescribeNanSubtotals(object):
    """Unit test suite for `cr.cube.matrix.NanSubtotals` object."""

    def it_can_compute_a_intersection_cell_value_to_help(self, request):
        assert np.isnan(NanSubtotals(None, None)._intersection(None, None))

    def it_can_compute_a_subtotal_column_to_help(self, request):
        property_mock(request, NanSubtotals, "_nrows", return_value=3)
        subtotals = NanSubtotals(None, None)

        np.testing.assert_equal(
            subtotals._subtotal_column(None),
            np.array([np.nan] * 3),
        )

    def it_can_compute_a_subtotal_row_to_help(self, request):
        property_mock(request, NanSubtotals, "_ncols", return_value=4)
        subtotals = NanSubtotals(None, None)

        np.testing.assert_equal(
            subtotals._subtotal_row(None),
            np.array([np.nan] * 4),
        )


class DescribeSumSubtotals(object):
    """Unit test suite for `cr.cube.matrix.SumSubtotals` object."""

    def it_provides_an_intersections_interface_method(
        self, request, dimensions_, _init_
    ):
        base_values = [[1, 5], [8, 0]]
        property_mock(
            request,
            SumSubtotals,
            "_intersections",
            return_value=np.array([[1, 2], [3, 4]]),
        )

        intersections = SumSubtotals.intersections(base_values, dimensions_)

        _init_.assert_called_once_with(ANY, [[1, 5], [8, 0]], dimensions_, False, False)
        assert intersections.tolist() == [[1, 2], [3, 4]]

    def it_provides_a_subtotal_columns_interface_method(
        self, request, dimensions_, _init_
    ):
        base_values = [[0, 4], [7, 9]]
        property_mock(
            request,
            SumSubtotals,
            "_subtotal_columns",
            return_value=np.array([[1, 2], [3, 4]]),
        )

        subtotal_columns = SumSubtotals.subtotal_columns(base_values, dimensions_)

        _init_.assert_called_once_with(ANY, [[0, 4], [7, 9]], dimensions_, False, False)
        assert subtotal_columns.tolist() == [[1, 2], [3, 4]]

    def it_provides_a_subtotal_rows_interface_method(
        self, request, dimensions_, _init_
    ):
        base_values = [[4, 1], [3, 5]]
        property_mock(
            request,
            SumSubtotals,
            "_subtotal_rows",
            return_value=np.array([[4, 3], [2, 1]]),
        )

        subtotal_rows = SumSubtotals.subtotal_rows(base_values, dimensions_)

        _init_.assert_called_once_with(ANY, [[4, 1], [3, 5]], dimensions_, False, False)
        assert subtotal_rows.tolist() == [[4, 3], [2, 1]]

    @pytest.mark.parametrize(
        (
            "row_add_idxs",
            "row_sub_idxs",
            "col_add_idxs",
            "col_sub_idxs",
            "diff_cols_nan",
            "diff_rows_nan",
            "expected_value",
        ),
        (
            ([1, 2], [], [0, 1], [], False, False, 26),
            ([0, 1], [], [0, 1], [], False, False, 10),
            ([1, 2], [], [2, 3], [], False, False, 34),
            ([1, 2], [0], [0, 1], [], False, False, 25),
            ([1, 2], [], [0, 1], [2, 3], False, False, -8),
            ([0, 1], [2], [2, 3], [0, 1], False, False, np.nan),
            ([], [1, 2], [], [0, 1], False, False, np.nan),
            ([1, 2], [0], [0, 1], [], False, True, np.nan),
            ([1, 2], [], [0, 1], [2, 3], True, False, np.nan),
        ),
    )
    def it_can_compute_a_subtotal_intersection_value(
        self,
        request,
        row_add_idxs,
        row_sub_idxs,
        col_add_idxs,
        col_sub_idxs,
        diff_cols_nan,
        diff_rows_nan,
        expected_value,
    ):
        col_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=col_add_idxs,
            subtrahend_idxs=col_sub_idxs,
        )
        row_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=row_add_idxs,
            subtrahend_idxs=row_sub_idxs,
        )
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None, diff_cols_nan, diff_rows_nan)

        np.testing.assert_equal(
            subtotals._intersection(row_subtotal_, col_subtotal_), expected_value
        )

    @pytest.mark.parametrize(
        ("addend_idxs", "subtrahend_idxs", "diff_cols_nan", "expected_value"),
        (
            ([1, 2], [], False, [3, 11, 19]),
            ([1, 3], [], False, [4, 12, 20]),
            ([0, 3], [], False, [3, 11, 19]),
            ([], [1, 2], False, [-3, -11, -19]),
            ([1], [3], False, [-2, -2, -2]),
            ([1], [3], True, [np.nan, np.nan, np.nan]),
        ),
    )
    def it_can_compute_a_subtotal_column_to_help(
        self, subtotal_, addend_idxs, subtrahend_idxs, diff_cols_nan, expected_value
    ):
        subtotal_.addend_idxs = addend_idxs
        subtotal_.subtrahend_idxs = subtrahend_idxs
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None, diff_cols_nan=diff_cols_nan)

        assert subtotals._subtotal_column(subtotal_).tolist() == pytest.approx(
            expected_value, nan_ok=True
        )

    @pytest.mark.parametrize(
        ("addend_idxs", "subtrahend_idxs", "diff_rows_nan", "expected_value"),
        (
            ([1, 2], [], False, [12, 14, 16, 18]),
            ([0, 1], [], False, [4, 6, 8, 10]),
            ([0, 2], [], False, [8, 10, 12, 14]),
            ([], [1, 2], False, [-12, -14, -16, -18]),
            ([0], [2], False, [-8, -8, -8, -8]),
            ([0], [2], True, [np.nan, np.nan, np.nan, np.nan]),
        ),
    )
    def it_can_compute_a_subtotal_row_to_help(
        self, subtotal_, addend_idxs, subtrahend_idxs, diff_rows_nan, expected_value
    ):
        subtotal_.addend_idxs = addend_idxs
        subtotal_.subtrahend_idxs = subtrahend_idxs
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None, diff_rows_nan=diff_rows_nan)

        assert subtotals._subtotal_row(subtotal_).tolist() == pytest.approx(
            expected_value, nan_ok=True
        )

    # --- fixture components -----------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def _init_(self, request):
        return initializer_mock(request, SumSubtotals)

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)


class DescribeTableStdErrSubtotals(object):
    """Unit test suite for `cr.cube.matrix.TableStdErrSubtotals` object."""

    def it_provides_access_to_the_base_counts_to_help(self, cube_result_matrix_):
        cube_result_matrix_.weighted_counts = [[1, 2], [3, 4]]
        subtotals = TableStdErrSubtotals(None, None, cube_result_matrix_)
        assert subtotals._base_counts == [[1, 2], [3, 4]]

    def it_can_compute_a_subtotal_column_to_help(
        self,
        subtotal_,
        _base_counts_prop_,
        _table_margin_prop_,
    ):
        subtotal_.addend_idxs = np.array([0, 1])
        subtotal_.subtrahend_idxs = np.array([])
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        _table_margin_prop_.return_value = 67
        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_almost_equal(
            subtotals._subtotal_column(subtotal_),
            np.array([0.0148136, 0.0416604, 0.0531615]),
        )

    def it_handles_subtrahends_when_computing_subtotal_column(
        self,
        request,
        subtotal_,
    ):
        subtotal_.subtrahend_idxs = np.array([1])
        property_mock(request, TableStdErrSubtotals, "_nrows", return_value=3)

        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_equal(
            subtotals._subtotal_column(subtotal_),
            np.array([np.nan, np.nan, np.nan]),
        )

    def it_can_compute_a_subtotal_row_to_help(
        self,
        subtotal_,
        _base_counts_prop_,
        _table_margin_prop_,
    ):
        subtotal_.addend_idxs = np.array([0, 1])
        subtotal_.subtrahend_idxs = np.array([])
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        _table_margin_prop_.return_value = 67
        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_almost_equal(
            subtotals._subtotal_row(subtotal_),
            np.array([0.0289460, 0.0348842, 0.0396149, 0.0435337]),
        )

    def it_handles_subtrahends_when_computing_subtotal_row(
        self,
        request,
        subtotal_,
    ):
        subtotal_.subtrahend_idxs = np.array([1])
        property_mock(request, TableStdErrSubtotals, "_ncols", return_value=4)

        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_equal(
            subtotals._subtotal_row(subtotal_),
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )

    def it_can_compute_a_subtotal_intersection_to_help(
        self,
        request,
        _base_counts_prop_,
        _table_margin_prop_,
    ):
        row_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array([]),
        )
        col_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([1, 2]),
            subtrahend_idxs=np.array([]),
        )
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        _table_margin_prop_.return_value = 67
        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_almost_equal(
            subtotals._intersection(row_subtotal_, col_subtotal_), 0.0496694
        )

    @pytest.mark.parametrize(
        ("row_subtrahend", "col_subtrahend"),
        (
            ([], [0]),
            ([2], []),
            ([2], [0]),
        ),
    )
    def it_handles_subtrahends_when_computing_subtotal_intersection(
        self, request, row_subtrahend, col_subtrahend
    ):
        row_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array(row_subtrahend),
        )
        col_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([1, 2]),
            subtrahend_idxs=np.array(col_subtrahend),
        )

        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_equal(
            subtotals._intersection(row_subtotal_, col_subtotal_), np.nan
        )

    def it_provides_access_to_the_table_margin_to_help(self, cube_result_matrix_):
        cube_result_matrix_.table_margin = 42
        subtotals = TableStdErrSubtotals(None, None, cube_result_matrix_)
        assert subtotals._table_margin == 42

    # --- fixture components -----------------------------------------

    @pytest.fixture
    def _base_counts_prop_(self, request):
        return property_mock(request, TableStdErrSubtotals, "_base_counts")

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, BaseCubeResultMatrix)

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)

    @pytest.fixture
    def _table_margin_prop_(self, request):
        return property_mock(request, TableStdErrSubtotals, "_table_margin")


class DescribeZscoreSubtotals(object):
    """Unit test suite for `cr.cube.matrix.ZscoreSubtotals` object."""

    def it_provides_a_blocks_interface_method(self, request, _init_, weighted_counts_):
        weighted_counts_.weighted_counts = [[1, 2], [2, 3]]
        weighted_counts_.rows_margin = None
        weighted_counts_.columns_margin = [3, 4]
        weighted_counts_.table_margin = 12
        property_mock(
            request,
            ZscoreSubtotals,
            "_blocks",
            return_value=np.array([[[1], [2]], [[3], [4]]]),
        )

        blocks = ZscoreSubtotals.blocks(weighted_counts_, None)

        _init_.assert_called_once_with(ANY, weighted_counts_, None, True, True)
        assert blocks.tolist() == [[[1], [2]], [[3], [4]]]

    def it_provides_an_intersections_interface_method(
        self, request, _init_, weighted_counts_
    ):
        weighted_counts_.weighted_counts = [[1, 2], [2, 3]]
        weighted_counts_.rows_margin = None
        weighted_counts_.columns_margin = [3, 4]
        weighted_counts_.table_margin = 12
        property_mock(
            request,
            ZscoreSubtotals,
            "_intersections",
            return_value=np.array([[1, 2], [3, 4]]),
        )

        intersections = ZscoreSubtotals.intersections(weighted_counts_, None)

        _init_.assert_called_once_with(ANY, weighted_counts_, None, True, True)
        assert intersections.tolist() == [[1, 2], [3, 4]]

    def it_can_compute_a_subtotal_intersection_to_help(self, request, weighted_counts_):
        row_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array([]),
        )
        col_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array([]),
        )
        weighted_counts_.weighted_counts = np.arange(12).reshape(3, 4)
        weighted_counts_.columns_margin = np.array([12, 15, 18, 21])
        weighted_counts_.rows_margin = np.array([10, 2, 34])
        weighted_counts_.table_margin = 66
        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._intersection(row_subtotal_, col_subtotal_) == pytest.approx(
            -0.7368146
        )

    @pytest.mark.parametrize(
        ("row_subtrahends", "col_subtrahends"),
        (
            ([], [2]),
            ([2], []),
            ([2], [2]),
        ),
    )
    def it_can_compute_a_subtotal_intersection_with_subtrahends(
        self, request, row_subtrahends, col_subtrahends, weighted_counts_
    ):
        row_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array(row_subtrahends),
        )
        col_subtotal_ = instance_mock(
            request,
            _Subtotal,
            addend_idxs=np.array([0, 1]),
            subtrahend_idxs=np.array(col_subtrahends),
        )

        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._intersection(row_subtotal_, col_subtotal_) == pytest.approx(
            np.nan, nan_ok=True
        )

    def it_can_compute_a_subtotal_column_to_help(self, subtotal_, weighted_counts_):
        subtotal_.addend_idxs = np.array([0, 1])
        subtotal_.subtrahend_idxs = np.array([])
        weighted_counts_.weighted_counts = np.arange(12).reshape(3, 4)
        weighted_counts_.rows_margin = np.array([6, 22, 38])
        weighted_counts_.table_margin = 66
        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._subtotal_column(subtotal_) == pytest.approx(
            [-1.2667117, 0.0, 0.7368146]
        )

    def it_can_compute_a_subtotal_column_with_subtrahends(
        self, request, subtotal_, weighted_counts_
    ):
        subtotal_.subtrahend_idxs = np.array([0])
        property_mock(request, ZscoreSubtotals, "_nrows", return_value=3)
        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._subtotal_column(subtotal_) == pytest.approx(
            [np.nan, np.nan, np.nan], nan_ok=True
        )

    def it_can_compute_a_subtotal_row_to_help(self, subtotal_, weighted_counts_):
        subtotal_.addend_idxs = np.array([0, 1])
        subtotal_.subtrahend_idxs = np.array([])
        weighted_counts_.weighted_counts = np.arange(12).reshape(3, 4)
        weighted_counts_.columns_margin = np.array([12, 15, 18, 21])
        weighted_counts_.table_margin = 66
        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._subtotal_row(subtotal_) == pytest.approx(
            [-0.7044435, -0.2161134, 0.2033553, 0.5833346]
        )

    def it_can_compute_a_subtotal_row_with_subtrahends(
        self, request, subtotal_, weighted_counts_
    ):
        subtotal_.subtrahend_idxs = np.array([0])
        property_mock(request, ZscoreSubtotals, "_ncols", return_value=4)
        subtotals = ZscoreSubtotals(weighted_counts_, None)

        assert subtotals._subtotal_row(subtotal_) == pytest.approx(
            [np.nan, np.nan, np.nan, np.nan], nan_ok=True
        )

    # --- fixture components -----------------------------------------

    @pytest.fixture
    def weighted_counts_(self, request):
        return instance_mock(request, _BaseWeightedCubeCounts)

    @pytest.fixture
    def _init_(self, request):
        return initializer_mock(request, ZscoreSubtotals)

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)
