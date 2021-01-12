# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.subtotals` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.matrix.cubemeasure import BaseCubeResultMatrix
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
        ("ncols", "row_subtotals", "dtype", "shape", "expected_value"),
        (
            (3, [], np.float64, (0, 3), []),
            (3, [1, 2], np.int64, (2, 3), [[1, 2, 3], [1, 2, 3]]),
        ),
    )
    def it_assembles_its_subtotal_rows_to_help(
        self,
        _row_subtotals_prop_,
        _ncols_prop_,
        _subtotal_row_,
        ncols,
        row_subtotals,
        dtype,
        shape,
        expected_value,
    ):
        _row_subtotals_prop_.return_value = row_subtotals
        _subtotal_row_.return_value = np.array([1, 2, 3])
        _ncols_prop_.return_value = ncols

        subtotal_rows = _BaseSubtotals(None, None)._subtotal_rows

        assert subtotal_rows.dtype == dtype
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

        _init_.assert_called_once_with(ANY, [[1, 5], [8, 0]], dimensions_)
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

        _init_.assert_called_once_with(ANY, [[0, 4], [7, 9]], dimensions_)
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

        _init_.assert_called_once_with(ANY, [[4, 1], [3, 5]], dimensions_)
        assert subtotal_rows.tolist() == [[4, 3], [2, 1]]

    @pytest.mark.parametrize(
        ("row_idxs", "col_idxs", "expected_value"),
        (
            ([1, 2], [0, 1], 26),
            ([0, 1], [0, 1], 10),
            ([1, 2], [2, 3], 34),
        ),
    )
    def it_can_compute_a_subtotal_intersection_value(
        self, request, row_idxs, col_idxs, expected_value
    ):
        col_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=col_idxs)
        row_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=row_idxs)
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None)

        assert subtotals._intersection(row_subtotal_, col_subtotal_) == expected_value

    @pytest.mark.parametrize(
        ("addend_idxs", "expected_value"),
        (([1, 2], [3, 11, 19]), ([1, 3], [4, 12, 20]), ([0, 3], [3, 11, 19])),
    )
    def it_can_compute_a_subtotal_column_to_help(
        self, subtotal_, addend_idxs, expected_value
    ):
        subtotal_.addend_idxs = addend_idxs
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None)

        assert subtotals._subtotal_column(subtotal_).tolist() == expected_value

    @pytest.mark.parametrize(
        ("addend_idxs", "expected_value"),
        (
            ([1, 2], [12, 14, 16, 18]),
            ([0, 1], [4, 6, 8, 10]),
            ([0, 2], [8, 10, 12, 14]),
        ),
    )
    def it_can_compute_a_subtotal_row_to_help(
        self, subtotal_, addend_idxs, expected_value
    ):
        subtotal_.addend_idxs = addend_idxs
        base_values = np.arange(12).reshape(3, 4)
        subtotals = SumSubtotals(base_values, None)

        assert subtotals._subtotal_row(subtotal_).tolist() == expected_value

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
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        _table_margin_prop_.return_value = 67
        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_almost_equal(
            subtotals._subtotal_column(subtotal_),
            np.array([0.0148136, 0.0416604, 0.0531615]),
        )

    def it_can_compute_a_subtotal_row_to_help(
        self,
        request,
        _base_counts_prop_,
        _table_margin_prop_,
    ):
        row_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=np.array([0, 1]))
        col_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=np.array([1, 2]))
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        _table_margin_prop_.return_value = 67
        subtotals = TableStdErrSubtotals(None, None, None)

        np.testing.assert_almost_equal(
            subtotals._intersection(row_subtotal_, col_subtotal_), 0.0496694
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

    def it_provides_access_to_the_base_counts_to_help(self, cube_result_matrix_):
        cube_result_matrix_.weighted_counts = [[1, 2], [3, 4]]
        subtotals = ZscoreSubtotals(None, None, cube_result_matrix_)
        assert subtotals._base_counts == [[1, 2], [3, 4]]

    def it_can_compute_a_subtotal_intersection_to_help(
        self,
        request,
        _base_counts_prop_,
        cube_result_matrix_,
        _table_margin_prop_,
    ):
        row_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=np.array([0, 1]))
        col_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=np.array([0, 1]))
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        cube_result_matrix_.columns_margin = np.array([12, 15, 18, 21])
        _table_margin_prop_.return_value = 66
        subtotals = ZscoreSubtotals(None, None, cube_result_matrix_)

        np.testing.assert_almost_equal(
            subtotals._intersection(row_subtotal_, col_subtotal_),
            -0.7368146,
        )

    def it_can_compute_a_subtotal_column_to_help(
        self,
        subtotal_,
        _base_counts_prop_,
        _table_margin_prop_,
        cube_result_matrix_,
    ):
        subtotal_.addend_idxs = np.array([0, 1])
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        cube_result_matrix_.rows_margin = np.array([6, 22, 38])
        _table_margin_prop_.return_value = 66
        subtotals = ZscoreSubtotals(None, None, cube_result_matrix_)

        np.testing.assert_almost_equal(
            subtotals._subtotal_column(subtotal_),
            np.array([-1.2667117, 0.0, 0.7368146]),
        )

    def it_can_compute_a_subtotal_row_to_help(
        self,
        subtotal_,
        _base_counts_prop_,
        _table_margin_prop_,
        cube_result_matrix_,
    ):
        subtotal_.addend_idxs = np.array([0, 1])
        _base_counts_prop_.return_value = np.arange(12).reshape(3, 4)
        cube_result_matrix_.columns_margin = np.array([12, 15, 18, 21])
        _table_margin_prop_.return_value = 66
        subtotals = ZscoreSubtotals(None, None, cube_result_matrix_)

        np.testing.assert_almost_equal(
            subtotals._subtotal_row(subtotal_),
            np.array([-0.7044435, -0.2161134, 0.2033553, 0.5833346]),
        )

    def it_provides_access_to_the_table_margin_to_help(self, cube_result_matrix_):
        cube_result_matrix_.table_margin = 42
        subtotals = ZscoreSubtotals(None, None, cube_result_matrix_)
        assert subtotals._table_margin == 42

    # --- fixture components -----------------------------------------

    @pytest.fixture
    def _base_counts_prop_(self, request):
        return property_mock(request, ZscoreSubtotals, "_base_counts")

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, BaseCubeResultMatrix)

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)

    @pytest.fixture
    def _table_margin_prop_(self, request):
        return property_mock(request, ZscoreSubtotals, "_table_margin")
