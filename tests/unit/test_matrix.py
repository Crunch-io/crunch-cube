# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Subtotal, _Subtotals
from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT
from cr.cube.matrix import (
    Assembler,
    _BaseCubeResultMatrix,
    _BaseSubtotals,
    _CatXCatMatrix,
    _CatXCatMeansMatrix,
    _CatXMrMatrix,
    _MrXCatMatrix,
    _MrXMrMatrix,
    _SumSubtotals,
)

from ..unitutil import (
    class_mock,
    instance_mock,
    method_mock,
    property_mock,
    initializer_mock,
)


class DescribeAssembler(object):
    """Unit test suite for `cr.cube.matrix.Assembler` object."""

    def it_knows_the_columns_base(
        self,
        _rows_dimension_prop_,
        dimension_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _column_subtotals_prop_,
        _column_order_prop_,
        _assemble_vector_,
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.CAT
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.columns_base = [1, 2, 3]
        _column_subtotals_prop_.return_value = [3, 5]
        _column_order_prop_.return_value = [0, -2, 1, 2, -1]
        _assemble_vector_.return_value = np.array([1, 3, 2, 3, 5])
        assembler = Assembler(None, None, None)

        columns_base = assembler.columns_base

        _assemble_vector_.assert_called_once_with(
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1]
        )
        np.testing.assert_equal(columns_base, np.array([1, 3, 2, 3, 5]))

    def but_it_provides_a_2D_columns_base_for_an_MR_X_cube_result(
        self,
        _rows_dimension_prop_,
        dimension_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _SumSubtotals_,
        _assemble_matrix_,
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.MR_SUBVAR
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        _SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        columns_base = assembler.columns_base

        _SumSubtotals_.blocks.assert_called_once_with(
            cube_result_matrix_, "columns_base"
        )
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert columns_base == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_2D_table_margin_of_an_MR_X_MR_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _row_order_prop_,
        _column_order_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_margin = np.array([[1, 2, 3], [4, 5, 6]])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _row_order_prop_.return_value = np.array([1, 0])
        _column_order_prop_.return_value = np.array([1, 0, 2])
        assembler = Assembler(None, None, None)

        np.testing.assert_equal(assembler.table_margin, [[5, 4, 6], [2, 1, 3]])

    def and_it_knows_the_1D_table_margin_of_an_MR_X_CAT_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _row_order_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_margin = np.array([1, 2, 3])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _row_order_prop_.return_value = np.array([1, 0, 2])
        assembler = Assembler(None, None, None)

        np.testing.assert_equal(assembler.table_margin, [2, 1, 3])

    def and_it_knows_the_1D_table_margin_of_a_CAT_X_MR_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _column_order_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_margin = np.array([1, 2, 3])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _column_order_prop_.return_value = np.array([2, 0, 1])
        assembler = Assembler(None, None, None)

        np.testing.assert_equal(assembler.table_margin, [3, 1, 2])

    def and_it_knows_the_scalar_table_margin_of_a_CAT_X_CAT_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_margin = 4242
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        assembler = Assembler(None, None, None)

        assert assembler.table_margin == 4242

    def it_knows_the_unweighted_counts(
        self,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _SumSubtotals_,
        _assemble_matrix_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        _SumSubtotals_.blocks.return_value = [["A", "B"], ["C", "D"]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        unweighted_counts = assembler.unweighted_counts

        _SumSubtotals_.blocks.assert_called_once_with(
            cube_result_matrix_, "unweighted_counts"
        )
        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert unweighted_counts == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_weighted_counts(
        self,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _SumSubtotals_,
        _assemble_matrix_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        _SumSubtotals_.blocks.return_value = [["A", "B"], ["C", "D"]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        weighted_counts = assembler.weighted_counts

        _SumSubtotals_.blocks.assert_called_once_with(
            cube_result_matrix_, "weighted_counts"
        )
        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert weighted_counts == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @pytest.mark.parametrize(
        "row_order, col_order, blocks, expected_value",
        (
            ([0, 1], [0, 1, 2], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
            ([0, 1], [2, 1, 0], [[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]]),
            ([0, 1], [1, 0], [[1, 2, 3], [4, 5, 6]], [[2, 1], [5, 4]]),
            ([0, 1], [2, 0], [[1, 2, 3], [4, 5, 6]], [[3, 1], [6, 4]]),
            ([1, 0], [0, 1, 2], [[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3]]),
            ([0], [0, 1, 2], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3]]),
            ([0], [0], [[1, 2, 3], [4, 5, 6]], [[1]]),
        ),
    )
    def it_can_assemble_a_matrix_from_blocks(
        self,
        _row_order_prop_,
        row_order,
        _column_order_prop_,
        col_order,
        blocks,
        expected_value,
    ):
        _row_order_prop_.return_value = row_order
        _column_order_prop_.return_value = col_order
        assembler = Assembler(None, None, None)

        assert assembler._assemble_matrix(blocks).tolist() == expected_value

    def it_can_assemble_a_vector(self, request):
        base_vector = np.array([1, 2, 3, 4])
        subtotals_ = tuple(
            instance_mock(request, _Subtotal, addend_idxs=np.array(addend_idxs))
            for addend_idxs in ((0, 1), (1, 2), (2, 3))
        )
        order = np.array([-3, 1, 0, -2, 3, 2, -1])
        assembler = Assembler(None, None, None)

        np.testing.assert_equal(
            assembler._assemble_vector(base_vector, subtotals_, order),
            np.array([3, 2, 1, 5, 4, 3, 7]),
        )

    @pytest.mark.parametrize(
        "order, prune_subtotal_columns, expected_value",
        (
            (np.array([-1, 1, -2, 2, -3, 3]), False, [-1, 1, -2, 2, -3, 3]),
            (np.array([-1, 1, -2, 2, -3, 3]), True, [1, 2, 3]),
            (np.array([], dtype=int), True, []),
        ),
    )
    def it_knows_the_column_order_to_help(
        self,
        request,
        _columns_dimension_prop_,
        dimension_,
        _empty_column_idxs_prop_,
        _dimension_order_,
        order,
        prune_subtotal_columns,
        expected_value,
    ):
        _columns_dimension_prop_.return_value = dimension_
        _empty_column_idxs_prop_.return_value = (4, 2)
        _dimension_order_.return_value = order
        property_mock(
            request,
            Assembler,
            "_prune_subtotal_columns",
            return_value=prune_subtotal_columns,
        )
        assembler = Assembler(None, None, None)

        column_order = assembler._column_order

        _dimension_order_.assert_called_once_with(assembler, dimension_, (4, 2))
        assert column_order.dtype == int
        np.testing.assert_equal(column_order, np.array(expected_value))

    def it_provides_access_to_the_column_subtotals_to_help(
        self, _columns_dimension_prop_, dimension_, subtotals_
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.subtotals = subtotals_

        assert Assembler(None, None, None)._column_subtotals is subtotals_

    def it_provides_access_to_the_columns_dimension_to_help(self, dimension_):
        assembler = Assembler(None, (None, dimension_), None)
        assert assembler._columns_dimension is dimension_

    def it_constructs_the_cube_result_matrix_to_help(
        self, request, cube_, dimension_, cube_result_matrix_
    ):
        _BaseCubeResultMatrix_ = class_mock(
            request, "cr.cube.matrix._BaseCubeResultMatrix"
        )
        _BaseCubeResultMatrix_.factory.return_value = cube_result_matrix_
        assembler = Assembler(cube_, (dimension_, dimension_), slice_idx=42)

        cube_result_matrix = assembler._cube_result_matrix

        _BaseCubeResultMatrix_.factory.assert_called_once_with(
            cube_, (dimension_, dimension_), 42
        )
        assert cube_result_matrix is cube_result_matrix_

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def it_computes_the_order_for_a_dimension_to_help(
        self, request, dimension_, collation_method, collator_class_name
    ):
        CollatorCls_ = class_mock(request, "cr.cube.matrix.%s" % collator_class_name)
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        dimension_.collation_method = collation_method
        assembler = Assembler(None, None, None)

        dimension_order = assembler._dimension_order(dimension_, empty_idxs=[2, 4, 6])

        CollatorCls_.display_order.assert_called_once_with(dimension_, [2, 4, 6])
        assert dimension_order.shape == (5,)
        np.testing.assert_equal(dimension_order, np.array([1, -2, 3, 5, -1]))

    @pytest.mark.parametrize(
        "base, expected_value",
        (((1, 1, 1), ()), ((1, 0, 1), (1,)), ((0, 0, 0), (0, 1, 2))),
    )
    def it_knows_its_empty_column_idxs_to_help(
        self, _cube_result_matrix_prop_, cube_result_matrix_, base, expected_value
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.columns_pruning_base = np.array(base)

        assert Assembler(None, None, None)._empty_column_idxs == expected_value

    @pytest.mark.parametrize(
        "base, expected_value",
        (((1, 1, 1), ()), ((1, 0, 1), (1,)), ((0, 0, 0), (0, 1, 2))),
    )
    def it_knows_its_empty_row_idxs_to_help(
        self, _cube_result_matrix_prop_, cube_result_matrix_, base, expected_value
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.rows_pruning_base = np.array(base)

        assert Assembler(None, None, None)._empty_row_idxs == expected_value

    @pytest.mark.parametrize(
        "prune, empty_row_idxs, element_ids, expected_value",
        (
            (False, (), (), False),
            (False, (1, 2, 3), (4, 5, 6), False),
            (False, (1, 2), (7, 8, 9), False),
            (True, (), (), True),
            (True, (1, 2, 3), (4, 5, 6), True),
            (True, (1, 2), (7, 8, 9), False),
        ),
    )
    def it_knows_whether_its_subtotal_columns_should_be_pruned_to_help(
        self,
        _rows_dimension_prop_,
        dimension_,
        prune,
        _empty_row_idxs_prop_,
        empty_row_idxs,
        element_ids,
        expected_value,
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.element_ids = element_ids
        dimension_.prune = prune
        _empty_row_idxs_prop_.return_value = empty_row_idxs

        assert Assembler(None, None, None)._prune_subtotal_columns is expected_value

    @pytest.mark.parametrize(
        "prune, empty_col_idxs, element_ids, expected_value",
        (
            (False, (), (), False),
            (False, (1, 2, 3), (4, 5, 6), False),
            (False, (1, 2), (7, 8, 9), False),
            (True, (), (), True),
            (True, (1, 2, 3), (4, 5, 6), True),
            (True, (1, 2), (7, 8, 9), False),
        ),
    )
    def it_knows_whether_its_subtotal_rows_should_be_pruned_to_help(
        self,
        _columns_dimension_prop_,
        dimension_,
        prune,
        _empty_column_idxs_prop_,
        empty_col_idxs,
        element_ids,
        expected_value,
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.element_ids = element_ids
        dimension_.prune = prune
        _empty_column_idxs_prop_.return_value = empty_col_idxs

        assert Assembler(None, None, None)._prune_subtotal_rows is expected_value

    @pytest.mark.parametrize(
        "order, prune, expected",
        (
            # --- False -> not pruned ---
            ([0, 1], False, [0, 1]),
            # --- True, but no negative indices -> not pruned ---
            ([0, 1], True, [0, 1]),
            # --- False -> not pruned ---
            ([0, -1, 1, -2], False, [0, -1, 1, -2]),
            # --- True, with negative indices -> pruned ---
            ([0, -1, 1, -2], True, [0, 1]),
        ),
    )
    def it_knows_its_row_order_to_help(
        self,
        request,
        _dimension_order_,
        dimension_,
        order,
        prune,
        expected,
    ):
        # --- Prepare mocks and return values ---
        property_mock(request, Assembler, "_rows_dimension", return_value=dimension_)
        fake_row_idxs = [0, 1, 2]
        property_mock(request, Assembler, "_empty_row_idxs", return_value=fake_row_idxs)
        _dimension_order_.return_value = np.array(order)
        property_mock(request, Assembler, "_prune_subtotal_rows", return_value=prune)
        assembler = Assembler(None, None, None)

        # --- Call the tested property ---
        row_order = assembler._row_order

        # --- Perform assertions
        assert row_order.tolist() == expected
        _dimension_order_.assert_called_once_with(assembler, dimension_, fake_row_idxs)

    def it_knows_its_rows_dimension_to_help(self, dimension_):
        assembler = Assembler(None, (dimension_, None), None)
        assert assembler._rows_dimension is dimension_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assemble_matrix_(self, request):
        return method_mock(request, Assembler, "_assemble_matrix")

    @pytest.fixture
    def _assemble_vector_(self, request):
        return method_mock(request, Assembler, "_assemble_vector")

    @pytest.fixture
    def _column_order_prop_(self, request):
        return property_mock(request, Assembler, "_column_order")

    @pytest.fixture
    def _column_subtotals_prop_(self, request):
        return property_mock(request, Assembler, "_column_subtotals")

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(request, Assembler, "_columns_dimension")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, _BaseCubeResultMatrix)

    @pytest.fixture
    def _cube_result_matrix_prop_(self, request):
        return property_mock(request, Assembler, "_cube_result_matrix")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _dimension_order_(self, request):
        return method_mock(request, Assembler, "_dimension_order")

    @pytest.fixture
    def _empty_column_idxs_prop_(self, request):
        return property_mock(request, Assembler, "_empty_column_idxs")

    @pytest.fixture
    def _empty_row_idxs_prop_(self, request):
        return property_mock(request, Assembler, "_empty_row_idxs")

    @pytest.fixture
    def _row_order_prop_(self, request):
        return property_mock(request, Assembler, "_row_order")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, Assembler, "_rows_dimension")

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def _SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix._SumSubtotals")


# === SUBTOTALS OBJECTS ===


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

    def it_knows_how_to_assemble_its_blocks(self, request):
        property_mock(request, _BaseSubtotals, "_base_values")
        property_mock(request, _BaseSubtotals, "_subtotal_columns")
        property_mock(request, _BaseSubtotals, "_subtotal_rows")
        property_mock(request, _BaseSubtotals, "_intersections")
        base_subtotals = _BaseSubtotals(None, None)

        blocks = base_subtotals._blocks

        assert blocks == [
            [base_subtotals._base_values, base_subtotals._subtotal_columns],
            [base_subtotals._subtotal_rows, base_subtotals._intersections],
        ]

    def it_provides_access_to_the_column_subtotals(
        self, cube_result_matrix_, dimension_
    ):
        cube_result_matrix_.columns_dimension = dimension_
        subtotals = _BaseSubtotals(cube_result_matrix_, None)
        assert subtotals._column_subtotals is dimension_.subtotals

    def it_knows_how_many_columns_are_in_the_base_matrix(self, _base_values_prop_):
        _base_values_prop_.return_value = np.arange(12).reshape(3, 4)
        assert _BaseSubtotals(None, None)._ncols == 4

    def it_knows_how_many_rows_are_in_the_base_matrix(self, _base_values_prop_):
        _base_values_prop_.return_value = np.arange(12).reshape(3, 4)
        assert _BaseSubtotals(None, None)._nrows == 3

    def it_provides_access_to_the_row_subtotals(self, cube_result_matrix_, dimension_):
        cube_result_matrix_.rows_dimension = dimension_
        subtotals = _BaseSubtotals(cube_result_matrix_, None)
        assert subtotals._row_subtotals is dimension_.subtotals

    @pytest.mark.parametrize(
        ("nrows", "n_subtotals", "expected_value"),
        (
            (3, 0, np.empty((3, 0), dtype=np.float64)),
            (3, 2, np.array([[1, 1], [2, 2], [3, 3]])),
        ),
    )
    def it_knows_its_subtotal_columns(
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
        ("ncols", "row_subtotals", "expected_value"),
        (
            (3, [], np.empty((0, 3), dtype=np.float64)),
            (3, [1, 2], np.array([[1, 2, 3], [1, 2, 3]])),
        ),
    )
    def it_knows_its_subtotal_rows(
        self,
        _row_subtotals_prop_,
        _ncols_prop_,
        _subtotal_row_,
        ncols,
        row_subtotals,
        expected_value,
    ):
        _row_subtotals_prop_.return_value = row_subtotals
        _subtotal_row_.return_value = np.array([1, 2, 3])
        _ncols_prop_.return_value = ncols

        np.testing.assert_equal(
            _BaseSubtotals(None, None)._subtotal_rows, expected_value
        )

    def it_knows_its_intersections(
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

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_base_values")

    @pytest.fixture
    def _column_subtotals_prop_(self, request):
        return property_mock(request, _BaseSubtotals, "_column_subtotals")

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, _BaseCubeResultMatrix)

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


class Describe_SumSubtotals(object):
    """Unit test suite for `cr.cube.matrix._SubSubtotals` object."""

    @pytest.mark.parametrize(
        ("prop_name",),
        (
            ("columns_base",),
            ("column_proportions",),
            ("columns_pruning_base",),
            ("rows_pruning_base",),
            ("table_margin",),
        ),
    )
    def it_returns_correct_base_values(self, cube_result_matrix_, prop_name):
        subtotals = _SumSubtotals(cube_result_matrix_, prop_name)
        assert subtotals._base_values == getattr(cube_result_matrix_, prop_name)

    @pytest.mark.parametrize(
        ("row_idxs", "col_idxs", "expected_value"),
        (
            ([1, 2], [0, 1], 26),
            ([0, 1], [0, 1], 10),
            ([1, 2], [2, 3], 34),
        ),
    )
    def it_can_compute_a_subtotal_intersection_value(
        self, request, _base_values_prop_, row_idxs, col_idxs, expected_value
    ):
        _base_values_prop_.return_value = np.arange(12).reshape(3, 4)
        col_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=col_idxs)
        row_subtotal_ = instance_mock(request, _Subtotal, addend_idxs=row_idxs)
        subtotals = _SumSubtotals(None, None)

        assert subtotals._intersection(row_subtotal_, col_subtotal_) == expected_value

    @pytest.mark.parametrize(
        ("addend_idxs", "expected_value"),
        (([1, 2], [3, 11, 19]), ([1, 3], [4, 12, 20]), ([0, 3], [3, 11, 19])),
    )
    def it_can_compute_the_subtotal_column_for_a_given_column_subtotal(
        self, _base_values_prop_, subtotal_, addend_idxs, expected_value
    ):
        _base_values_prop_.return_value = np.arange(12).reshape(3, 4)
        subtotal_.addend_idxs = addend_idxs
        subtotals = _SumSubtotals(None, None)

        assert subtotals._subtotal_column(subtotal_).tolist() == expected_value

    @pytest.mark.parametrize(
        ("addend_idxs", "expected_value"),
        (
            ([1, 2], [12, 14, 16, 18]),
            ([0, 1], [4, 6, 8, 10]),
            ([0, 2], [8, 10, 12, 14]),
        ),
    )
    def it_can_compute_the_subtotal_row_for_a_given_row_subtotal(
        self, _base_values_prop_, subtotal_, addend_idxs, expected_value
    ):
        _base_values_prop_.return_value = np.arange(12).reshape(3, 4)
        subtotal_.addend_idxs = addend_idxs
        subtotals = _SumSubtotals(None, None)

        assert subtotals._subtotal_row(subtotal_).tolist() == expected_value

    # --- fixture components -----------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _SumSubtotals, "_base_values")

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, _BaseCubeResultMatrix)

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)


# === CUBE-RESULT MATRIX OBJECTS ===


class Describe_BaseCubeResultMatrix(object):
    """Unit test suite for `cr.cube.matrix._BaseCubeResultMatrix` object."""

    @pytest.mark.parametrize(
        "has_means, factory_method_name",
        ((True, "_means_matrix_factory"), (False, "_regular_matrix_factory")),
    )
    def it_calls_the_correct_factory_method_for_appropriate_matrix_type(
        self, request, cube_, dimension_, has_means, factory_method_name
    ):
        cube_.has_means = has_means
        cube_result_matrix_ = instance_mock(request, _BaseCubeResultMatrix)
        factory_method = method_mock(
            request,
            _BaseCubeResultMatrix,
            factory_method_name,
            return_value=cube_result_matrix_,
        )

        cube_result_matrix = _BaseCubeResultMatrix.factory(
            cube_, (dimension_, dimension_), slice_idx=71
        )

        factory_method.assert_called_once_with(cube_, (dimension_, dimension_), 71)
        assert cube_result_matrix is cube_result_matrix_

    def it_knows_its_columns_dimension(self, dimension_):
        matrix = _BaseCubeResultMatrix([None, dimension_], None, None)
        assert matrix.columns_dimension == dimension_

    def it_knows_its_rows_dimension(self, dimension_):
        matrix = _BaseCubeResultMatrix([dimension_, None], None, None)
        assert matrix.rows_dimension == dimension_

    @pytest.mark.parametrize(
        "dimension_types, matrix_class_name",
        (
            ((DT.MR, DT.CAT), "_MrXCatMeansMatrix"),
            ((DT.CAT, DT.MR), "_CatXMrMeansMatrix"),
            ((DT.CAT, DT.CAT), "_CatXCatMeansMatrix"),
        ),
    )
    def it_can_construct_a_means_matrix_for_a_2D_slice_to_help(
        self, request, cube_, dimension_types, dimension_, matrix_class_name
    ):
        cube_.dimension_types = dimension_types
        cube_.ndim = 2
        cube_.counts = [1, 2, 3, 4]
        cube_.unweighted_counts = [5, 6, 7, 8]
        MatrixCls_ = class_mock(request, "cr.cube.matrix.%s" % matrix_class_name)

        matrix = _BaseCubeResultMatrix._means_matrix_factory(
            cube_, (dimension_, dimension_), None
        )

        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [1, 2, 3, 4], [5, 6, 7, 8]
        )
        assert matrix is MatrixCls_.return_value

    @pytest.mark.parametrize(
        "dimension_types, matrix_class_name",
        (
            ((None, DT.MR, DT.CAT), "_MrXCatMeansMatrix"),
            ((None, DT.CAT, DT.MR), "_CatXMrMeansMatrix"),
            ((None, DT.CAT, DT.CAT), "_CatXCatMeansMatrix"),
        ),
    )
    def and_it_can_construct_a_means_matrix_for_a_3D_slice_to_help(
        self, request, cube_, dimension_types, dimension_, matrix_class_name
    ):
        cube_.dimension_types = dimension_types
        cube_.ndim = 3
        cube_.counts = [None, [1, 2, 3, 4], None]
        cube_.unweighted_counts = [None, [5, 6, 7, 8], None]
        MatrixCls_ = class_mock(request, "cr.cube.matrix.%s" % matrix_class_name)

        matrix = _BaseCubeResultMatrix._means_matrix_factory(
            cube_, (dimension_, dimension_), slice_idx=1
        )

        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [1, 2, 3, 4], [5, 6, 7, 8]
        )
        assert matrix is MatrixCls_.return_value

    def but_it_raises_on_MEANS_MR_X_MR(self, cube_):
        cube_.dimension_types = (DT.MR, DT.MR)

        with pytest.raises(NotImplementedError) as e:
            _BaseCubeResultMatrix._means_matrix_factory(cube_, None, None)

        assert str(e.value) == "MR x MR with means is not implemented"

    @pytest.mark.parametrize(
        "dimension_types, expected_value",
        (
            ((DT.MR, DT.MR), _MrXMrMatrix),
            ((DT.MR, DT.CAT), _MrXCatMatrix),
            ((DT.CAT, DT.MR), _CatXMrMatrix),
            ((DT.CAT, DT.CAT), _CatXCatMatrix),
        ),
    )
    def it_knows_its_regular_matrix_class_to_help(
        self, dimension_types, expected_value
    ):
        assert (
            _BaseCubeResultMatrix._regular_matrix_class(dimension_types)
            is expected_value
        )

    @pytest.mark.parametrize(
        ("slice_idx", "dim_types", "expected"),
        (
            # --- <= 2D ---
            (0, (DT.CAT,), np.s_[:]),
            (1, (DT.CAT,), np.s_[:]),
            (0, (DT.CAT, DT.CAT), np.s_[:]),
            (1, (DT.CAT, DT.CAT), np.s_[:]),
            # --- 3D, no MR as tabs ---
            (0, (DT.CAT, DT.CAT, DT.CAT), np.s_[0]),
            (1, (DT.CAT, DT.CAT, DT.CAT), np.s_[1]),
            (2, (DT.CAT, DT.CAT, DT.CAT), np.s_[2]),
            # --- 3D, MR as tabs ---
            (0, (DT.MR, DT.CAT, DT.CAT), np.s_[0, 0]),
            (1, (DT.MR, DT.CAT, DT.CAT), np.s_[1, 0]),
            (2, (DT.MR, DT.CAT, DT.CAT), np.s_[2, 0]),
        ),
    )
    def it_knows_its_regular_matrix_counts_slice_to_help(
        self, cube_, slice_idx, dim_types, expected
    ):
        cube_.dimension_types = dim_types
        cube_.ndim = len(dim_types)

        s = _BaseCubeResultMatrix._regular_matrix_counts_slice(cube_, slice_idx)

        assert s == expected

    @pytest.mark.parametrize(
        "matrix_class_name",
        (
            "_CatXCatMatrix",
            "_MrXCatMatrix",
            "_CatXMrMatrix",
            "_MrXMrMatrix",
        ),
    )
    def it_can_construct_a_regular_matrix_to_help(
        self, request, cube_, dimension_, matrix_class_name
    ):
        cube_.dimension_types = (DT.CAT, DT.MR, DT.CAT)
        MatrixCls_ = class_mock(request, "cr.cube.matrix.%s" % matrix_class_name)
        _regular_matrix_class = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_regular_matrix_class",
            return_value=MatrixCls_,
        )
        _sliced_counts = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_sliced_counts",
            return_value=([[1], [2]], [[3], [4]]),
        )

        matrix = _BaseCubeResultMatrix._regular_matrix_factory(
            cube_, (dimension_, dimension_), slice_idx=17
        )

        _regular_matrix_class.assert_called_once_with((DT.MR, DT.CAT))
        _sliced_counts.assert_called_once_with(cube_, 17)
        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [[1], [2]], [[3], [4]]
        )
        assert matrix is MatrixCls_.return_value

    @pytest.mark.parametrize(
        ("counts", "counts_slice", "expected"),
        (
            ([[1, 2, 3], [4, 5, 6]], np.s_[:], [[1, 2, 3], [4, 5, 6]]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0], [1, 2, 3]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0, 0], 1),
        ),
    )
    def it_knows_its_sliced_counts_to_help(
        self, request, cube_, counts, counts_slice, expected
    ):
        counts = np.array(counts)
        cube_.counts = counts
        cube_.unweighted_counts = counts
        cube_.counts_with_missings = counts
        _regular_matrix_counts_slice = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_regular_matrix_counts_slice",
            return_value=counts_slice,
        )

        sliced_counts = _BaseCubeResultMatrix._sliced_counts(cube_, slice_idx=23)

        _regular_matrix_counts_slice.assert_called_once_with(cube_, 23)
        counts, unweighted, with_missing = sliced_counts
        assert counts.tolist() == expected
        assert unweighted.tolist() == expected

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_CatXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXCatMatrix` object."""

    def it_knows_its_columns_base(self):
        matrix = _CatXCatMatrix(None, None, np.array([[1, 2, 3], [4, 5, 6]]))
        np.testing.assert_equal(matrix.columns_base, np.array([5, 7, 9]))

    @pytest.mark.parametrize(
        ("unweighted_counts", "expected"),
        (
            # --- 1 "row", 0 cols ---
            ([[]], []),
            # --- 1 row, 1 col ---
            ([[1]], [1]),
            # --- 1 row, 3 cols ---
            ([[1, 2, 3]], [1, 2, 3]),
            # --- 3 rows, 0 cols ---
            ([[], [], []], []),
            # --- 3 rows, 1 col ---
            ([[1], [2], [3]], [6]),
            # --- 3 rows, 3 cols ---
            ([[1, 2, 3], [4, 5, 6]], [5, 7, 9]),
        ),
    )
    def it_knows_its_columns_pruning_base(self, unweighted_counts, expected):
        matrix = _CatXCatMatrix(None, None, np.array(unweighted_counts))

        columns_pruning_base = matrix.columns_pruning_base

        assert columns_pruning_base.shape == (len(expected),)
        np.testing.assert_equal(columns_pruning_base, np.array(expected))

    @pytest.mark.parametrize(
        ("unweighted_counts", "expected"),
        (
            ([[1, 2, 3]], [6]),
            ([[1, 2, 3], [4, 5, 6]], [6, 15]),
            ([[1], [2], [3]], [1, 2, 3]),
        ),
    )
    def it_knows_its_rows_pruning_base(self, unweighted_counts, expected):
        matrix = _CatXCatMatrix(None, None, unweighted_counts)
        assert matrix.rows_pruning_base.tolist() == expected

    def it_knows_its_table_margin(self):
        weighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        assert _CatXCatMatrix(None, weighted_counts, None).table_margin == 21

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(
            _CatXCatMatrix(None, None, unweighted_counts).unweighted_counts,
            unweighted_counts,
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array([[3, 2, 1], [6, 5, 4]])
        np.testing.assert_equal(
            _CatXCatMatrix(None, weighted_counts, None).weighted_counts,
            weighted_counts,
        )


class Describe_CatXCatMeansMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXCatMeansMatrix` object."""

    def it_knows_its_weighted_counts(self):
        means = np.array([[3, 2, 1], [6, 5, 4]])
        np.testing.assert_equal(
            _CatXCatMeansMatrix(None, means, None).weighted_counts,
            np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
        )


class Describe_CatXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXMrMatrix` object."""

    def it_knows_its_columns_pruning_base(self):
        unweighted_counts = np.array(
            [
                [[1, 6], [2, 5], [3, 5]],  # --- row 0 ---
                [[5, 3], [6, 3], [7, 2]],  # --- row 1 ---
            ]
        )
        np.testing.assert_equal(
            _CatXMrMatrix(None, None, unweighted_counts, None).columns_pruning_base,
            np.array([15, 16, 17]),
        )

    def it_knows_its_rows_pruning_base(self):
        unweighted_counts = np.array(
            [[[1, 6], [2, 5], [3, 4]], [[5, 3], [6, 2], [7, 1]]]
        )
        np.testing.assert_equal(
            _CatXMrMatrix(None, None, unweighted_counts, None).rows_pruning_base,
            np.array([21, 24]),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _CatXMrMatrix(None, None, unweighted_counts).unweighted_counts,
            np.array([[1, 2, 3], [4, 5, 6]]),
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _CatXMrMatrix(None, weighted_counts, None).weighted_counts,
            np.array([[1, 2, 3], [4, 5, 6]]),
        )


class Describe_MrXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXCatMatrix` object."""

    def it_knows_its_columns_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [3, 2, 1],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).columns_base,
            np.array([[5, 7, 9], [10, 10, 10]]),
        )

    def it_knows_its_columns_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [3, 2, 1],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).columns_pruning_base,
            np.array([15, 17, 19]),
        )

    def it_knows_its_rows_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 2, 3],  # -- selected --
                    [4, 5, 6],  # -- not --
                ],
                [  # -- row 1 ------------
                    [7, 8, 9],  # -- selected --
                    [3, 2, 1],  # -- not --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).rows_pruning_base,
            np.array([21, 30]),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 2, 3],  # -- selected --
                    [4, 5, 6],  # -- not --
                ],
                [  # -- row 1 ------------
                    [7, 8, 9],  # -- selected --
                    [0, 4, 2],  # -- not --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts).unweighted_counts,
            np.array([[1, 2, 3], [7, 8, 9]]),
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [0, 4, 2],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, weighted_counts, None).weighted_counts,
            np.array([[1, 2, 3], [7, 8, 9]]),
        )


class Describe_MrXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXMrMatrix` object."""

    def it_knows_its_columns_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).columns_base,
            np.array([[2, 8, 4], [10, 4, 13]]),
        )

    def it_knows_its_columns_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).columns_pruning_base,
            np.array([12, 12, 17]),
        )

    def it_knows_its_rows_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).rows_pruning_base,
            np.array([25, 27]),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 8],  # -- col 0
                        [1, 7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 6],  # -- col 0
                        [3, 5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 4],  # -- col 0
                        [5, 3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 2],  # -- col 0
                        [7, 1],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).unweighted_counts,
            np.array([[0, 1], [4, 5]]),
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 8],  # -- col 0
                        [1, 7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 6],  # -- col 0
                        [3, 5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 4],  # -- col 0
                        [5, 3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 2],  # -- col 0
                        [7, 1],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, weighted_counts, None, None).weighted_counts,
            np.array([[0, 1], [4, 5]]),
        )
