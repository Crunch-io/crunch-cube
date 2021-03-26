# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.assembler` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Element, _Subtotal, _Subtotals
from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT
from cr.cube.matrix.assembler import (
    Assembler,
    _BaseOrderHelper,
    _ColumnOrderHelper,
    _RowOrderHelper,
    _SortRowsByColumnValueHelper,
)
from cr.cube.matrix.cubemeasure import (
    BaseCubeResultMatrix,
    _CatXCatMatrix,
)
from cr.cube.matrix.measure import (
    _BaseSecondOrderMeasure,
    _ColumnProportions,
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

from ...unitutil import (
    class_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class DescribeAssembler(object):
    """Unit test suite for `cr.cube.matrix.assembler.Assembler` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("column_proportions", _ColumnProportions),
            ("column_unweighted_bases", _ColumnUnweightedBases),
            ("column_weighted_bases", _ColumnWeightedBases),
            ("row_unweighted_bases", _RowUnweightedBases),
            ("row_weighted_bases", _RowWeightedBases),
            ("table_unweighted_bases", _TableUnweightedBases),
            ("table_weighted_bases", _TableWeightedBases),
            ("weighted_counts", _WeightedCounts),
            ("unweighted_counts", _UnweightedCounts),
        ),
    )
    def it_assembles_various_measures(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        _assemble_matrix_,
        measure_prop_name,
        MeasureCls,
    ):
        _measures_prop_.return_value = second_order_measures_
        setattr(
            second_order_measures_,
            measure_prop_name,
            instance_mock(request, MeasureCls, blocks=[["A", "B"], ["C", "D"]]),
        )
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        value = getattr(assembler, measure_prop_name)

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert value == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_column_index(
        self,
        request,
        _cube_result_matrix_prop_,
        dimensions_,
        NanSubtotals_,
        _assemble_matrix_,
    ):
        cube_result_matrix_ = instance_mock(
            request, _CatXCatMatrix, column_index=[[1, 2], [3, 4]]
        )
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        NanSubtotals_.blocks.return_value = [[[1], [np.nan]], [[3], []]]
        _assemble_matrix_.return_value = [[1, np.nan, 3], [4, np.nan, 6]]
        assembler = Assembler(None, dimensions_, None)

        column_index = assembler.column_index

        NanSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        _assemble_matrix_.assert_called_once_with(
            assembler, [[[1], [np.nan]], [[3], []]]
        )
        assert column_index == [[1, np.nan, 3], [4, np.nan, 6]]

    def it_knows_the_column_labels(
        self,
        _columns_dimension_prop_,
        dimension_,
        _column_order_prop_,
        _dimension_labels_,
    ):
        _columns_dimension_prop_.return_value = dimension_
        _column_order_prop_.return_value = [0, 1, 2]
        _dimension_labels_.return_value = np.array(["Alpha", "Baker", "Charlie"])
        assembler = Assembler(None, None, None)

        column_labels = assembler.column_labels

        _dimension_labels_.assert_called_once_with(assembler, dimension_, [0, 1, 2])
        assert column_labels.tolist() == ["Alpha", "Baker", "Charlie"]

    def it_provides_a_1D_columns_base_for_a_CAT_X_cube_result(
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
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1], diffs_nan=True
        )
        assert columns_base.tolist() == [1, 3, 2, 3, 5]

    def but_it_provides_a_2D_columns_base_for_an_MR_X_cube_result(
        self,
        _rows_dimension_prop_,
        dimensions_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        _rows_dimension_prop_.return_value = dimensions_[0]
        dimensions_[0].dimension_type = DT.MR_SUBVAR
        cube_result_matrix_.columns_base = np.array([[1, 2], [3, 4]])
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, dimensions_, None)

        columns_base = assembler.columns_base

        assert SumSubtotals_.blocks.call_count == 1
        np.testing.assert_array_equal(
            SumSubtotals_.blocks.call_args_list[0][0][0], [[1, 2], [3, 4]]
        )
        assert SumSubtotals_.blocks.call_args_list[0][0][1] == dimensions_
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert columns_base == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_columns_dimension_numeric_values(
        self, request, _columns_dimension_prop_, dimension_, _column_order_prop_
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.valid_elements = tuple(
            instance_mock(request, _Element, numeric_value=numeric_value)
            for numeric_value in (1, 2, 3)
        )
        _column_order_prop_.return_value = [2, -1, 0, -2]
        assembler = Assembler(None, None, None)

        np.testing.assert_almost_equal(
            assembler.columns_dimension_numeric_values,
            [3.0, np.nan, 1.0, np.nan],
        )

    def it_provides_a_1D_columns_margin_for_a_CAT_X_cube_result(
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
        cube_result_matrix_.columns_margin = [1, 2, 3]
        _column_subtotals_prop_.return_value = [3, 5]
        _column_order_prop_.return_value = [0, -2, 1, 2, -1]
        _assemble_vector_.return_value = np.array([1, 3, 2, 3, 5])
        assembler = Assembler(None, None, None)

        columns_margin = assembler.columns_margin

        _assemble_vector_.assert_called_once_with(
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1]
        )
        assert columns_margin.tolist() == [1, 3, 2, 3, 5]

    def but_it_provides_a_2D_columns_margin_for_an_MR_X_cube_result(
        self,
        _rows_dimension_prop_,
        dimensions_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        _rows_dimension_prop_.return_value = dimensions_[0]
        dimensions_[0].dimension_type = DT.MR_SUBVAR
        cube_result_matrix_.columns_margin = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, dimensions_, None)

        columns_margin = assembler.columns_margin

        SumSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert columns_margin == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_inserted_column_idxs(self, _column_order_prop_):
        _column_order_prop_.return_value = [2, -1, 0, -2]
        assert Assembler(None, None, None).inserted_column_idxs == (1, 3)

    def it_knows_the_inserted_row_idxs(self, _row_order_prop_):
        _row_order_prop_.return_value = [0, 1, -2, 2, -1, 3]
        assert Assembler(None, None, None).inserted_row_idxs == (2, 4)

    def it_knows_the_means(self, request, dimensions_):
        property_mock(
            request, Assembler, "means", return_value=np.array([1.2, 1.34, 3.3])
        )
        assembler = Assembler(None, dimensions_, None)

        assert assembler.means == pytest.approx([1.2, 1.34, 3.3])

    def it_knows_the_sum(self, request, dimensions_):
        property_mock(request, Assembler, "sums", return_value=np.array([4, 5, 6]))
        assembler = Assembler(None, dimensions_, None)

        assert assembler.sums == pytest.approx([4, 5, 6])

    def it_knows_the_pvalues(self, request):
        property_mock(
            request, Assembler, "zscores", return_value=np.array([0.7, 0.8, 0.9])
        )
        assembler = Assembler(None, None, None)

        np.testing.assert_almost_equal(
            assembler.pvalues, np.array([0.4839273, 0.4237108, 0.3681203])
        )

    def it_knows_the_row_labels(
        self,
        _rows_dimension_prop_,
        dimension_,
        _row_order_prop_,
        _dimension_labels_,
    ):
        _rows_dimension_prop_.return_value = dimension_
        _row_order_prop_.return_value = [0, 1, 2]
        _dimension_labels_.return_value = np.array(["Alpha", "Baker", "Charlie"])
        assembler = Assembler(None, None, None)

        row_labels = assembler.row_labels

        _dimension_labels_.assert_called_once_with(assembler, dimension_, [0, 1, 2])
        assert row_labels.tolist() == ["Alpha", "Baker", "Charlie"]

    def it_provides_a_1D_rows_base_for_an_X_CAT_cube_result(
        self,
        _columns_dimension_prop_,
        dimension_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _row_subtotals_prop_,
        _row_order_prop_,
        _assemble_vector_,
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.CAT
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.rows_base = [1, 2, 3]
        _row_subtotals_prop_.return_value = [3, 5]
        _row_order_prop_.return_value = [0, -2, 1, 2, -1]
        _assemble_vector_.return_value = [1, 3, 2, 3, 5]
        assembler = Assembler(None, None, None)

        rows_base = assembler.rows_base

        _assemble_vector_.assert_called_once_with(
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1], diffs_nan=True
        )
        assert rows_base == [1, 3, 2, 3, 5]

    def but_it_provides_a_2D_rows_base_for_an_X_MR_cube_result(
        self,
        _columns_dimension_prop_,
        dimensions_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        _columns_dimension_prop_.return_value = dimensions_[1]
        dimensions_[1].dimension_type = DT.MR_SUBVAR
        cube_result_matrix_.rows_base = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, dimensions_, None)

        rows_base = assembler.rows_base

        SumSubtotals_.blocks.assert_called_once_with(
            [[1, 2], [3, 4]], dimensions_, diff_rows_nan=True
        )
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert rows_base == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_rows_dimension_fills(
        self, request, _rows_dimension_prop_, dimension_, _row_order_prop_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.valid_elements = tuple(
            instance_mock(request, _Element, fill=fill)
            for fill in ("#000000", "#111111", "#f00ba5")
        )
        _row_order_prop_.return_value = [2, -1, 0, -2]
        assembler = Assembler(None, None, None)

        assert assembler.rows_dimension_fills == ("#f00ba5", None, "#000000", None)

    def it_knows_the_rows_dimension_numeric_values(
        self, request, _rows_dimension_prop_, dimension_, _row_order_prop_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.valid_elements = tuple(
            instance_mock(request, _Element, numeric_value=numeric_value)
            for numeric_value in (1, 2, 3)
        )
        _row_order_prop_.return_value = [2, -1, 0, -2]
        assembler = Assembler(None, None, None)

        np.testing.assert_almost_equal(
            assembler.rows_dimension_numeric_values,
            [3.0, np.nan, 1.0, np.nan],
        )

    def it_provides_a_1D_rows_margin_for_an_X_CAT_cube_result(
        self,
        _columns_dimension_prop_,
        dimension_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _row_subtotals_prop_,
        _row_order_prop_,
        _assemble_vector_,
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.CAT
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.rows_margin = [1, 2, 3]
        _row_subtotals_prop_.return_value = [3, 5]
        _row_order_prop_.return_value = [0, -2, 1, 2, -1]
        _assemble_vector_.return_value = [1, 3, 2, 3, 5]
        assembler = Assembler(None, None, None)

        rows_margin = assembler.rows_margin

        _assemble_vector_.assert_called_once_with(
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1]
        )
        assert rows_margin == [1, 3, 2, 3, 5]

    def but_it_provides_a_2D_rows_margin_for_an_X_MR_cube_result(
        self,
        _columns_dimension_prop_,
        dimensions_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        _columns_dimension_prop_.return_value = dimensions_[1]
        dimensions_[1].dimension_type = DT.MR_SUBVAR
        cube_result_matrix_.rows_margin = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, dimensions_, None)

        rows_margin = assembler.rows_margin

        SumSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert rows_margin == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_2D_table_base_of_an_MR_X_MR_matrix(
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
        cube_result_matrix_.table_base = np.array([[1, 2, 3], [4, 5, 6]])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _row_order_prop_.return_value = np.array([1, 0])
        _column_order_prop_.return_value = np.array([1, 0, 2])
        assembler = Assembler(None, None, None)

        assert assembler.table_base.tolist() == [[5, 4, 6], [2, 1, 3]]

    def and_it_knows_the_1D_table_base_of_a_CAT_X_MR_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _column_order_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_base = np.array([1, 2, 3])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _column_order_prop_.return_value = np.array([2, 0, 1])
        assembler = Assembler(None, None, None)

        assert assembler.table_base.tolist() == [3, 1, 2]

    def and_it_knows_the_1D_table_base_of_an_MR_X_CAT_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _row_order_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_base = np.array([1, 2, 3])
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.MR
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _row_order_prop_.return_value = np.array([1, 0, 2])
        assembler = Assembler(None, None, None)

        assert assembler.table_base.tolist() == [2, 1, 3]

    def and_it_knows_the_scalar_table_base_of_a_CAT_X_CAT_matrix(
        self,
        request,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        cube_result_matrix_.table_base = 4242
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        assembler = Assembler(None, None, None)

        assert assembler.table_base == 4242

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

        assert assembler.table_margin.tolist() == [[5, 4, 6], [2, 1, 3]]

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

        assert assembler.table_margin.tolist() == [2, 1, 3]

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

        assert assembler.table_margin.tolist() == [3, 1, 2]

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

    def it_knows_the_table_stderrs(
        self,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        dimensions_,
        TableStdErrSubtotals_,
        _assemble_matrix_,
    ):
        cube_result_matrix_.table_stderrs = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        TableStdErrSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 3, 2], [4, 6, 5]]
        assembler = Assembler(None, dimensions_, None)

        table_stderrs = assembler.table_stderrs

        TableStdErrSubtotals_.blocks.assert_called_once_with(
            [[1, 2], [3, 4]], dimensions_, cube_result_matrix_
        )
        _assemble_matrix_.assert_called_once_with(assembler, [[[1], [2]], [[3], [4]]])
        assert table_stderrs == [[1, 3, 2], [4, 6, 5]]

    def it_computes_zscores_for_a_CAT_X_CAT_slice(
        self,
        request,
        dimensions_,
        _rows_dimension_prop_,
        _columns_dimension_prop_,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        ZscoreSubtotals_,
        _assemble_matrix_,
    ):
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=DT.CAT
        )
        cube_result_matrix_.zscores = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        ZscoreSubtotals_.blocks.return_value = [[[4], [3]], [[2], [1]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6]]
        assembler = Assembler(None, dimensions_, None)

        zscores = assembler.zscores

        ZscoreSubtotals_.blocks.assert_called_once_with(
            [[1, 2], [3, 4]], dimensions_, cube_result_matrix_
        )
        _assemble_matrix_.assert_called_once_with(assembler, [[[4], [3]], [[2], [1]]])
        assert zscores == [[1, 2, 3], [4, 5, 6]]

    @pytest.mark.parametrize(
        "rows_dim_type, cols_dim_type",
        (
            (DT.CAT, DT.MR),
            (DT.MR, DT.CAT),
            (DT.MR, DT.MR),
        ),
    )
    def but_it_provides_zscores_with_NaN_subtotals_when_cube_has_an_MR_dimension(
        self,
        request,
        dimensions_,
        _rows_dimension_prop_,
        rows_dim_type,
        _columns_dimension_prop_,
        cols_dim_type,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        NanSubtotals_,
        _assemble_matrix_,
    ):
        _rows_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=rows_dim_type
        )
        _columns_dimension_prop_.return_value = instance_mock(
            request, Dimension, dimension_type=cols_dim_type
        )
        cube_result_matrix_.zscores = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        NanSubtotals_.blocks.return_value = [[[1], [np.nan]], [[3], []]]
        _assemble_matrix_.return_value = [[1, np.nan, 3], [4, np.nan, 6]]
        assembler = Assembler(None, dimensions_, None)

        zscores = assembler.zscores

        NanSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        _assemble_matrix_.assert_called_once_with(
            assembler, [[[1], [np.nan]], [[3], []]]
        )
        assert zscores == [[1, np.nan, 3], [4, np.nan, 6]]

    # === implementation methods/properties ===

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
            instance_mock(
                request,
                _Subtotal,
                addend_idxs=np.array(addend_idxs, dtype=int),
                subtrahend_idxs=np.array(subtrahend_idxs, dtype=int),
            )
            for addend_idxs, subtrahend_idxs in (
                ((0, 1), ()),
                ((1, 2), ()),
                ((2, 3), ()),
            )
        )
        order = np.array([-3, 1, 0, -2, 3, 2, -1])
        assembler = Assembler(None, None, None)

        vector = assembler._assemble_vector(base_vector, subtotals_, order)

        assert vector.tolist() == [3, 2, 1, 5, 4, 3, 7]

    def it_knows_the_column_order_to_help(
        self,
        request,
        _BaseOrderHelper_,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        _BaseOrderHelper_.column_display_order.return_value = np.array(
            [-1, 1, -2, 2, -3, 3]
        )
        assembler = Assembler(None, dimensions_, None)

        column_order = assembler._column_order

        _BaseOrderHelper_.column_display_order.assert_called_once_with(
            dimensions_, second_order_measures_
        )
        assert column_order.tolist() == [-1, 1, -2, 2, -3, 3]

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
        self, request, cube_, dimensions_, cube_result_matrix_
    ):
        BaseCubeResultMatrix_ = class_mock(
            request, "cr.cube.matrix.assembler.BaseCubeResultMatrix"
        )
        BaseCubeResultMatrix_.factory.return_value = cube_result_matrix_
        assembler = Assembler(cube_, dimensions_, slice_idx=42)

        cube_result_matrix = assembler._cube_result_matrix

        BaseCubeResultMatrix_.factory.assert_called_once_with(cube_, dimensions_, 42)
        assert cube_result_matrix is cube_result_matrix_

    def it_assembles_the_dimension_labels_to_help(self, request, dimension_):
        dimension_.valid_elements = tuple(
            instance_mock(request, _Element, label=label)
            for label in ("Alpha", "Bravo", "Charlie", "Delta")
        )
        dimension_.subtotals = tuple(
            instance_mock(request, _Subtotal, label=label) for label in ("Top 2", "All")
        )
        order = np.array([1, 3, -2, 2, -1])
        assembler = Assembler(None, None, None)

        labels = assembler._dimension_labels(dimension_, order)

        assert labels.tolist() == ["Bravo", "Delta", "Top 2", "Charlie", "All"]

    def it_constructs_its_measures_collaborator_object_to_help(
        self, request, cube_, dimensions_, second_order_measures_
    ):
        SecondOrderMeasures_ = class_mock(
            request,
            "cr.cube.matrix.assembler.SecondOrderMeasures",
            return_value=second_order_measures_,
        )
        assembler = Assembler(cube_, dimensions_, slice_idx=17)

        measures = assembler._measures

        SecondOrderMeasures_.assert_called_once_with(cube_, dimensions_, 17)
        assert measures == second_order_measures_

    def it_knows_the_row_order_to_help(
        self,
        request,
        _BaseOrderHelper_,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        _BaseOrderHelper_.row_display_order.return_value = np.array(
            [-1, 1, -2, 2, -3, 3]
        )
        assembler = Assembler(None, dimensions_, None)

        row_order = assembler._row_order

        _BaseOrderHelper_.row_display_order.assert_called_once_with(
            dimensions_, second_order_measures_
        )
        assert row_order.tolist() == [-1, 1, -2, 2, -3, 3]

    def it_provides_access_to_the_row_subtotals_to_help(
        self, _rows_dimension_prop_, dimension_, subtotals_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.subtotals = subtotals_

        assert Assembler(None, None, None)._row_subtotals is subtotals_

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
    def _BaseOrderHelper_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler._BaseOrderHelper")

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
        return instance_mock(request, BaseCubeResultMatrix)

    @pytest.fixture
    def _cube_result_matrix_prop_(self, request):
        return property_mock(request, Assembler, "_cube_result_matrix")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _dimension_labels_(self, request):
        return method_mock(request, Assembler, "_dimension_labels")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def _measures_prop_(self, request):
        return property_mock(request, Assembler, "_measures")

    @pytest.fixture
    def NanSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.NanSubtotals")

    @pytest.fixture
    def _row_order_prop_(self, request):
        return property_mock(request, Assembler, "_row_order")

    @pytest.fixture
    def _row_subtotals_prop_(self, request):
        return property_mock(request, Assembler, "_row_subtotals")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, Assembler, "_rows_dimension")

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.SumSubtotals")

    @pytest.fixture
    def TableStdErrSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.TableStdErrSubtotals")

    @pytest.fixture
    def ZscoreSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.ZscoreSubtotals")


class Describe_BaseOrderHelper(object):
    """Unit test suite for `cr.cube.matrix.assembler._BaseOrderHelper` object."""

    def it_dispatches_to_the_right_column_order_helper(
        self, request, dimensions_, second_order_measures_
    ):
        column_order_helper_ = instance_mock(
            request, _ColumnOrderHelper, _display_order=np.array([-2, 1, -1, 2])
        )
        _ColumnOrderHelper_ = class_mock(
            request,
            "cr.cube.matrix.assembler._ColumnOrderHelper",
            return_value=column_order_helper_,
        )

        column_order = _BaseOrderHelper.column_display_order(
            dimensions_, second_order_measures_
        )

        _ColumnOrderHelper_.assert_called_once_with(dimensions_, second_order_measures_)
        assert column_order.tolist() == [-2, 1, -1, 2]

    @pytest.mark.parametrize(
        "collation_method, HelperCls",
        (
            (CM.OPPOSING_ELEMENT, _SortRowsByColumnValueHelper),
            (CM.EXPLICIT_ORDER, _RowOrderHelper),
            (CM.PAYLOAD_ORDER, _RowOrderHelper),
        ),
    )
    def it_dispatches_to_the_right_row_order_helper(
        self, request, dimensions_, second_order_measures_, collation_method, HelperCls
    ):
        dimensions_[0].collation_method = collation_method
        helper_ = instance_mock(
            request, HelperCls, _display_order=np.array([-1, 1, -2, 2])
        )
        HelperCls_ = class_mock(
            request,
            "cr.cube.matrix.assembler.%s" % HelperCls.__name__,
            return_value=helper_,
        )

        row_order = _BaseOrderHelper.row_display_order(
            dimensions_, second_order_measures_
        )

        HelperCls_.assert_called_once_with(dimensions_, second_order_measures_)
        assert row_order.tolist() == [-1, 1, -2, 2]

    def it_provides_access_to_the_columns_dimension_to_help(self, dimension_):
        order_helper = _BaseOrderHelper((None, dimension_), None)
        assert order_helper._columns_dimension is dimension_

    @pytest.mark.parametrize(
        "prune_subtotals, order, expected_value",
        (
            (True, (-1, 1, -2, 2, -3, 3), [1, 2, 3]),
            (True, (1, 2, 3), [1, 2, 3]),
            (False, (-1, 1, -2, 2, -3, 3), [-1, 1, -2, 2, -3, 3]),
        ),
    )
    def it_post_processes_the_display_order_to_help(
        self, request, prune_subtotals, order, expected_value
    ):
        property_mock(
            request, _BaseOrderHelper, "_prune_subtotals", return_value=prune_subtotals
        )
        property_mock(request, _BaseOrderHelper, "_order", return_value=order)
        order_helper = _BaseOrderHelper(None, None)

        display_order = order_helper._display_order

        assert display_order.tolist() == expected_value

    @pytest.mark.parametrize(
        "base, expected_value",
        (([1, 1, 1], ()), ([1, 0, 1], (1,)), ([0, 0, 0], (0, 1, 2))),
    )
    def it_knows_its_empty_column_idxs_to_help(
        self, second_order_measures_, base, expected_value
    ):
        second_order_measures_.columns_pruning_base = np.array(base)
        order_helper = _BaseOrderHelper(None, second_order_measures_)

        assert order_helper._empty_column_idxs == expected_value

    @pytest.mark.parametrize(
        "base, expected_value",
        (([1, 1, 1], ()), ([1, 0, 1], (1,)), ([0, 0, 0], (0, 1, 2))),
    )
    def it_knows_its_empty_row_idxs_to_help(
        self, second_order_measures_, base, expected_value
    ):
        second_order_measures_.rows_pruning_base = np.array(base)
        order_helper = _BaseOrderHelper(None, second_order_measures_)

        assert order_helper._empty_row_idxs == expected_value

    def it_provides_access_to_the_rows_dimension_to_help(self, dimension_):
        order_helper = _BaseOrderHelper((dimension_, None), None)
        assert order_helper._rows_dimension is dimension_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)


class Describe_ColumnOrderHelper(object):
    """Unit test suite for `cr.cube.matrix.assembler._ColumnOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def it_computes_the_order_of_a_columns_dimension_to_help(
        self, request, dimension_, collation_method, collator_class_name
    ):
        property_mock(
            request, _ColumnOrderHelper, "_columns_dimension", return_value=dimension_
        )
        dimension_.collation_method = collation_method
        CollatorCls_ = class_mock(
            request, "cr.cube.matrix.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (3, -1, 5, 1, -2)
        property_mock(
            request, _ColumnOrderHelper, "_empty_column_idxs", return_value=(1, 3)
        )
        order_helper = _ColumnOrderHelper(None, None)

        order = order_helper._order

        CollatorCls_.display_order.assert_called_once_with(dimension_, (1, 3))
        assert order == (3, -1, 5, 1, -2)

    @pytest.mark.parametrize(
        "prune, empty_row_idxs, expected_value",
        (
            (False, None, False),
            (True, (3,), False),
            (True, (0, 1, 2), True),
        ),
    )
    def it_knows_whether_to_prune_the_subtotal_columns_to_help(
        self, request, dimension_, prune, empty_row_idxs, expected_value
    ):
        property_mock(
            request, _ColumnOrderHelper, "_rows_dimension", return_value=dimension_
        )
        property_mock(
            request,
            _ColumnOrderHelper,
            "_empty_row_idxs",
            return_value=empty_row_idxs,
        )
        dimension_.prune = prune
        dimension_.element_ids = (1, 2, 3)
        order_helper = _ColumnOrderHelper(None, None)

        assert order_helper._prune_subtotals is expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_RowOrderHelper(object):
    """Unit test suite for `cr.cube.matrix.assembler._RowOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def it_computes_the_order_of_a_rows_dimension_to_help(
        self, request, dimension_, collation_method, collator_class_name
    ):
        property_mock(
            request, _RowOrderHelper, "_rows_dimension", return_value=dimension_
        )
        dimension_.collation_method = collation_method
        CollatorCls_ = class_mock(
            request, "cr.cube.matrix.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        property_mock(
            request, _RowOrderHelper, "_empty_row_idxs", return_value=(2, 4, 6)
        )
        order_helper = _RowOrderHelper(None, None)

        order = order_helper._order

        CollatorCls_.display_order.assert_called_once_with(dimension_, (2, 4, 6))
        assert order == (1, -2, 3, 5, -1)

    @pytest.mark.parametrize(
        "prune, empty_column_idxs, expected_value",
        (
            (False, None, False),
            (True, (3,), False),
            (True, (0, 1, 2), True),
        ),
    )
    def it_knows_whether_to_prune_the_subtotal_rows_to_help(
        self, request, dimension_, prune, empty_column_idxs, expected_value
    ):
        property_mock(
            request, _RowOrderHelper, "_columns_dimension", return_value=dimension_
        )
        property_mock(
            request,
            _RowOrderHelper,
            "_empty_column_idxs",
            return_value=empty_column_idxs,
        )
        dimension_.prune = prune
        dimension_.element_ids = (1, 2, 3)
        order_helper = _RowOrderHelper(None, None)

        assert order_helper._prune_subtotals is expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_SortRowsByColumnValueHelper(object):
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByColumnValueHelper`."""

    def it_extracts_the_sort_column_idx_from_the_order_spec_dict_to_help(
        self, _rows_dimension_prop_, dimension_, _order_dict_prop_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.element_ids = (1, 2, 3, 4, 5)
        _order_dict_prop_.return_value = {"element_id": 3}
        order_helper = _SortRowsByColumnValueHelper(None, None)

        assert order_helper._column_idx == 2

    def it_extracts_the_element_values_to_help(
        self, _measure_prop_, measure_, _column_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[np.arange(20).reshape(4, 5), None], [None, None]]
        _column_idx_prop_.return_value = 2
        order_helper = _SortRowsByColumnValueHelper(None, None)

        assert order_helper._element_values.tolist() == [2, 7, 12, 17]

    def it_retrieves_the_measure_object_to_help(self, request, _order_dict_prop_):
        column_proportions_ = instance_mock(request, _ColumnProportions)
        second_order_measures_ = instance_mock(
            request, SecondOrderMeasures, column_proportions=column_proportions_
        )
        _order_dict_prop_.return_value = {"measure": "col_percent"}
        order_helper = _SortRowsByColumnValueHelper(None, second_order_measures_)

        assert order_helper._measure is column_proportions_

    def but_it_raises_when_an_unsupported_sort_by_value_measure_is_requested(
        self, _order_dict_prop_
    ):
        _order_dict_prop_.return_value = {"measure": "foobar"}
        order_helper = _SortRowsByColumnValueHelper(None, None)

        with pytest.raises(NotImplementedError) as e:
            order_helper._measure

        assert str(e.value) == "sort-by-value for measure 'foobar' is not yet supported"

    def it_computes_the_sorted_element_order_to_help(
        self, request, _rows_dimension_prop_, dimension_
    ):
        _rows_dimension_prop_.return_value = dimension_
        property_mock(
            request,
            _SortRowsByColumnValueHelper,
            "_element_values",
            # --- return type is ndarray in real life, but assert_called_once_with()
            # --- won't match on those, so use list instead.
            return_value=[16, 3, 12],
        )
        property_mock(
            request,
            _SortRowsByColumnValueHelper,
            "_subtotal_values",
            return_value=[15, 19],  # --- ndarray in real life ---
        )
        property_mock(
            request, _SortRowsByColumnValueHelper, "_empty_row_idxs", return_value=()
        )
        SortByValueCollator_ = class_mock(
            request, "cr.cube.matrix.assembler.SortByValueCollator"
        )
        SortByValueCollator_.display_order.return_value = (-1, -2, 0, 2, 1)
        order_helper = _SortRowsByColumnValueHelper(None, None)

        order = order_helper._order

        SortByValueCollator_.display_order.assert_called_once_with(
            dimension_, [16, 3, 12], [15, 19], ()
        )
        assert order == (-1, -2, 0, 2, 1)

    def it_provides_access_to_the_order_dict_to_help(
        self, _rows_dimension_prop_, dimension_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.order_dict = {"order": "dict"}
        order_helper = _SortRowsByColumnValueHelper(None, None)

        assert order_helper._order_dict == {"order": "dict"}

    def it_extracts_the_subtotal_values_to_help(
        self, _measure_prop_, measure_, _column_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[None, None], [np.arange(10, 101, 10).reshape(2, 5), None]]
        _column_idx_prop_.return_value = 2
        order_helper = _SortRowsByColumnValueHelper(None, None)

        assert order_helper._subtotal_values.tolist() == [30, 80]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _column_idx_prop_(self, request):
        return property_mock(request, _SortRowsByColumnValueHelper, "_column_idx")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def _measure_prop_(self, request):
        return property_mock(request, _SortRowsByColumnValueHelper, "_measure")

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, _SortRowsByColumnValueHelper, "_order_dict")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _SortRowsByColumnValueHelper, "_rows_dimension")
