# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.assembler` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Element, _Subtotal, _Subtotals
from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT
from cr.cube.matrix.assembler import Assembler
from cr.cube.matrix.cubemeasure import (
    BaseCubeResultMatrix,
    _CatXCatMatrix,
    _CatXCatMeansMatrix,
)
from cr.cube.matrix.measure import (
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

from ...unitutil import class_mock, instance_mock, method_mock, property_mock


class DescribeAssembler(object):
    """Unit test suite for `cr.cube.matrix.assembler.Assembler` object."""

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

    def it_knows_the_column_unweighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        column_unweighted_bases_ = instance_mock(
            request, _ColumnUnweightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.column_unweighted_bases = column_unweighted_bases_
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        column_unweighted_bases = assembler.column_unweighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert column_unweighted_bases == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_column_weighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        column_weighted_bases_ = instance_mock(
            request, _ColumnWeightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.column_weighted_bases = column_weighted_bases_
        _assemble_matrix_.return_value = [[7, 8, 9], [4, 5, 6], [1, 2, 3]]
        assembler = Assembler(None, None, None)

        column_weighted_bases = assembler.column_weighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert column_weighted_bases == [[7, 8, 9], [4, 5, 6], [1, 2, 3]]

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
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1]
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
        cube_result_matrix_.columns_base = [[1, 2], [3, 4]]
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, dimensions_, None)

        columns_base = assembler.columns_base

        SumSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
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

    def it_knows_the_means(
        self,
        request,
        cube_,
        _cube_result_matrix_prop_,
        dimensions_,
        NanSubtotals_,
        _assemble_matrix_,
    ):
        cube_.has_means = True
        cube_result_matrix_ = instance_mock(
            request, _CatXCatMeansMatrix, means=[[1, 2], [3, 4]]
        )
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        NanSubtotals_.blocks.return_value = [[[3], [2]], [[4], [1]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6]]
        assembler = Assembler(cube_, dimensions_, None)

        means = assembler.means

        NanSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        _assemble_matrix_.assert_called_once_with(assembler, [[[3], [2]], [[4], [1]]])
        assert means == [[1, 2, 3], [4, 5, 6]]

    def but_it_raises_when_the_cube_result_does_not_contain_means_measure(self, cube_):
        cube_.has_means = False
        with pytest.raises(ValueError) as e:
            Assembler(cube_, None, None).means
        assert str(e.value) == "cube-result does not include a means cube-measure"

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

    def it_knows_the_row_unweighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        row_unweighted_bases_ = instance_mock(
            request, _RowUnweightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.row_unweighted_bases = row_unweighted_bases_
        _assemble_matrix_.return_value = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        assembler = Assembler(None, None, None)

        row_unweighted_bases = assembler.row_unweighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert row_unweighted_bases == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

    def it_knows_the_row_weighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        row_weighted_bases_ = instance_mock(
            request, _RowWeightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.row_weighted_bases = row_weighted_bases_
        _assemble_matrix_.return_value = [[9.9, 8.8, 7.7], [6.6, 5.5, 4.4]]
        assembler = Assembler(None, None, None)

        row_weighted_bases = assembler.row_weighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert row_weighted_bases == [[9.9, 8.8, 7.7], [6.6, 5.5, 4.4]]

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
            assembler, [1, 2, 3], [3, 5], [0, -2, 1, 2, -1]
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

        SumSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
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

    def it_knows_the_table_unweighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        table_unweighted_bases_ = instance_mock(
            request, _TableUnweightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.table_unweighted_bases = table_unweighted_bases_
        _assemble_matrix_.return_value = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        assembler = Assembler(None, None, None)

        table_unweighted_bases = assembler.table_unweighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert table_unweighted_bases == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

    def it_knows_the_table_weighted_bases(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        table_weighted_bases_ = instance_mock(
            request, _TableWeightedBases, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.table_weighted_bases = table_weighted_bases_
        _assemble_matrix_.return_value = [
            [9.9, 8.8, 7.7],
            [6.6, 5.5, 4.4],
            [3.3, 2.2, 1.1],
        ]
        assembler = Assembler(None, None, None)

        table_weighted_bases = assembler.table_weighted_bases

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert table_weighted_bases == [
            [9.9, 8.8, 7.7],
            [6.6, 5.5, 4.4],
            [3.3, 2.2, 1.1],
        ]

    def it_knows_the_unweighted_counts(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        unweighted_counts_ = instance_mock(
            request, _UnweightedCounts, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.unweighted_counts = unweighted_counts_
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        unweighted_counts = assembler.unweighted_counts

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert unweighted_counts == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_knows_the_weighted_counts(
        self, request, _measures_prop_, second_order_measures_, _assemble_matrix_
    ):
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[["A", "B"], ["C", "D"]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.weighted_counts = weighted_counts_
        _assemble_matrix_.return_value = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        assembler = Assembler(None, None, None)

        weighted_counts = assembler.weighted_counts

        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert weighted_counts == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

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
            instance_mock(request, _Subtotal, addend_idxs=np.array(addend_idxs))
            for addend_idxs in ((0, 1), (1, 2), (2, 3))
        )
        order = np.array([-3, 1, 0, -2, 3, 2, -1])
        assembler = Assembler(None, None, None)

        vector = assembler._assemble_vector(base_vector, subtotals_, order)

        assert vector.tolist() == [3, 2, 1, 5, 4, 3, 7]

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
        assert column_order.tolist() == expected_value

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
        CollatorCls_ = class_mock(
            request, "cr.cube.matrix.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        dimension_.collation_method = collation_method
        assembler = Assembler(None, None, None)

        dimension_order = assembler._dimension_order(dimension_, empty_idxs=[2, 4, 6])

        CollatorCls_.display_order.assert_called_once_with(dimension_, [2, 4, 6])
        assert dimension_order.shape == (5,)
        assert dimension_order.tolist() == [1, -2, 3, 5, -1]

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
    def _dimension_order_(self, request):
        return method_mock(request, Assembler, "_dimension_order")

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def _empty_column_idxs_prop_(self, request):
        return property_mock(request, Assembler, "_empty_column_idxs")

    @pytest.fixture
    def _empty_row_idxs_prop_(self, request):
        return property_mock(request, Assembler, "_empty_row_idxs")

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
