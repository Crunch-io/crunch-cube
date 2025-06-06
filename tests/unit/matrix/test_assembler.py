# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart assembly` module."""

import mock
import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice
from cr.cube.dimension import Dimension, Element, _OrderSpec, _Subtotal, _Subtotals
from cr.cube.enums import (
    COLLATION_METHOD as CM,
    DIMENSION_TYPE as DT,
    MARGINAL,
    MEASURE as M,
    ORDER_FORMAT,
)
from cr.cube.matrix.assembler import (
    _BaseOrderHelper,
    _BaseSortColumnsByValueHelper,
    _BaseSortRowsByValueHelper,
    _ColumnOrderHelper,
    _RowOrderHelper,
    _SortColumnsByLabelHelper,
    _SortColumnsByBaseRowHelper,
    _SortRowsByBaseColumnHelper,
    _SortRowsByDerivedColumnHelper,
    _SortRowsByInsertedColumnHelper,
    _SortRowsByLabelHelper,
    _SortRowsByMarginalHelper,
)
from cr.cube.matrix.measure import (
    _BaseMarginal,
    _BaseSecondOrderMeasure,
    _ColumnIndex,
    _ColumnProportions,
    _AudienceRatio,
    _ColumnShareSum,
    _ColumnStandardError,
    _ColumnUnweightedBases,
    _ColumnWeightedBases,
    _Means,
    _PopulationProportions,
    _PopulationStandardError,
    _MarginTableProportion,
    _MarginTableBase,
    _ProportionVariances,
    _Pvalues,
    _RowProportions,
    _RowStandardError,
    _RowShareSum,
    _RowUnweightedBases,
    _RowWeightedBases,
    _ScaleMean,
    _ScaleMeanStddev,
    _ScaleMeanStderr,
    _ScaleMedian,
    SecondOrderMeasures,
    _StdDev,
    _MarginWeightedBase,
    _Sums,
    _TotalShareSum,
    _TableProportions,
    _TableStandardError,
    _TableUnweightedBases,
    _TableBase,
    _TableBasesRange,
    _TableWeightedBases,
    _MarginUnweightedBase,
    _UnweightedCounts,
    _WeightedCounts,
    _Zscores,
)

from ...unitutil import (
    ANY,
    class_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class TestAssembler:
    """Unit test suite for `cr.cube.cubepart` assembly methods."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("column_index", _ColumnIndex),
            ("column_proportions", _ColumnProportions),
            ("audience_ratio", _AudienceRatio),
            ("column_proportion_variances", _ProportionVariances),
            ("column_share_sum", _ColumnShareSum),
            ("column_std_err", _ColumnStandardError),
            ("column_unweighted_bases", _ColumnUnweightedBases),
            ("column_weighted_bases", _ColumnWeightedBases),
            ("means", _Means),
            ("population_proportions", _PopulationProportions),
            ("population_std_err", _PopulationStandardError),
            ("pvalues", _Pvalues),
            ("row_proportions", _RowProportions),
            ("row_proportion_variances", _ProportionVariances),
            ("row_share_sum", _RowShareSum),
            ("row_std_err", _RowStandardError),
            ("row_unweighted_bases", _RowUnweightedBases),
            ("row_weighted_bases", _RowWeightedBases),
            ("stddev", _StdDev),
            ("sums", _Sums),
            ("table_proportions", _TableProportions),
            ("table_proportion_variances", _ProportionVariances),
            ("table_std_err", _TableStandardError),
            ("table_unweighted_bases", _TableUnweightedBases),
            ("table_weighted_bases", _TableWeightedBases),
            ("total_share_sum", _TotalShareSum),
            ("weighted_counts", _WeightedCounts),
            ("unweighted_counts", _UnweightedCounts),
            ("zscores", _Zscores),
        ),
    )
    def test_it_assembles_various_measures(
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
        property_mock(request, _Slice, "diff_row_idxs", return_value=())
        property_mock(request, _Slice, "diff_column_idxs", return_value=())
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        slice_ = _Slice(None, None, None, None, None)

        value = getattr(slice_, measure_prop_name)

        _assemble_matrix_.assert_called_once_with(slice_, [["A", "B"], ["C", "D"]])
        assert value == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("rows_scale_mean", _ScaleMean),
            ("columns_scale_mean", _ScaleMean),
            ("rows_scale_mean_stddev", _ScaleMeanStddev),
            ("columns_scale_mean_stddev", _ScaleMeanStddev),
            ("rows_scale_mean_stderr", _ScaleMeanStderr),
            ("columns_scale_mean_stderr", _ScaleMeanStderr),
            ("rows_scale_median", _ScaleMedian),
            ("columns_scale_median", _ScaleMedian),
        ),
    )
    def test_it_assembles_various_marginals(
        self,
        request,
        _assemble_marginal_,
        _measures_prop_,
        second_order_measures_,
        measure_prop_name,
        MeasureCls,
    ):
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        measure_ = instance_mock(request, MeasureCls)
        setattr(second_order_measures_, measure_prop_name, measure_)
        slice_ = _Slice(None, None, None, None, None)

        value = getattr(slice_, measure_prop_name)

        _assemble_marginal_.assert_called_once_with(slice_, measure_)
        assert value == [[1, 2, 3], [4, 5, 6]]

    def test_it_knows_the_column_labels(self, dimension_, _column_order_prop_):
        _column_order_prop_.return_value = [0, 2, 1]
        dimension_.element_labels = ("Alpha", "Baker")
        dimension_.subtotal_labels = ("Charlie",)
        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            column_labels = slice_.column_labels
            assert column_labels.tolist() == ["Alpha", "Charlie", "Baker"]

    def test_it_provides_a_1D_columns_base_for_an_X_CAT_cube_result(
        self,
        _measures_prop_,
        second_order_measures_,
        _assemble_marginal_,
        margin_unweighted_base_,
    ):
        margin_unweighted_base_.is_defined = True
        second_order_measures_.columns_unweighted_base = margin_unweighted_base_
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        slice_ = _Slice(None, None, None, None, None)

        rows_base = slice_.columns_base

        _assemble_marginal_.assert_called_once_with(slice_, margin_unweighted_base_)
        assert rows_base == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_columns_base_for_an_X_MR_cube_result(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        margin_unweighted_base_,
    ):
        margin_unweighted_base_.is_defined = False
        second_order_measures_.columns_unweighted_base = margin_unweighted_base_
        _measures_prop_.return_value = second_order_measures_
        property_mock(
            request,
            _Slice,
            "column_unweighted_bases",
            return_value=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        slice_ = _Slice(None, None, None, None, None)

        rows_base = slice_.columns_base

        assert rows_base == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_it_knows_the_columns_dimension_numeric_values(
        self, request, dimension_, _column_order_prop_
    ):
        dimension_.valid_elements = tuple(
            instance_mock(request, Element, numeric_value=numeric_value)
            for numeric_value in (1, 2, 3)
        )
        _column_order_prop_.return_value = [2, -1, 0, -2]

        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            np.testing.assert_almost_equal(
                slice_._columns_dimension_numeric_values,
                [3.0, np.nan, 1.0, np.nan],
            )

    def test_it_provides_a_1D_columns_margin_for_a_CAT_X_cube_result(
        self,
        _rows_dimension_prop_,
        dimension_,
        _measures_prop_,
        _assemble_marginal_,
        second_order_measures_,
        margin_weighted_base_,
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.CAT
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.columns_weighted_base = margin_weighted_base_
        slice_ = _Slice(None, None, None, None, None)

        columns_margin = slice_.columns_margin

        _assemble_marginal_.assert_called_once_with(slice_, margin_weighted_base_)
        assert columns_margin == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_columns_margin_for_an_MR_X_cube_result(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        margin_weighted_base_,
    ):
        property_mock(
            request, _Slice, "column_weighted_bases", return_value=[[1, 2], [3, 4]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.columns_weighted_base = margin_weighted_base_
        margin_weighted_base_.is_defined = False
        slice_ = _Slice(None, None, None, None, None)

        columns_margin = slice_.columns_margin

        assert columns_margin == [[1, 2], [3, 4]]

    def test_it_provides_a_1D_columns_margin_proportion_for_a_CAT_X_cube_result(
        self,
        _rows_dimension_prop_,
        dimension_,
        _assemble_marginal_,
        _measures_prop_,
        margin_table_proportion_,
        second_order_measures_,
    ):
        margin_table_proportion_.is_defined = True
        _rows_dimension_prop_.return_value = dimension_
        dimension_.dimension_type = DT.CAT
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.columns_table_proportion = margin_table_proportion_
        slice_ = _Slice(None, None, None, None, None)

        columns_margin_proportion = slice_.columns_margin_proportion

        _assemble_marginal_.assert_called_once_with(slice_, margin_table_proportion_)
        assert columns_margin_proportion == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_columns_margin_proportion_for_an_MR_X_cube_result(
        self,
        request,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
        margin_table_proportion_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        margin_table_proportion_.is_defined = False
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.columns_table_proportion = margin_table_proportion_
        property_mock(
            request,
            _Slice,
            "columns_margin",
            return_value=np.array([[1, 2], [3, 4]]),
        )
        property_mock(
            request,
            _Slice,
            "table_weighted_bases",
            return_value=np.array([4.0, 6.0]),
        )
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        with mock.patch("cr.cube.cubepart._Slice._dimensions", new=dimensions_):
            slice_ = _Slice(None, None, None, None, None)

            columns_margin_proportion = slice_.columns_margin_proportion

            SumSubtotals_.blocks.assert_called_once_with(ANY, dimensions_)
            assert SumSubtotals_.blocks.call_args.args[0].tolist() == [
                pytest.approx([1 / 4.0, 2 / 6.0]),
                pytest.approx([3 / 4.0, 4 / 6.0]),
            ]
            _assemble_matrix_.assert_called_once_with(slice_, [[[1], [2]], [[3], [4]]])
            assert columns_margin_proportion == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_it_knows_the_inserted_column_idxs(self, _column_order_prop_):
        _column_order_prop_.return_value = [2, -1, 0, -2]
        assert _Slice(None, None, None, None, None).inserted_column_idxs == (1, 3)

    def test_it_knows_the_inserted_row_idxs(self, _row_order_prop_):
        _row_order_prop_.return_value = [0, 1, -2, 2, -1, 3]
        assert _Slice(None, None, None, None, None).inserted_row_idxs == (2, 4)

    def test_it_knows_the_means(self, request, dimensions_):
        property_mock(request, _Slice, "means", return_value=np.array([1.2, 1.34, 3.3]))
        slice_ = _Slice(None, dimensions_, None, None, None)

        assert slice_.means == pytest.approx([1.2, 1.34, 3.3])

    def test_it_knows_the_sum(self, request, dimensions_):
        property_mock(request, _Slice, "sums", return_value=np.array([4, 5, 6]))
        slice_ = _Slice(None, dimensions_, None, None, None)

        assert slice_.sums == pytest.approx([4, 5, 6])

    def test_it_knows_the_row_labels(self, dimension_, _row_order_prop_):
        _row_order_prop_.return_value = [0, 2, 1]
        dimension_.element_labels = ("Alpha", "Baker")
        dimension_.subtotal_labels = ("Charlie",)
        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            row_labels = slice_.row_labels
            assert row_labels.tolist() == ["Alpha", "Charlie", "Baker"]

    def test_it_provides_a_1D_rows_base_for_an_X_CAT_cube_result(
        self,
        _measures_prop_,
        second_order_measures_,
        _assemble_marginal_,
        margin_unweighted_base_,
    ):
        margin_unweighted_base_.is_defined = True
        second_order_measures_.rows_unweighted_base = margin_unweighted_base_
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        slice_ = _Slice(None, None, None, None, None)

        rows_base = slice_.rows_base

        _assemble_marginal_.assert_called_once_with(slice_, margin_unweighted_base_)
        assert rows_base == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_rows_base_for_an_X_MR_cube_result(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        margin_unweighted_base_,
    ):
        margin_unweighted_base_.is_defined = False
        second_order_measures_.rows_unweighted_base = margin_unweighted_base_
        _measures_prop_.return_value = second_order_measures_
        property_mock(
            request,
            _Slice,
            "row_unweighted_bases",
            return_value=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        slice_ = _Slice(None, None, None, None, None)

        rows_base = slice_.rows_base

        assert rows_base == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @pytest.mark.parametrize(
        "order, expected_fills",
        (
            ((2, -2, 0, -1), ("#f00ba5", "STF1", "#000000", "STF2")),
            ((0, 1, 2, -2, -1), ("#000000", "#111111", "#f00ba5", "STF1", "STF2")),
            ((-2, -1, 0, 1, 2), ("STF1", "STF2", "#000000", "#111111", "#f00ba5")),
            ((-1, -2, 2, 1, 0), ("STF2", "STF1", "#f00ba5", "#111111", "#000000")),
        ),
    )
    def test_it_knows_the_rows_dimension_fills(
        self,
        request,
        _rows_dimension_prop_,
        dimension_,
        _row_order_prop_,
        order,
        expected_fills,
    ):
        element_fills = ("#000000", "#111111", "#f00ba5")
        subtotal_fills = ("STF1", "STF2")
        _rows_dimension_prop_.return_value = dimension_
        dimension_.valid_elements = tuple(
            instance_mock(request, Element, fill=fill) for fill in element_fills
        )
        dimension_.subtotals = tuple(
            instance_mock(request, _Subtotal, fill=fill) for fill in subtotal_fills
        )
        _row_order_prop_.return_value = order
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.rows_dimension_fills == expected_fills

    def test_it_knows_the_rows_dimension_numeric_values(
        self, request, dimension_, _row_order_prop_
    ):
        dimension_.valid_elements = tuple(
            instance_mock(request, Element, numeric_value=numeric_value)
            for numeric_value in (1, 2, 3)
        )
        _row_order_prop_.return_value = [2, -1, 0, -2]

        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            np.testing.assert_almost_equal(
                slice_._rows_dimension_numeric_values,
                [3.0, np.nan, 1.0, np.nan],
            )

    def test_it_provides_a_1D_rows_margin_for_an_X_CAT_cube_result(
        self,
        dimension_,
        _measures_prop_,
        _assemble_marginal_,
        second_order_measures_,
        margin_weighted_base_,
    ):
        dimension_.dimension_type = DT.CAT
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.rows_weighted_base = margin_weighted_base_

        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            rows_margin = slice_.rows_margin
            _assemble_marginal_.assert_called_once_with(slice_, margin_weighted_base_)
            assert rows_margin == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_rows_margin_for_an_X_MR_cube_result(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        margin_weighted_base_,
    ):
        property_mock(
            request, _Slice, "row_weighted_bases", return_value=[[1, 2], [3, 4]]
        )
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.rows_weighted_base = margin_weighted_base_
        margin_weighted_base_.is_defined = False
        slice_ = _Slice(None, None, None, None, None)

        rows_margin = slice_.rows_margin

        assert rows_margin == [[1, 2], [3, 4]]

    def test_it_provides_a_1D_rows_margin_proportion_for_an_X_CAT_cube_result(
        self,
        request,
        dimension_,
        _assemble_marginal_,
        _measures_prop_,
        second_order_measures_,
    ):
        dimension_.dimension_type = DT.CAT
        _assemble_marginal_.return_value = [[1, 2, 3], [4, 5, 6]]
        _measures_prop_.return_value = second_order_measures_
        measure_ = instance_mock(request, _MarginTableProportion)
        second_order_measures_.rows_table_proportion = measure_
        with mock.patch(
            "cr.cube.cubepart._Slice._dimensions", new=(dimension_, dimension_)
        ):
            slice_ = _Slice(None, None, None, None, None)
            rows_margin_proportion = slice_.rows_margin_proportion
            _assemble_marginal_.assert_called_once_with(slice_, measure_)
            assert rows_margin_proportion == [[1, 2, 3], [4, 5, 6]]

    def test_but_it_provides_a_2D_rows_margin_proportion_for_an_X_MR_cube_result(
        self,
        request,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
        margin_table_proportion_,
        SumSubtotals_,
        _assemble_matrix_,
    ):
        margin_table_proportion_.is_defined = False
        _measures_prop_.return_value = second_order_measures_
        second_order_measures_.rows_table_proportion = margin_table_proportion_
        property_mock(
            request, _Slice, "rows_margin", return_value=np.array([[1, 2], [3, 4]])
        )
        property_mock(
            request,
            _Slice,
            "table_weighted_bases",
            return_value=np.array([4.0, 6.0]),
        )
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        with mock.patch("cr.cube.cubepart._Slice._dimensions", new=dimensions_):
            slice_ = _Slice(None, None, None, None, None)

            rows_margin_proportion = slice_.rows_margin_proportion

            SumSubtotals_.blocks.assert_called_once_with(ANY, dimensions_)
            assert SumSubtotals_.blocks.call_args.args[0].tolist() == [
                pytest.approx([1 / 4.0, 2 / 6.0]),
                pytest.approx([3 / 4.0, 4 / 6.0]),
            ]
            _assemble_matrix_.assert_called_once_with(slice_, [[[1], [2]], [[3], [4]]])
            assert rows_margin_proportion == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_it_knows_the_2D_table_base_of_an_ARRAY_X_ARRAY_matrix(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        table_unweighted_base_,
        rows_table_unweighted_base_,
        columns_table_unweighted_base_,
    ):
        table_unweighted_base_.is_defined = False
        rows_table_unweighted_base_.is_defined = False
        columns_table_unweighted_base_.is_defined = False
        second_order_measures_.table_unweighted_base = table_unweighted_base_
        second_order_measures_.rows_table_unweighted_base = rows_table_unweighted_base_
        second_order_measures_.columns_table_unweighted_base = (
            columns_table_unweighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        property_mock(
            request,
            _Slice,
            "table_unweighted_bases",
            return_value=[[5, 4, 6], [2, 1, 3]],
        )
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_base == [[5, 4, 6], [2, 1, 3]]

    def test_and_it_knows_the_1D_table_base_of_an_ARRAY_X_CAT_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_unweighted_base_,
        columns_table_unweighted_base_,
        _assemble_marginal_,
    ):
        table_unweighted_base_.is_defined = False
        columns_table_unweighted_base_.is_defined = True
        second_order_measures_.table_unweighted_base = table_unweighted_base_
        second_order_measures_.columns_table_unweighted_base = (
            columns_table_unweighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        _assemble_marginal_.return_value = [2, 1, 3]
        slice_ = _Slice(None, None, None, None, None)

        table_base = slice_.table_base

        _assemble_marginal_.assert_called_once_with(
            slice_, columns_table_unweighted_base_
        )
        assert table_base == [2, 1, 3]

    def test_and_it_knows_the_1D_table_base_of_a_CAT_X_ARRAY_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_unweighted_base_,
        rows_table_unweighted_base_,
        columns_table_unweighted_base_,
        _assemble_marginal_,
    ):
        table_unweighted_base_.is_defined = False
        rows_table_unweighted_base_.is_defined = True
        columns_table_unweighted_base_.is_defined = False
        second_order_measures_.table_unweighted_base = table_unweighted_base_
        second_order_measures_.rows_table_unweighted_base = rows_table_unweighted_base_
        second_order_measures_.columns_table_unweighted_base = (
            columns_table_unweighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        _assemble_marginal_.return_value = [2, 1, 3]
        slice_ = _Slice(None, None, None, None, None)

        table_base = slice_.table_base

        _assemble_marginal_.assert_called_once_with(slice_, rows_table_unweighted_base_)
        assert table_base == [2, 1, 3]

    def test_and_it_knows_the_scalar_table_base_of_a_CAT_X_CAT_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_unweighted_base_,
    ):
        table_unweighted_base_.is_defined = True
        table_unweighted_base_.value = 4242
        second_order_measures_.table_unweighted_base = table_unweighted_base_
        _measures_prop_.return_value = second_order_measures_
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_base == 4242

    def test_it_knows_the_2D_table_margin_of_an_ARRAY_X_ARRAY_matrix(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
        table_weighted_base_,
        rows_table_weighted_base_,
        columns_table_weighted_base_,
    ):
        table_weighted_base_.is_defined = False
        rows_table_weighted_base_.is_defined = False
        columns_table_weighted_base_.is_defined = False
        second_order_measures_.table_weighted_base = table_weighted_base_
        second_order_measures_.rows_table_weighted_base = rows_table_weighted_base_
        second_order_measures_.columns_table_weighted_base = (
            columns_table_weighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        property_mock(
            request,
            _Slice,
            "table_weighted_bases",
            return_value=[[5, 4, 6], [2, 1, 3]],
        )
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_margin == [[5, 4, 6], [2, 1, 3]]

    def test_and_it_knows_the_1D_table_margin_of_an_ARRAY_X_CAT_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_weighted_base_,
        columns_table_weighted_base_,
        _assemble_marginal_,
    ):
        table_weighted_base_.is_defined = False
        columns_table_weighted_base_.is_defined = True
        second_order_measures_.table_weighted_base = table_weighted_base_
        second_order_measures_.columns_table_weighted_base = (
            columns_table_weighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        _assemble_marginal_.return_value = [2, 1, 3]
        slice_ = _Slice(None, None, None, None, None)

        table_margin = slice_.table_margin

        _assemble_marginal_.assert_called_once_with(
            slice_, columns_table_weighted_base_
        )
        assert table_margin == [2, 1, 3]

    def test_and_it_knows_the_1D_table_margin_of_a_CAT_X_ARRAY_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_weighted_base_,
        rows_table_weighted_base_,
        columns_table_weighted_base_,
        _assemble_marginal_,
    ):
        table_weighted_base_.is_defined = False
        rows_table_weighted_base_.is_defined = True
        columns_table_weighted_base_.is_defined = False
        second_order_measures_.table_weighted_base = table_weighted_base_
        second_order_measures_.rows_table_weighted_base = rows_table_weighted_base_
        second_order_measures_.columns_table_weighted_base = (
            columns_table_weighted_base_
        )
        _measures_prop_.return_value = second_order_measures_
        _assemble_marginal_.return_value = [2, 1, 3]
        slice_ = _Slice(None, None, None, None, None)

        table_margin = slice_.table_margin

        _assemble_marginal_.assert_called_once_with(slice_, rows_table_weighted_base_)
        assert table_margin == [2, 1, 3]

    def test_and_it_knows_the_scalar_table_margin_of_a_CAT_X_CAT_matrix(
        self,
        _measures_prop_,
        second_order_measures_,
        table_weighted_base_,
    ):
        table_weighted_base_.is_defined = True
        table_weighted_base_.value = 4242
        second_order_measures_.table_weighted_base = table_weighted_base_
        _measures_prop_.return_value = second_order_measures_
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_margin == 4242

    def test_it_knows_the_table_base_range(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        measure_ = instance_mock(request, _TableBasesRange, value=42)
        second_order_measures_.table_unweighted_bases_range = measure_
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_base_range == 42

    def test_it_knows_the_table_margin_range(
        self,
        request,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        measure_ = instance_mock(request, _TableBasesRange, value=42)
        second_order_measures_.table_weighted_bases_range = measure_
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.table_margin_range == 42

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
    def test_it_can_assemble_a_matrix_from_blocks(
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
        slice_ = _Slice(None, None, None, None, None)

        assert slice_._assemble_matrix(blocks).tolist() == expected_value

    def test_it_can_assemble_a_vector(self, request):
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
        slice_ = _Slice(None, None, None, None, None)

        vector = slice_._assemble_vector(base_vector, subtotals_, order)

        assert vector.tolist() == [3, 2, 1, 5, 4, 3, 7]

    def test_it_knows_the_column_order_to_help(
        self,
        _BaseOrderHelper_,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        _BaseOrderHelper_.column_display_order.return_value = np.array(
            [-1, 1, -2, 2, -3, 3]
        )
        with mock.patch("cr.cube.cubepart._Slice._dimensions", new=dimensions_):
            slice_ = _Slice(None, None, None, None, None)

            column_order = slice_._column_order_signed_indexes

            _BaseOrderHelper_.column_display_order.assert_called_once_with(
                dimensions_, second_order_measures_, ORDER_FORMAT.SIGNED_INDEXES
            )
            assert column_order.tolist() == [-1, 1, -2, 2, -3, 3]

    @pytest.mark.parametrize(
        "format, row_order",
        (
            (ORDER_FORMAT.SIGNED_INDEXES, (-1, 1, -2, 2, -3, 3)),
            (ORDER_FORMAT.BOGUS_IDS, ("ins_1", 1, "ins_2", 2, "ins_3", 3)),
        ),
    )
    def test_it_knows_the_row_order_to_help(
        self,
        _BaseOrderHelper_,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
        format,
        row_order,
    ):
        _measures_prop_.return_value = second_order_measures_
        _BaseOrderHelper_.row_display_order.return_value = np.array(row_order)
        with mock.patch("cr.cube.cubepart._Slice._dimensions", new=dimensions_):
            slice_ = _Slice(None, dimensions_, None, None, None)

            row_order = slice_.row_order(format)

            _BaseOrderHelper_.row_display_order.assert_called_once_with(
                dimensions_, second_order_measures_, format
            )
            assert row_order.tolist() == list(row_order)

    def test_it_knows_the_payload_order_to_help(
        self,
        request,
        dimensions_,
        _measures_prop_,
        second_order_measures_,
    ):
        _measures_prop_.return_value = second_order_measures_
        _OrderHelper_ = class_mock(
            request, "cr.cube.matrix.assembler.PayloadOrderCollator"
        )
        _OrderHelper_.display_order.return_value = np.array([1, 2, 3])
        with mock.patch("cr.cube.cubepart._Slice._dimensions", new=dimensions_):
            slice_ = _Slice(None, None, None, None, None)

            row_order = slice_.row_order()

            _OrderHelper_.display_order.assert_called_once_with(
                dimensions_[0], (), ORDER_FORMAT.SIGNED_INDEXES
            )

            assert row_order.tolist() == [1, 2, 3]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assemble_marginal_(self, request):
        return method_mock(request, _Slice, "_assemble_marginal")

    @pytest.fixture
    def _assemble_matrix_(self, request):
        return method_mock(request, _Slice, "_assemble_matrix")

    @pytest.fixture
    def _BaseOrderHelper_(self, request):
        return class_mock(request, "cr.cube.cubepart._BaseOrderHelper")

    @pytest.fixture
    def _column_order_prop_(self, request):
        return property_mock(request, _Slice, "_column_order_signed_indexes")

    @pytest.fixture
    def columns_table_unweighted_base_(self, request):
        return instance_mock(request, _MarginTableBase)

    @pytest.fixture
    def columns_table_weighted_base_(self, request):
        return instance_mock(request, _MarginTableBase)

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def margin_table_proportion_(self, request):
        return instance_mock(request, _MarginTableProportion)

    @pytest.fixture
    def margin_unweighted_base_(self, request):
        return instance_mock(request, _MarginUnweightedBase)

    @pytest.fixture
    def margin_weighted_base_(self, request):
        return instance_mock(request, _MarginWeightedBase)

    @pytest.fixture
    def _measures_prop_(self, request):
        return property_mock(request, _Slice, "_measures")

    @pytest.fixture
    def _row_order_prop_(self, request):
        return property_mock(request, _Slice, "_row_order_signed_indexes")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _Slice, "_rows_dimension")

    @pytest.fixture
    def rows_table_unweighted_base_(self, request):
        return instance_mock(request, _MarginTableBase)

    @pytest.fixture
    def rows_table_weighted_base_(self, request):
        return instance_mock(request, _MarginTableBase)

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.cubepart.SumSubtotals")

    @pytest.fixture
    def table_unweighted_base_(self, request):
        return instance_mock(request, _TableBase)

    @pytest.fixture
    def table_weighted_base_(self, request):
        return instance_mock(request, _TableBase)


class Test_BaseOrderHelper:
    """Unit test suite for `cr.cube.matrix.assembler._BaseOrderHelper` object."""

    def test_it_dispatches_to_the_right_column_order_helper(
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
            dimensions_, second_order_measures_, ORDER_FORMAT.SIGNED_INDEXES
        )

        _ColumnOrderHelper_.assert_called_once_with(
            dimensions_, second_order_measures_, ORDER_FORMAT.SIGNED_INDEXES
        )
        assert column_order.tolist() == [-2, 1, -1, 2]

    @pytest.mark.parametrize(
        "collation_method, HelperCls",
        (
            (CM.EXPLICIT_ORDER, _RowOrderHelper),
            (CM.LABEL, _SortRowsByLabelHelper),
            (CM.MARGINAL, _SortRowsByMarginalHelper),
            (CM.OPPOSING_ELEMENT, _SortRowsByBaseColumnHelper),
            (CM.PAYLOAD_ORDER, _RowOrderHelper),
        ),
    )
    def test_it_dispatches_to_the_right_row_order_helper(
        self, request, dimensions_, second_order_measures_, collation_method, HelperCls
    ):
        dimensions_[0].order_spec = instance_mock(
            request, _OrderSpec, collation_method=collation_method
        )
        helper_ = instance_mock(
            request, HelperCls, _display_order=np.array([-1, 1, -2, 2])
        )
        HelperCls_ = class_mock(
            request,
            "cr.cube.matrix.assembler.%s" % HelperCls.__name__,
            return_value=helper_,
        )

        row_order = _BaseOrderHelper.row_display_order(
            dimensions_, second_order_measures_, ORDER_FORMAT.SIGNED_INDEXES
        )

        HelperCls_.assert_called_once_with(
            dimensions_, second_order_measures_, ORDER_FORMAT.SIGNED_INDEXES
        )
        assert row_order.tolist() == [-1, 1, -2, 2]

    def test_it_provides_access_to_the_columns_dimension_to_help(self, dimension_):
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
    def test_it_post_processes_the_display_order_to_help(
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
        (
            ([False, False, False], ()),
            ([False, True, False], (1,)),
            ([True, True, True], (0, 1, 2)),
        ),
    )
    def test_it_knows_its_empty_column_idxs_to_help(
        self, second_order_measures_, base, expected_value
    ):
        second_order_measures_.columns_pruning_mask = np.array(base)
        order_helper = _BaseOrderHelper(None, second_order_measures_)

        assert order_helper._empty_column_idxs == expected_value

    @pytest.mark.parametrize(
        "base, expected_value",
        (
            ([False, False, False], ()),
            ([False, True, False], (1,)),
            ([True, True, True], (0, 1, 2)),
        ),
    )
    def test_it_knows_its_empty_row_idxs_to_help(
        self, second_order_measures_, base, expected_value
    ):
        second_order_measures_.rows_pruning_mask = np.array(base)
        order_helper = _BaseOrderHelper(None, second_order_measures_)

        assert order_helper._empty_row_idxs == expected_value

    @pytest.mark.parametrize(
        "json_name, internal_name",
        (
            ("col_base_unweighted", "column_unweighted_bases"),
            ("col_base_weighted", "column_weighted_bases"),
            ("col_index", "column_index"),
            ("col_percent", "column_proportions"),
            ("col_percent_moe", "column_std_err"),
            ("col_share_sum", "column_share_sum"),
            ("col_std_dev", "column_proportion_variances"),
            ("col_std_err", "column_std_err"),
            ("count_unweighted", "unweighted_counts"),
            ("count_weighted", "weighted_counts"),
            ("mean", "means"),
            ("population", "population_proportions"),
            ("population_moe", "population_std_err"),
            ("p_value", "pvalues"),
            ("row_base_unweighted", "row_unweighted_bases"),
            ("row_base_weighted", "row_weighted_bases"),
            ("row_percent", "row_proportions"),
            ("row_percent_moe", "row_std_err"),
            ("row_share_sum", "row_share_sum"),
            ("row_std_dev", "row_proportion_variances"),
            ("row_std_err", "row_std_err"),
            ("stddev", "stddev"),
            ("sum", "sums"),
            ("table_base_unweighted", "table_unweighted_bases"),
            ("table_base_weighted", "table_weighted_bases"),
            ("table_percent", "table_proportions"),
            ("table_percent_moe", "table_std_err"),
            ("table_std_dev", "table_proportion_variances"),
            ("table_std_err", "table_std_err"),
            ("total_share_sum", "total_share_sum"),
            ("valid_count_unweighted", "unweighted_counts"),
            ("valid_count_weighted", "weighted_counts"),
            ("z_score", "zscores"),
        ),
    )
    def test_it_retrieves_the_measure_object_to_help(
        self, request, second_order_measures_, json_name, internal_name
    ):
        property_mock(
            request,
            _BaseOrderHelper,
            "_order_spec",
            return_value=instance_mock(request, _OrderSpec, measure=M(json_name)),
        )
        measure_ = instance_mock(request, _BaseSecondOrderMeasure)
        setattr(second_order_measures_, internal_name, measure_)
        order_helper = _BaseOrderHelper(None, second_order_measures_)

        assert order_helper._measure is measure_

    def test_it_provides_access_to_the_rows_dimension_to_help(self, dimension_):
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


class Test_ColumnOrderHelper:
    """Unit test suite for `cr.cube.matrix.assembler._ColumnOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def test_it_computes_the_order_of_a_columns_dimension_to_help(
        self, request, dimension_, collation_method, collator_class_name
    ):
        property_mock(
            request, _ColumnOrderHelper, "_columns_dimension", return_value=dimension_
        )
        dimension_.order_spec = instance_mock(
            request, _OrderSpec, collation_method=collation_method
        )
        CollatorCls_ = class_mock(
            request, "cr.cube.matrix.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (3, -1, 5, 1, -2)
        property_mock(
            request, _ColumnOrderHelper, "_empty_column_idxs", return_value=(1, 3)
        )
        order_helper = _ColumnOrderHelper(None, None)

        order = order_helper._order

        CollatorCls_.display_order.assert_called_once_with(
            dimension_, (1, 3), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (3, -1, 5, 1, -2)

    @pytest.mark.parametrize(
        "prune, empty_row_idxs, expected_value",
        (
            (False, None, False),
            (True, (3,), False),
            (True, (0, 1, 2), True),
        ),
    )
    def test_it_knows_whether_to_prune_the_subtotal_columns_to_help(
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


class Test_RowOrderHelper:
    """Unit test suite for `cr.cube.matrix.assembler._RowOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def test_it_computes_the_order_of_a_rows_dimension_to_help(
        self, request, dimension_, collation_method, collator_class_name
    ):
        property_mock(
            request, _RowOrderHelper, "_rows_dimension", return_value=dimension_
        )
        dimension_.order_spec = instance_mock(
            request, _OrderSpec, collation_method=collation_method
        )
        CollatorCls_ = class_mock(
            request, "cr.cube.matrix.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        property_mock(
            request, _RowOrderHelper, "_empty_row_idxs", return_value=(2, 4, 6)
        )
        order_helper = _RowOrderHelper(None, None)

        order = order_helper._order

        CollatorCls_.display_order.assert_called_once_with(
            dimension_, (2, 4, 6), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (1, -2, 3, 5, -1)

    def test_it_provides_access_to_the_order_spec_to_help(self, request, dimension_):
        property_mock(
            request, _RowOrderHelper, "_rows_dimension", return_value=dimension_
        )
        order_spec_ = instance_mock(request, _OrderSpec)
        dimension_.order_spec = order_spec_
        order_helper = _SortRowsByBaseColumnHelper(None, None)

        assert order_helper._order_spec is order_spec_

    @pytest.mark.parametrize(
        "prune, empty_column_idxs, expected_value",
        (
            (False, None, False),
            (True, (3,), False),
            (True, (0, 1, 2), True),
        ),
    )
    def test_it_knows_whether_to_prune_the_subtotal_rows_to_help(
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


class Test_BaseSortRowsByValueHelper:
    """Unit test suite for `cr.cube.matrix.assembler._BaseSortRowsByValueHelper`."""

    def test_it_provides_the_order(
        self,
        SortByValueCollator_,
        _rows_dimension_prop_,
        _element_values_prop_,
        _subtotal_values_prop_,
        _empty_row_idxs_prop_,
    ):
        _BaseSortRowsByValueHelper(None, None)._order

        SortByValueCollator_.display_order.assert_called_once_with(
            _rows_dimension_prop_(),
            _element_values_prop_(),
            _subtotal_values_prop_(),
            _empty_row_idxs_prop_(),
            ORDER_FORMAT.SIGNED_INDEXES,
        )

    def test_but_it_falls_back_to_payload_order_on_value_error(
        self,
        request,
        dimensions_,
        _element_values_prop_,
        _subtotal_values_prop_,
        _empty_row_idxs_prop_,
        SortByValueCollator_,
    ):
        _element_values_prop_.return_value = None
        _subtotal_values_prop_.return_value = None
        _empty_row_idxs_prop_.return_value = (4, 2)
        SortByValueCollator_.display_order.side_effect = ValueError
        PayloadOrderCollator_ = class_mock(
            request, "cr.cube.matrix.assembler.PayloadOrderCollator"
        )
        PayloadOrderCollator_.display_order.return_value = (1, 2, 3, 4)
        order_helper = _BaseSortRowsByValueHelper(dimensions_, None)

        order = order_helper._order

        PayloadOrderCollator_.display_order.assert_called_once_with(
            dimensions_[0], (4, 2), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (1, 2, 3, 4)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def _element_values_prop_(self, request):
        return property_mock(request, _BaseSortRowsByValueHelper, "_element_values")

    @pytest.fixture
    def _empty_row_idxs_prop_(self, request):
        return property_mock(request, _BaseSortRowsByValueHelper, "_empty_row_idxs")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _BaseSortRowsByValueHelper, "_rows_dimension")

    @pytest.fixture
    def SortByValueCollator_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.SortByValueCollator")

    @pytest.fixture
    def _subtotal_values_prop_(self, request):
        return property_mock(request, _BaseSortRowsByValueHelper, "_subtotal_values")


class Test_BaseSortColumnsByValueHelper:
    """Unit test suite for `cr.cube.matrix.assembler._BaseSortColumnsByValueHelper`."""

    def test_it_provides_the_order(
        self,
        SortByValueCollator_,
        _columns_dimension_prop_,
        _element_values_prop_,
        _subtotal_values_prop_,
        _empty_column_idxs_prop_,
    ):
        _BaseSortColumnsByValueHelper(None, None)._order

        SortByValueCollator_.display_order.assert_called_once_with(
            _columns_dimension_prop_(),
            _element_values_prop_(),
            _subtotal_values_prop_(),
            _empty_column_idxs_prop_(),
            ORDER_FORMAT.SIGNED_INDEXES,
        )

    def test_but_it_falls_back_to_payload_order_on_value_error(
        self,
        request,
        dimensions_,
        _element_values_prop_,
        _subtotal_values_prop_,
        _empty_column_idxs_prop_,
        SortByValueCollator_,
    ):
        _element_values_prop_.return_value = None
        _subtotal_values_prop_.return_value = None
        _empty_column_idxs_prop_.return_value = (4, 2)
        SortByValueCollator_.display_order.side_effect = ValueError
        PayloadOrderCollator_ = class_mock(
            request, "cr.cube.matrix.assembler.PayloadOrderCollator"
        )
        PayloadOrderCollator_.display_order.return_value = (1, 2, 3, 4)
        order_helper = _BaseSortColumnsByValueHelper(dimensions_, None)

        order = order_helper._order

        PayloadOrderCollator_.display_order.assert_called_once_with(
            dimensions_[1], (4, 2), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (1, 2, 3, 4)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def _element_values_prop_(self, request):
        return property_mock(request, _BaseSortColumnsByValueHelper, "_element_values")

    @pytest.fixture
    def _empty_column_idxs_prop_(self, request):
        return property_mock(
            request, _BaseSortColumnsByValueHelper, "_empty_column_idxs"
        )

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(
            request, _BaseSortColumnsByValueHelper, "_columns_dimension"
        )

    @pytest.fixture
    def SortByValueCollator_(self, request):
        return class_mock(request, "cr.cube.matrix.assembler.SortByValueCollator")

    @pytest.fixture
    def _subtotal_values_prop_(self, request):
        return property_mock(request, _BaseSortColumnsByValueHelper, "_subtotal_values")


class Test_SortRowsByBaseColumnHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByBaseColumnHelper`."""

    def test_it_derives_the_sort_column_idx_from_the_order_spec_to_help(
        self, _columns_dimension_prop_, dimension_, _order_spec_prop_, order_spec_
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.element_ids = (1, 2, 3, 4, 5)
        dimension_.translate_element_id.return_value = 3
        _order_spec_prop_.return_value = order_spec_
        order_spec_.element_id = "c"
        order_helper = _SortRowsByBaseColumnHelper(None, None)

        column_idx = order_helper._column_idx

        dimension_.translate_element_id.assert_called_once_with("c")
        assert column_idx == 2

    def test_it_extracts_the_element_values_to_help(
        self, _measure_prop_, measure_, _column_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[np.arange(20).reshape(4, 5), None], [None, None]]
        _column_idx_prop_.return_value = 2
        order_helper = _SortRowsByBaseColumnHelper(None, None)

        assert order_helper._element_values.tolist() == [2, 7, 12, 17]

    def test_but_it_raises_when_an_unsupported_sort_by_value_measure_is_requested(
        self, _order_spec_prop_, order_spec_
    ):
        _order_spec_prop_.return_value = order_spec_
        order_spec_.measure = "foo"
        order_helper = _SortRowsByBaseColumnHelper(None, None)

        with pytest.raises(NotImplementedError) as e:
            order_helper._measure

        assert str(e.value) == ("sort-by-value for measure 'foo' is not yet supported")

    def test_it_extracts_the_subtotal_values_to_help(
        self, _measure_prop_, measure_, _column_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[None, None], [np.arange(10, 101, 10).reshape(2, 5), None]]
        _column_idx_prop_.return_value = 2
        order_helper = _SortRowsByBaseColumnHelper(None, None)

        assert order_helper._subtotal_values.tolist() == [30, 80]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _column_idx_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_column_idx")

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_columns_dimension")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def _measure_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_measure")

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_order_spec")


class Test_SortRowsByDerivedColumnHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByDerivedColumnHelper`."""

    def test_it_derives_the_sort_column_idx_from_the_order_spec_to_help(
        self, _columns_dimension_prop_, dimension_, _order_spec_prop_, order_spec_
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.element_ids = (1, 2, 3, 4, 5)
        dimension_.translate_element_id.return_value = 3
        _order_spec_prop_.return_value = order_spec_
        order_spec_.insertion_id = "c"
        order_helper = _SortRowsByDerivedColumnHelper(None, None)

        column_idx = order_helper._column_idx

        dimension_.translate_element_id.assert_called_once_with("c")
        assert column_idx == 2

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_columns_dimension")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortRowsByBaseColumnHelper, "_order_spec")


class Test_SortRowsByInsertedColumnHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByInsertedColumnHelper`"""

    def test_it_extracts_the_element_values_to_help(
        self, _measure_prop_, measure_, _insertion_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[None, np.arange(12).reshape(4, 3)], [None, None]]
        _insertion_idx_prop_.return_value = 1
        order_helper = _SortRowsByInsertedColumnHelper(None, None)

        assert order_helper._element_values.tolist() == [1, 4, 7, 10]

    def test_it_derives_the_sort_insertion_idx_from_the_order_spec_to_help(
        self, _columns_dimension_prop_, dimension_, _order_spec_prop_, order_spec_
    ):
        _columns_dimension_prop_.return_value = dimension_
        dimension_.insertion_ids = (1, 2, 3)
        _order_spec_prop_.return_value = order_spec_
        order_spec_.insertion_id = 2
        order_helper = _SortRowsByInsertedColumnHelper(None, None)

        assert order_helper._insertion_idx == 1

    def test_it_extracts_the_subtotal_values_to_help(
        self, _measure_prop_, measure_, _insertion_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[None, None], [None, np.arange(10, 61, 10).reshape(2, 3)]]
        _insertion_idx_prop_.return_value = 1
        order_helper = _SortRowsByInsertedColumnHelper(None, None)

        assert order_helper._subtotal_values.tolist() == [20, 50]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(
            request, _SortRowsByInsertedColumnHelper, "_columns_dimension"
        )

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _insertion_idx_prop_(self, request):
        return property_mock(request, _SortRowsByInsertedColumnHelper, "_insertion_idx")

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def _measure_prop_(self, request):
        return property_mock(request, _SortRowsByInsertedColumnHelper, "_measure")

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortRowsByInsertedColumnHelper, "_order_spec")


class Test_SortRowsByLabelHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByLabelHelper`."""

    def test_it_provides_the_element_values_to_help(self, dimensions_):
        dimensions_[0].element_labels = ("c", "a", "b")

        assert _SortRowsByLabelHelper(dimensions_, None)._element_values.tolist() == [
            "c",
            "a",
            "b",
        ]

    def test_it_provides_the_subtotal_values_to_help(self, dimensions_):
        dimensions_[0].subtotal_labels = ("c", "a", "b")

        assert _SortRowsByLabelHelper(dimensions_, None)._subtotal_values.tolist() == [
            "c",
            "a",
            "b",
        ]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))


class Test_SortColumnsByBaseRowHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortColumnsByBaseRowHelper`."""

    def test_it_derives_the_sort_column_idx_from_the_order_spec_to_help(
        self, _rows_dimension_prop_, dimension_, _order_spec_prop_, order_spec_
    ):
        _rows_dimension_prop_.return_value = dimension_
        dimension_.element_ids = (1, 2, 3, 4, 5)
        dimension_.translate_element_id.return_value = 3
        _order_spec_prop_.return_value = order_spec_
        order_spec_.element_id = "c"
        order_helper = _SortColumnsByBaseRowHelper(None, None)

        row_idx = order_helper._row_idx

        dimension_.translate_element_id.assert_called_once_with("c")
        assert row_idx == 2

    def test_it_extracts_the_element_values_to_help(
        self, _measure_prop_, measure_, _row_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[np.arange(20).reshape(4, 5), None], [None, None]]
        _row_idx_prop_.return_value = 2
        order_helper = _SortColumnsByBaseRowHelper(None, None)

        assert order_helper._element_values.tolist() == [10, 11, 12, 13, 14]

    def test_but_it_raises_when_an_unsupported_sort_by_value_measure_is_requested(
        self, _order_spec_prop_, order_spec_
    ):
        _order_spec_prop_.return_value = order_spec_
        order_spec_.measure = "foo"
        order_helper = _SortColumnsByBaseRowHelper(None, None)

        with pytest.raises(NotImplementedError) as e:
            order_helper._measure

        assert str(e.value) == ("sort-by-value for measure 'foo' is not yet supported")

    def test_it_extracts_the_subtotal_values_to_help222(
        self, _measure_prop_, measure_, _row_idx_prop_
    ):
        _measure_prop_.return_value = measure_
        measure_.blocks = [[None, np.arange(10, 101, 10).reshape(2, 5)], [None, None]]
        _row_idx_prop_.return_value = 1
        order_helper = _SortColumnsByBaseRowHelper(None, None)

        assert order_helper._subtotal_values.tolist() == [60, 70, 80, 90, 100]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _row_idx_prop_(self, request):
        return property_mock(request, _SortColumnsByBaseRowHelper, "_row_idx")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _SortColumnsByBaseRowHelper, "_rows_dimension")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def _measure_prop_(self, request):
        return property_mock(request, _SortColumnsByBaseRowHelper, "_measure")

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortColumnsByBaseRowHelper, "_order_spec")


class Test_SortColumnsByLabelHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortColumnsByLabelHelper`."""

    def test_it_provides_the_element_values_to_help(self, dimensions_):
        dimensions_[1].element_labels = ("c", "a", "b")

        assert _SortColumnsByLabelHelper(
            dimensions_, None
        )._element_values.tolist() == [
            "c",
            "a",
            "b",
        ]

    def test_it_provides_the_subtotal_values_to_help(self, dimensions_):
        dimensions_[1].subtotal_labels = ("c", "a", "b")

        assert _SortColumnsByLabelHelper(
            dimensions_, None
        )._subtotal_values.tolist() == [
            "c",
            "a",
            "b",
        ]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))


class Test_SortRowsByMarginalHelper:
    """Unit test suite for `cr.cube.matrix.assembler._SortRowsByMarginalHelper`."""

    def test_it_provides_the_element_values_to_help(self, _marginal_prop_, marginal_):
        marginal_.blocks = ["a", "b"]
        _marginal_prop_.return_value = marginal_

        assert _SortRowsByMarginalHelper(None, None)._element_values == "a"

    @pytest.mark.parametrize(
        "marginal, marginal_prop_name",
        (
            (MARGINAL.BASE, "rows_unweighted_base"),
            (MARGINAL.MARGIN, "rows_weighted_base"),
            (MARGINAL.MARGIN_PROPORTION, "rows_table_proportion"),
            (MARGINAL.SCALE_MEAN, "rows_scale_mean"),
            (MARGINAL.SCALE_MEAN_STDDEV, "rows_scale_mean_stddev"),
            (MARGINAL.SCALE_MEAN_STDERR, "rows_scale_mean_stderr"),
            (MARGINAL.SCALE_MEDIAN, "rows_scale_median"),
        ),
    )
    def test_it_provides_the_marginal_to_help(
        self,
        second_order_measures_,
        marginal,
        marginal_prop_name,
        _order_spec_,
        _order_spec_prop_,
    ):
        setattr(second_order_measures_, marginal_prop_name, "foo")
        _order_spec_.marginal = marginal
        _order_spec_prop_.return_value = _order_spec_
        order_helper = _SortRowsByMarginalHelper(None, second_order_measures_)

        assert order_helper._marginal == "foo"

    def test_but_it_raises_on_unknown_marginal(
        self, second_order_measures_, _order_spec_, _order_spec_prop_
    ):
        _order_spec_.marginal = "bar"
        _order_spec_prop_.return_value = _order_spec_
        order_helper = _SortRowsByMarginalHelper(None, second_order_measures_)

        with pytest.raises(NotImplementedError) as e:
            order_helper._marginal

        assert str(e.value) == "sort-by-value for marginal 'bar' is not yet supported"

    def test_it_provides_the_subtotal_values_to_help(self, _marginal_prop_, marginal_):
        marginal_.blocks = ["a", "b"]
        _marginal_prop_.return_value = marginal_

        assert _SortRowsByMarginalHelper(None, None)._subtotal_values == "b"

    # fixture components ---------------------------------------------

    @pytest.fixture
    def marginal_(self, request):
        return instance_mock(request, _BaseMarginal)

    @pytest.fixture
    def _marginal_prop_(self, request):
        return property_mock(request, _SortRowsByMarginalHelper, "_marginal")

    @pytest.fixture
    def _order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortRowsByMarginalHelper, "_order_spec")

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)
