# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.measure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT, MARGINAL_ORIENTATION as MO
from cr.cube.matrix.cubemeasure import (
    CubeMeasures,
    _BaseCubeCounts,
    _BaseUnweightedCubeCounts,
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
    _PairwiseSigPvals,
    _PairwiseSigTstats,
    _PairwiseMeansSigPVals,
    _PairwiseMeansSigTStats,
    _PopulationProportions,
    _MarginTableProportion,
    _MarginTableWeightedBase,
    _RowComparableCounts,
    _RowProportions,
    _RowShareSum,
    _RowUnweightedBases,
    _RowWeightedBases,
    _ScaleMean,
    _ScaleMeanStddev,
    _ScaleMedian,
    SecondOrderMeasures,
    _MarginWeightedBase,
    _Sums,
    _TableProportionVariances,
    _TableProportions,
    _TableStandardError,
    _TableUnweightedBases,
    _TableWeightedBase,
    _TableWeightedBases,
    _TableWeightedBasesRange,
    _TotalShareSum,
    _MarginUnweightedBase,
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
            ("population_proportions", _PopulationProportions),
            ("row_proportions", _RowProportions),
            ("row_unweighted_bases", _RowUnweightedBases),
            ("row_weighted_bases", _RowWeightedBases),
            ("table_proportion_variances", _TableProportionVariances),
            ("table_std_err", _TableStandardError),
            ("table_proportions", _TableProportions),
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
            ("rows", "rows_weighted_base", _MarginWeightedBase),
            ("columns", "columns_weighted_base", _MarginWeightedBase),
            ("rows", "rows_table_proportion", _MarginTableProportion),
            ("columns", "columns_table_proportion", _MarginTableProportion),
            ("rows", "rows_scale_median", _ScaleMedian),
            ("columns", "columns_scale_median", _ScaleMedian),
            ("rows", "rows_scale_mean", _ScaleMean),
            ("columns", "columns_scale_mean", _ScaleMean),
            ("rows", "rows_scale_mean_stddev", _ScaleMeanStddev),
            ("columns", "columns_scale_mean_stddev", _ScaleMeanStddev),
        ),
    )
    def it_provides_access_to_the_marginals(
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

    def it_provides_access_to_the_table_weighted_base_scalar(
        self,
        request,
        dimensions_,
        _cube_measures_prop_,
        cube_measures_,
    ):
        table_weighted_base_ = instance_mock(request, _TableWeightedBase)
        _TableWeightedBase_ = class_mock(
            request,
            "cr.cube.matrix.measure._TableWeightedBase",
            return_value=table_weighted_base_,
        )
        _cube_measures_prop_.return_value = cube_measures_
        measures = SecondOrderMeasures(None, dimensions_, None)

        table_weighted_base = measures.table_weighted_base

        _TableWeightedBase_.assert_called_once_with(
            dimensions_, measures, cube_measures_
        )
        assert table_weighted_base is table_weighted_base_

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("pairwise_t_stats", _PairwiseSigTstats),
            ("pairwise_p_vals", _PairwiseSigPvals),
            ("pairwise_significance_means_p_vals", _PairwiseMeansSigPVals),
            ("pairwise_significance_means_t_stats", _PairwiseMeansSigTStats),
        ),
    )
    def it_provides_access_to_pairwise_significance_measure_objects(
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
        second_order_measures = SecondOrderMeasures(None, dimensions_, None)
        col_idx = 0

        pairwise_sig_measure = getattr(second_order_measures, measure_prop_name)

        assert pairwise_sig_measure(col_idx) is measure_
        MeasureCls_.assert_called_once_with(
            dimensions_, second_order_measures, cube_measures_, False
        )

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
        cube_counts_ = instance_mock(request, _BaseCubeCounts)
        cube_measures_.weighted_cube_counts = cube_counts_
        measure = _BaseSecondOrderMeasure(None, None, cube_measures_)

        weighted_cube_counts = measure._weighted_cube_counts

        assert weighted_cube_counts is cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)


class Describe_ColumnComparableCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnComparableCounts` object."""

    def it_computes_its_blocks(self, request, is_defined_prop_, dimensions_):
        is_defined_prop_.return_value = True
        counts = np.arange(12).reshape(3, 4).tolist()
        cube_counts_ = instance_mock(request, _BaseCubeCounts, counts=counts)
        property_mock(
            request,
            _BaseSecondOrderMeasure,
            "_weighted_cube_counts",
            return_value=cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        col_comparable_counts = _ColumnComparableCounts(dimensions_, None, None)

        blocks = col_comparable_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(
            counts, dimensions_, diff_rows_nan=True
        )
        assert blocks == [[[1], [2]], [[3], [4]]]

    def but_blocks_raises_when_is_not_defined(self, is_defined_prop_):
        is_defined_prop_.return_value = False

        with pytest.raises(ValueError) as e:
            _ColumnComparableCounts(None, None, None).blocks

        assert (
            str(e.value) == "column_comparable_counts not defined across subvariables."
        )

    @pytest.mark.parametrize(
        "dim_types, expected",
        (
            ((DT.CAT, DT.CAT), True),
            ((DT.NUM_ARRAY, DT.CAT), True),
            ((DT.CAT, DT.MR_SUBVAR), False),
            ((DT.CAT, DT.NUM_ARRAY), False),
        ),
    )
    def it_knows_when_it_is_defined(self, dimensions_, dim_types, expected):
        dimensions_[0].dimension_type = dim_types[0]
        dimensions_[1].dimension_type = dim_types[1]
        col_comparable_counts = _ColumnComparableCounts(dimensions_, None, None)

        assert col_comparable_counts.is_defined == expected

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _ColumnComparableCounts, "is_defined")


class Describe_ColumnProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._ColumnProportions` object."""

    def it_computes_its_blocks(self, request):
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[[5.0, 12.0], [21.0, 32.0]]
        )
        column_weighted_bases_ = instance_mock(
            request, _ColumnWeightedBases, blocks=[[5.0, 6.0], [7.0, 8.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            weighted_counts=weighted_counts_,
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
        _base_values_prop_.return_value = np.array([[4.4, 7.7], [4.4, 7.7]])
        SumSubtotals_.subtotal_rows.return_value = np.array([[8.8, 3.3], [6.6, 4.4]])
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        column_weighted_bases = _ColumnWeightedBases(dimensions_, None, None)

        subtotal_rows = column_weighted_bases._subtotal_rows

        SumSubtotals_.subtotal_rows.assert_called_once_with(
            ANY, dimensions_, diff_cols_nan=True
        )
        assert SumSubtotals_.subtotal_rows.call_args.args[0].tolist() == [
            [4.4, 7.7],
            [4.4, 7.7],
        ]
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
        return instance_mock(request, _BaseCubeCounts)

    @pytest.fixture
    def _weighted_cube_counts_prop_(self, request):
        return property_mock(request, _ColumnWeightedBases, "_weighted_cube_counts")


class Describe_PopulationProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._PopulationProportions` object."""

    @pytest.mark.parametrize(
        "dimension_types, expected",
        (
            ((DT.CAT, DT.CAT), "table"),
            ((DT.CAT_DATE, DT.CAT), "row"),
            ((DT.CAT, DT.CAT_DATE), "column"),
            ((DT.CA_SUBVAR, DT.CA_CAT), "row"),
            ((DT.CA_CAT, DT.CA_SUBVAR), "column"),
            ((DT.CAT, DT.CAT_DATE, DT.CAT), "row"),
            ((DT.CAT, DT.CA_SUBVAR, DT.CA_CAT), "row"),
        ),
    )
    def it_computes_its_blocks(object, request, dimension_types, expected):
        dimensions_ = [
            instance_mock(request, Dimension, dimension_type=dt)
            for dt in dimension_types
        ]
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            row_proportions=instance_mock(request, _RowProportions, blocks="row"),
            column_proportions=instance_mock(request, _RowProportions, blocks="column"),
            table_proportions=instance_mock(request, _RowProportions, blocks="table"),
        )

        assert (
            _PopulationProportions(dimensions_, second_order_measures_, None).blocks
            == expected
        )


class Describe_RowComparableCounts(object):
    """Unit test suite for `cr.cube.matrix.measure._RowComparableCounts` object."""

    def it_computes_its_blocks(self, request, is_defined_prop_, dimensions_):
        is_defined_prop_.return_value = True
        counts = np.arange(12).reshape(3, 4).tolist()
        cube_counts_ = instance_mock(request, _BaseCubeCounts, counts=counts)
        property_mock(
            request,
            _BaseSecondOrderMeasure,
            "_weighted_cube_counts",
            return_value=cube_counts_,
        )
        SumSubtotals_ = class_mock(request, "cr.cube.matrix.measure.SumSubtotals")
        SumSubtotals_.blocks.return_value = [[[1], [2]], [[3], [4]]]
        row_comparable_counts = _RowComparableCounts(dimensions_, None, None)

        blocks = row_comparable_counts.blocks

        SumSubtotals_.blocks.assert_called_once_with(
            counts, dimensions_, diff_cols_nan=True
        )
        assert blocks == [[[1], [2]], [[3], [4]]]

    def but_blocks_raises_when_is_not_defined(self, is_defined_prop_):
        is_defined_prop_.return_value = False

        with pytest.raises(ValueError) as e:
            _RowComparableCounts(None, None, None).blocks

        assert str(e.value) == "row_comparable_counts not defined across subvariables."

    @pytest.mark.parametrize(
        "dim_types, expected",
        (
            ((DT.CAT, DT.CAT), True),
            ((DT.CAT, DT.NUM_ARRAY), True),
            ((DT.MR_SUBVAR, DT.CAT), False),
            ((DT.CA_SUBVAR, DT.CA_CAT), False),
        ),
    )
    def it_knows_when_it_is_defined(self, dimensions_, dim_types, expected):
        dimensions_[0].dimension_type = dim_types[0]
        dimensions_[1].dimension_type = dim_types[1]
        row_comparable_counts = _RowComparableCounts(dimensions_, None, None)

        assert row_comparable_counts.is_defined == expected

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _RowComparableCounts, "is_defined")


class Describe_RowProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._RowProportions` object."""

    def it_computes_its_blocks(self, request):
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[[5.0, 12.0], [21.0, 32.0]]
        )
        row_weighted_bases_ = instance_mock(
            request, _RowWeightedBases, blocks=[[5.0, 6.0], [7.0, 8.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            row_weighted_bases=row_weighted_bases_,
            weighted_counts=weighted_counts_,
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
        _base_values_prop_.return_value = np.array([[7.7, 7.7], [4.4, 4.4]])
        SumSubtotals_.subtotal_columns.return_value = np.array([[3.3, 8.8], [4.4, 6.6]])
        _weighted_cube_counts_prop_.return_value = weighted_cube_counts_
        row_weighted_bases = _RowWeightedBases(dimensions_, None, None)

        subtotal_columns = row_weighted_bases._subtotal_columns

        SumSubtotals_.subtotal_columns.assert_called_once_with(
            ANY, dimensions_, diff_rows_nan=True
        )
        assert SumSubtotals_.subtotal_columns.call_args.args[0].tolist() == [
            [7.7, 7.7],
            [4.4, 4.4],
        ]
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
        return instance_mock(request, _BaseCubeCounts)

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


class Describe_TableProportions(object):
    """Unit test suite for `cr.cube.matrix.measure._TableProportions` object."""

    def it_computes_its_blocks(self, request):
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[[5.0, 12.0], [21.0, 32.0]]
        )
        table_weighted_bases_ = instance_mock(
            request, _TableWeightedBases, blocks=[[5.0, 6.0], [7.0, 8.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            table_weighted_bases=table_weighted_bases_,
            weighted_counts=weighted_counts_,
        )

        table_proportions = _TableProportions(None, second_order_measures_, None)

        assert table_proportions.blocks == [[1.0, 2.0], [3.0, 4.0]]


class Describe_TableProportionVariances(object):
    """Unit test suite for `cr.cube.matrix.measure._TableProportionVariances` object."""

    def it_computes_its_blocks(self, request):
        table_proportions_ = instance_mock(
            request, _TableProportions, blocks=[[0.1, 0.3], [0.25, 0.35]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            table_proportions=table_proportions_,
        )

        variances = _TableProportionVariances(None, second_order_measures_, None)

        assert variances.blocks == [
            pytest.approx([0.09, 0.21]),
            pytest.approx([0.1875, 0.2275]),
        ]


class Describe_TableStandardError(object):
    """Unit test suite for `cr.cube.matrix.measure._TableStandardError` object."""

    def it_computes_its_blocks(self, request):
        table_proportion_variances_ = instance_mock(
            request, _TableProportionVariances, blocks=[[4.0, 18.0], [48.0, 100.0]]
        )
        table_weighted_bases_ = instance_mock(
            request, _TableWeightedBases, blocks=[[1.0, 2.0], [3.0, 4.0]]
        )
        second_order_measures_ = instance_mock(
            request,
            SecondOrderMeasures,
            table_proportion_variances=table_proportion_variances_,
            table_weighted_bases=table_weighted_bases_,
        )

        stderrs = _TableStandardError(None, second_order_measures_, None)

        assert stderrs.blocks == [[2.0, 3.0], [4.0, 5.0]]


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
        "intersections_shape, expected_value",
        (
            ((0, 0), []),
            ((0, 2), []),
            ((2, 0), [[], []]),
            ((4, 1), [[6.6], [6.6], [6.6], [6.6]]),
        ),
    )
    def it_computes_its_intersections_block_to_help(
        self,
        _base_values_prop_,
        _intersections_shape_prop_,
        intersections_shape,
        expected_value,
    ):
        _base_values_prop_.return_value = np.array(
            [
                [6.6, 5.5, 4.4],
                [9.9, 8.8, 7.7],
                [3.3, 2.2, 1.1],
            ]
        )
        _intersections_shape_prop_.return_value = intersections_shape

        intersections = _TableWeightedBases(None, None, None)._intersections

        assert intersections.tolist() == expected_value
        assert intersections.shape == intersections_shape
        assert intersections.dtype == np.float64

    def it_computes_its_intersections_shape_to_help(self, request):
        pass

    @pytest.mark.parametrize(
        "intersections_shape, expected_value",
        (
            ((0, 0), [[], [], []]),
            ((0, 2), [[6.6, 6.6], [9.9, 9.9], [3.3, 3.3]]),
            ((2, 0), [[], [], []]),
            ((4, 1), [[6.6], [9.9], [3.3]]),
        ),
    )
    def it_computes_its_subtotal_columns_to_help(
        self,
        _base_values_prop_,
        _intersections_shape_prop_,
        intersections_shape,
        expected_value,
    ):
        _base_values_prop_.return_value = np.array(
            [
                [6.6, 6.6, 6.6],
                [9.9, 9.9, 9.9],
                [3.3, 3.3, 3.3],
            ]
        )
        _intersections_shape_prop_.return_value = intersections_shape

        subtotal_columns = _TableWeightedBases(None, None, None)._subtotal_columns

        assert subtotal_columns.tolist() == expected_value
        assert subtotal_columns.shape[0] == 3
        assert subtotal_columns.shape[1] == intersections_shape[1]
        assert subtotal_columns.dtype == np.float64

    @pytest.mark.parametrize(
        "intersections_shape, expected_value",
        (
            ((0, 0), []),
            ((0, 2), []),
            ((2, 0), [[6.6, 5.5, 4.4], [6.6, 5.5, 4.4]]),
            (
                (4, 1),
                [[6.6, 5.5, 4.4], [6.6, 5.5, 4.4], [6.6, 5.5, 4.4], [6.6, 5.5, 4.4]],
            ),
        ),
    )
    def it_computes_its_subtotal_rows_to_help(
        self,
        _base_values_prop_,
        _intersections_shape_prop_,
        intersections_shape,
        expected_value,
    ):
        _base_values_prop_.return_value = np.array(
            [
                [6.6, 5.5, 4.4],
                [6.6, 5.5, 4.4],
                [6.6, 5.5, 4.4],
            ]
        )
        _intersections_shape_prop_.return_value = intersections_shape

        subtotal_rows = _TableWeightedBases(None, None, None)._subtotal_rows

        assert subtotal_rows.tolist() == expected_value
        assert subtotal_rows.shape[0] == intersections_shape[0]
        assert subtotal_rows.shape[1] == 3
        assert subtotal_rows.dtype == np.float64

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _TableWeightedBases, "_base_values")

    @pytest.fixture
    def _intersections_shape_prop_(self, request):
        return property_mock(request, _TableWeightedBases, "_intersections_shape")

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseCubeCounts)

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
        cube_counts_ = instance_mock(
            request, _BaseCubeCounts, counts=counts, diff_nans=False
        )
        property_mock(
            request,
            _WeightedCounts,
            "_weighted_cube_counts",
            return_value=cube_counts_,
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

    def it_computes_its_blocks_for_non_mr_dimensions(
        self, request, _base_values_prop_, dimensions_
    ):
        _base_values_prop_.return_value = "A"
        property_mock(request, _Zscores, "_subtotal_columns", return_value="B")
        property_mock(request, _Zscores, "_subtotal_rows", return_value="C")
        property_mock(request, _Zscores, "_intersections", return_value="D")

        blocks = _Zscores(dimensions_, None, None).blocks

        assert blocks == [["A", "B"], ["C", "D"]]

    def it_computes_its_blocks_if_mr_in_dimensions(
        self, request, _base_values_prop_, dimensions_
    ):
        dimensions_[0].dimension_type = DT.MR_SUBVAR
        _base_values_prop_.return_value = [[1, 2], [3, 4]]
        NanSubtotals_ = class_mock(request, "cr.cube.matrix.measure.NanSubtotals")
        NanSubtotals_.blocks.return_value = [[[1], [np.nan]], [[np.nan], [np.nan]]]

        blocks = _Zscores(dimensions_, None, None).blocks

        NanSubtotals_.blocks.assert_called_once_with([[1, 2], [3, 4]], dimensions_)
        assert blocks == [[[1], [np.nan]], [[np.nan], [np.nan]]]

    def it_can_calculate_a_zscore(self, _is_defective_prop_):
        _is_defective_prop_.return_value = False
        zscores = _Zscores(None, None, None)

        actual = zscores._calculate_zscores(
            np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]]),
            np.array([[23.1, 23.1, 23.1], [23.1, 23.1, 23.1]]),
            np.array([[6.6, 6.6, 6.6], [16.5, 16.5, 16.5]]),
            np.array([[9.9, 7.7, 5.5], [9.9, 7.7, 5.5]]),
        )

        assert actual.tolist() == [
            pytest.approx([0.4387482, 4.33878897e-16, -0.5097793]),
            pytest.approx([-0.4387482, 8.67757795e-16, 0.5097793]),
        ]

    def but_it_does_not_calculate_zscore_if_defective(self, _is_defective_prop_):
        _is_defective_prop_.return_value = True
        counts = np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]])
        zscores = _Zscores(None, None, None)

        actual = zscores._calculate_zscores(counts, None, None, None)

        assert actual == pytest.approx(np.full(counts.shape, np.nan), nan_ok=True)

    def and_it_is_nan_if_rows_base_equal_table_base(self, _is_defective_prop_):
        _is_defective_prop_.return_value = False
        counts = np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]])
        zscores = _Zscores(None, None, None)

        actual = zscores._calculate_zscores(
            np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]]),
            np.array([[6.6, 6.6, 6.6], [16.5, 16.5, 16.5]]),
            np.array([[6.6, 6.6, 6.6], [16.5, 16.5, 16.5]]),
            np.array([[9.9, 7.7, 5.5], [9.9, 7.7, 5.5]]),
        )

        assert actual == pytest.approx(np.full(counts.shape, np.nan), nan_ok=True)

    def and_it_is_nan_if_columns_base_equal_table_base(self, _is_defective_prop_):
        _is_defective_prop_.return_value = False
        counts = np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]])
        zscores = _Zscores(None, None, None)

        actual = zscores._calculate_zscores(
            np.array([[3.3, 2.2, 1.1], [6.6, 5.5, 4.4]]),
            np.array([[9.9, 7.7, 5.5], [9.9, 7.7, 5.5]]),
            np.array([[6.6, 6.6, 6.6], [16.5, 16.5, 16.5]]),
            np.array([[9.9, 7.7, 5.5], [9.9, 7.7, 5.5]]),
        )

        assert actual == pytest.approx(np.full(counts.shape, np.nan), nan_ok=True)

    @pytest.mark.parametrize(
        "counts, expected",
        (
            ([[1, 2], [4]], True),
            ([[], []], True),
            ([[1, 2], [4, 5]], False),
            ([[1, 0, 0], [4, 0, 0]], True),
        ),
    )
    def it_knows_if_it_is_defective(
        self, request, second_order_measures_, counts, expected
    ):
        counts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = False
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[[np.array(counts), None], [None, None]]
        )
        second_order_measures_.weighted_counts = weighted_counts_
        zscores = _Zscores(None, second_order_measures_, None)

        assert zscores._is_defective == expected

    @pytest.mark.parametrize(
        "prop_name, index",
        (
            ("_base_values", 1),
            ("_subtotal_columns", 2),
            ("_subtotal_rows", 3),
            ("_intersections", 4),
        ),
    )
    def it_provides_the_block_components_to_help(
        self, request, _calculate_zscores_, second_order_measures_, prop_name, index
    ):
        _calculate_zscores_.return_value = [[1, 2], [3, 4]]
        weighted_counts_ = instance_mock(
            request, _WeightedCounts, blocks=[["w1", "w2"], ["w3", "w4"]]
        )
        table_weighted_bases_ = instance_mock(
            request, _TableWeightedBases, blocks=[["t1", "t2"], ["t3", "t4"]]
        )
        row_weighted_bases_ = instance_mock(
            request, _RowWeightedBases, blocks=[["r1", "r2"], ["r3", "r4"]]
        )
        column_weighted_bases_ = instance_mock(
            request, _WeightedCounts, blocks=[["c1", "c2"], ["c3", "c4"]]
        )
        second_order_measures_.weighted_counts = weighted_counts_
        second_order_measures_.table_weighted_bases = table_weighted_bases_
        second_order_measures_.row_weighted_bases = row_weighted_bases_
        second_order_measures_.column_weighted_bases = column_weighted_bases_
        zscores = _Zscores(None, second_order_measures_, None)

        actual = getattr(zscores, prop_name)

        assert actual == [[1, 2], [3, 4]]
        _calculate_zscores_.assert_called_once_with(
            zscores, "w%s" % index, "t%s" % index, "r%s" % index, "c%s" % index
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _Zscores, "_base_values")

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)

    @pytest.fixture
    def _calculate_zscores_(self, request):
        return method_mock(request, _Zscores, "_calculate_zscores")

    @pytest.fixture
    def _is_defective_prop_(self, request):
        return property_mock(request, _Zscores, "_is_defective")

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)


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

    def it_gets_the_right_counts_for_rows(self, measure_, second_order_measures_):
        measure_.blocks = [["a", "b"], ["c", "d"]]
        second_order_measures_.column_comparable_counts = measure_
        marginal = _BaseMarginal(None, second_order_measures_, None, MO.ROWS)

        assert marginal._counts == ["a", "c"]

    def it_gets_the_right_counts_for_columns(self, measure_, second_order_measures_):
        measure_.blocks = [["a", "b"], ["c", "d"]]
        second_order_measures_.row_comparable_counts = measure_
        marginal = _BaseMarginal(None, second_order_measures_, None, MO.COLUMNS)

        assert marginal._counts == ["a", "b"]

    def it_knows_when_counts_are_defined_columns(self, second_order_measures_):
        marginal = _BaseMarginal(None, second_order_measures_, None, MO.COLUMNS)

        actual = marginal._counts_are_defined

        assert actual == second_order_measures_.row_comparable_counts.is_defined

    def it_knows_when_counts_are_defined_rows(self, second_order_measures_):
        marginal = _BaseMarginal(None, second_order_measures_, None, MO.ROWS)

        actual = marginal._counts_are_defined

        assert actual == second_order_measures_.column_comparable_counts.is_defined

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)


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


class Describe_MarginTableWeightedBase(object):
    """Unit test suite for `cr.cube.matrix.measure._MarginTableWeightedBase` object."""

    def it_provides_blocks_if_table_weighted_base(
        self,
        request,
        _base_values_prop_,
        is_defined_prop_,
    ):
        is_defined_prop_.return_value = True
        _base_values_prop_.return_value = np.array([1.0, 1.0])
        property_mock(
            request, _MarginTableWeightedBase, "_subtotal_shape", return_value=3
        )
        table_weighted_base = _MarginTableWeightedBase(None, None, None, None)

        results = table_weighted_base.blocks

        assert results[0].tolist() == [1.0, 1.0]
        assert results[1].tolist() == [1.0, 1.0, 1.0]

    def but_it_raises_if_undefined(self, is_defined_prop_):
        is_defined_prop_.return_value = False

        with pytest.raises(ValueError) as e:
            _MarginTableWeightedBase(None, None, None, None).blocks

        assert (
            str(e.value)
            == "Could not calculate weighted-table-base-margin across subvariables"
        )

    @pytest.mark.parametrize(
        "base_values, expected",
        (
            (None, False),
            (np.array([1, 2]), True),
        ),
    )
    def it_can_tell_if_it_is_defined(self, _base_values_prop_, base_values, expected):
        _base_values_prop_.return_value = base_values
        table_weighted_base = _MarginTableWeightedBase(None, None, None, None)

        assert table_weighted_base.is_defined == expected

    def it_provides_the_base_values_for_rows_to_help(
        self, weighted_cube_counts_, cube_measures_
    ):
        weighted_cube_counts_.rows_table_base = [1, 2]
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        table_weighted_base = _MarginTableWeightedBase(
            None, None, cube_measures_, MO.ROWS
        )

        assert table_weighted_base._base_values == [1, 2]

    def it_provides_the_base_values_for_columns_to_help(
        self, weighted_cube_counts_, cube_measures_
    ):
        weighted_cube_counts_.columns_table_base = [1, 2]
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        table_weighted_base = _MarginTableWeightedBase(
            None, None, cube_measures_, MO.COLUMNS
        )

        assert table_weighted_base._base_values == [1, 2]

    def it_provides_subtotal_shape_for_rows_orientation_to_help(self, dimensions_):
        dimensions_[0].subtotals = (1, 2, 3)
        table_weighted_base = _MarginTableWeightedBase(dimensions_, None, None, MO.ROWS)

        assert table_weighted_base._subtotal_shape == 3

    def it_provides_subtotal_shape_for_columns_orientation_to_help(self, dimensions_):
        dimensions_[1].subtotals = (1,)
        table_weighted_base = _MarginTableWeightedBase(
            dimensions_, None, None, MO.COLUMNS
        )

        assert table_weighted_base._subtotal_shape == 1

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _MarginTableWeightedBase, "_base_values")

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _MarginTableWeightedBase, "is_defined")

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseCubeCounts)


class Describe_MarginTableProportion(object):
    """Unit test suite for `cr.cube.matrix.measure._MarginTableProportion` object."""

    def it_provides_blocks(self, request):
        property_mock(
            request,
            _MarginTableProportion,
            "_proportion_numerators",
            return_value=[np.array([2.0, 3.0]), np.array([], dtype=float)],
        )
        property_mock(
            request,
            _MarginTableProportion,
            "_proportion_denominators",
            return_value=[np.array([5.0, 5.0]), np.array([], dtype=float)],
        )

        margin_proportion = _MarginTableProportion(None, None, None, None).blocks

        assert len(margin_proportion) == 2
        assert margin_proportion[0] == pytest.approx([0.4, 0.6])
        assert margin_proportion[1].shape == (0,)

    def it_can_tell_if_it_is_defined(self, request):
        property_mock(request, _BaseMarginal, "_counts_are_defined")
        margin_proportion = _MarginTableProportion(None, None, None, None)
        assert margin_proportion.is_defined == margin_proportion._counts_are_defined

    def it_provides_the_right_denominator_for_rows(self):
        pass

    def it_provides_the_right_numerator_for_rows(
        self, second_order_measures_, weighted_counts_, _apply_along_orientation_
    ):
        second_order_measures_.weighted_counts = weighted_counts_
        weighted_counts_.blocks = [[1, 2], [3, 4]]
        margin = _MarginTableProportion(None, second_order_measures_, None, MO.ROWS)

        margin._proportion_numerators

        assert _apply_along_orientation_.call_args_list == [
            call(margin, np.sum, 1),
            call(margin, np.sum, 3),
        ]

    def but_it_raises_if_column_counts_are_not_comparable(
        self, request, second_order_measures_, weighted_counts_
    ):
        second_order_measures_.weighted_counts = weighted_counts_
        second_order_measures_.column_comparable_counts = instance_mock(
            request,
            _ColumnComparableCounts,
        )
        property_mock(
            request, _ColumnComparableCounts, "blocks", side_effect=ValueError
        )
        margin = _MarginTableProportion(None, second_order_measures_, None, MO.ROWS)

        with pytest.raises(ValueError):
            margin._proportion_numerators

    def it_provides_the_right_numerator_for_columns(
        self, second_order_measures_, weighted_counts_, _apply_along_orientation_
    ):
        second_order_measures_.weighted_counts = weighted_counts_
        weighted_counts_.blocks = [[1, 2], [3, 4]]
        margin = _MarginTableProportion(None, second_order_measures_, None, MO.COLUMNS)

        margin._proportion_numerators

        assert _apply_along_orientation_.call_args_list == [
            call(margin, np.sum, 1),
            call(margin, np.sum, 2),
        ]

    def but_it_raises_if_row_counts_are_not_comparable(
        self, request, second_order_measures_, weighted_counts_
    ):
        second_order_measures_.weighted_counts = weighted_counts_
        second_order_measures_.row_comparable_counts = instance_mock(
            request,
            _RowComparableCounts,
        )
        property_mock(request, _RowComparableCounts, "blocks", side_effect=ValueError)
        margin = _MarginTableProportion(None, second_order_measures_, None, MO.ROWS)

        with pytest.raises(ValueError):
            margin._proportion_numerators

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _apply_along_orientation_(self, request):
        return method_mock(
            request,
            _MarginTableProportion,
            "_apply_along_orientation",
        )

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)

    @pytest.fixture
    def weighted_counts_(self, request):
        return instance_mock(request, _WeightedCounts)


class Describe_MarginUnweightedBase(object):
    """Unit test suite for `cr.cube.matrix.measure._MarginUnweightedBase` object."""

    def it_provides_blocks_for_rows(
        self,
        request,
        is_defined_prop_,
        orientation_prop_,
        second_order_measures_,
        blocks_,
    ):
        is_defined_prop_.return_value = True
        orientation_prop_.return_value = MO.ROWS
        row_unweighted_bases_ = instance_mock(
            request, _RowUnweightedBases, blocks=blocks_
        )
        second_order_measures_.row_unweighted_bases = row_unweighted_bases_
        margin_ = _MarginUnweightedBase(None, second_order_measures_, None, None)

        actual = margin_.blocks

        assert actual[0].tolist() == [0, 2]
        assert actual[1].tolist() == [6]

    def it_provides_blocks_for_columns(
        self,
        request,
        is_defined_prop_,
        orientation_prop_,
        second_order_measures_,
        blocks_,
    ):
        is_defined_prop_.return_value = True
        orientation_prop_.return_value = MO.COLUMNS
        column_unweighted_bases_ = instance_mock(
            request, _RowUnweightedBases, blocks=blocks_
        )
        second_order_measures_.column_unweighted_bases = column_unweighted_bases_
        margin_ = _MarginUnweightedBase(None, second_order_measures_, None, None)

        actual = margin_.blocks

        assert actual[0].tolist() == [0, 1]
        assert actual[1].tolist() == [4]

    def but_blocks_raises_if_undefined(self, is_defined_prop_):
        is_defined_prop_.return_value = False
        margin_ = _MarginUnweightedBase(None, None, None, None)

        with pytest.raises(ValueError) as e:
            margin_.blocks

        assert str(e.value) == "Cannot calculate base across subvariables dimension."

    def it_can_tell_if_it_is_defined(self, request):
        property_mock(request, _BaseMarginal, "_counts_are_defined")
        margin_unweighted_base = _MarginUnweightedBase(None, None, None, None)
        assert (
            margin_unweighted_base.is_defined
            == margin_unweighted_base._counts_are_defined
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def blocks_(self):
        return [
            [np.array([[0, 1], [2, 3]]), np.array([[4], [5]])],
            [np.array([[6, 7]]), np.array([8])],
        ]

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _MarginUnweightedBase, "is_defined")

    @pytest.fixture
    def orientation_prop_(self, request):
        return property_mock(request, _MarginUnweightedBase, "orientation")

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)


class Describe_MarginWeightedBase(object):
    """Unit test suite for `cr.cube.matrix.measure._MarginWeightedBase` object."""

    def it_provides_blocks_for_rows(
        self,
        request,
        is_defined_prop_,
        orientation_prop_,
        second_order_measures_,
        blocks_,
    ):
        is_defined_prop_.return_value = True
        orientation_prop_.return_value = MO.ROWS
        row_weighted_bases_ = instance_mock(request, _RowWeightedBases, blocks=blocks_)
        second_order_measures_.row_weighted_bases = row_weighted_bases_
        margin_ = _MarginWeightedBase(None, second_order_measures_, None, None)

        actual = margin_.blocks

        assert actual[0].tolist() == [0, 2]
        assert actual[1].tolist() == [6]

    def it_provides_blocks_for_columns(
        self,
        request,
        is_defined_prop_,
        orientation_prop_,
        second_order_measures_,
        blocks_,
    ):
        is_defined_prop_.return_value = True
        orientation_prop_.return_value = MO.COLUMNS
        column_weighted_bases_ = instance_mock(
            request, _RowWeightedBases, blocks=blocks_
        )
        second_order_measures_.column_weighted_bases = column_weighted_bases_
        margin_ = _MarginWeightedBase(None, second_order_measures_, None, None)

        actual = margin_.blocks

        assert actual[0].tolist() == [0, 1]
        assert actual[1].tolist() == [4]

    def but_blocks_raises_if_undefined(self, is_defined_prop_):
        is_defined_prop_.return_value = False
        margin_ = _MarginWeightedBase(None, None, None, None)

        with pytest.raises(ValueError) as e:
            margin_.blocks

        assert (
            str(e.value)
            == "Could not calculate weighted-base-margin across subvariables"
        )

    @pytest.mark.parametrize(
        "base_values, expected",
        (
            (None, False),
            (np.array([1, 2]), True),
        ),
    )
    def it_can_tell_if_it_is_defined(self, _base_values_prop_, base_values, expected):
        _base_values_prop_.return_value = base_values
        table_weighted_base = _MarginWeightedBase(None, None, None, None)

        assert table_weighted_base.is_defined == expected

    def it_provides_the_base_values_for_rows_to_help(
        self, weighted_cube_counts_, cube_measures_
    ):
        weighted_cube_counts_.rows_base = [1, 2]
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        table_weighted_base = _MarginWeightedBase(None, None, cube_measures_, MO.ROWS)

        assert table_weighted_base._base_values == [1, 2]

    def it_provides_the_base_values_for_columns_to_help(
        self, weighted_cube_counts_, cube_measures_
    ):
        weighted_cube_counts_.columns_base = [1, 2]
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        table_weighted_base = _MarginWeightedBase(
            None, None, cube_measures_, MO.COLUMNS
        )

        assert table_weighted_base._base_values == [1, 2]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, _MarginWeightedBase, "_base_values")

    @pytest.fixture
    def blocks_(self):
        return [
            [np.array([[0, 1], [2, 3]]), np.array([[4], [5]])],
            [np.array([[6, 7]]), np.array([8])],
        ]

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _MarginWeightedBase, "is_defined")

    @pytest.fixture
    def orientation_prop_(self, request):
        return property_mock(request, _MarginWeightedBase, "orientation")

    @pytest.fixture
    def second_order_measures_(self, request):
        return instance_mock(request, SecondOrderMeasures)

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseCubeCounts)


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


# === Scalars ===


class Describe_TableWeightedBase(object):
    """Unit test suite for `cr.cube.matrix.measure._TableWeightedBase` object."""

    @pytest.mark.parametrize(
        "table_base, expected",
        (
            (None, False),
            (1, True),
        ),
    )
    def it_knows_if_it_is_defined(
        self, cube_measures_, weighted_cube_counts_, table_base, expected
    ):
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        weighted_cube_counts_.table_base = table_base

        assert _TableWeightedBase(None, None, cube_measures_).is_defined == expected

    def it_provides_its_value(
        self, request, cube_measures_, weighted_cube_counts_, is_defined_prop_
    ):
        is_defined_prop_.return_value = True
        cube_measures_.weighted_cube_counts = weighted_cube_counts_
        weighted_cube_counts_.table_base = 2.0
        table_margin = _TableWeightedBase(None, None, cube_measures_)

        assert table_margin.value == 2.0

    def but_it_raises_if_it_is_not_defined(self, is_defined_prop_):
        is_defined_prop_.return_value = False

        with pytest.raises(ValueError) as e:
            _TableWeightedBase(None, None, None).value

        assert (
            str(e.value)
            == "Cannot sum across subvariables dimension for table weighted base scalar"
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def is_defined_prop_(self, request):
        return property_mock(request, _TableWeightedBase, "is_defined")

    @pytest.fixture
    def cube_measures_(self, request):
        return instance_mock(request, CubeMeasures)

    @pytest.fixture
    def weighted_cube_counts_(self, request):
        return instance_mock(request, _BaseCubeCounts)


class Describe_TableWeightedBasesRange(object):
    """Unit test suite for `cr.cube.matrix.measure._TableWeightedBasesRange` object."""

    def it_is_always_defined(self):
        assert _TableWeightedBasesRange(None, None, None).is_defined

    @pytest.mark.parametrize(
        "bases, expected",
        (
            ([[2, 2, 2], [2, 2, 2]], [2, 2]),
            ([[0, 1, 2], [3, 4, 5]], [0, 5]),
        ),
    )
    def it_knows_its_values(self, request, bases, expected):
        weighted_cube_counts_ = instance_mock(
            request, _BaseCubeCounts, table_bases=bases
        )
        cube_measures_ = instance_mock(
            request, CubeMeasures, weighted_cube_counts=weighted_cube_counts_
        )
        range = _TableWeightedBasesRange(None, None, cube_measures_)

        assert range.value.tolist() == expected
