# encoding: utf-8 """Unit test suite for `cr.cube.stripe.assembler` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Element, _Subtotal
from cr.cube.enums import COLLATION_METHOD as CM
from cr.cube.stripe.assembler import (
    _BaseOrderHelper,
    _OrderHelper,
    _SortByMeasureHelper,
    StripeAssembler,
)
from cr.cube.stripe.measure import (
    _Means,
    _ScaledCounts,
    StripeMeasures,
    _TableProportionStddevs,
    _TableProportionStderrs,
    _TableProportions,
    _UnweightedBases,
    _UnweightedCounts,
    _WeightedBases,
    _WeightedCounts,
)

from ...unitutil import (
    class_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class DescribeStripeAssembler(object):
    """Unit test suite for `cr.cube.stripe.assembler.StripeAssembler` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("means", _Means),
            ("table_proportion_stddevs", _TableProportionStddevs),
            ("table_proportion_stderrs", _TableProportionStderrs),
            ("table_proportions", _TableProportions),
            ("unweighted_bases", _UnweightedBases),
            ("unweighted_counts", _UnweightedCounts),
            ("weighted_bases", _WeightedBases),
            ("weighted_counts", _WeightedCounts),
        ),
    )
    def it_assembles_various_measures(
        self,
        request,
        _measures_prop_,
        measures_,
        _assemble_vector_,
        measure_prop_name,
        MeasureCls,
    ):
        _measures_prop_.return_value = measures_
        setattr(
            measures_,
            measure_prop_name,
            instance_mock(request, MeasureCls, blocks=("A", "B")),
        )
        _assemble_vector_.return_value = np.array([1, 2, 3, 4, 5])
        assembler = StripeAssembler(None, None, None, None)

        value = getattr(assembler, measure_prop_name)

        _assemble_vector_.assert_called_once_with(assembler, ("A", "B"))
        assert value.tolist() == [1, 2, 3, 4, 5]

    def it_knows_the_inserted_row_idxs(self, _row_order_prop_):
        _row_order_prop_.return_value = np.array([-1, 0, 3, -2, 4, 1])
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.inserted_row_idxs == (0, 3)

    def it_knows_the_row_count(self, _row_order_prop_):
        _row_order_prop_.return_value = np.array([1, 2, 3, 4, 5])
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.row_count == 5

    def it_knows_the_row_labels(self, request, rows_dimension_, _row_order_prop_):
        rows_dimension_.valid_elements = tuple(
            instance_mock(request, _Element, label=label)
            for label in ("baz", "foo", "bar")
        )
        rows_dimension_.subtotals = tuple(
            instance_mock(request, _Subtotal, label=label) for label in ("bing", "bada")
        )
        _row_order_prop_.return_value = np.array([1, 2, 0, -1, -2])
        assembler = StripeAssembler(None, rows_dimension_, None, None)

        assert assembler.row_labels.tolist() == ["foo", "bar", "baz", "bada", "bing"]

    def it_knows_the_rows_dimension_fills(
        self, request, rows_dimension_, _row_order_prop_
    ):
        rows_dimension_.valid_elements = tuple(
            instance_mock(request, _Element, fill=fill)
            for fill in ("cdef01", "6789ab", "012345")
        )
        _row_order_prop_.return_value = np.array([2, -2, 1, -1, 0])
        assembler = StripeAssembler(None, rows_dimension_, None, None)

        print(assembler.rows_dimension_fills)
        assert assembler.rows_dimension_fills == (
            "012345",
            None,
            "6789ab",
            None,
            "cdef01",
        )

    def it_knows_the_scale_mean(self, _measures_prop_, measures_, scaled_counts_):
        scaled_counts_.scale_mean = 3
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.scale_mean == 3

    def it_knows_the_scale_median(self, _measures_prop_, measures_, scaled_counts_):
        scaled_counts_.scale_median = 4
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.scale_median == 4

    def it_knows_the_scale_stddev(self, _measures_prop_, measures_, scaled_counts_):
        scaled_counts_.scale_stddev = 5
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.scale_stddev == 5

    def it_knows_the_scale_stderr(self, _measures_prop_, measures_, scaled_counts_):
        scaled_counts_.scale_stderr = 6
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.scale_stderr == 6

    def it_knows_the_table_base_range(self, request, _measures_prop_, measures_):
        measures_.unweighted_bases = instance_mock(
            request, _UnweightedBases, table_base_range=np.array([50, 100])
        )
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.table_base_range.tolist() == [50, 100]

    def it_knows_the_table_margin_range(self, request, _measures_prop_, measures_):
        measures_.weighted_bases = instance_mock(
            request, _WeightedBases, table_margin_range=np.array([50.5, 100.1])
        )
        _measures_prop_.return_value = measures_
        assembler = StripeAssembler(None, None, None, None)

        assert assembler.table_margin_range.tolist() == [50.5, 100.1]

    def it_can_assemble_a_vector_to_help(self, _row_order_prop_):
        base_values = np.array([1, 2, 3, 4])
        subtotal_values = (3, 5, 7)
        blocks = (base_values, subtotal_values)
        _row_order_prop_.return_value = np.array([-3, 1, 0, -2, 3, 2, -1])
        assembler = StripeAssembler(None, None, None, None)

        assert assembler._assemble_vector(blocks).tolist() == [3, 2, 1, 5, 4, 3, 7]

    def it_constructs_its_measures_collaborator_object_to_help(
        self, request, cube_, rows_dimension_, measures_
    ):
        StripeMeasures_ = class_mock(
            request,
            "cr.cube.stripe.assembler.StripeMeasures",
            return_value=measures_,
        )
        assembler = StripeAssembler(
            cube_, rows_dimension_, ca_as_0th=False, slice_idx=7
        )

        measures = assembler._measures

        StripeMeasures_.assert_called_once_with(cube_, rows_dimension_, False, 7)
        assert measures is measures_

    def it_knows_the_row_order_to_help(
        self, request, rows_dimension_, _measures_prop_, measures_
    ):
        _measures_prop_.return_value = measures_
        _BaseOrderHelper_ = class_mock(
            request, "cr.cube.stripe.assembler._BaseOrderHelper"
        )
        _BaseOrderHelper_.display_order.return_value = (-1, 1, -2, 2, -3, 3)
        assembler = StripeAssembler(None, rows_dimension_, None, None)

        row_order = assembler._row_order

        _BaseOrderHelper_.display_order.assert_called_once_with(
            rows_dimension_, measures_
        )
        assert row_order.tolist() == [-1, 1, -2, 2, -3, 3]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assemble_vector_(self, request):
        return method_mock(request, StripeAssembler, "_assemble_vector")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)

    @pytest.fixture
    def _measures_prop_(self, request):
        return property_mock(request, StripeAssembler, "_measures")

    @pytest.fixture
    def _row_order_prop_(self, request):
        return property_mock(request, StripeAssembler, "_row_order")

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def scaled_counts_(self, request):
        return instance_mock(request, _ScaledCounts)


class Describe_BaseOrderHelper(object):
    """Unit-test suite for `cr.cube.stripe.assembler._BaseOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, HelperCls",
        (
            (CM.UNIVARIATE_MEASURE, _SortByMeasureHelper),
            (CM.EXPLICIT_ORDER, _OrderHelper),
            (CM.PAYLOAD_ORDER, _OrderHelper),
        ),
    )
    def it_dispatches_to_the_right_order_helper(
        self, request, measures_, collation_method, HelperCls
    ):
        rows_dimension_ = instance_mock(
            request, Dimension, collation_method=collation_method
        )
        order_helper_ = instance_mock(
            request, HelperCls, _display_order=np.array([-2, 1, -1, 2])
        )
        HelperCls_ = class_mock(
            request,
            "cr.cube.stripe.assembler.%s" % HelperCls.__name__,
            return_value=order_helper_,
        )

        display_order = _BaseOrderHelper.display_order(rows_dimension_, measures_)

        HelperCls_.assert_called_once_with(rows_dimension_, measures_)
        assert display_order.tolist() == [-2, 1, -1, 2]

    @pytest.mark.parametrize(
        "pruning_base, expected_value",
        (([1, 1, 1], ()), ([1, 0, 1], (1,)), ([0, 0, 0], (0, 1, 2))),
    )
    def it_knows_the_empty_row_idxs_to_help(
        self, measures_, pruning_base, expected_value
    ):
        measures_.pruning_base = np.array(pruning_base)
        order_helper = _BaseOrderHelper(None, measures_)

        assert order_helper._empty_row_idxs == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)


class Describe_OrderHelper(object):
    """Unit test suite for `cr.cube.stripe.assembler._OrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def it_computes_the_order_of_a_rows_dimension_to_help(
        self, request, collation_method, collator_class_name
    ):
        rows_dimension_ = instance_mock(
            request, Dimension, collation_method=collation_method
        )
        CollatorCls_ = class_mock(
            request, "cr.cube.stripe.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        property_mock(request, _OrderHelper, "_empty_row_idxs", return_value=(2, 4, 6))
        order_helper = _OrderHelper(rows_dimension_, None)

        display_order = order_helper._display_order

        CollatorCls_.display_order.assert_called_once_with(rows_dimension_, (2, 4, 6))
        assert display_order == (1, -2, 3, 5, -1)
