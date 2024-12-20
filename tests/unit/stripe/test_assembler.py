# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.assembler` module."""

import mock
import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Strand
from cr.cube.dimension import Dimension, Element, _OrderSpec, _Subtotal
from cr.cube.enums import COLLATION_METHOD as CM, ORDER_FORMAT
from cr.cube.stripe.assembler import (
    _BaseOrderHelper,
    _BaseSortByValueHelper,
    _OrderHelper,
    _SortByLabelHelper,
    _SortByMeasureHelper,
)
from cr.cube.stripe.measure import (
    StripeMeasures,
    _BaseSecondOrderMeasure,
    _Means,
    _PopulationProportions,
    _PopulationProportionStderrs,
    _ScaledCounts,
    _TableProportions,
    _TableProportionStddevs,
    _TableProportionStderrs,
    _UnweightedBases,
    _UnweightedCounts,
    _WeightedBases,
    _WeightedCounts,
)

from ...unitutil import class_mock, instance_mock, method_mock, property_mock


class TestStripeAssembler:
    """Unit test suite for `cr.cube.stripe.assembler.StripeAssembler` object."""

    @pytest.mark.parametrize(
        "measure_prop_name, MeasureCls",
        (
            ("means", _Means),
            ("population_proportions", _PopulationProportions),
            ("population_proportion_stderrs", _PopulationProportionStderrs),
            ("table_proportion_stddevs", _TableProportionStddevs),
            ("table_proportion_stderrs", _TableProportionStderrs),
            ("table_proportions", _TableProportions),
            ("unweighted_bases", _UnweightedBases),
            ("unweighted_counts", _UnweightedCounts),
            ("weighted_bases", _WeightedBases),
            ("weighted_counts", _WeightedCounts),
        ),
    )
    def test_it_assembles_various_measures(
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
        property_mock(request, _Strand, "diff_row_idxs", return_value=())
        _assemble_vector_.return_value = np.array([1, 2, 3, 4, 5])
        strand = _Strand(None, None, None, None, None, None)

        value = getattr(strand, measure_prop_name)

        _assemble_vector_.assert_called_once_with(strand, ("A", "B"))
        assert value.tolist() == [1, 2, 3, 4, 5]

    def test_it_knows_the_inserted_row_idxs(self, _row_order_prop_):
        _row_order_prop_.return_value = np.array([-1, 0, 3, -2, 4, 1])
        strand = _Strand(None, None, None, None, None, None)

        assert strand.inserted_row_idxs == (0, 3)

    def test_it_knows_the_row_count(self, _row_order_prop_):
        _row_order_prop_.return_value = np.array([1, 2, 3, 4, 5])
        strand = _Strand(None, None, None, None, None, None)

        assert strand.row_count == 5

    def test_it_knows_the_row_labels(self, rows_dimension_, _row_order_prop_):
        rows_dimension_.element_labels = ("baz", "foo", "bar")
        rows_dimension_.subtotal_labels = ("bing", "bada")
        _row_order_prop_.return_value = np.array([1, 2, 0, -1, -2])
        with mock.patch(
            "cr.cube.cubepart._Strand._rows_dimension", new=rows_dimension_
        ):
            strand = _Strand(None, None, None, None, None, None)

            assert strand.row_labels.tolist() == ["foo", "bar", "baz", "bada", "bing"]

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
        self, request, rows_dimension_, _row_order_prop_, order, expected_fills
    ):
        element_fills = ("#000000", "#111111", "#f00ba5")
        subtotal_fills = ("STF1", "STF2")
        rows_dimension_.valid_elements = tuple(
            instance_mock(request, Element, fill=fill) for fill in element_fills
        )
        rows_dimension_.subtotals = tuple(
            instance_mock(request, _Subtotal, fill=fill) for fill in subtotal_fills
        )
        _row_order_prop_.return_value = order
        with mock.patch(
            "cr.cube.cubepart._Strand._rows_dimension", new=rows_dimension_
        ):
            strand = _Strand(None, None, None, None, None, None)

            assert strand.rows_dimension_fills == expected_fills

    def test_it_knows_the_scale_mean(self, _measures_prop_, measures_, scaled_counts_):
        scaled_counts_.scale_mean = 3
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.scale_mean == 3

    def test_it_knows_the_scale_median(
        self, _measures_prop_, measures_, scaled_counts_
    ):
        scaled_counts_.scale_median = 4
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.scale_median == 4

    def test_it_knows_the_scale_stddev(
        self, _measures_prop_, measures_, scaled_counts_
    ):
        scaled_counts_.scale_stddev = 5
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.scale_stddev == 5

    def test_it_knows_the_scale_stderr(
        self, _measures_prop_, measures_, scaled_counts_
    ):
        scaled_counts_.scale_stderr = 6
        measures_.scaled_counts = scaled_counts_
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.scale_stderr == 6

    def test_it_knows_the_table_base_range(self, request, _measures_prop_, measures_):
        measures_.unweighted_bases = instance_mock(
            request, _UnweightedBases, table_base_range=np.array([50, 100])
        )
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.table_base_range.tolist() == [50, 100]

    def test_it_knows_the_table_margin_range(self, request, _measures_prop_, measures_):
        measures_.weighted_bases = instance_mock(
            request, _WeightedBases, table_margin_range=np.array([50.5, 100.1])
        )
        _measures_prop_.return_value = measures_
        strand = _Strand(None, None, None, None, None, None)

        assert strand.table_margin_range.tolist() == [50.5, 100.1]

    def test_it_can_assemble_a_vector_to_help(self, _row_order_prop_):
        base_values = np.array([1, 2, 3, 4])
        subtotal_values = (3, 5, 7)
        blocks = (base_values, subtotal_values)
        _row_order_prop_.return_value = np.array([-3, 1, 0, -2, 3, 2, -1])
        strand = _Strand(None, None, None, None, None, None)

        assert strand._assemble_vector(blocks).tolist() == [3, 2, 1, 5, 4, 3, 7]

    def test_it_constructs_its_measures_collaborator_object_to_help(
        self, request, cube_, rows_dimension_, measures_
    ):
        StripeMeasures_ = class_mock(
            request,
            "cr.cube.cubepart.StripeMeasures",
            return_value=measures_,
        )
        with mock.patch(
            "cr.cube.cubepart._Strand._rows_dimension", new=rows_dimension_
        ):
            strand = _Strand(
                cube_, None, None, ca_as_0th=False, slice_idx=7, mask_size=None
            )

            measures = strand._measures

            StripeMeasures_.assert_called_once_with(cube_, rows_dimension_, False, 7)
            assert measures is measures_

    @pytest.mark.parametrize(
        "format, row_order",
        (
            (ORDER_FORMAT.SIGNED_INDEXES, (-1, 1, -2, 2, -3, 3)),
            (ORDER_FORMAT.BOGUS_IDS, ("ins_1", 1, "ins_2", 2, "ins_3", 3)),
        ),
    )
    def test_it_knows_the_row_order_to_help(
        self, request, rows_dimension_, _measures_prop_, measures_, format, row_order
    ):
        _measures_prop_.return_value = measures_
        _BaseOrderHelper_ = class_mock(
            request, "cr.cube.cubepart.stripe_BaseOrderHelper"
        )
        _BaseOrderHelper_.display_order.return_value = row_order
        with mock.patch(
            "cr.cube.cubepart._Strand._rows_dimension", new=rows_dimension_
        ):
            strand = _Strand(None, None, None, None, None, None)

            row_order = strand.row_order(format)

            _BaseOrderHelper_.display_order.assert_called_once_with(
                rows_dimension_, measures_, format
            )
            assert row_order.tolist() == list(row_order)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assemble_vector_(self, request):
        return method_mock(request, _Strand, "_assemble_vector")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)

    @pytest.fixture
    def _measures_prop_(self, request):
        return property_mock(request, _Strand, "_measures")

    @pytest.fixture
    def _row_order_prop_(self, request):
        return property_mock(request, _Strand, "_row_order_signed_indexes")

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def scaled_counts_(self, request):
        return instance_mock(request, _ScaledCounts)


class Test_BaseOrderHelper:
    """Unit-test suite for `cr.cube.stripe.assembler._BaseOrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, HelperCls",
        (
            (CM.UNIVARIATE_MEASURE, _SortByMeasureHelper),
            (CM.LABEL, _SortByLabelHelper),
            (CM.EXPLICIT_ORDER, _OrderHelper),
            (CM.PAYLOAD_ORDER, _OrderHelper),
        ),
    )
    def test_it_dispatches_to_the_right_order_helper(
        self, request, measures_, collation_method, HelperCls
    ):
        rows_dimension_ = instance_mock(
            request,
            Dimension,
            order_spec=instance_mock(
                request, _OrderSpec, collation_method=collation_method
            ),
        )
        order_helper_ = instance_mock(
            request, HelperCls, _display_order=np.array([-2, 1, -1, 2])
        )
        HelperCls_ = class_mock(
            request,
            "cr.cube.stripe.assembler.%s" % HelperCls.__name__,
            return_value=order_helper_,
        )

        display_order = _BaseOrderHelper.display_order(
            rows_dimension_, measures_, format=ORDER_FORMAT.SIGNED_INDEXES
        )

        HelperCls_.assert_called_once_with(
            rows_dimension_, measures_, ORDER_FORMAT.SIGNED_INDEXES
        )
        assert display_order.tolist() == [-2, 1, -1, 2]

    @pytest.mark.parametrize(
        "pruning_base, expected_value",
        (([1, 1, 1], ()), ([1, 0, 1], (1,)), ([0, 0, 0], (0, 1, 2))),
    )
    def test_it_knows_the_empty_row_idxs_to_help(
        self, measures_, pruning_base, expected_value
    ):
        measures_.pruning_base = np.array(pruning_base)
        order_helper = _BaseOrderHelper(None, measures_)

        assert order_helper._empty_row_idxs == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, StripeMeasures)


class Test_OrderHelper:
    """Unit test suite for `cr.cube.stripe.assembler._OrderHelper` object."""

    @pytest.mark.parametrize(
        "collation_method, collator_class_name",
        (
            (CM.PAYLOAD_ORDER, "PayloadOrderCollator"),
            (CM.EXPLICIT_ORDER, "ExplicitOrderCollator"),
        ),
    )
    def test_it_computes_the_order_of_a_rows_dimension_to_help(
        self, request, collation_method, collator_class_name
    ):
        rows_dimension_ = instance_mock(
            request,
            Dimension,
            order_spec=instance_mock(
                request, _OrderSpec, collation_method=collation_method
            ),
        )
        CollatorCls_ = class_mock(
            request, "cr.cube.stripe.assembler.%s" % collator_class_name
        )
        CollatorCls_.display_order.return_value = (1, -2, 3, 5, -1)
        property_mock(request, _OrderHelper, "_empty_row_idxs", return_value=(2, 4, 6))
        order_helper = _OrderHelper(rows_dimension_, None)

        display_order = order_helper._display_order

        CollatorCls_.display_order.assert_called_once_with(
            rows_dimension_, (2, 4, 6), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert display_order == (1, -2, 3, 5, -1)


class Test_BaseSortByValueHelper:
    """Unit test suite for `cr.cube.strip.assembler._BaseSortByValueHelper`."""

    def test_it_computes_the_display_order_to_help(
        self,
        dimension_,
        _element_values_prop_,
        _subtotal_values_prop_,
        _empty_row_idxs_prop_,
        SortByValueCollator_,
    ):
        # --- return type of first two is ndarray in real life, but
        # --- assert_called_once_with() won't match on those, so use list instead.
        _element_values_prop_.return_value = [16, 3, 12]
        _subtotal_values_prop_.return_value = [15, 19]
        _empty_row_idxs_prop_.return_value = ()
        SortByValueCollator_.display_order.return_value = (-1, -2, 0, 2, 1)
        order_helper = _BaseSortByValueHelper(dimension_, None)

        order = order_helper._display_order

        SortByValueCollator_.display_order.assert_called_once_with(
            dimension_, [16, 3, 12], [15, 19], (), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (-1, -2, 0, 2, 1)

    def test_but_it_falls_back_to_payload_order_on_value_error(
        self,
        request,
        dimension_,
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
            request, "cr.cube.stripe.assembler.PayloadOrderCollator"
        )
        PayloadOrderCollator_.display_order.return_value = (1, 2, 3, 4)
        order_helper = _BaseSortByValueHelper(dimension_, None)

        order = order_helper._display_order

        PayloadOrderCollator_.display_order.assert_called_once_with(
            dimension_, (4, 2), ORDER_FORMAT.SIGNED_INDEXES
        )
        assert order == (1, 2, 3, 4)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _element_values_prop_(self, request):
        return property_mock(request, _BaseSortByValueHelper, "_element_values")

    @pytest.fixture
    def _empty_row_idxs_prop_(self, request):
        return property_mock(request, _BaseSortByValueHelper, "_empty_row_idxs")

    @pytest.fixture
    def SortByValueCollator_(self, request):
        return class_mock(request, "cr.cube.stripe.assembler.SortByValueCollator")

    @pytest.fixture
    def _subtotal_values_prop_(self, request):
        return property_mock(request, _BaseSortByValueHelper, "_subtotal_values")


class Test_SortByLabelHelper:
    """Unit test suite for `cr.cube.strip.assembler._SortByLabelHelper`."""

    def test_it_extracts_the_element_values_to_help(self, dimension_):
        dimension_.element_labels = ["b", "a", "c"]
        order_helper = _SortByLabelHelper(dimension_, None)

        assert order_helper._element_values.tolist() == ["b", "a", "c"]

    def test_it_extracts_the_subtotal_values_to_help(self, dimension_):
        dimension_.subtotal_labels = ["b", "a", "c"]
        order_helper = _SortByLabelHelper(dimension_, None)

        assert order_helper._subtotal_values.tolist() == ["b", "a", "c"]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Test_SortByMeasureHelper:
    """Unit test suite for `cr.cube.strip.assembler._SortByMeasureHelper`."""

    def test_it_extracts_the_element_values_to_help(self, _measure_prop_, measure_):
        _measure_prop_.return_value = measure_
        measure_.blocks = [np.arange(5), None]
        order_helper = _SortByMeasureHelper(None, None)

        assert order_helper._element_values.tolist() == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize(
        "json_name, internal_name",
        (
            ("base_unweighted", "unweighted_bases"),
            ("base_weighted", "weighted_bases"),
            ("count_unweighted", "unweighted_counts"),
            ("count_weighted", "weighted_counts"),
            ("mean", "means"),
            ("percent", "table_proportions"),
            ("percent_stddev", "table_proportion_stddevs"),
            ("percent_moe", "table_proportion_stderrs"),
            ("population", "population_proportions"),
            ("population_moe", "population_proportion_stderrs"),
            ("share_sum", "share_sum"),
            ("sum", "sums"),
        ),
    )
    def test_it_retrieves_the_right_measure_object_to_help(
        self,
        request,
        _order_spec_prop_,
        order_spec_,
        measure_,
        json_name,
        internal_name,
    ):
        measures_ = instance_mock(request, StripeMeasures)
        setattr(measures_, internal_name, measure_)
        _order_spec_prop_.return_value = order_spec_
        order_spec_.measure_keyname = json_name
        order_helper = _SortByMeasureHelper(None, measures_)

        assert order_helper._measure is measure_

    def test_but_it_raises_when_the_sort_measure_is_not_supported(
        self, _order_spec_prop_, order_spec_
    ):
        _order_spec_prop_.return_value = order_spec_
        order_spec_.measure_keyname = "foobar"
        order_helper = _SortByMeasureHelper(None, None)

        with pytest.raises(ValueError) as e:
            order_helper._measure

        assert str(e.value) == "sort-by-value for measure 'foobar' is not yet supported"

    def test_it_extracts_the_subtotal_values_to_help(self, _measure_prop_, measure_):
        _measure_prop_.return_value = measure_
        measure_.blocks = [None, np.arange(3)]
        order_helper = _SortByMeasureHelper(None, None)

        assert order_helper._subtotal_values.tolist() == [0, 1, 2]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def measure_(self, request):
        return instance_mock(request, _BaseSecondOrderMeasure)

    @pytest.fixture
    def _measure_prop_(self, request):
        return property_mock(request, _SortByMeasureHelper, "_measure")

    @pytest.fixture
    def order_spec_(self, request):
        return instance_mock(request, _OrderSpec)

    @pytest.fixture
    def _order_spec_prop_(self, request):
        return property_mock(request, _SortByMeasureHelper, "_order_spec")
