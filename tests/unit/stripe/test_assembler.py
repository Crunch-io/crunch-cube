# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.assembler` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.stripe.assembler import StripeAssembler
from cr.cube.stripe.measure import StripeMeasures, _UnweightedCounts

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
        (("unweighted_counts", _UnweightedCounts),),
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
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)
