# encoding: utf-8

"""Unit test suite for `cr.cube.noa.smoothing` module."""

import pytest

from cr.cube.cubepart import _Slice
from cr.cube.noa.smoothing import SingleSidedMovingAvgSmoother

from ..unitutil import initializer_mock, instance_mock


class DescribeSingleSideMovingAvgSmoother(object):
    def it_constructs_single_sided_moving_avg_to_help(self, request, slice_):
        _init_ = initializer_mock(request, SingleSidedMovingAvgSmoother)

        single_sided_miving_avg = SingleSidedMovingAvgSmoother(slice_, **{})

        _init_.assert_called_once_with(single_sided_miving_avg, slice_, **{})
        assert isinstance(single_sided_miving_avg, SingleSidedMovingAvgSmoother)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def slice_(self, request):
        return instance_mock(request, _Slice)
