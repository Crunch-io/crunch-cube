# encoding: utf-8

"""Unit test suite for `cr.cube.noa.smoothing` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.smoothing import SingleSidedMovingAvgSmoother

from ..unitutil import initializer_mock, instance_mock


class DescribeSingleSideMovingAvgSmoother:
    def it_constructs_single_sided_moving_avg_to_help(self, request, slice_):
        _init_ = initializer_mock(request, SingleSidedMovingAvgSmoother)

        single_sided_miving_avg = SingleSidedMovingAvgSmoother(
            window=3, dimension=dimension_
        )

        _init_.assert_called_once_with(single_sided_miving_avg, 3, dimension_)
        assert isinstance(single_sided_miving_avg, SingleSidedMovingAvgSmoother)

    @pytest.mark.parametrize(
        "window, base_values, dimension_type, expected_value",
        (
            (2, np.random.rand(3, 5), DT.CAT_DATE, True),
            (3, np.random.rand(3, 5), DT.CAT_DATE, True),
            (3, np.array([]), DT.CAT_DATE, False),
        ),
    )
    def it_knows_when_it_can_apply_smoothing(
        self, dimension_, window, base_values, dimension_type, expected_value
    ):
        dimension_.dimension_type = dimension_type
        smoother = SingleSidedMovingAvgSmoother(window, dimension_)

        assert smoother._can_smooth(base_values) is expected_value

    @pytest.mark.parametrize(
        "window, base_values, dimension_type",
        (
            (12, np.random.rand(3, 5), DT.CAT_DATE),
            (1, np.random.rand(3, 5), DT.CAT_DATE),
        ),
    )
    def but_it_warns_when_window_is_invalid(
        self, dimension_, window, base_values, dimension_type
    ):
        dimension_.dimension_type = dimension_type
        smoother = SingleSidedMovingAvgSmoother(window, dimension_)
        expected_warning_regex = (
            r"No smoothing performed. Smoothing window must be between 2 and the "
            r"number of periods \(%d\), got %d" % (base_values.shape[-1], window)
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            can_smooth = smoother._can_smooth(base_values)

        assert can_smooth is False

    @pytest.mark.parametrize(
        "window, base_values, dimension_type",
        (
            (2, np.random.rand(3, 5), DT.MR),
            (3, np.random.rand(3, 5), DT.CAT),
        ),
    )
    def and_it_warns_when_dim_is_not_a_categorical_date(
        self, dimension_, window, base_values, dimension_type
    ):
        dimension_.dimension_type = dimension_type
        smoother = SingleSidedMovingAvgSmoother(window, dimension_)
        expected_warning_regex = (
            r"No smoothing performed. Column dimension must be a categorical date."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            can_smooth = smoother._can_smooth(base_values)

        assert can_smooth is False

    @pytest.mark.parametrize(
        "base_values, window, expected_value",
        (
            (np.array([1, 4, 6, 7, 8, 10]), 2, [np.nan, 2.5, 5.0, 6.5, 7.5, 9.0]),
            (
                np.array([1, 4, 6, 7, 8, 10]),
                3,
                [np.nan, np.nan, 3.66666667, 5.66666667, 7.0, 8.33333333],
            ),
            (np.array([]), 3, []),
        ),
    )
    def it_provides_complete_smoothed_values(
        self, dimension_, base_values, window, expected_value
    ):
        dimension_.dimension_type = DT.CAT_DATE
        smoother = SingleSidedMovingAvgSmoother(window, dimension_)

        assert smoother.smooth(base_values) == pytest.approx(
            expected_value, nan_ok=True
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)
