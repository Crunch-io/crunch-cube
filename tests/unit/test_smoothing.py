# encoding: utf-8

"""Unit test suite for `cr.cube.smoothing` module."""

import numpy as np
import pytest

from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.smoothing import Smoother, _SingleSidedMovingAvgSmoother

from ..unitutil import class_mock, initializer_mock, instance_mock


class DescribeSmoother:
    def it_provides_a_factory_for_constructing_smoother_objects(self, request):
        _SingleSidedMovingAvgSmoother_ = class_mock(
            request,
            "cr.cube.smoothing._SingleSidedMovingAvgSmoother",
            return_value=_SingleSidedMovingAvgSmoother,
        )
        dimension_ = instance_mock(request, Dimension)
        dimension_.dimension_type = DT.CAT_DATE
        dimension_.smoothing_dict = {"function": "one_sided_moving_avg", "window": 3}
        smoother = Smoother.factory(dimension_)

        assert smoother.__name__ == "_SingleSidedMovingAvgSmoother"
        _SingleSidedMovingAvgSmoother_.assert_called_once_with(
            {"function": "one_sided_moving_avg", "window": 3}, DT.CAT_DATE
        )

    def but_it_raises_an_exception_when_function_is_not_implemented(self, request):
        dimension_ = instance_mock(request, Dimension)
        dimension_.smoothing_dict = {"function": "foo", "window": 3}

        with pytest.raises(NotImplementedError) as e:
            Smoother.factory(dimension_)

        assert str(e.value) == "Function foo is not available."


class Describe_SingleSideMovingAvgSmoother(object):
    def it_constructs_single_sided_moving_avg_to_help(self, request):
        _init_ = initializer_mock(request, _SingleSidedMovingAvgSmoother)
        single_sided_miving_avg = _SingleSidedMovingAvgSmoother(
            smoothing_dict=3, dimension_type=DT.CAT_DATE
        )

        _init_.assert_called_once_with(single_sided_miving_avg, 3, DT.CAT_DATE)
        assert isinstance(single_sided_miving_avg, _SingleSidedMovingAvgSmoother)

    @pytest.mark.parametrize(
        "smoothing_dict, base_values, dimension_type, expected_value",
        (
            ({"window": 2}, np.random.rand(3, 5), DT.CAT_DATE, True),
            ({"window": 3}, np.random.rand(3, 5), DT.CAT_DATE, True),
            ({"window": 3}, np.array([]), DT.CAT_DATE, False),
        ),
    )
    def it_knows_when_it_can_apply_smoothing(
        self, smoothing_dict, base_values, dimension_type, expected_value
    ):
        smoother = _SingleSidedMovingAvgSmoother(smoothing_dict, dimension_type)

        assert smoother._can_smooth(base_values) is expected_value

    @pytest.mark.parametrize(
        "window, base_values, dimension_type",
        (
            (12, np.random.rand(3, 5), DT.CAT_DATE),
            (1, np.random.rand(3, 5), DT.CAT_DATE),
        ),
    )
    def but_it_warns_when_window_is_invalid(self, window, base_values, dimension_type):
        smoother = _SingleSidedMovingAvgSmoother(
            smoothing_dict={"window": window}, dimension_type=dimension_type
        )
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
        self, window, base_values, dimension_type
    ):
        smoother = _SingleSidedMovingAvgSmoother(
            smoothing_dict={"window": window}, dimension_type=dimension_type
        )
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
    def it_provides_complete_smoothed_values(self, base_values, window, expected_value):
        smoother = _SingleSidedMovingAvgSmoother(
            smoothing_dict={"window": window}, dimension_type=DT.CAT_DATE
        )

        assert smoother.smooth(base_values) == pytest.approx(
            expected_value, nan_ok=True
        )
