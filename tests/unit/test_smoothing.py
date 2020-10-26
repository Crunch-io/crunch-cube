# encoding: utf-8

"""Unit test suite for `cr.cube.noa.smoothing` module."""

import numpy as np
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.noa.smoothing import SingleSidedMovingAvgSmoother

from ..unitutil import initializer_mock, instance_mock, property_mock


class DescribeSingleSideMovingAvgSmoother(object):
    def it_constructs_single_sided_moving_avg_to_help(self, request, slice_):
        _init_ = initializer_mock(request, SingleSidedMovingAvgSmoother)

        single_sided_miving_avg = SingleSidedMovingAvgSmoother(slice_, {})

        _init_.assert_called_once_with(single_sided_miving_avg, slice_, {})
        assert isinstance(single_sided_miving_avg, SingleSidedMovingAvgSmoother)

    @pytest.mark.parametrize(
        "categories, expected_value",
        (
            ([], False),
            ([{"date": "2019-10-10"}, {}], False),
            ([{"date": "2019-10-10"}, {"missing": False}], False),
            ([{"date": "2019-10-10"}, {"date": "2019-11-10"}], True),
            ([{"date": "2019-10-10"}, {"date": "2019-11-10"}, {"missing": True}], True),
        ),
    )
    def it_knows_its_is_cat_date_prop(self, categories, expected_value, slice_):
        dimension_dict = {"type": {"categories": categories}}
        slice_.smoothed_dimension_dict = dimension_dict

        _is_cat_date = SingleSidedMovingAvgSmoother(slice_, {})._is_cat_date

        assert _is_cat_date is expected_value

    @pytest.mark.parametrize(
        "window, is_cat_date, expected_value",
        (
            (2, True, True),
            (2, False, False),
            (3, False, False),
            (3, False, False),
            (1, True, False),
            (0, True, False),
            (3, True, True),
        ),
    )
    def it_knows_its_can_smooth_property(
        self,
        _is_cat_date_prop_,
        _base_values_prop_,
        _window_prop_,
        window,
        is_cat_date,
        expected_value,
    ):
        _is_cat_date_prop_.return_value = is_cat_date
        _window_prop_.return_value = window
        _base_values_prop_.return_value = np.random.rand(3, 5)

        can_smooth = SingleSidedMovingAvgSmoother(None, {})._can_smooth

        assert can_smooth is expected_value

    def but_it_raises_a_warning_when_window_is_invalid(
        self, _base_values_prop_, _window_prop_
    ):
        _base_values_prop_.return_value = np.random.rand(2, 3)
        _window_prop_.return_value = 1

        with pytest.warns(UserWarning) as record:
            can_smooth = SingleSidedMovingAvgSmoother(None, {})._can_smooth

        assert (
            record[0].message.args[0]
            == "No smoothing performed. Window (value: 1) parameter is not valid: "
            "window must be less than equal to the total period (value: 3) and "
            "positive"
        )
        assert can_smooth is False

    def and_it_raises_a_warning_when_dim_is_not_a_categorical_date(
        self, _is_cat_date_prop_, _base_values_prop_, _window_prop_
    ):
        _is_cat_date_prop_.return_value = False
        _base_values_prop_.return_value = np.random.rand(4, 5)
        _window_prop_.return_value = 3

        with pytest.warns(UserWarning) as record:
            can_smooth = SingleSidedMovingAvgSmoother(None, {})._can_smooth

        assert (
            record[0].message.args[0]
            == "No smoothing performed. Column dimension must be a categorical date"
        )
        assert can_smooth is False

    @pytest.mark.parametrize(
        "base_values, window, expected_value",
        (
            (np.array([1, 4, 6, 7, 8, 10]), 3, [3.666667, 5.666667, 7.0, 8.333333]),
            (np.array([[3, 4, 5, 6], [7, 8, 9, 1]]), 1, [[3, 4, 5, 6], [7, 8, 9, 1]]),
            (
                np.array([[3, 4, 5, 6], [7, 8, 9, 1]]),
                2,
                [[3.5, 4.5, 5.5], [7.5, 8.5, 5.0]],
            ),
            (np.array([[3, 4, 5, 6], [7, 8, 9, 1]]), 3, [[4.0, 5.0], [8.0, 6.0]]),
            (np.array([[3, 4, 5, 6], [7, 8, 9, 1]]), 4, [[4.5], [6.25]]),
        ),
    )
    def it_knows_its_smoothed_values(
        self, _base_values_prop_, base_values, window, expected_value
    ):
        _base_values_prop_.return_value = base_values
        function_spec = {"window": window}

        _smoothed_values_ = SingleSidedMovingAvgSmoother(
            None, function_spec
        )._smoothed_values

        np.testing.assert_array_almost_equal(_smoothed_values_, expected_value)

    def it_knows_its_base_values(self, slice_):
        slice_.column_percentages = np.array([[0.3, 0.4], [0.5, 0.6]])
        function_spec = {"base_measure": "column_percentages"}

        _base_vaules = SingleSidedMovingAvgSmoother(slice_, function_spec)._base_values

        np.testing.assert_array_almost_equal(_base_vaules, [[0.3, 0.4], [0.5, 0.6]])

    @pytest.mark.parametrize(
        "base_values, show_smoothing, smoothed_values, expected_value",
        (
            (
                np.array([1, 4, 6, 7, 8, 10]),
                True,
                np.array([3.666667, 5.666667, 7.0, 8.333333]),
                [np.nan, np.nan, 3.666667, 5.666667, 7.0, 8.333333],
            ),
            (np.array([1, 4, 6, 7, 8, 10]), False, [], [1, 4, 6, 7, 8, 10]),
        ),
    )
    def it_provides_complete_smoothed_values(
        self,
        _smoothed_values_prop_,
        _base_values_prop_,
        _can_smooth_prop_,
        base_values,
        show_smoothing,
        smoothed_values,
        expected_value,
    ):
        _base_values_prop_.return_value = base_values
        _smoothed_values_prop_.return_value = smoothed_values
        _can_smooth_prop_.return_value = show_smoothing

        values = SingleSidedMovingAvgSmoother(None, {}).values

        np.testing.assert_array_almost_equal(values, expected_value)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def slice_(self, request):
        return instance_mock(request, _Slice)

    @pytest.fixture
    def _is_cat_date_prop_(self, request):
        return property_mock(request, SingleSidedMovingAvgSmoother, "_is_cat_date")

    @pytest.fixture
    def _base_values_prop_(self, request):
        return property_mock(request, SingleSidedMovingAvgSmoother, "_base_values")

    @pytest.fixture
    def _smoothed_values_prop_(self, request):
        return property_mock(request, SingleSidedMovingAvgSmoother, "_smoothed_values")

    @pytest.fixture
    def _can_smooth_prop_(self, request):
        return property_mock(request, SingleSidedMovingAvgSmoother, "_can_smooth")

    @pytest.fixture
    def _window_prop_(self, request):
        return property_mock(request, SingleSidedMovingAvgSmoother, "_window")
