# encoding: utf-8

"""Integration-test suite for smoothing feature."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR
from ..util import load_python_expression


class DescribeSliceSmoothing(object):
    """Integration-test suite for _Slice.evaluate() method."""

    @pytest.mark.parametrize(
        "fixture, window, expectation",
        (
            (CR.CAT_X_CAT_DATE_WGTD, 4, "cat-x-cat-date-wgtd-smoothed-col-idx-w4"),
            (CR.CAT_X_MR_X_CAT_DATE, 3, "cat-x-mr-x-cat-date-smoothed-col-idx-w3"),
            (
                CR.CA_SUBVAR_X_CA_CAT_X_CAT_DATE,
                3,
                "ca-subvar-x-ca-cat-cat-date-smoothed-col-idx-w3",
            ),
            (CR.CAT_HS_X_CAT_DATE, 3, "cat-hs-x-cat-date-smoothed-col-idx-w3"),
            (CR.MR_X_CAT_DATE, 3, "mr-x-cat-date-smoothed-col-idx-w3"),
            (CR.NUMERIC_X_CAT_DATE, 3, "numeric-x-cat-date-smoothed-col-idx-w3"),
            (CR.TXT_X_CAT_DATE, 3, "txt-x-cat-date-smoothed-col-idx-w3"),
            (CR.DATETIME_X_CAT_DATE, 3, "datetime-x-cat-date-smoothed-col-idx-w3"),
        ),
    )
    def it_provides_smoothed_col_index_for_compatible_cubes(
        self, fixture, window, expectation
    ):
        slice_ = Cube(fixture).partitions[0]
        col_index = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "col_index",
                "window": window,
            }
        )
        np.testing.assert_array_almost_equal(
            col_index, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, window, expectation",
        (
            (CR.CAT_X_CAT_DATE_WGTD, 4, "cat-x-cat-date-wgtd-smoothed-col-pct-w4"),
            (CR.CAT_X_MR_X_CAT_DATE, 3, "cat-x-mr-x-cat-date-smoothed-col-pct-w3"),
            (
                CR.CA_SUBVAR_X_CA_CAT_X_CAT_DATE,
                3,
                "ca-subvar-x-ca-cat-cat-date-smoothed-col-pct-w3",
            ),
            (CR.CAT_HS_X_CAT_DATE, 3, "cat-hs-x-cat-date-smoothed-col-pct-w3"),
            (CR.MR_X_CAT_DATE, 3, "mr-x-cat-date-smoothed-col-pct-w3"),
            (CR.NUMERIC_X_CAT_DATE, 3, "numeric-x-cat-date-smoothed-col-pct-w3"),
            (CR.TXT_X_CAT_DATE, 3, "txt-x-cat-date-smoothed-col-pct-w3"),
            (CR.DATETIME_X_CAT_DATE, 3, "datetime-x-cat-date-smoothed-col-pct-w3"),
        ),
    )
    def it_provides_smoothed_col_percent_for_compatible_cubes(
        self, fixture, window, expectation
    ):
        slice_ = Cube(fixture).partitions[0]
        col_percent = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "col_percent",
                "window": window,
            }
        )
        np.testing.assert_array_almost_equal(
            col_percent, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (CR.CAT_X_CAT_DATE, "cat-x-cat-date-smoothed-scale-means-w2"),
            (CR.CAT_X_CAT_DATE_WGTD, "cat-x-cat-date-smoothed-scale-means-w2"),
            (
                CR.CA_SUBVAR_X_CA_CAT_X_CAT_DATE,
                "ca-subvar-ca-cat-x-cat-date-scale-means-w2",
            ),
        ),
    )
    def it_provides_smoothed_scale_means_for_compatible_cubes(
        self, fixture, expectation
    ):

        slice_ = Cube(fixture).partitions[0]
        # --- window not expressed get the default value : 2
        scale_mean = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "scale_mean",
                "window": 2,
            }
        )
        np.testing.assert_array_almost_equal(
            scale_mean, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture",
        (CR.CAT_X_MR, CR.MR_X_MR, CR.MR_X_CA_CAT_X_CA_SUBVAR, CR.CAT_DATE_X_CAT),
    )
    def it_warns_and_does_not_smooth_when_dimension_is_not_smoothable(self, fixture):
        slice_ = Cube(fixture).partitions[0]
        base_values = slice_.column_percentages
        expected_warning_regex = (
            r"No smoothing performed. Column dimension must be a categorical date."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            smoothed_values = slice_.evaluate(
                {
                    "function": "one_sided_moving_avg",
                    "base_measure": "col_percent",
                    "window": 3,
                }
            )

        np.testing.assert_array_almost_equal(smoothed_values, base_values)

    @pytest.mark.parametrize(
        "fixture, base_measure, prop_name, periods, window",
        (
            (CR.CAT_X_CAT_DATE, "col_percent", "column_percentages", 4, 1),
            (CR.CAT_X_CAT_DATE, "col_index", "column_index", 4, 1),
            (CR.CAT_X_CAT, "col_percent", "column_percentages", 2, 3),
        ),
    )
    def it_warns_and_does_not_smooth_when_window_is_invalid(
        self, fixture, base_measure, prop_name, periods, window
    ):
        slice_ = Cube(fixture).partitions[0]
        base_values = getattr(slice_, prop_name)
        expected_warning_regex = (
            r"No smoothing performed. Smoothing window must be between 2 and the "
            r"number of periods \(%d\), got %d" % (periods, window)
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            smoothed_values = slice_.evaluate(
                {
                    "function": "one_sided_moving_avg",
                    "base_measure": base_measure,
                    "window": window,
                }
            )

        np.testing.assert_array_almost_equal(smoothed_values, base_values)


class DescribeStrandMeansSmoothing(object):
    """Integration-test suite for _Strand.evaluate() method."""

    def it_provides_smoothed_means_cat_date(self):
        strand_ = Cube(CR.CAT_DATE_MEAN).partitions[0]
        means = strand_.evaluate(
            {"function": "one_sided_moving_avg", "base_measure": "mean", "window": 3}
        )
        np.testing.assert_array_almost_equal(
            means, [np.nan, np.nan, 2.65670765025029, 2.5774816240050358]
        )

    def it_does_not_smooth_means_mr_mean_filt_wgtd(self):
        strand_ = Cube(CR.MR_MEAN_FILT_WGTD).partitions[0]
        expected_warning_regex = (
            r"No smoothing performed. Column dimension must be a categorical date."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            means = strand_.evaluate(
                {
                    "function": "one_sided_moving_avg",
                    "base_measure": "mean",
                    "window": 3,
                }
            )
        np.testing.assert_array_almost_equal(
            means, [3.724051, 2.578429, 2.218593, 1.865335]
        )
