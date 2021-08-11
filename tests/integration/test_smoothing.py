# encoding: utf-8

"""Integration-test suite for smoothing feature."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR, NA
from ..util import load_python_expression


class DescribeSliceSmoothing:
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
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": window,
                }
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.smoothed_column_index, load_python_expression(expectation)
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
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": window,
                }
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.smoothed_column_percentages, load_python_expression(expectation)
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
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": 2,
                }
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]
        # --- window not expressed get the default value : 2
        np.testing.assert_array_almost_equal(
            slice_.smoothed_columns_scale_mean, load_python_expression(expectation)
        )

    def it_provides_smoothed_means_for_numeric_array(self):
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": 2,
                }
            }
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_DATE, transforms=transforms
        ).partitions[0]

        expectation = np.array(
            load_python_expression("num-array-means-grouped-by-cat-date-smoothed")
        )
        assert slice_.smoothed_means == pytest.approx(expectation, nan_ok=True)

    @pytest.mark.parametrize(
        "fixture",
        (CR.CAT_X_MR, CR.MR_X_MR, CR.MR_X_CA_CAT_X_CA_SUBVAR, CR.CAT_DATE_X_CAT),
    )
    def it_warns_and_does_not_smooth_when_dimension_is_not_smoothable(self, fixture):
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": 3,
                }
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]
        expected_warning_regex = (
            r"No smoothing performed. Column dimension must be a categorical date."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            smoothed_values = slice_.smoothed_column_percentages

        slice_ = Cube(fixture).partitions[0]
        base_values = slice_.column_percentages

        assert smoothed_values.tolist() == base_values.tolist()

    @pytest.mark.parametrize(
        "fixture, smoothed_prop_name, prop_name, periods, window",
        (
            (
                CR.CAT_X_CAT_DATE,
                "smoothed_column_percentages",
                "smoothed_column_percentages",
                4,
                1,
            ),
            (CR.CAT_X_CAT_DATE, "smoothed_column_index", "column_index", 4, 1),
            (
                CR.CAT_X_CAT_DATE,
                "smoothed_columns_scale_mean",
                "columns_scale_mean",
                4,
                1,
            ),
        ),
    )
    def it_warns_and_does_not_smooth_when_window_is_invalid(
        self, fixture, smoothed_prop_name, prop_name, periods, window
    ):
        transforms = {
            "columns_dimension": {
                "smoother": {
                    "function": "one_sided_moving_avg",
                    "window": window,
                }
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]
        expected_warning_regex = (
            r"No smoothing performed. Smoothing window must be between 2 and the "
            r"number of periods \(%d\), got %d" % (periods, window)
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            smoothed_values = getattr(slice_, smoothed_prop_name)

        base_values = getattr(slice_, prop_name)

        np.testing.assert_array_almost_equal(smoothed_values, base_values)

    def it_uses_default_smoothing_if_smoother_is_not_specified(self):
        slice_ = Cube(CR.CAT_X_CAT_DATE).partitions[0]

        assert slice_.smoothed_column_percentages == pytest.approx(
            np.array(
                [
                    [np.nan, 28.4013529661, 30.877106856, 35.7038771792],
                    [np.nan, 42.5027849229, 47.5045000818, 47.491543065],
                    [np.nan, 13.8136228302, 13.0829651448, 11.170960187],
                    [np.nan, 5.12648613614, 3.74406807396, 1.54306531355],
                    [np.nan, 10.1557531444, 4.7913598429, 4.0905542544],
                ]
            ),
            nan_ok=True,
        )


class DescribeStrandMeansSmoothing:
    """Integration-test suite for _Strand method."""

    def it_provides_smoothed_means_cat_date(self):
        transforms = {
            "rows_dimension": {
                "smoother": {"function": "one_sided_moving_avg", "window": 3}
            }
        }
        strand_ = Cube(CR.CAT_DATE_MEAN, transforms=transforms).partitions[0]

        np.testing.assert_array_almost_equal(
            strand_.smoothed_means,
            [np.nan, np.nan, 2.65670765025029, 2.5774816240050358],
        )

    def it_does_not_smooth_means_mr_mean_filt_wgtd(self):
        transforms = {
            "rows_dimension": {
                "smoother": {"function": "one_sided_moving_avg", "window": 3}
            }
        }
        strand_ = Cube(CR.MR_MEAN_FILT_WGTD, transforms=transforms).partitions[0]
        expected_warning_regex = (
            r"No smoothing performed. Row dimension must be a categorical date."
        )

        with pytest.warns(UserWarning, match=expected_warning_regex):
            means = strand_.smoothed_means

        assert means == pytest.approx([3.724051, 2.578429, 2.218593, 1.865335])
