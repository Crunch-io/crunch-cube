# encoding: utf-8

"""Integration-test suite for smoothing feature."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR
from ..util import load_python_expression


class DescribeSliceSmoothing(object):
    @pytest.mark.parametrize(
        "fixture, window, expectation",
        (
            (CR.CAT_X_CAT_DATE, 1, "cat-x-cat-date-smoothed-col-pct-w1"),
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
    def it_provides_smoothed_col_pct_for_compatible_cubes(
        self, fixture, window, expectation
    ):
        slice_ = Cube(fixture).partitions[0]
        col_percent = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "column_percentages",
                "window": window,
            }
        )
        np.testing.assert_array_almost_equal(
            col_percent, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, window, expectation",
        (
            (CR.CAT_X_CAT_DATE, 3, "cat-x-cat-date-smoothed-scale-means-w3"),
            (CR.CAT_X_CAT_DATE_WGTD, 3, "cat-x-cat-date-smoothed-scale-means-w3"),
            (
                CR.CA_SUBVAR_X_CA_CAT_X_CAT_DATE,
                3,
                "ca-subvar-ca-cat-x-cat-date-scale-means-w3",
            ),
        ),
    )
    def it_provides_smoothed_scale_means_for_compatible_cubes(
        self, fixture, window, expectation
    ):

        slice_ = Cube(fixture).partitions[0]
        scale_means_row = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "scale_means_row",
                "window": window,
            }
        )
        np.testing.assert_array_almost_equal(
            scale_means_row, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (CR.CAT_X_MR, "cat-x-mr-unsmoothed-col-pct"),
            (CR.MR_X_MR, "mr-x-mr-unsmoothed-col-pct"),
            (CR.MR_X_CA_CAT_X_CA_SUBVAR, "mr-x-ca-cat-x-ca-subvar-unsmoothed-col-pct"),
            (CR.CAT_X_CAT, "cat-x-cat-unsmoothed-col-pct"),
            (CR.CAT_DATE_X_CAT, "cat-date-x-cat-unsmoothed-col-pct"),
        ),
    )
    def it_does_not_smooth_col_pct_for_incompatible_cubes(self, fixture, expectation):
        cube = Cube(fixture)
        slice_ = cube.partitions[0]
        col_percent = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "column_percentages",
                "window": 3,
            }
        )
        np.testing.assert_array_almost_equal(
            col_percent, load_python_expression(expectation)
        )

    def it_doesnt_smooth_counts_when_window_is_not_valid(self):
        slice_ = Cube(CR.CAT_X_CAT_DATE).partitions[0]
        col_percent = slice_.evaluate(
            {
                "function": "one_sided_moving_avg",
                "base_measure": "column_percentages",
                "window": 1,
            }
        )
        slice2_ = Cube(CR.CAT_X_CAT_DATE).partitions[0]
        col_percent2 = slice2_.column_percentages

        np.testing.assert_array_almost_equal(col_percent, col_percent2)


class DescribeStrandMeansSmoothing(object):
    def it_provides_smoothed_means_cat_date(self):
        strand_ = Cube(CR.CAT_DATE_MEAN).partitions[0]
        means = strand_.evaluate(
            {"function": "one_sided_moving_avg", "base_measure": "means", "window": 3}
        )
        np.testing.assert_array_almost_equal(
            means, [np.nan, np.nan, 2.65670765025029, 2.5774816240050358]
        )

    def it_doesnt_smoot_means_mr_mean_filt_wgtd(self):
        strand_ = Cube(CR.MR_MEAN_FILT_WGTD).partitions[0]
        means = strand_.evaluate(
            {"function": "one_sided_moving_avg", "base_measure": "means", "window": 3}
        )
        np.testing.assert_array_almost_equal(
            means, [3.724051, 2.578429, 2.218593, 1.865335]
        )
