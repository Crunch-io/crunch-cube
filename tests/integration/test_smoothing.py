# encoding: utf-8

"""Integration-test suite for smoothing feature."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cube import Cube

# ---mnemonic: CR = 'cube-response'---
# ---mnemonic: TR = 'transforms'---
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
        transforms = {
            "smoothing": {
                "method": "one_side_moving_avg",
                "window": window,
                "show": True,
            }
        }
        slice_ = Cube(fixture, transforms=transforms).partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
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
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(fixture, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
        )

    def it_doesnt_smooth_counts_when_window_is_not_valid(self):
        transforms = {"smoothing": {"window": 30, "show": True}}
        slice_ = Cube(CR.CAT_X_CAT_DATE, transforms=transforms).partitions[0]
        slice2_ = Cube(CR.CAT_X_CAT_DATE).partitions[0]
        np.testing.assert_array_almost_equal(slice_.counts, slice2_.counts)


class DescribeStrandMeansSmoothing(object):
    def it_provides_smoothed_means_cat_date(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        strand_ = Cube(CR.CAT_DATE_MEAN, transforms=transforms).partitions[0]
        np.testing.assert_array_almost_equal(
            strand_.means, [np.nan, np.nan, 2.65670765025029, 2.5774816240050358]
        )

    def it_doesnt_smoot_means_mr_mean_filt_wgtd(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        strand_ = Cube(CR.MR_MEAN_FILT_WGTD, transforms=transforms).partitions[0]
        np.testing.assert_array_almost_equal(
            strand_.means, [3.724051, 2.578429, 2.218593, 1.865335]
        )
