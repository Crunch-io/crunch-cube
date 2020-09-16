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
    def it_provides_smoothed_col_percent_for_cat_x_cat_date(
        self, cat_x_cat_date_col_percent_fixture
    ):
        window, expectation = cat_x_cat_date_col_percent_fixture
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": window,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
        )

    def it_provides_smoothed_col_percent_for_cat_x_cat_date_wgtd(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_CAT_DATE_WGTD, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages,
            [
                [np.nan, np.nan, 29.622066, 33.699492],
                [np.nan, np.nan, 43.7849254, 47.88095998],
                [np.nan, np.nan, 14.28844697, 11.08991848],
                [np.nan, np.nan, 4.26421827, 2.67819475],
                [np.nan, np.nan, 8.04034337, 4.6514348],
            ],
        )

    def it_provides_smoothed_col_percent_for_cat_x_mr_x_cat_date(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_MR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages,
            load_python_expression("cat-x-mr-x-cat-date-smoothed-col-pct"),
        )

    def it_provides_smoothed_col_percent_for_ca_x_ca_subvar_x_cat_date(
        self, ca_x_ca_subvar_x_cat_date_col_percent_fixture
    ):
        window, expectation = ca_x_ca_subvar_x_cat_date_col_percent_fixture
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": window,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CA_X_CA_SUBVAR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
        )

    def it_provides_smoothed_col_percent_for_mr_x_cat_date(
        self, mr_x_cat_date_col_percent_fixture
    ):
        window, expectation = mr_x_cat_date_col_percent_fixture
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": window,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.MR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
        )

    def it_doesnt_smooth_counts_when_window_is_not_valid(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 30,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]

        cube2 = Cube(CR.CAT_X_CAT_DATE)
        slice2_ = cube2.partitions[0]

        np.testing.assert_array_almost_equal(slice_.counts, slice2_.counts)

    def it_provides_smoothed_col_percent_for_cat_hs_x_cat_date(
        self, cat_hs_x_cat_date_col_percent_fixture
    ):
        window, expectation = cat_hs_x_cat_date_col_percent_fixture
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": window,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_HS_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages, load_python_expression(expectation)
        )

    def it_doesnt_smooth_col_percent_for_cat_x_mr(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_MR, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages,
            [[30.0, 35.29411765, 31.57894737], [70.0, 64.70588235, 68.42105263]],
        )

    def it_doesnt_smooth_col_percent_for_mr_x_mr(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.MR_X_MR, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages,
            [
                [100.0, 13.3024034, 12.3912447, 22.804396],
                [28.5669365, 100.0, 23.4988046, 47.7518371],
                [43.4566976, 34.959546, 100.0, 72.8388746],
                [100.0, 100.0, 100.0, 100.0],
            ],
        )

    def it_doesnt_smooth_col_percent_for_mr_x_ca_cat_x_ca_subvar(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.MR_X_CA_CAT_X_CA_SUBVAR, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages,
            load_python_expression("mr-x-ca-cat-x-ca-subvar-smoothed-col-pct"),
        )

    def it_doesnt_smooth_col_percent_cat_x_cat(self):
        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_X_CAT, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_percentages, [[50.0, 40.0], [50.0, 60.0]]
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (1, "cat-hs-x-cat-date-smoothed-col-pct-window-1"),
            (2, "cat-hs-x-cat-date-smoothed-col-pct-window-2"),
            (3, "cat-hs-x-cat-date-smoothed-col-pct-window-3"),
            (4, "cat-hs-x-cat-date-smoothed-col-pct-window-4"),
        ]
    )
    def cat_hs_x_cat_date_col_percent_fixture(self, request):
        window, expectation = request.param
        return window, expectation

    @pytest.fixture(
        params=[
            (1, "cat-x-cat-date-smoothed-col-pct-window-1"),
            (2, "cat-x-cat-date-smoothed-col-pct-window-2"),
            (3, "cat-x-cat-date-smoothed-col-pct-window-3"),
            (4, "cat-x-cat-date-smoothed-col-pct-window-4"),
        ]
    )
    def cat_x_cat_date_col_percent_fixture(self, request):
        window, expectation = request.param
        return window, expectation

    @pytest.fixture(
        params=[
            (1, "mr-x-cat-date-smoothed-col-pct-window-1"),
            (2, "mr-x-cat-date-smoothed-col-pct-window-2"),
            (3, "mr-x-cat-date-smoothed-col-pct-window-3"),
            (4, "mr-x-cat-date-smoothed-col-pct-window-4"),
        ]
    )
    def mr_x_cat_date_col_percent_fixture(self, request):
        window, expectation = request.param
        return window, expectation

    @pytest.fixture(
        params=[
            (1, "ca-x-ca-subvar-x-cat-date-smoothed-col-pct-window-1"),
            (2, "ca-x-ca-subvar-x-cat-date-smoothed-col-pct-window-2"),
            (3, "ca-x-ca-subvar-x-cat-date-smoothed-col-pct-window-3"),
            (4, "ca-x-ca-subvar-x-cat-date-smoothed-col-pct-window-4"),
        ]
    )
    def ca_x_ca_subvar_x_cat_date_col_percent_fixture(self, request):
        window, expectation = request.param
        return window, expectation


class DescribeStrandMeansSmoothing(object):
    def it_provides_smoothed_means_cat_date(self):
        transforms = {
            "rows_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.CAT_DATE_MEAN, transforms=transforms)
        strand_ = cube.partitions[0]

        np.testing.assert_array_almost_equal(
            strand_.means, [np.nan, np.nan, 2.65670765025029, 2.5774816240050358]
        )

    def it_doesnt_smoot_means_mr_mean_filt_wgtd(self):
        transforms = {
            "rows_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": True,
                }
            }
        }
        cube = Cube(CR.MR_MEAN_FILT_WGTD, transforms=transforms)
        strand_ = cube.partitions[0]

        np.testing.assert_array_almost_equal(
            strand_.means, [3.724051, 2.578429, 2.218593, 1.865335]
        )
