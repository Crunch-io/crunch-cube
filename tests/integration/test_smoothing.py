# encoding: utf-8

"""Integration-test suite for smoothing feature."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cube import Cube

# ---mnemonic: CR = 'cube-response'---
# ---mnemonic: TR = 'transforms'---
from ..fixtures import CR


class DescribeSliceSmoothing(object):
    def it_provides_smoothed_counts_for_cat_x_cat_date(
        self, cat_x_cat_date_counts_fixture
    ):
        window, expected_value = cat_x_cat_date_counts_fixture
        transforms = {
            "smoothing": {
                "method": "one_side_moving_avg",
                "window": window,
                "show": True,
            }
        }
        cube = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(slice_.counts, expected_value)

        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": False}
        }
        cube2 = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [138, 144, 101, 72],
                [185, 236, 146, 89],
                [85, 53, 48, 13],
                [27, 24, 8, 1],
                [74, 28, 12, 8],
            ],
        )
        assert slice_.shape == slice2_.shape

    def it_provides_smoothed_counts_for_cat_x_cat_date_wgtd(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(CR.CAT_X_CAT_DATE_WGTD, transforms=transforms)
        slice_ = cube.partitions[0]

        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [np.nan, np.nan, 131.35327524, 100.25662157],
                [np.nan, np.nan, 196.09532984, 152.55817888],
                [np.nan, np.nan, 63.00400068, 36.14178289],
                [np.nan, np.nan, 21.21684549, 11.91461353],
                [np.nan, np.nan, 40.41337078, 16.21162515],
            ],
        )

        cube2 = Cube(CR.CAT_X_CAT_DATE_WGTD)
        slice2_ = cube2.partitions[0]

        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [146.40471513, 175.96741344, 71.68769716, 53.1147541],
                [196.26719057, 288.3910387, 103.62776025, 65.6557377],
                [90.17681729, 64.76578411, 34.06940063, 9.59016393],
                [28.64440079, 29.32790224, 5.67823344, 0.73770492],
                [78.50687623, 34.21588595, 8.51735016, 5.90163934],
            ],
        )
        assert slice_.shape == slice2_.shape

    def it_provides_smoothed_counts_for_cat_x_mr_x_cat_date(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(CR.CAT_X_MR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.counts,
            [
                [np.nan, np.nan, 26.66666667, 19.66666667],
                [np.nan, np.nan, 74.0, 62.66666667],
                [np.nan, np.nan, 34.66666667, 26.33333333],
                [np.nan, np.nan, 42.33333333, 34.33333333],
                [np.nan, np.nan, 7.0, 6.0],
                [np.nan, np.nan, 10.33333333, 8.0],
                [np.nan, np.nan, 2.33333333, 1.33333333],
                [np.nan, np.nan, 65.0, 51.33333333],
                [np.nan, np.nan, 10.0, 11.33333333],
                [np.nan, np.nan, 22.33333333, 20.33333333],
                [np.nan, np.nan, 34.33333333, 26.33333333],
                [np.nan, np.nan, 17.33333333, 13.0],
                [np.nan, np.nan, 12.33333333, 11.66666667],
                [np.nan, np.nan, 6.33333333, 6.33333333],
                [np.nan, np.nan, 24.0, 19.33333333],
                [np.nan, np.nan, 1.66666667, 0.33333333],
            ],
        )

        transforms = {"smoothing": {"method": "one_side_moving_avg", "window": 3}}
        cube2 = Cube(CR.CAT_X_MR_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [35, 27, 18, 14],
                [77, 86, 59, 43],
                [43, 38, 23, 18],
                [47, 49, 31, 23],
                [5, 12, 4, 2],
                [10, 17, 4, 3],
                [3, 3, 1, 0],
                [74, 75, 46, 33],
                [5, 14, 11, 9],
                [24, 23, 20, 18],
                [41, 39, 23, 17],
                [19, 23, 10, 6],
                [12, 12, 13, 10],
                [5, 8, 6, 5],
                [23, 25, 24, 9],
                [4, 0, 1, 0],
            ],
        )
        for i, p in enumerate(cube2.partitions):
            assert cube.partitions[i].counts.shape == cube2.partitions[i].counts.shape

    def it_provides_smoothed_counts_for_ca_x_ca_subvar_x_cat_date(
        self, ca_x_ca_subvar_x_cat_date_counts_fixture
    ):
        window, expected_value = ca_x_ca_subvar_x_cat_date_counts_fixture
        transforms = {
            "smoothing": {
                "method": "one_side_moving_avg",
                "window": window,
                "show": True,
            }
        }
        cube = Cube(CR.CA_X_CA_SUBVAR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(slice_.counts, expected_value)

        cube2 = Cube(CR.CA_X_CA_SUBVAR_X_CAT_DATE)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [296, 220, 137, 84],
                [127, 87, 40, 27],
                [169, 133, 97, 57],
                [70, 107, 69, 53],
                [75, 120, 94, 36],
                [61, 36, 14, 9],
            ],
        )

    def it_provides_smoothed_counts_for_mr_x_cat_date(
        self, mr_x_cat_date_counts_fixture
    ):
        window, expected_value = mr_x_cat_date_counts_fixture
        transforms = {
            "smoothing": {
                "method": "one_side_moving_avg",
                "window": window,
                "show": True,
            }
        }
        cube = Cube(CR.MR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(slice_.counts, expected_value)

        transforms = {"smoothing": {"method": "one_side_moving_avg", "window": 0}}
        cube2 = Cube(CR.MR_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [87, 94, 53, 33],
                [272, 265, 184, 117],
                [118, 106, 69, 38],
                [134, 147, 91, 56],
                [22, 38, 14, 3],
                [40, 39, 25, 10],
                [17, 19, 9, 2],
                [264, 241, 167, 98],
                [24, 33, 29, 19],
                [70, 77, 61, 39],
                [115, 133, 82, 43],
                [49, 58, 38, 20],
                [34, 49, 46, 18],
                [28, 27, 20, 6],
                [101, 94, 57, 31],
                [17, 5, 5, 2],
            ],
        )
        assert slice_.shape == slice2_.shape

    def it_doesnt_smooth_counts_when_window_is_not_valid(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 30, "show": True}
        }
        cube = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]

        cube2 = Cube(CR.CAT_X_CAT_DATE)
        slice2_ = cube2.partitions[0]

        np.testing.assert_array_almost_equal(slice_.counts, slice2_.counts)

    def it_provides_smoothed_col_percent_for_cat_x_cat_date(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
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

        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": False}
        }
        cube2 = Cube(CR.CAT_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.column_percentages,
            [
                [27.11198428, 29.69072165, 32.06349206, 39.3442623],
                [36.34577603, 48.65979381, 46.34920635, 48.63387978],
                [16.69941061, 10.92783505, 15.23809524, 7.10382514],
                [5.30451866, 4.94845361, 2.53968254, 0.54644809],
                [14.53831041, 5.77319588, 3.80952381, 4.3715847],
            ],
        )
        assert slice_.shape == slice2_.shape

    def it_provides_smoothed_col_percent_for_mr_x_cat_date(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(CR.MR_X_CAT_DATE, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            slice_.column_percentages,
            [
                [np.nan, np.nan, 17.65206122, 17.96554421],
                [np.nan, np.nan, 55.15125492, 58.65002568],
                [np.nan, np.nan, 22.17928914, 21.37339451],
                [np.nan, np.nan, 28.32388483, 29.74887257],
                [np.nan, np.nan, 5.49263724, 4.59835186],
                [np.nan, np.nan, 7.89598501, 7.09796324],
                [np.nan, np.nan, 3.34955087, 2.60055555],
                [np.nan, np.nan, 51.21043193, 51.77226788],
                [np.nan, np.nan, 6.86145676, 8.75058541],
                [np.nan, np.nan, 16.22587969, 18.74555289],
                [np.nan, np.nan, 25.1828015, 25.48411734],
                [np.nan, np.nan, 11.14224268, 11.57632358],
                [np.nan, np.nan, 10.39014622, 11.44224666],
                [np.nan, np.nan, 5.76970408, 5.02893949],
                [np.nan, np.nan, 18.98950149, 18.02185537],
                [np.nan, np.nan, 1.97849971, 1.22950439],
            ],
        )

        transforms = {"smoothing": {"method": "one_side_moving_avg", "window": 0}}
        cube2 = Cube(CR.MR_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.column_percentages,
            [
                [17.09233792, 19.14460285, 16.7192429, 18.03278689],
                [53.43811395, 53.97148676, 58.04416404, 63.93442623],
                [23.1827112, 21.5885947, 21.76656151, 20.76502732],
                [26.32612967, 29.9389002, 28.70662461, 30.6010929],
                [4.32220039, 7.73930754, 4.41640379, 1.63934426],
                [7.85854617, 7.94297352, 7.88643533, 5.46448087],
                [3.33988212, 3.86965377, 2.83911672, 1.09289617],
                [51.86640472, 49.08350305, 52.68138801, 53.55191257],
                [4.7151277, 6.7209776, 9.14826498, 10.38251366],
                [13.7524558, 15.68228106, 19.24290221, 21.31147541],
                [22.59332024, 27.08757637, 25.86750789, 23.49726776],
                [9.62671906, 11.81262729, 11.9873817, 10.92896175],
                [6.67976424, 9.9796334, 14.51104101, 9.83606557],
                [5.50098232, 5.49898167, 6.30914826, 3.27868852],
                [19.84282908, 19.14460285, 17.98107256, 16.93989071],
                [3.33988212, 1.01832994, 1.57728707, 1.09289617],
            ],
        )
        assert slice_.shape == slice2_.shape

    def it_provides_smoothed_counts_for_cat_x_cat_date_hs(
        self, cat_x_cat_date_hs_fixture
    ):
        window, expected_value = cat_x_cat_date_hs_fixture
        transforms = {
            "smoothing": {
                "method": "one_side_moving_avg",
                "window": window,
                "show": True,
            }
        }
        cube = Cube(CR.CAT_X_CAT_DATE_HS, transforms=transforms)
        slice_ = cube.partitions[0]
        np.testing.assert_almost_equal(slice_.counts, expected_value)

        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": False}
        }
        cube2 = Cube(CR.CAT_X_CAT_DATE_HS, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.counts,
            [
                [213.0, 212.0, 425.0, 128.0, 78.0],
                [73.0, 69.0, 142.0, 59.0, 31.0],
                [286.0, 281.0, 567.0, 187.0, 109.0],
                [31.0, 20.0, 51.0, 17.0, 9.0],
                [33.0, 34.0, 67.0, 20.0, 11.0],
                [25.0, 16.0, 41.0, 7.0, 3.0],
                [23.0, 29.0, 52.0, 22.0, 13.0],
                [48.0, 45.0, 93.0, 29.0, 16.0],
                [90.0, 93.0, 183.0, 58.0, 31.0],
                [17.0, 9.0, 26.0, 4.0, 5.0],
            ],
        )
        assert slice_.shape == slice2_.shape

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [213.0, 212.0, 425.0, 128.0, 78.0],
                    [73.0, 69.0, 142.0, 59.0, 31.0],
                    [286.0, 281.0, 567.0, 187.0, 109.0],
                    [31.0, 20.0, 51.0, 17.0, 9.0],
                    [33.0, 34.0, 67.0, 20.0, 11.0],
                    [25.0, 16.0, 41.0, 7.0, 3.0],
                    [23.0, 29.0, 52.0, 22.0, 13.0],
                    [48.0, 45.0, 93.0, 29.0, 16.0],
                    [90.0, 93.0, 183.0, 58.0, 31.0],
                    [17.0, 9.0, 26.0, 4.0, 5.0],
                ],
            ),
            (
                2,
                [
                    [np.nan, 212.5, 318.5, 276.5, 103.0],
                    [np.nan, 71.0, 105.5, 100.5, 45.0],
                    [np.nan, 283.5, 424.0, 377.0, 148.0],
                    [np.nan, 25.5, 35.5, 34.0, 13.0],
                    [np.nan, 33.5, 50.5, 43.5, 15.5],
                    [np.nan, 20.5, 28.5, 24.0, 5.0],
                    [np.nan, 26.0, 40.5, 37.0, 17.5],
                    [np.nan, 46.5, 69.0, 61.0, 22.5],
                    [np.nan, 91.5, 138.0, 120.5, 44.5],
                    [np.nan, 13.0, 17.5, 15.0, 4.5],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 283.33333333, 255.0, 210.33333333],
                    [np.nan, np.nan, 94.66666667, 90.0, 77.33333333],
                    [np.nan, np.nan, 378.0, 345.0, 287.66666667],
                    [np.nan, np.nan, 34.0, 29.33333333, 25.66666667],
                    [np.nan, np.nan, 44.66666667, 40.33333333, 32.66666667],
                    [np.nan, np.nan, 27.33333333, 21.33333333, 17.0],
                    [np.nan, np.nan, 34.66666667, 34.33333333, 29.0],
                    [np.nan, np.nan, 62.0, 55.66666667, 46.0],
                    [np.nan, np.nan, 122.0, 111.33333333, 90.66666667],
                    [np.nan, np.nan, 17.33333333, 13.0, 11.66666667],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 244.5, 210.75],
                    [np.nan, np.nan, np.nan, 85.75, 75.25],
                    [np.nan, np.nan, np.nan, 330.25, 286.0],
                    [np.nan, np.nan, np.nan, 29.75, 24.25],
                    [np.nan, np.nan, np.nan, 38.5, 33.0],
                    [np.nan, np.nan, np.nan, 22.25, 16.75],
                    [np.nan, np.nan, np.nan, 31.5, 29.0],
                    [np.nan, np.nan, np.nan, 53.75, 45.75],
                    [np.nan, np.nan, np.nan, 106.0, 91.25],
                    [np.nan, np.nan, np.nan, 14.0, 11.0],
                ],
            ),
            (
                5,
                [
                    [np.nan, np.nan, np.nan, np.nan, 211.2],
                    [np.nan, np.nan, np.nan, np.nan, 74.8],
                    [np.nan, np.nan, np.nan, np.nan, 286.0],
                    [np.nan, np.nan, np.nan, np.nan, 25.6],
                    [np.nan, np.nan, np.nan, np.nan, 33.0],
                    [np.nan, np.nan, np.nan, np.nan, 18.4],
                    [np.nan, np.nan, np.nan, np.nan, 27.8],
                    [np.nan, np.nan, np.nan, np.nan, 46.2],
                    [np.nan, np.nan, np.nan, np.nan, 91.0],
                    [np.nan, np.nan, np.nan, np.nan, 12.2],
                ],
            ),
        ]
    )
    def cat_x_cat_date_hs_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [138.0, 144.0, 101.0, 72.0],
                    [185.0, 236.0, 146.0, 89.0],
                    [85.0, 53.0, 48.0, 13.0],
                    [27.0, 24.0, 8.0, 1.0],
                    [74.0, 28.0, 12.0, 8.0],
                ],
            ),
            (
                2,
                [
                    [np.nan, 141.0, 122.5, 86.5],
                    [np.nan, 210.5, 191.0, 117.5],
                    [np.nan, 69.0, 50.5, 30.5],
                    [np.nan, 25.5, 16.0, 4.5],
                    [np.nan, 51.0, 20.0, 10.0],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 127.66666667, 105.66666667],
                    [np.nan, np.nan, 189.0, 157.0],
                    [np.nan, np.nan, 62.0, 38.0],
                    [np.nan, np.nan, 19.66666667, 11.0],
                    [np.nan, np.nan, 38.0, 16.0],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 113.75],
                    [np.nan, np.nan, np.nan, 164.0],
                    [np.nan, np.nan, np.nan, 49.75],
                    [np.nan, np.nan, np.nan, 15.0],
                    [np.nan, np.nan, np.nan, 30.5],
                ],
            ),
        ]
    )
    def cat_x_cat_date_counts_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [87, 94, 53, 33],
                    [272, 265, 184, 117],
                    [118, 106, 69, 38],
                    [134, 147, 91, 56],
                    [22, 38, 14, 3],
                    [40, 39, 25, 10],
                    [17, 19, 9, 2],
                    [264, 241, 167, 98],
                    [24, 33, 29, 19],
                    [70, 77, 61, 39],
                    [115, 133, 82, 43],
                    [49, 58, 38, 20],
                    [34, 49, 46, 18],
                    [28, 27, 20, 6],
                    [101, 94, 57, 31],
                    [17, 5, 5, 2],
                ],
            ),
            (
                2,
                [
                    [np.nan, 90.5, 73.5, 43.0],
                    [np.nan, 268.5, 224.5, 150.5],
                    [np.nan, 112.0, 87.5, 53.5],
                    [np.nan, 140.5, 119.0, 73.5],
                    [np.nan, 30.0, 26.0, 8.5],
                    [np.nan, 39.5, 32.0, 17.5],
                    [np.nan, 18.0, 14.0, 5.5],
                    [np.nan, 252.5, 204.0, 132.5],
                    [np.nan, 28.5, 31.0, 24.0],
                    [np.nan, 73.5, 69.0, 50.0],
                    [np.nan, 124.0, 107.5, 62.5],
                    [np.nan, 53.5, 48.0, 29.0],
                    [np.nan, 41.5, 47.5, 32.0],
                    [np.nan, 27.5, 23.5, 13.0],
                    [np.nan, 97.5, 75.5, 44.0],
                    [np.nan, 11.0, 5.0, 3.5],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 78.0, 60.0],
                    [np.nan, np.nan, 240.33333333, 188.66666667],
                    [np.nan, np.nan, 97.66666667, 71.0],
                    [np.nan, np.nan, 124.0, 98.0],
                    [np.nan, np.nan, 24.66666667, 18.33333333],
                    [np.nan, np.nan, 34.66666667, 24.66666667],
                    [np.nan, np.nan, 15.0, 10.0],
                    [np.nan, np.nan, 224.0, 168.66666667],
                    [np.nan, np.nan, 28.66666667, 27.0],
                    [np.nan, np.nan, 69.33333333, 59.0],
                    [np.nan, np.nan, 110.0, 86.0],
                    [np.nan, np.nan, 48.33333333, 38.66666667],
                    [np.nan, np.nan, 43.0, 37.66666667],
                    [np.nan, np.nan, 25.0, 17.66666667],
                    [np.nan, np.nan, 84.0, 60.66666667],
                    [np.nan, np.nan, 9.0, 4.0],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 66.75],
                    [np.nan, np.nan, np.nan, 209.5],
                    [np.nan, np.nan, np.nan, 82.75],
                    [np.nan, np.nan, np.nan, 107.0],
                    [np.nan, np.nan, np.nan, 19.25],
                    [np.nan, np.nan, np.nan, 28.5],
                    [np.nan, np.nan, np.nan, 11.75],
                    [np.nan, np.nan, np.nan, 192.5],
                    [np.nan, np.nan, np.nan, 26.25],
                    [np.nan, np.nan, np.nan, 61.75],
                    [np.nan, np.nan, np.nan, 93.25],
                    [np.nan, np.nan, np.nan, 41.25],
                    [np.nan, np.nan, np.nan, 36.75],
                    [np.nan, np.nan, np.nan, 20.25],
                    [np.nan, np.nan, np.nan, 70.75],
                    [np.nan, np.nan, np.nan, 7.25],
                ],
            ),
        ]
    )
    def mr_x_cat_date_counts_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [296, 220, 137, 84],
                    [127, 87, 40, 27],
                    [169, 133, 97, 57],
                    [70, 107, 69, 53],
                    [75, 120, 94, 36],
                    [61, 36, 14, 9],
                ],
            ),
            (
                2,
                [
                    [np.nan, 258.0, 178.5, 110.5],
                    [np.nan, 107.0, 63.5, 33.5],
                    [np.nan, 151.0, 115.0, 77.0],
                    [np.nan, 88.5, 88.0, 61.0],
                    [np.nan, 97.5, 107.0, 65.0],
                    [np.nan, 48.5, 25.0, 11.5],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 217.66666667, 147.0],
                    [np.nan, np.nan, 84.66666667, 51.33333333],
                    [np.nan, np.nan, 133.0, 95.66666667],
                    [np.nan, np.nan, 82.0, 76.33333333],
                    [np.nan, np.nan, 96.33333333, 83.33333333],
                    [np.nan, np.nan, 37.0, 19.66666667],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 184.25],
                    [np.nan, np.nan, np.nan, 70.25],
                    [np.nan, np.nan, np.nan, 114.0],
                    [np.nan, np.nan, np.nan, 74.75],
                    [np.nan, np.nan, np.nan, 81.25],
                    [np.nan, np.nan, np.nan, 30.0],
                ],
            ),
        ]
    )
    def ca_x_ca_subvar_x_cat_date_counts_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value


class DescribeStrandSmoothing(object):
    def it_provides_smoothed_counts_univariate_date(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 3, "show": True}
        }
        cube = Cube(CR.CAT_DATE, transforms=transforms)
        strand_ = cube.partitions[0]
        np.testing.assert_array_almost_equal(
            strand_.counts,
            (
                np.nan,
                np.nan,
                24.666666666666668,
                22.0,
                76.33333333333333,
                71.66666666666667,
                69.0,
                18.0,
                12.333333333333334,
                51.0,
                58.333333333333336,
                56.333333333333336,
                16.0,
                21.666666666666668,
                30.333333333333332,
                31.666666666666668,
                19.333333333333332,
                17.0,
                17.0,
                14.333333333333334,
                12.666666666666666,
                18.0,
                30.333333333333332,
                32.0,
                25.333333333333332,
                20.0,
                13.0,
                11.333333333333334,
                8.333333333333334,
                9.0,
                18.666666666666668,
                16.666666666666668,
                48.333333333333336,
                52.0,
                48.666666666666664,
                39.0,
                29.333333333333332,
                38.0,
                43.333333333333336,
                40.666666666666664,
                36.666666666666664,
                7.666666666666667,
                16.333333333333332,
                38.333333333333336,
                42.333333333333336,
                31.666666666666668,
                17.0,
                25.0,
                27.666666666666668,
                26.666666666666668,
                15.333333333333334,
            ),
        )

        cube2 = Cube(CR.CAT_DATE)
        strand2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            strand2_.counts,
            (
                30,
                2,
                42,
                22,
                165,
                28,
                14,
                12,
                11,
                130,
                34,
                5,
                9,
                51,
                31,
                13,
                14,
                24,
                13,
                6,
                19,
                29,
                43,
                24,
                9,
                27,
                3,
                4,
                18,
                5,
                33,
                12,
                100,
                44,
                2,
                71,
                15,
                28,
                87,
                7,
                16,
                0,
                33,
                82,
                12,
                1,
                38,
                36,
                9,
                35,
                2,
            ),
        )
        assert len(strand_.counts) == len(strand2_.counts)

    def it_doesnt_smooth_counts_when_window_is_not_valid(self):
        transforms = {
            "smoothing": {"method": "one_side_moving_avg", "window": 100, "show": True}
        }
        cube = Cube(CR.CAT_DATE, transforms=transforms)
        strand_ = cube.partitions[0]

        cube2 = Cube(CR.CAT_DATE)
        strand2_ = cube2.partitions[0]

        np.testing.assert_array_almost_equal(strand_.counts, strand2_.counts)
