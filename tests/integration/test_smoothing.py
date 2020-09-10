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
    def it_provides_smoothed_col_percent_for_cat_x_cat_date(
        self, cat_x_cat_date_col_percent_fixture
    ):
        window, expected_value = cat_x_cat_date_col_percent_fixture
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
        np.testing.assert_almost_equal(slice_.column_percentages, expected_value)

        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": False,
                }
            }
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

        cube2 = Cube(CR.CAT_X_CAT_DATE_WGTD)
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
            [
                [np.nan, np.nan, 20.64470034, 18.67207554],
                [np.nan, np.nan, 57.97838842, 59.28676201],
                [np.nan, np.nan, 26.7735288, 24.72038871],
                [np.nan, np.nan, 32.9262727, 32.22176384],
                [np.nan, np.nan, 5.30563926, 5.02383572],
                [np.nan, np.nan, 7.67077614, 6.64420609],
                [np.nan, np.nan, 1.74911513, 1.02447745],
                [np.nan, np.nan, 50.4170254, 47.82040704],
                [np.nan, np.nan, 8.07883325, 11.03777044],
                [np.nan, np.nan, 17.72183559, 20.25806747],
                [np.nan, np.nan, 26.5219185, 24.48890722],
                [np.nan, np.nan, 13.21377609, 11.40218188],
                [np.nan, np.nan, 9.96675755, 11.69783645],
                [np.nan, np.nan, 5.03977934, 6.14686469],
                [np.nan, np.nan, 19.26338467, 17.87449578],
                [np.nan, np.nan, 1.29621658, 0.330033],
            ],
        )

        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": 3,
                    "show": False,
                }
            }
        }
        cube2 = Cube(CR.CAT_X_MR_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.column_percentages,
            [
                [25.36231884, 18.75, 17.82178218, 19.44444444],
                [55.79710145, 59.72222222, 58.41584158, 59.72222222],
                [31.15942029, 26.38888889, 22.77227723, 25.0],
                [34.05797101, 34.02777778, 30.69306931, 31.94444444],
                [3.62318841, 8.33333333, 3.96039604, 2.77777778],
                [7.24637681, 11.80555556, 3.96039604, 4.16666667],
                [2.17391304, 2.08333333, 0.99009901, 0.0],
                [53.62318841, 52.08333333, 45.54455446, 45.83333333],
                [3.62318841, 9.72222222, 10.89108911, 12.5],
                [17.39130435, 15.97222222, 19.8019802, 25.0],
                [29.71014493, 27.08333333, 22.77227723, 23.61111111],
                [13.76811594, 15.97222222, 9.9009901, 8.33333333],
                [8.69565217, 8.33333333, 12.87128713, 13.88888889],
                [3.62318841, 5.55555556, 5.94059406, 6.94444444],
                [16.66666667, 17.36111111, 23.76237624, 12.5],
                [2.89855072, 0.0, 0.99009901, 0.0],
            ],
        )
        for i, p in enumerate(cube2.partitions):
            assert cube.partitions[i].counts.shape == cube2.partitions[i].counts.shape

    def it_provides_smoothed_col_percent_for_ca_x_ca_subvar_x_cat_date(
        self, ca_x_ca_subvar_x_cat_date_col_percent_fixture
    ):
        window, expected_value = ca_x_ca_subvar_x_cat_date_col_percent_fixture
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
        np.testing.assert_array_almost_equal(slice_.column_percentages, expected_value)

        cube2 = Cube(CR.CA_X_CA_SUBVAR_X_CAT_DATE)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.column_percentages,
            [
                [58.96414343, 45.54865424, 43.63057325, 46.15384615],
                [25.29880478, 18.01242236, 12.7388535, 14.83516484],
                [33.66533865, 27.53623188, 30.89171975, 31.31868132],
                [13.94422311, 22.15320911, 21.97452229, 29.12087912],
                [14.94023904, 24.8447205, 29.93630573, 19.78021978],
                [12.15139442, 7.45341615, 4.45859873, 4.94505495],
            ],
        )

    def it_provides_smoothed_col_percent_for_mr_x_cat_date(
        self, mr_x_cat_date_col_percent_fixture
    ):
        window, expected_value = mr_x_cat_date_col_percent_fixture
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
        np.testing.assert_array_almost_equal(slice_.column_percentages, expected_value)

        transforms = {
            "columns_dimension": {
                "smoothing": {"method": "one_side_moving_avg", "window": 0}
            }
        }
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
        window, expected_value = cat_hs_x_cat_date_col_percent_fixture
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
        np.testing.assert_almost_equal(slice_.column_percentages, expected_value)

        transforms = {
            "columns_dimension": {
                "smoothing": {
                    "method": "one_side_moving_avg",
                    "window": window,
                    "show": False,
                }
            }
        }
        cube2 = Cube(CR.CAT_HS_X_CAT_DATE, transforms=transforms)
        slice2_ = cube2.partitions[0]
        np.testing.assert_array_almost_equal(
            slice2_.column_percentages,
            [
                [73.28094303, 68.83910387, 78.5488959, 81.96721311],
                [72.69155206, 66.19144603, 74.44794953, 80.32786885],
                [12.77013752, 13.64562118, 7.57097792, 8.19672131],
                [12.37721022, 11.20162933, 7.2555205, 6.01092896],
                [25.14734774, 24.84725051, 14.82649842, 14.20765027],
                [0.58939096, 2.64765784, 4.10094637, 1.63934426],
                [0.58939096, 0.61099796, 0.63091483, 0.0],
                [0.78585462, 2.44399185, 4.7318612, 1.63934426],
                [0.19646365, 2.64765784, 1.26182965, 1.63934426],
                [0.0, 0.61099796, 0.0, 0.54644809],
                [0.19646365, 3.2586558, 1.26182965, 2.18579235],
            ],
        )
        assert slice_.shape == slice2_.shape

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
            [
                [
                    14.28571429,
                    10.20408163,
                    20.51282051,
                    16.36363636,
                    16.43835616,
                    13.7254902,
                    18.18181818,
                    29.91452991,
                    32.0,
                    44.7761194,
                ],
                [
                    7.14285714,
                    23.46938776,
                    17.94871795,
                    14.54545455,
                    20.54794521,
                    9.80392157,
                    27.27272727,
                    11.11111111,
                    35.2,
                    23.88059701,
                ],
                [
                    12.85714286,
                    19.3877551,
                    10.25641026,
                    16.36363636,
                    13.69863014,
                    15.68627451,
                    25.0,
                    17.09401709,
                    13.6,
                    14.92537313,
                ],
                [
                    15.71428571,
                    15.30612245,
                    14.1025641,
                    5.45454545,
                    17.80821918,
                    9.80392157,
                    18.18181818,
                    20.51282051,
                    6.4,
                    5.2238806,
                ],
                [
                    12.85714286,
                    12.24489796,
                    10.25641026,
                    5.45454545,
                    15.06849315,
                    7.84313725,
                    6.06060606,
                    10.25641026,
                    6.4,
                    5.97014925,
                ],
                [
                    5.71428571,
                    9.18367347,
                    20.51282051,
                    9.09090909,
                    9.5890411,
                    11.76470588,
                    3.03030303,
                    2.56410256,
                    3.2,
                    1.49253731,
                ],
                [
                    8.57142857,
                    4.08163265,
                    5.12820513,
                    7.27272727,
                    1.36986301,
                    11.76470588,
                    1.51515152,
                    5.12820513,
                    2.4,
                    2.23880597,
                ],
                [
                    17.14285714,
                    4.08163265,
                    1.28205128,
                    3.63636364,
                    2.73972603,
                    1.96078431,
                    0.75757576,
                    0.85470085,
                    0.8,
                    0.74626866,
                ],
                [
                    1.42857143,
                    2.04081633,
                    0.0,
                    14.54545455,
                    1.36986301,
                    11.76470588,
                    0.0,
                    0.0,
                    0.0,
                    0.74626866,
                ],
                [
                    4.28571429,
                    0.0,
                    0.0,
                    7.27272727,
                    1.36986301,
                    5.88235294,
                    0.0,
                    2.56410256,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
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
            (
                1,
                [
                    [73.28094303, 68.83910387, 78.5488959, 81.96721311],
                    [72.69155206, 66.19144603, 74.44794953, 80.32786885],
                    [12.77013752, 13.64562118, 7.57097792, 8.19672131],
                    [12.37721022, 11.20162933, 7.2555205, 6.01092896],
                    [25.14734774, 24.84725051, 14.82649842, 14.20765027],
                    [0.58939096, 2.64765784, 4.10094637, 1.63934426],
                    [0.58939096, 0.61099796, 0.63091483, 0.0],
                    [0.78585462, 2.44399185, 4.7318612, 1.63934426],
                    [0.19646365, 2.64765784, 1.26182965, 1.63934426],
                    [0.0, 0.61099796, 0.0, 0.54644809],
                    [0.19646365, 3.2586558, 1.26182965, 2.18579235],
                ],
            ),
            (
                2,
                [
                    [np.nan, 71.06002345, 73.69399988, 80.25805451],
                    [np.nan, 69.44149905, 70.31969778, 77.38790919],
                    [np.nan, 13.20787935, 10.60829955, 7.88384961],
                    [np.nan, 11.78941977, 9.22857492, 6.63322473],
                    [np.nan, 24.99729912, 19.83687447, 14.51707435],
                    [np.nan, 1.6185244, 3.37430211, 2.87014532],
                    [np.nan, 0.60019446, 0.62095639, 0.31545741],
                    [np.nan, 1.61492324, 3.58792653, 3.18560273],
                    [np.nan, 1.42206075, 1.95474375, 1.45058696],
                    [np.nan, 0.30549898, 0.30549898, 0.27322404],
                    [np.nan, 1.72755973, 2.26024273, 1.723811],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 73.55631426, 76.45173763],
                    [np.nan, np.nan, 71.11031587, 73.6557548],
                    [np.nan, np.nan, 11.32891221, 9.80444014],
                    [np.nan, np.nan, 10.27812002, 8.15602626],
                    [np.nan, np.nan, 21.60703222, 17.9604664],
                    [np.nan, np.nan, 2.44599839, 2.79598283],
                    [np.nan, np.nan, 0.61043458, 0.41397093],
                    [np.nan, np.nan, 2.65390256, 2.9383991],
                    [np.nan, np.nan, 1.36865038, 1.84961059],
                    [np.nan, np.nan, 0.20366599, 0.38581535],
                    [np.nan, np.nan, 1.57231637, 2.23542594],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 75.65903898],
                    [np.nan, np.nan, np.nan, 73.41470412],
                    [np.nan, np.nan, np.nan, 10.54586448],
                    [np.nan, np.nan, np.nan, 9.21132225],
                    [np.nan, np.nan, np.nan, 19.75718674],
                    [np.nan, np.nan, np.nan, 2.24433486],
                    [np.nan, np.nan, np.nan, 0.45782594],
                    [np.nan, np.nan, np.nan, 2.40026298],
                    [np.nan, np.nan, np.nan, 1.43632385],
                    [np.nan, np.nan, np.nan, 0.28936151],
                    [np.nan, np.nan, np.nan, 1.72568537],
                ],
            ),
        ]
    )
    def cat_hs_x_cat_date_col_percent_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [27.11198428, 29.69072165, 32.06349206, 39.3442623],
                    [36.34577603, 48.65979381, 46.34920635, 48.63387978],
                    [16.69941061, 10.92783505, 15.23809524, 7.10382514],
                    [5.30451866, 4.94845361, 2.53968254, 0.54644809],
                    [14.53831041, 5.77319588, 3.80952381, 4.3715847],
                ],
            ),
            (
                2,
                [
                    [np.nan, 28.40135297, 30.87710686, 35.70387718],
                    [np.nan, 42.50278492, 47.50450008, 47.49154307],
                    [np.nan, 13.81362283, 13.08296514, 11.17096019],
                    [np.nan, 5.12648614, 3.74406807, 1.54306531],
                    [np.nan, 10.15575314, 4.79135984, 4.09055425],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 29.622066, 33.699492],
                    [np.nan, np.nan, 43.7849254, 47.88095998],
                    [np.nan, np.nan, 14.28844697, 11.08991848],
                    [np.nan, np.nan, 4.26421827, 2.67819475],
                    [np.nan, np.nan, 8.04034337, 4.6514348],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 32.05261507],
                    [np.nan, np.nan, np.nan, 44.99716399],
                    [np.nan, np.nan, np.nan, 12.49229151],
                    [np.nan, np.nan, np.nan, 3.33477572],
                    [np.nan, np.nan, np.nan, 7.1231537],
                ],
            ),
        ]
    )
    def cat_x_cat_date_col_percent_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
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
            ),
            (
                2,
                [
                    [np.nan, 18.11847038, 17.93192288, 17.37601489],
                    [np.nan, 53.70480036, 56.0078254, 60.98929513],
                    [np.nan, 22.38565295, 21.67757811, 21.26579442],
                    [np.nan, 28.13251493, 29.3227624, 29.65385875],
                    [np.nan, 6.03075396, 6.07785566, 3.02787402],
                    [np.nan, 7.90075985, 7.91470443, 6.6754581],
                    [np.nan, 3.60476794, 3.35438524, 1.96600645],
                    [np.nan, 50.47495389, 50.88244553, 53.11665029],
                    [np.nan, 5.71805265, 7.93462129, 9.76538932],
                    [np.nan, 14.71736843, 17.46259163, 20.27718881],
                    [np.nan, 24.84044831, 26.47754213, 24.68238782],
                    [np.nan, 10.71967317, 11.9000045, 11.45817173],
                    [np.nan, 8.32969882, 12.24533721, 12.17355329],
                    [np.nan, 5.49998199, 5.90406497, 4.79391839],
                    [np.nan, 19.49371596, 18.5628377, 17.46048163],
                    [np.nan, 2.17910603, 1.2978085, 1.33509162],
                ],
            ),
            (
                3,
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
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 17.74724264],
                    [np.nan, np.nan, np.nan, 57.34704774],
                    [np.nan, np.nan, np.nan, 21.82572368],
                    [np.nan, np.nan, np.nan, 28.89318684],
                    [np.nan, np.nan, np.nan, 4.52931399],
                    [np.nan, np.nan, np.nan, 7.28810897],
                    [np.nan, np.nan, np.nan, 2.7853872],
                    [np.nan, np.nan, np.nan, 51.79580209],
                    [np.nan, np.nan, np.nan, 7.74172099],
                    [np.nan, np.nan, np.nan, 17.49727862],
                    [np.nan, np.nan, np.nan, 24.76141806],
                    [np.nan, np.nan, np.nan, 11.08892245],
                    [np.nan, np.nan, np.nan, 10.25162606],
                    [np.nan, np.nan, np.nan, 5.14695019],
                    [np.nan, np.nan, np.nan, 18.4770988],
                    [np.nan, np.nan, np.nan, 1.75709883],
                ],
            ),
        ]
    )
    def mr_x_cat_date_col_percent_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [58.96414343, 45.54865424, 43.63057325, 46.15384615],
                    [25.29880478, 18.01242236, 12.7388535, 14.83516484],
                    [33.66533865, 27.53623188, 30.89171975, 31.31868132],
                    [13.94422311, 22.15320911, 21.97452229, 29.12087912],
                    [14.94023904, 24.8447205, 29.93630573, 19.78021978],
                    [12.15139442, 7.45341615, 4.45859873, 4.94505495],
                ],
            ),
            (
                2,
                [
                    [np.nan, 52.25639884, 44.58961375, 44.8922097],
                    [np.nan, 21.65561357, 15.37563793, 13.78700917],
                    [np.nan, 30.60078526, 29.21397581, 31.10520053],
                    [np.nan, 18.04871611, 22.0638657, 25.54770071],
                    [np.nan, 19.89247977, 27.39051311, 24.85826276],
                    [np.nan, 9.80240529, 5.95600744, 4.70182684],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 49.38112364, 45.11102455],
                    [np.nan, np.nan, 18.68336021, 15.19548023],
                    [np.nan, np.nan, 30.69776342, 29.91554432],
                    [np.nan, np.nan, 19.35731817, 24.41620351],
                    [np.nan, np.nan, 23.24042176, 24.85374867],
                    [np.nan, np.nan, 8.02113643, 5.61902327],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 48.57430427],
                    [np.nan, np.nan, np.nan, 17.72131137],
                    [np.nan, np.nan, np.nan, 30.8529929],
                    [np.nan, np.nan, np.nan, 21.79820841],
                    [np.nan, np.nan, np.nan, 22.37537126],
                    [np.nan, np.nan, np.nan, 7.25211606],
                ],
            ),
        ]
    )
    def ca_x_ca_subvar_x_cat_date_col_percent_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value
