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
        self, cat_x_cat_date_counts_fixture
    ):
        window, expected_value = cat_x_cat_date_counts_fixture
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
        self, ca_x_ca_subvar_x_cat_date_counts_fixture
    ):
        window, expected_value = ca_x_ca_subvar_x_cat_date_counts_fixture
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
        self, mr_x_cat_date_counts_fixture
    ):
        window, expected_value = mr_x_cat_date_counts_fixture
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
        self, cat_x_cat_date_hs_fixture
    ):
        window, expected_value = cat_x_cat_date_hs_fixture
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
                [42.17821782, 43.98340249, 40.63492063, 43.09392265],
                [14.45544554, 14.3153527, 18.73015873, 17.12707182],
                [56.63366337, 58.29875519, 59.36507937, 60.22099448],
                [6.13861386, 4.14937759, 5.3968254, 4.97237569],
                [6.53465347, 7.05394191, 6.34920635, 6.07734807],
                [4.95049505, 3.31950207, 2.22222222, 1.65745856],
                [4.55445545, 6.01659751, 6.98412698, 7.18232044],
                [9.5049505, 9.33609959, 9.20634921, 8.83977901],
                [17.82178218, 19.29460581, 18.41269841, 17.12707182],
                [3.36633663, 1.86721992, 1.26984127, 2.76243094],
            ],
        )
        assert slice_.shape == slice2_.shape

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                1,
                [
                    [42.17821782, 43.98340249, 40.63492063, 43.09392265],
                    [14.45544554, 14.3153527, 18.73015873, 17.12707182],
                    [56.63366337, 58.29875519, 59.36507937, 60.22099448],
                    [6.13861386, 4.14937759, 5.3968254, 4.97237569],
                    [6.53465347, 7.05394191, 6.34920635, 6.07734807],
                    [4.95049505, 3.31950207, 2.22222222, 1.65745856],
                    [4.55445545, 6.01659751, 6.98412698, 7.18232044],
                    [9.5049505, 9.33609959, 9.20634921, 8.83977901],
                    [17.82178218, 19.29460581, 18.41269841, 17.12707182],
                    [3.36633663, 1.86721992, 1.26984127, 2.76243094],
                ],
            ),
            (
                2,
                [
                    [np.nan, 43.08081016, 42.30916156, 41.86442164],
                    [np.nan, 14.38539912, 16.52275571, 17.92861528],
                    [np.nan, 57.46620928, 58.83191728, 59.79303692],
                    [np.nan, 5.14399573, 4.7731015, 5.18460054],
                    [np.nan, 6.79429769, 6.70157413, 6.21327721],
                    [np.nan, 4.13499856, 2.77086215, 1.93984039],
                    [np.nan, 5.28552648, 6.50036225, 7.08322371],
                    [np.nan, 9.42052504, 9.2712244, 9.02306411],
                    [np.nan, 18.55819399, 18.85365211, 17.76988512],
                    [np.nan, 2.61677828, 1.56853059, 2.0161361],
                ],
            ),
            (
                3,
                [
                    [np.nan, np.nan, 42.26551365, 42.57074859],
                    [np.nan, np.nan, 15.83365232, 16.72419442],
                    [np.nan, np.nan, 58.09916597, 59.29494301],
                    [np.nan, np.nan, 5.22827228, 4.83952623],
                    [np.nan, np.nan, 6.64593391, 6.49349877],
                    [np.nan, np.nan, 3.49740645, 2.39972762],
                    [np.nan, np.nan, 5.85172665, 6.72768165],
                    [np.nan, np.nan, 9.3491331, 9.12740927],
                    [np.nan, np.nan, 18.50969547, 18.27812535],
                    [np.nan, np.nan, 2.16779927, 1.96649738],
                ],
            ),
            (
                4,
                [
                    [np.nan, np.nan, np.nan, 42.4726159],
                    [np.nan, np.nan, np.nan, 16.1570072],
                    [np.nan, np.nan, np.nan, 58.6296231],
                    [np.nan, np.nan, np.nan, 5.16429814],
                    [np.nan, np.nan, np.nan, 6.50378745],
                    [np.nan, np.nan, np.nan, 3.03741948],
                    [np.nan, np.nan, np.nan, 6.1843751],
                    [np.nan, np.nan, np.nan, 9.22179457],
                    [np.nan, np.nan, np.nan, 18.16403956],
                    [np.nan, np.nan, np.nan, 2.31645719],
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
    def cat_x_cat_date_counts_fixture(self, request):
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
    def mr_x_cat_date_counts_fixture(self, request):
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
    def ca_x_ca_subvar_x_cat_date_counts_fixture(self, request):
        window, expected_value = request.param
        return window, expected_value
