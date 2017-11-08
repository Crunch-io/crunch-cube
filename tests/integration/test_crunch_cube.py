from unittest import TestCase

import numpy as np

from .fixtures import (
    fixt_cat_x_cat,
    fixt_cat_x_datetime,
    fixt_cat_x_num_x_datetime,
    fixt_cat_x_mr,
    fixt_econ_gender_x_ideology_weighted,
    fixt_univariate_categorical,
    fixt_voter_registration,
    fixt_simple_datetime,
    fixt_simple_text,
    fixt_simple_cat_array,
    fixt_simple_mr,
)
from cr.cube.crunch_cube import CrunchCube


class TestCrunchCube(TestCase):
    def test_crunch_cube_loads_data(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = fixt_cat_x_cat['value']
        actual = cube._cube
        self.assertEqual(actual, expected)

    def test_as_array_univariate_cat_exclude_missing(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([10, 5])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_numeric(self):
        cube = CrunchCube(fixt_voter_registration)
        expected = np.array([885, 105, 10])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_datetime(self):
        cube = CrunchCube(fixt_simple_datetime)
        expected = np.array([1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_text(self):
        cube = CrunchCube(fixt_simple_text)
        expected = np.array([1, 1, 1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_univariate_cat_include_missing(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([10, 5, 5, 0])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [5, 2],
            [5, 3],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_unweighted(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [5, 2],
            [5, 3],
        ])
        actual = cube._as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_include_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [5, 3, 2, 0],
            [5, 2, 3, 0],
            [0, 0, 0, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_datetime_include_missing(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_univariate_cat_axis_none(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_numeric(self):
        cube = CrunchCube(fixt_voter_registration)
        expected = np.array([1000])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_datetime(self):
        cube = CrunchCube(fixt_simple_datetime)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_text(self):
        cube = CrunchCube(fixt_simple_text)
        expected = np.array([6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_none(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([10, 5])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_0(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([1, 1, 1, 1])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([7, 8])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_1(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([1, 1, 1, 1, 0])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_univariate_cat_axis_none(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([0.6666667, 0.3333333])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_numeric(self):
        cube = CrunchCube(fixt_voter_registration)
        expected = np.array([0.885, 0.105, 0.010])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_datetime(self):
        cube = CrunchCube(fixt_simple_datetime)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_text(self):
        cube = CrunchCube(fixt_simple_text)
        expected = np.array([
            0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)


    def test_proportions_cat_x_cat_axis_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.3333333, 0.1333333],
            [0.3333333, 0.2000000],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_none(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([
            [0., 0., 0.25, 0.],
            [0., 0., 0., 0.25],
            [0., 0.25, 0., 0.],
            [0.25, 0., 0., 0.],
            [0., 0., 0., 0.],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_cat_axis_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.5, 0.4],
            [0.5, 0.6],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_0(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_cat_axis_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.7142857, 0.2857143],
            [0.6250000, 0.3750000],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_1(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_univariate_cat_axis_none(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([66.6666667, 33.3333333])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_numeric(self):
        cube = CrunchCube(fixt_voter_registration)
        expected = np.array([88.5, 10.5, 1.0])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_datetime(self):
        cube = CrunchCube(fixt_simple_datetime)
        expected = np.array([25., 25., 25., 25.])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_text(self):
        cube = CrunchCube(fixt_simple_text)
        expected = np.array([
            16.6666667,
            16.6666667,
            16.6666667,
            16.6666667,
            16.6666667,
            16.6666667,
        ])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [33.3333333, 13.3333333],
            [33.3333333, 20.],
        ])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [50, 40],
            [50, 60],
        ])
        actual = cube.percentages(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [71.4285714, 28.5714286],
            [62.50000, 37.50000],
        ])
        actual = cube.percentages(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = [
            ['B', 'C'],
            ['C', 'E'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_cat_include_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = [
            ['B', 'C', 'No Data'],
            ['C', 'D', 'E', 'No Data'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = [
            ['red', 'green', 'blue', '4', '9'],
            [
                '1776-07-04T00:00:00',
                '1950-12-24T00:00:00',
                '2000-01-01T00:00:00',
                '2000-01-02T00:00:00'
            ],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_datetime_include_missing(self):
        cube = CrunchCube(fixt_cat_x_datetime)
        expected = [
            ['red', 'green', 'blue', '4', '8', '9', 'No Data'],
            [
                '1776-07-04T00:00:00',
                '1950-12-24T00:00:00',
                '2000-01-01T00:00:00',
                '2000-01-02T00:00:00',
                None,
            ],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_simple_cat_array_include_missing(self):
        cube = CrunchCube(fixt_simple_cat_array)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'c', 'd', 'No Data'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(fixt_simple_cat_array)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'c', 'd'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_as_array_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(fixt_simple_cat_array)
        expected = np.array([
            [3, 3, 0, 0],
            [1, 3, 2, 0],
            [0, 2, 1, 3],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_simple_cat_array_include_missing(self):
        cube = CrunchCube(fixt_simple_cat_array)
        expected = np.array([
            [3, 3, 0, 0, 0],
            [1, 3, 2, 0, 0],
            [0, 2, 1, 3, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_num_x_datetime(self):
        '''Test 3D cube, slicing accross first (numerical) variable.'''
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([
            [[1, 1],
             [0, 0],
             [0, 0],
             [0, 0]],

            [[2, 1],
             [1, 1],
             [0, 0],
             [0, 0]],

            [[0, 0],
             [2, 3],
             [0, 0],
             [0, 0]],

            [[0, 0],
             [0, 0],
             [3, 2],
             [0, 0]],

            [[0, 0],
             [0, 0],
             [1, 1],
             [0, 1]]
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_num_datetime(self):
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([
            [[0.05, 0.05],
             [0., 0.],
             [0., 0.],
             [0., 0.]],

            [[0.1, 0.05],
             [0.05, 0.05],
             [0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0.1, 0.15],
             [0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0., 0.],
             [0.15, 0.1],
             [0., 0.]],

            [[0., 0.],
             [0., 0.],
             [0.05, 0.05],
             [0., 0.05]],
        ])
        actual = cube.proportions()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_none(self):
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([20])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_0(self):
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([
            [3, 2],
            [3, 4],
            [4, 3],
            [0, 1],
        ])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_1(self):
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([
            [1, 1],
            [3, 2],
            [2, 3],
            [3, 2],
            [1, 2],
        ])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_2(self):
        cube = CrunchCube(fixt_cat_x_num_x_datetime)
        expected = np.array([
            [2, 0, 0, 0],
            [3, 2, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 2, 1],
        ])
        actual = cube.margin(axis=2)
        np.testing.assert_array_equal(actual, expected)

    def test_labels_simple_mr_exclude_missing(self):
        cube = CrunchCube(fixt_simple_mr)
        expected = [['Response #1', 'Response #2', 'Response #3']]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_simple_mr_include_missing(self):
        cube = CrunchCube(fixt_simple_mr)
        expected = [
            ['Response #1', 'Response #2', 'Response #3']
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_simple_mr_exclude_missing(self):
        cube = CrunchCube(fixt_simple_mr)
        expected = np.array([3, 4, 0])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_simple_mr_axis_none(self):
        cube = CrunchCube(fixt_simple_mr)
        expected = np.array([5, 6, 6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_simple_mr(self):
        cube = CrunchCube(fixt_simple_mr)
        expected = np.array([0.6, 0.6666667, 0.])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_mr_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = [
            ['rambutan', 'satsuma'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_mr_include_missing(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = [
            ['rambutan', 'satsuma', 'No Data'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_cat_x_mr(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = np.array([
            [12, 12, 12],
            [28, 22, 26],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_none(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = np.array([80, 79, 70])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_0(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = np.array([
            [28, 25, 23],
            [52, 54, 47]
        ])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_none(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = np.array([
            [0.15, 0.1518987, 0.1714286],
            [0.35, 0.2784810, 0.3714286],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_0(self):
        cube = CrunchCube(fixt_cat_x_mr)
        expected = np.array([
            [0.4285714, 0.4800000, 0.5217391],
            [0.5384615, 0.4074074, 0.5531915],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_as_array_unweighted_gender_x_ideology(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([
            [32, 85, 171, 114, 70, 13],
            [40, 97, 205, 106, 40, 27]
        ])
        actual = cube.as_array(weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_weighted_gender_x_ideology(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([
            [
                32.98969072,
                87.62886598,
                176.28865979,
                117.5257732,
                72.16494845,
                13.40206186
            ],
            [
                38.83495146,
                94.17475728,
                199.02912621,
                102.91262136,
                38.83495146,
                26.21359223
            ]
        ])
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_margin_weighted_gender_x_ideology_axis_0(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([71.82464218, 181.80362326, 375.31778601,
                             220.43839456, 110.99989991, 39.61565409])
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_0(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([72, 182, 376, 220, 110, 40])
        actual = cube.margin(axis=0, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([485, 515])
        actual = cube.margin(axis=1, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_weighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([500, 500])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_standard_error_axis_0(self):
        '''Calculate standard error across columns.'''
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([
            [
                0.0566273415723,
                0.0334615203051,
                0.0202778232899,
                0.0296144360169,
                0.0428948082269,
                0.0738148387071,
            ],
            [
                0.0566273415723,
                0.0334615203051,
                0.0202778232899,
                0.0296144360169,
                0.0428948082269,
                0.0738148387071,
            ],
        ])
        actual = cube._calculate_standard_error(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_standard_error_axis_1(self):
        '''Calculate standard error across rows.'''
        cube = CrunchCube(fixt_econ_gender_x_ideology_weighted)
        expected = np.array([
            [
                0.00816282401384,
                0.012194597867,
                0.0152950062668,
                0.0131008306818,
                0.00987765323496,
                0.00615484514258,
            ],
            [
                0.00816282401384,
                0.012194597867,
                0.0152950062668,
                0.0131008306818,
                0.00987765323496,
                0.00615484514258,
            ],
        ])
        actual = cube._calculate_standard_error(axis=1)
        np.testing.assert_almost_equal(actual, expected)
