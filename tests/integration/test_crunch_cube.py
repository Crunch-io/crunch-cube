from unittest import TestCase

import numpy as np

from .fixtures import (
    FIXT_CAT_X_CAT,
    FIXT_CAT_X_CAT_GERMAN_WEIGHTED,
    FIXT_CAT_X_DATETIME,
    FIXT_CAT_X_NUM_X_DATETIME,
    FIXT_CAT_X_MR,
    FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED,
    FIXT_UNIVARIATE_CATEGORICAL,
    FIXT_VOTER_REGISTRATION,
    FIXT_SIMPLE_DATETIME,
    FIXT_SIMPLE_TEXT,
    FIXT_SIMPLE_CAT_ARRAY,
    FIXT_SIMPLE_MR,
    FIXT_STATS_TEST,
    FIXT_ECON_MEAN_AGE_BLAME_X_GENDER,
)
from cr.cube.crunch_cube import CrunchCube


class TestCrunchCube(TestCase):
    def test_crunch_cube_loads_data(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = FIXT_CAT_X_CAT['value']
        actual = cube._cube
        self.assertEqual(actual, expected)

    def test_as_array_univariate_cat_exclude_missing(self):
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([10, 5])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_univariate_cat_exclude_missing_adjusted(self):
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([11, 6])
        actual = cube.as_array(adjusted=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_numeric(self):
        cube = CrunchCube(FIXT_VOTER_REGISTRATION)
        expected = np.array([885, 105, 10])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_numeric_adjusted(self):
        cube = CrunchCube(FIXT_VOTER_REGISTRATION)
        expected = np.array([886, 106, 11])
        actual = cube.as_array(adjusted=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_datetime(self):
        cube = CrunchCube(FIXT_SIMPLE_DATETIME)
        expected = np.array([1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_datetime_adjusted(self):
        cube = CrunchCube(FIXT_SIMPLE_DATETIME)
        expected = np.array([2, 2, 2, 2])
        actual = cube.as_array(adjusted=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_text(self):
        cube = CrunchCube(FIXT_SIMPLE_TEXT)
        expected = np.array([1, 1, 1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_univariate_cat_include_missing(self):
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([10, 5, 5, 0])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [5, 2],
            [5, 3],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_exclude_missing_adjusted(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [6, 3],
            [6, 4],
        ])
        actual = cube.as_array(adjusted=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_unweighted(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [5, 2],
            [5, 3],
        ])
        actual = cube._as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [5, 3, 2, 0],
            [5, 2, 3, 0],
            [0, 0, 0, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_datetime_include_missing(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_numeric(self):
        cube = CrunchCube(FIXT_VOTER_REGISTRATION)
        expected = np.array([1000])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_datetime(self):
        cube = CrunchCube(FIXT_SIMPLE_DATETIME)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_text(self):
        cube = CrunchCube(FIXT_SIMPLE_TEXT)
        expected = np.array([6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([10, 5])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
        expected = np.array([1, 1, 1, 1])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([7, 8])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
        expected = np.array([1, 1, 1, 1, 0])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_univariate_cat_axis_none(self):
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([0.6666667, 0.3333333])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_numeric(self):
        cube = CrunchCube(FIXT_VOTER_REGISTRATION)
        expected = np.array([0.885, 0.105, 0.010])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_datetime(self):
        cube = CrunchCube(FIXT_SIMPLE_DATETIME)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_text(self):
        cube = CrunchCube(FIXT_SIMPLE_TEXT)
        expected = np.array([
            0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)


    def test_proportions_cat_x_cat_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [0.3333333, 0.1333333],
            [0.3333333, 0.2000000],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [0.5, 0.4],
            [0.5, 0.6],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [0.7142857, 0.2857143],
            [0.6250000, 0.3750000],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_UNIVARIATE_CATEGORICAL)
        expected = np.array([66.6666667, 33.3333333])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_numeric(self):
        cube = CrunchCube(FIXT_VOTER_REGISTRATION)
        expected = np.array([88.5, 10.5, 1.0])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_datetime(self):
        cube = CrunchCube(FIXT_SIMPLE_DATETIME)
        expected = np.array([25., 25., 25., 25.])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_text(self):
        cube = CrunchCube(FIXT_SIMPLE_TEXT)
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
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [33.3333333, 13.3333333],
            [33.3333333, 20.],
        ])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [50, 40],
            [50, 60],
        ])
        actual = cube.percentages(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = np.array([
            [71.4285714, 28.5714286],
            [62.50000, 37.50000],
        ])
        actual = cube.percentages(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = [
            ['B', 'C'],
            ['C', 'E'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_cat_include_missing(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = [
            ['B', 'C', 'No Data'],
            ['C', 'D', 'E', 'No Data'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_DATETIME)
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
        cube = CrunchCube(FIXT_SIMPLE_CAT_ARRAY)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'c', 'd', 'No Data'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(FIXT_SIMPLE_CAT_ARRAY)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'c', 'd'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_as_array_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(FIXT_SIMPLE_CAT_ARRAY)
        expected = np.array([
            [3, 3, 0, 0],
            [1, 3, 2, 0],
            [0, 2, 1, 3],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_simple_cat_array_include_missing(self):
        cube = CrunchCube(FIXT_SIMPLE_CAT_ARRAY)
        expected = np.array([
            [3, 3, 0, 0, 0],
            [1, 3, 2, 0, 0],
            [0, 2, 1, 3, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_num_x_datetime(self):
        '''Test 3D cube, slicing accross first (numerical) variable.'''
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
        expected = np.array([20])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
        expected = np.array([
            [3, 2],
            [3, 4],
            [4, 3],
            [0, 1],
        ])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
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
        cube = CrunchCube(FIXT_CAT_X_NUM_X_DATETIME)
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
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = [['Response #1', 'Response #2', 'Response #3']]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_simple_mr_include_missing(self):
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = [
            ['Response #1', 'Response #2', 'Response #3']
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_simple_mr_exclude_missing(self):
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = np.array([3, 4, 0])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_simple_mr_axis_none(self):
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = np.array([5, 6, 6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_simple_mr(self):
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = np.array([0.6, 0.6666667, 0.])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_mr_exclude_missing(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = [
            ['rambutan', 'satsuma'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_mr_include_missing(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = [
            ['rambutan', 'satsuma', 'No Data'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_cat_x_mr(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = np.array([
            [12, 12, 12],
            [28, 22, 26],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = np.array([80, 79, 70])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = np.array([
            [28, 25, 23],
            [52, 54, 47]
        ])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = np.array([
            [0.15, 0.1518987, 0.1714286],
            [0.35, 0.2784810, 0.3714286],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_MR)
        expected = np.array([
            [0.4285714, 0.4800000, 0.5217391],
            [0.5384615, 0.4074074, 0.5531915],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_as_array_unweighted_gender_x_ideology(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([
            [32, 85, 171, 114, 70, 13],
            [40, 97, 205, 106, 40, 27]
        ])
        actual = cube.as_array(weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_weighted_gender_x_ideology(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
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
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([71.82464218, 181.80362326, 375.31778601,
                             220.43839456, 110.99989991, 39.61565409])
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_0(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([72, 182, 376, 220, 110, 40])
        actual = cube.margin(axis=0, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([485, 515])
        actual = cube.margin(axis=1, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_weighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([500, 500])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_standard_error_axis_0(self):
        '''Calculate standard error across columns.'''
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        axis = 0
        expected = np.array([
            [
                0.0556176,
                0.0332214,
                0.0202155,
                0.0295626,
                0.0430823,
                0.0713761
            ],
            [
                0.0556176,
                0.0332214,
                0.0202155,
                0.0295626,
                0.0430823,
                0.0713761,
            ],
        ])
        actual = cube._calculate_standard_error(axis=axis)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_standard_error_axis_1(self):
        '''Calculate standard error across rows.'''
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        axis = 1
        expected = np.array([
            [
                0.0084098,
                0.0124771,
                0.0156354,
                0.0134104,
                0.010181,
                0.0063893,
            ],
            [
                0.0079255,
                0.0117586,
                0.0147351,
                0.0126382,
                0.0095947,
                0.0060214,
            ],
        ])
        actual = cube._calculate_standard_error(axis=axis)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_statistics_raises_error_for_bad_axis(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        axis = 2
        with self.assertRaises(ValueError):
            cube._calculate_statistics(axis)

    def test_calculate_statistics_axis_0(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([
            [
                -0.7316249,
                -0.5418986,
                -1.4985995,
                1.1212007,
                3.4848471,
                -2.265434,
            ],
            [
                0.7316249,
                0.5418986,
                1.4985995,
                -1.1212007,
                -3.4848471,
                2.265434,
            ]
        ])
        actual = cube._calculate_statistics(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_statistics_axis_1(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([
            [
                -0.6950547,
                -0.524633,
                -1.454419,
                1.0896893,
                3.2737502,
                -2.0051536,
            ],
            [
                0.7375224,
                0.556688,
                1.5432837,
                -1.1562691,
                -3.4737757,
                2.1276681,
            ]
        ])
        actual = cube._calculate_statistics(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_GERMAN_WEIGHTED)
        expected = np.array([
            [
                -0.18420272141763716,
                0.000664966346674678,
                0.00037716285275513073,
                0.15399832449651685,
                0.015829610375772907,
                -8.697487174913476e-12,
                -0.8252001781531713,
                0.46978876911573253,
            ],
            [
                0.1842027214176405,
                -0.000664966346674678,
                -0.00037716285275513073,
                -0.15399832449651685,
                -0.01582961037577313,
                8.697487174913476e-12,
                0.8252001781531728,
                -0.46978876911573697
            ]
        ])
        actual = cube.pvals(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_GERMAN_WEIGHTED)
        expected = np.array([
            [
                -0.178514305553791,
                0.0008140062811921034,
                0.0006341605404542872,
                0.1684165900623824,
                0.021308763357477334,
                -5.784706047506916e-12,
                -0.8138010875690975,
                0.4674522428169867,
            ],
            [
                0.1829786364352084,
                -0.0007866365804722886,
                -0.0005731511734761163,
                -0.16600295360474404,
                -0.02070272847943566,
                1.226752033289813e-11,
                0.8158409095455534,
                -0.4675571877257023,
            ]
        ])
        actual = cube.pvals(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals_stats(self):
        cube = CrunchCube(FIXT_STATS_TEST)

        expected_axis_0 = np.array([
            [
                0.021992266759100376,
                0.00011398387657313158,
                0.08710488165760855,
                -0.6809458435023874,
                -0.4639470201751028,
                -5.861367553539054e-7,
            ],
            [
                -0.021992266759100376,
                -0.00011398387657313158,
                -0.08710488165760855,
                0.6809458435023874,
                0.4639470201751028,
                5.861367553539054e-7,
            ]
        ])
        actual = cube.pvals(0)
        np.testing.assert_almost_equal(actual, expected_axis_0)

        expected_axis_1 = np.array([
            [
                0.04565813781439898,
                0.00019875030267102467,
                0.0881878053993032,
                -0.682332970015509,
                -0.4667738621549087,
                -0.0000011049984953714898,
            ],
            [
                -0.04544250237653502,
                -0.0001958471500445036,
                -0.08787083482608415,
                0.6820331877195527,
                0.46632917677948105,
                0.000001078109930885418,
            ]
        ])
        actual = cube.pvals(1)
        np.testing.assert_almost_equal(actual, expected_axis_1)

    def test_mean_age_for_blame_x_gender(self):
        cube = CrunchCube(FIXT_ECON_MEAN_AGE_BLAME_X_GENDER)
        expected = np.array([
            [52.78205128205122, 49.9069767441861],
            [50.43654822335009, 48.20100502512572],
            [51.5643564356436, 47.602836879432715],
            [58, 29],
            [37.53846153846155, 39.45238095238095],
        ])
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)
