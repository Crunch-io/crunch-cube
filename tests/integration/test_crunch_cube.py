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
    FIXT_ECON_MEAN_NO_DIMS,
    FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED,
    FIXT_ADMIT_X_DEPT_UNWEIGHTED,
    FIXT_ADMIT_X_GENDER_WEIGHTED,
    FIXT_SELECTED_CROSSTAB_4,
    FIXT_PETS_X_PETS,
    FIXT_PETS_X_FRUIT,
    FIXT_PETS_ARRAY,
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
                -0.182449424191072,
                0.000632923708704169,
                0.000367602277025753,
                0.153659627087918,
                0.0157593386530601,
                -7.31288611096873e-12,
                -0.825177717845072,
                0.469687167401825
            ],
            [
                0.182449424191079,
                -0.000632923708704146,
                -0.000367602277025734,
                -0.153659627087916,
                -0.0157593386530601,
                7.31288611096927e-12,
                0.825177717845075,
                -0.469687167401828
            ]
        ])
        actual = cube.pvals(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals_axis_1(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_GERMAN_WEIGHTED)
        expected = np.array([
            [
                -0.174780896035399,
                0.00067832739208794,
                0.000508330650039071,
                0.161553975907204,
                0.0193301356937216,
                -4.33199265948476e-12,
                -0.812304843861103,
                0.46145538258239,
            ],
            [
                0.176109557731585,
                -0.0007046838390719,
                -0.000528959758456953,
                -0.162846203806368,
                -0.0197051728937385,
                5.03371135873422e-12,
                0.812870881353903,
                -0.462833246125983,
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

    def test_mean_no_dims(self):
        cube = CrunchCube(FIXT_ECON_MEAN_NO_DIMS)
        expected = np.array([49.095])
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_from_r_row_margin(self):
        cube = CrunchCube(FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED)
        expected = np.array([
            [
                -1.35700098973668,
                3.39819222765456,
                3.47632774910236,
                1.39986424142017,
                2.33910237706402,
                -6.92590429515317,
                -0.237453687452224,
                0.736452470486666
            ],
            [
                1.3528312160513,
                -3.38775031004662,
                -3.4656457377556,
                -1.39556275813377,
                -2.33191481595459,
                6.90462247318263,
                0.236724043078395,
                -0.734189509622821,
            ],
        ])
        actual = cube._calculate_statistics(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_from_r_col_margin(self):
        cube = CrunchCube(FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED)
        expected = np.array([
            [
                -1.33325107235154,
                3.4170985193131,
                3.56231261682056,
                1.42672343792323,
                2.41444184160409,
                -6.85140362038577,
                -0.220890470186746,
                0.722988145330955,
            ],
            [
                1.33325107235152,
                -3.41709851931311,
                -3.56231261682057,
                -1.42672343792324,
                -2.41444184160409,
                6.85140362038576,
                0.220890470186742,
                -0.72298814533095,
            ],
        ])
        actual = cube._calculate_statistics(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_dept_unweighted_rows(self):
        cube = CrunchCube(FIXT_ADMIT_X_DEPT_UNWEIGHTED)
        expected = np.array([
            [
                17.3006739150679,
                12.1555052876046,
                -2.61883165552036,
                -3.12585957287982,
                -7.73178794867428,
                -23.9433203846143,
            ],
            [
                -17.2790610621901,
                -12.1403200324679,
                2.6155600821955,
                3.12195459533981,
                7.72212901884083,
                23.9134092110139,
            ]
        ])
        actual = cube._calculate_statistics(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_dept_unweighted_cols(self):
        cube = CrunchCube(FIXT_ADMIT_X_DEPT_UNWEIGHTED)
        expected = np.array([
            [
                18.7216214725448,
                13.3291986335621,
                -2.67980030430232,
                -3.19261047229265,
                -8.09694682104735,
                -32.0139892315214,
            ],
            [
                -18.7216214725448,
                -13.3291986335621,
                2.67980030430231,
                3.19261047229265,
                8.09694682104735,
                32.0139892315214,
            ],
        ])
        actual = cube._calculate_statistics(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_gender_weighted_rows(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = np.array([
            [9.80281743121017, -9.80281743121017],
            [-9.71107624617507, 9.71107624617506],
        ])
        actual = cube._calculate_statistics(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_gender_weighted_cols(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = np.array([
            [9.75089877074671, -9.72361434000118],
            [-9.75089877074672, 9.72361434000117],
        ])
        actual = cube._calculate_statistics(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_dim_names(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = ['Statements agreed with about Climate', 'Gender']
        actual = [dim.name for dim in cube.dimensions]
        self.assertEqual(actual, expected)

    def test_selected_crosstab_dim_aliases(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = ['attitudes_recoded_klima_2', 'pdl_gender']
        actual = [dim.alias for dim in cube.dimensions]
        self.assertEqual(actual, expected)

    def test_selected_crosstab_as_array(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [9928.20954289002, 11524.821237084192],
            [9588.843313998908, 9801.254016136965],
            [11697.435357575358, 13095.670425525452],
            [9782.8995547749, 10531.918128023966],
            [4417.596222134318, 3448.380316269752],
            [6179.175512581436, 6490.427474934746],
        ])
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_by_rows(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            21453.03077997421,
            19390.097330135875,
            24793.105783100807,
            20314.817682798865,
            7865.976538404069,
            12669.602987516182,
        ])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_by_cols(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [14566.261567907562, 15607.301233922663],
            [14456.513325488017, 15450.609903833058],
            [14415.136475733132, 15405.898678070093],
            [11485.661204663904, 11912.588886491172],
            [11664.69933815247, 12110.196347286023],
            [11547.413553551738, 11961.575582997419],
        ])
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_total(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [30173.5628018302],
            [29907.1232293211],
            [29821.0351538032],
            [23398.2500911551],
            [23774.8956854385],
            [23508.9891365492],
        ])
        actual = cube.margin()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_by_rows(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [0.4627882020361299, 0.5372117979638701],
            [0.4945227014975337, 0.5054772985024663],
            [0.47180193800279874, 0.5281980619972013],
            [0.481564723224583, 0.5184352767754171],
            [0.5616081106479636, 0.4383918893520365],
            [0.48771658580541166, 0.5122834141945883],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_by_cols(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [0.6815894041587091, 0.7384249886863752],
            [0.6632887957217867, 0.6343603312193796],
            [0.8114689290154947, 0.8500426167391849],
            [0.8517489224566737, 0.8840998567462627],
            [0.3787149667617584, 0.28475015741941767],
            [0.535113381358101, 0.5426064007955989],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_total(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [0.329036700375595, 0.381950958618156],
            [0.320620717695708,  0.327723062528721],
            [0.392254504152701, 0.439142047148397],
            [0.418103897371069,  0.450115632023491],
            [0.185809278853744, 0.14504292098248],
            [0.262843097025161,  0.27608279697761],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_is_double_mr(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = False
        actual = cube._is_double_multiple_response()
        self.assertEqual(actual, expected)

    def test_pets_x_pets_is_double_mr(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = True
        actual = cube._is_double_multiple_response()
        self.assertEqual(actual, expected)

    def test_pets_x_pets_as_array(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = np.array([
            [40, 14, 18],
            [14, 34, 16],
            [18, 16, 38],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_x_pets_proportions_by_cell(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = np.array([
            [.5, .4, .5294118],
            [.4, .4303797, .4571429],
            [.5294118, .4571429, .5428571],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_proportions_by_col(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = np.array([
            [1., .4827586, .4736842],
            [.4117647, 1., 0.4210526],
            [.5294118, .5517241, 1.],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_proportions_by_row(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = np.array([
            [1., .4117647, .5294118],
            [.4827586, 1., .5517241],
            [.4736842, .4210526, 1.],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_as_array(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([
            [12, 28],
            [12, 22],
            [12, 26],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_x_fruit_margin_row(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([40, 34, 38])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_as_array(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([
            [45, 34],
            [40, 40],
            [32, 38],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_proportions_by_cell(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([
            [0.1965066, 0.1484716],
            [0.1746725, 0.1746725],
            [0.1397380, 0.1659389],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_proportions_by_row(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([
            [0.5696203, 0.4303797],
            [0.5000000, 0.500000],
            [0.4571429, 0.5428571],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_proportions_by_col(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([
            [0.3846154, 0.3035714],
            [0.3418803, 0.3571429],
            [0.2735043, 0.3392857],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_margin_total(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = 229
        actual = cube.margin()
        self.assertEqual(actual, expected)

    def test_pets_array_margin_by_row(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([79, 80, 70])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_margin_by_col(self):
        cube = CrunchCube(FIXT_PETS_ARRAY)
        expected = np.array([117, 112])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)
