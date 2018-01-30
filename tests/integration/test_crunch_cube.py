from unittest import TestCase

import numpy as np

from .fixtures import FIXT_CAT_X_CAT
from .fixtures import FIXT_CAT_X_CAT_GERMAN_WEIGHTED
from .fixtures import FIXT_CAT_X_DATETIME
from .fixtures import FIXT_CAT_X_NUM_X_DATETIME
from .fixtures import FIXT_CAT_X_MR_SIMPLE
from .fixtures import FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED
from .fixtures import FIXT_UNIVARIATE_CATEGORICAL
from .fixtures import FIXT_VOTER_REGISTRATION
from .fixtures import FIXT_SIMPLE_DATETIME
from .fixtures import FIXT_SIMPLE_TEXT
from .fixtures import FIXT_SIMPLE_CAT_ARRAY
from .fixtures import FIXT_SIMPLE_MR
from .fixtures import FIXT_STATS_TEST
from .fixtures import FIXT_ECON_MEAN_AGE_BLAME_X_GENDER
from .fixtures import FIXT_ECON_MEAN_NO_DIMS
from .fixtures import FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED
from .fixtures import FIXT_ADMIT_X_DEPT_UNWEIGHTED
from .fixtures import FIXT_ADMIT_X_GENDER_WEIGHTED
from .fixtures import FIXT_SELECTED_CROSSTAB_4
from .fixtures import FIXT_PETS_X_PETS
from .fixtures import FIXT_PETS_X_FRUIT
from .fixtures import FIXT_PETS_ARRAY
from .fixtures import FIXT_ECON_BLAME_WITH_HS
from .fixtures import FIXT_ECON_BLAME_WITH_HS_MISSING
from .fixtures import FIXT_FRUIT_X_PETS
from .fixtures import FIXT_FRUIT_X_PETS_ARRAY
from .fixtures import FIXT_GENDER_PARTY_RACE
from .fixtures import FIXT_SINGLE_COL_MARGIN_NOT_ITERABLE
from .fixtures import FIXT_BINNED
from .fixtures import FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM
from .fixtures import FIXT_CAT_X_CAT_WITH_EMPTY_COLS
from .fixtures import FIXT_ARRAY_X_MR
from .fixtures import FIXT_PROFILES_PERCENTS
from .fixtures import FIXT_IDENTITY_X_PERIOD
from .fixtures import FIXT_CA_SINGLE_CAT
from .fixtures import FIXT_MR_X_SINGLE_WAVE
from .fixtures import FIXT_PETS_ARRAY_X_PETS
from .fixtures import FIXT_PETS_X_PETS_ARRAY
from .fixtures import FIXT_SELECTED_3_WAY_2
from .fixtures import FIXT_SELECTED_3_WAY
from .fixtures import FIXT_SINGLE_CAT_MEANS
from .fixtures import FIXT_CA_X_SINGLE_CAT

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
            [[0.5, 0.5],
             [0., 0.],
             [0., 0.],
             [0., 0.]],

            [[0.4, 0.2],
             [0.2, 0.2],
             [0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0.4, 0.6],
             [0., 0.],
             [0., 0.]],

            [[0., 0.],
             [0., 0.],
             [0.6, 0.4],
             [0., 0.]],

            [[0., 0.],
             [0., 0.],
             [0.33333333, 0.33333333],
             [0., 0.33333333]],
        ])
        # Set axis to tuple (1, 2), since we want to do a total for each slice
        # of the 3D cube. This is consistent with how the np.sum works
        # (axis parameter), which is used internally in
        # 'proportions' calculation.
        actual = cube.proportions((1, 2))
        np.testing.assert_almost_equal(actual, expected)

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
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = [
            ['rambutan', 'satsuma'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_mr_include_missing(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = [
            ['rambutan', 'satsuma', 'No Data'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_cat_x_mr(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([
            [12, 12, 12],
            [28, 22, 26],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([80, 79, 70])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_by_col(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([40, 34, 38])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_none(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([
            [.42857143, .48,       .52173913],
            [.53846154, .40740741, .55319149],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_mr_axis_0(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([
            [.3, .3529412, .3157895],
            [.7, .6470588, .6842105],
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
            30173.5628018302,
            29907.1232293211,
            29821.0351538032,
            23398.2500911551,
            23774.8956854385,
            23508.9891365492
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
            [.6815894,  .73842499],
            [.6632888,  .63436033],
            [.81146893, .85004262],
            [.85174892, .88409986],
            [.37871497, .28475016],
            [.53511338, .5426064],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_is_double_mr(self):
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = False
        actual = cube.is_double_mr
        self.assertEqual(actual, expected)

    def test_pets_x_pets_is_double_mr(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = True
        actual = cube.is_double_mr
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
            [.5, .2, .2571429],
            [.2, .4303797, .2285714],
            [.2571429, .2285714, .5428571],
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

    def test_headings_econ_blame_one_subtotal(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=None)[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
            'Test Heading with Skipped',
        ]
        actual = cube.labels(include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=None)[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_with_missing(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
            'Skipped',
            'Test Heading with Skipped',
            'Not Asked',
            'No Data',
        ]
        actual = cube.labels(include_missing=True, include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_one_transform(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_one_transform_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=None)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_excluded(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([285, 396, 681, 242, 6, 68, 77])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_included(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([285, 396, 681, 242, 6, 68, 3, 77, 0, 0])
        actual = cube.as_array(include_missing=True, include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_proportions_one_transform(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .6830491, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_one_transform_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_excluded(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([
            .2858576,
            .3971916,
            .6830491,
            .2427282,
            .0060181,
            .0682046,
            .0772317,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_included(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([
            .2858576,
            .3971916,
            .6830491,
            .2427282,
            .0060181,
            .0682046,
            .003009,
            .0772317,
            0,
            0,
        ])
        actual = cube.proportions(include_missing=True,
                                  include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_count_unweighted(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = 4526
        actual = cube.count(weighted=False)
        self.assertEqual(actual, expected)

    def test_count_weighted(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = 4451.955438803242
        actual = cube.count(weighted=True)
        self.assertEqual(actual, expected)

    def test_econ_x_ideology_index_by_col(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([
            [
                .91861761,
                .96399471,
                .9394101,
                1.06629132,
                1.30027051,
                .67660435,
            ],
            [
                1.08138239,
                1.03600529,
                1.0605899,
                .93370868,
                .69972949,
                1.32339565,
            ],
        ])
        actual = cube.index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_econ_x_ideology_index_by_row(self):
        cube = CrunchCube(FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([
            [
                .91861761,
                .96399471,
                .9394101,
                1.06629132,
                1.30027051,
                .67660435,
            ],
            [
                1.08138239,
                1.03600529,
                1.0605899,
                .93370868,
                .69972949,
                1.32339565,
            ],
        ])
        actual = cube.index(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_cell(self):
        cube = CrunchCube(FIXT_FRUIT_X_PETS)
        expected = np.array([
            [.4285714, .48, .5217391],
            [.5384615, .4074074, .5531915],
        ])
        actual = cube.proportions(axis=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_cell(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([
            [.4285714, .5384615],
            [.48, .4074074],
            [.5217391, .5531915],
        ])
        actual = cube.proportions(axis=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_col(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([
            [.4285714, .5384615],
            [.48, .4074074],
            [.5217391, .5531915],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_row(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([
            [.3, .7],
            [.3529412, .6470588],
            [.3157895, .6842105],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_row(self):
        cube = CrunchCube(FIXT_FRUIT_X_PETS)
        expected = np.array([
            [.4285714, .48, .5217391],
            [.5384615, .4074074, .5531915],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_col(self):
        cube = CrunchCube(FIXT_FRUIT_X_PETS)
        expected = np.array([
            [.3, .3529412, .3157895],
            [.7, .6470588, .6842105],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_row(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([
            [.8571429,  1.1152941, .9610984],
            [1.0769231, .9466231,  1.019037],
        ])
        actual = cube.index(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_cell(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.array([
            [.8571429,  1.1152941, .9610984],
            [1.0769231, .9466231,  1.019037],
        ])
        actual = cube.index(axis=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_col(self):
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        expected = np.full(cube.as_array().shape, np.nan)
        actual = cube.index(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_index_by_col(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.array([
            [.8571429,  1.0769231],
            [1.1152941, .9466231],
            [.9610984,  1.019037],
        ])
        actual = cube.index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_cat_index_by_row(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.full(cube.as_array().shape, np.nan)
        # with self.assertRaises(ValueError) as ctx:
        actual = cube.index(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_index_by_cell(self):
        cube = CrunchCube(FIXT_PETS_X_FRUIT)
        expected = np.full(cube.as_array().shape, np.nan)
        # with self.assertRaises(ValueError) as ctx:
        actual = cube.index(axis=None)
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_mr_index_by_col(self):
        cube = CrunchCube(FIXT_PETS_X_PETS)
        expected = np.array([
            [2.06944444, .99904215, .98026316],
            [.95863971,  2.328125,  .98026316],
            [1.02205882, 1.0651341, 1.93055556],
        ])
        actual = cube.index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_array_proportions_by_row(self):
        '''Get the proportions for each slice of the 3D cube.

        The axis is equal to 2, since this is the dimensions across which
        we have to calculate the margin.
        '''
        cube = CrunchCube(FIXT_FRUIT_X_PETS_ARRAY)
        expected = ([[
            [0.52,       0.48],
            [0.57142857, 0.42857143],
            [0.47826087, 0.52173913]],

           [[0.59259259, 0.40740741],
            [0.46153846, 0.53846154],
            [0.44680851, 0.55319149]]])
        actual = cube.proportions(axis=2)
        np.testing.assert_almost_equal(actual, expected)

    def test_identity_x_period_axis_out_of_bounds(self):
        cube = CrunchCube(FIXT_IDENTITY_X_PERIOD)
        # There are margins that have 0 value in this cube. In whaam, they're
        # pruned, so they're not shown. CrunchCube is not responsible for
        # pruning (cr.exporter is).
        expected = np.array([94, 0, 248, 210, 102, 0, 0, 0, 286, 60])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_with_single_cat(self):
        cube = CrunchCube(FIXT_CA_SINGLE_CAT)
        # The last 0 of the expectation is not visible in whaam because of
        # pruning, which is not the responsibility of cr.cube.
        expected = np.array([79, 80, 70, 0])
        actual = cube.margin(axis=1, weighted=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_single_wave(self):
        cube = CrunchCube(FIXT_MR_X_SINGLE_WAVE)
        expected = np.array([
            308.32755712, 187.06825269, 424.82328071, 72.68885079,
            273.15993803, 467.62527785, 62.183386, 442.80441811,
            281.57825919, 0., 237.35065847, 233.19692455, 0., 0., 0., 0.,
            0., 0., 0., 38.05075633, 90.93234493, 123.22747266, 142.42909713,
        ])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_by_col(self):
        cube = CrunchCube(FIXT_PETS_ARRAY_X_PETS)
        expected = np.array([
            [0.59097127, 0., 0.55956679],
            [0.40902873, 1., 0.44043321],
        ])
        # Since cube is 3D, col dim is 1 (instead of 0)
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_row(self):
        cube = CrunchCube(FIXT_PETS_ARRAY_X_PETS)
        expected = np.array([
            [0.44836533, 0., 0.48261546],
            [0.39084967, 1., 0.47843137],
        ])
        # Since cube is 3D, row dim is 2 (instead of 0)
        actual = cube.proportions(axis=2)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_cell(self):
        cube = CrunchCube(FIXT_PETS_ARRAY_X_PETS)
        expected = np.array([
            [0.44836533, 0., 0.48261546],
            [0.39084967, 1., 0.47843137],
        ])
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_array_by_col(self):
        cube = CrunchCube(FIXT_PETS_X_PETS_ARRAY)
        expected = np.array([
            [0.55555556, 0.19444444],
            [0., 0.55555556],
            [0.44444444, 0.25],
        ])
        actual = cube.proportions(axis=1)[0]
        # Since cube is 3D, col dim is 1 (instead of 0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_array_by_row(self):
        cube = CrunchCube(FIXT_PETS_X_PETS_ARRAY)
        expected = np.array([
            [0.44444444, 0.41176471],
            [0., 1.],
            [0.5, 0.47368421],
        ])
        actual = cube.proportions(axis=2)[0]
        # Since cube is 3D, col dim is 1 (instead of 0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_array_by_cell(self):
        cube = CrunchCube(FIXT_PETS_X_PETS_ARRAY)
        expected = np.array([
            [0.44444444, 0.41176471],
            [0., 1.],
            [0.5, 0.47368421],
        ])
        actual = cube.proportions(axis=None)[0]
        # Since cube is 3D, col dim is 1 (instead of 0)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_cat_x_cat_by_row(self):
        cube = CrunchCube(FIXT_SELECTED_3_WAY_2)
        # Only compare 0 slice (parity with whaam tests)
        expected = np.array([
            [0.5923110874002918, 0.3758961399306439],
            [0, 0],
            [0.49431928922535223, 0.6091963925363675]
        ])
        actual = cube.proportions(axis=2)[0]
        # Since cube is 3D, row dim is 2 (instead of 1)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_col(self):
        cube = CrunchCube(FIXT_SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        expected = np.array([[1, 0], [1, 0], [1, 0]])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis 0 represents column direction.
        actual = cube.proportions(axis=0)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_row(self):
        cube = CrunchCube(FIXT_SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        expected = np.array([
            [0.0997975162008577, np.nan],
            [0.20327963774693497, np.nan],
            [0.3113417143573762, np.nan],
        ])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis 1 represents row direction.
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_cell(self):
        cube = CrunchCube(FIXT_SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        expected = np.array([
            [0.0997975162008577, np.nan],
            [0.20327963774693497, np.nan],
            [0.3113417143573762, np.nan],
        ])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis None represents cell direction.
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_col(self):
        cube = CrunchCube(FIXT_ARRAY_X_MR)
        expected = np.array([
            [0.5146153267487166, 0.04320534228100489, 0.5933354514113938],
            [0.4853846732512835, 0.9567946577189951, 0.4066645485886063],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_row(self):
        cube = CrunchCube(FIXT_ARRAY_X_MR)
        expected = np.array([
            [0.41922353375674093, 0.03471395310157275, 0.5832027484767315],
            [0.5143557893611596, 1, 0.5199603338915276],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=2)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_cell(self):
        cube = CrunchCube(FIXT_ARRAY_X_MR)
        expected = np.array([
            [0.41922353375674093, 0.03471395310157275, 0.5832027484767315],
            [0.5143557893611596, 1, 0.5199603338915276],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_profiles_percentages_add_up_to_100(self):
        cube = CrunchCube(FIXT_PROFILES_PERCENTS)
        props = cube.percentages(axis=1)
        actual_sum = np.sum(props, axis=1)
        expected_sum = np.ones(props.shape[0]) * 100
        np.testing.assert_almost_equal(actual_sum, expected_sum)

    def test_cat_x_cat_as_array_prune_cols(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array([
            [2, 2, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 2, 0, 0],
            [0, 2, 0, 1],
            [0, 1, 0, 0],
        ])
        actual = cube.as_array(prune=False)
        expected = np.array([
            [2, 2, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 1, 0],
        ])
        actual = cube.as_array(prune=True)
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_cat_props_by_col_prune_cols(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array([
            [1., 0.25, np.nan, 0.25],
            [0., 0., np.nan, 0.],
            [0., 0.125, np.nan, 0.5],
            [0., 0.25, np.nan, 0.],
            [0., 0.25, np.nan, 0.25],
            [0., 0.125, np.nan, 0.]
        ])
        actual = cube.proportions(axis=0, prune=False)
        expected = np.array([
            [1., 0.25, 0.25],
            [0., 0.125, 0.5],
            [0., 0.25, 0.],
            [0., 0.25, 0.25],
            [0., 0.125, 0.]
        ])
        actual = cube.proportions(axis=0, prune=True)
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_cat_props_by_row_prune_cols(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array([
            [0.4, 0.4, 0., 0.2],
            [np.nan, np.nan, np.nan, np.nan],
            [0., 0.33333333, 0., 0.66666667],
            [0., 1., 0., 0.],
            [0., 0.66666667, 0., 0.33333333],
            [0., 1., 0., 0.],
        ])
        actual = cube.proportions(axis=1, prune=False)
        expected = np.array([
            [0.4, 0.4, 0.2],
            [0., 0.33333333, 0.66666667],
            [0., 1., 0.],
            [0., 0.66666667, 0.33333333],
            [0., 1., 0.]
        ])
        actual = cube.proportions(axis=1, prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_props_by_cell_prune_cols(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array([
            [0.14285714, 0.14285714, 0., 0.07142857],
            [0., 0., 0., 0.],
            [0., 0.07142857, 0., 0.14285714],
            [0., 0.14285714, 0., 0.],
            [0., 0.14285714, 0., 0.07142857],
            [0., 0.07142857, 0., 0.],
        ])
        actual = cube.proportions(axis=None, prune=False)
        expected = np.array([
            [0.14285714, 0.14285714, 0.07142857],
            [0., 0.07142857, 0.14285714],
            [0., 0.14285714, 0.],
            [0., 0.14285714, 0.07142857],
            [0., 0.07142857, 0.],
        ])
        actual = cube.proportions(axis=None, prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_index_by_col_prune_cols(self):
        cube = CrunchCube(FIXT_CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array([
            [2.8, 0.7, np.nan, 0.7],
            [np.nan, np.nan, np.nan, np.nan],
            [0., 0.58333333, np.nan, 2.33333333],
            [0., 1.75, np.nan, 0.],
            [0., 1.16666667, np.nan, 1.16666667],
            [0., 1.75, np.nan, 0.]
        ])
        actual = cube.index(axis=0, prune=False)
        expected = np.array([
            [2.8, 0.7, 0.7],
            [0., 0.58333333, 2.33333333],
            [0., 1.75, 0.],
            [0., 1.16666667, 1.16666667],
            [0., 1.75, 0.]
        ])
        actual = cube.index(axis=0, prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_as_props_by_col_not_affected_by_prune(self):
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.93244626, 0.66023166],
            [0.63664278, 0.23166023],
            [0.29580348, 0.42857143],
            [0.04401228, 0.21428571],
            [0.00307062, 0.06177606],
            [0.02047083, 0.06370656],
        ])
        actual = cube.proportions(axis=0, include_transforms_for_dims=[0, 1],
                                  prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_as_props_by_row_not_affected_by_prune(self):
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.72705507, 0.27294493],
            [0.83827493, 0.16172507],
            [0.56555773, 0.43444227],
            [0.27922078, 0.72077922],
            [0.08571429, 0.91428571],
            [0.37735849, 0.62264151],
        ])
        actual = cube.proportions(axis=1, include_transforms_for_dims=[0, 1],
                                  prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_as_props_by_cell_not_affected_by_prune(self):
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.60936455, 0.22876254],
            [0.41605351, 0.08026756],
            [0.19331104, 0.14849498],
            [0.02876254, 0.07424749],
            [0.00200669, 0.02140468],
            [0.01337793, 0.02207358],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1],
                                  prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_prune_univariate_cat(self):
        cube = CrunchCube(FIXT_BINNED)
        expected = np.array([
            118504.40402204,
            155261.2723631,
            182923.95470245,
        ])
        actual = cube.as_array(prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_with_hs_and_prune_params(self):
        '''Test that HS and prune params don't break CAT x MR.'''
        cube = CrunchCube(FIXT_CAT_X_MR_SIMPLE)
        # Only ensure that the call doesn't break
        cube.as_array(
            include_transforms_for_dims=[0, 1],
            prune=True,
        )
        assert True

    def test_single_col_margin_not_iterable(self):
        cube = CrunchCube(FIXT_SINGLE_COL_MARGIN_NOT_ITERABLE)
        expected = (1,)
        actual = cube.margin(axis=0).shape
        self.assertEqual(actual, expected)

    def test_3d_percentages_by_col(self):
        cube = CrunchCube(FIXT_GENDER_PARTY_RACE)
        expected = np.array([
            [[.17647059, 0., 0., 0., 0., 0., 0., 0.],
             [.17647059, .05882353, 0., 0., 0., 0., 0., 0.],
             [.23529412, 0., 0., 0., 0., 0.05882353, 0., 0.],
             [.11764706, .05882353, 0., 0.05882353, 0., 0.05882353, 0., 0.]],

            [[.04761905, 0., 0., 0.04761905, 0., 0., 0., 0.],
             [.14285714, .04761905, .0952381, .04761905, 0., .04761905, 0., 0.],
             [.23809524, 0., 0.04761905, 0., 0., 0., 0., 0.],
             [.19047619, 0., 0.04761905, 0., 0., 0., 0., 0.]]
        ])
        # Set axis to tuple (1, 2), since we want to do a total for each slice
        # of the 3D cube. This is consistent with how the np.sum works
        # (axis parameter), which is used internally in
        # 'proportions' calculation.
        axis = (1, 2)
        actual = cube.proportions(axis=axis)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_dim_ind_for_cat_cube(self):
        cube = CrunchCube(FIXT_CAT_X_CAT)
        expected = None
        actual = cube.mr_dim_ind
        self.assertEqual(actual, expected)

    def test_simple_mr_margin_by_col(self):
        cube = CrunchCube(FIXT_SIMPLE_MR)
        expected = np.array([3, 4, 0])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_total_unweighted_margin_when_has_means(self):
        '''Tests that total margin is Unweighted N, when cube has means.'''
        cube = CrunchCube(FIXT_SINGLE_CAT_MEANS)
        expected = 17615
        actual = cube.margin(weighted=False)
        assert actual == expected

    def test_row_unweighted_margin_when_has_means(self):
        '''Tests that total margin is Unweighted N, when cube has means.'''
        cube = CrunchCube(FIXT_SINGLE_CAT_MEANS)
        expected = np.array([
            806, 14, 14, 28, 780, 42, 1114, 28, 24, 746, 2, 12, 6, 2178, 2026,
            571, 136, 16, 14, 1334, 1950, 26, 128, 4, 28, 3520, 1082, 36, 56,
            556, 38, 146, 114, 28, 12,
        ])
        actual = cube.margin(axis=1, weighted=False, prune=True)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_with_single_cat_pruning(self):
        cube = CrunchCube(FIXT_CA_SINGLE_CAT)
        # The last 0 of the expectation is not visible in whaam because of
        # pruning, which is not the responsibility of cr.cube.
        expected = np.array([79, 80, 70])
        actual = cube.as_array(weighted=False, prune=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_x_single_cat_col_margins(test):
        cube = CrunchCube(FIXT_CA_X_SINGLE_CAT)
        expected = np.array([25, 28, 23])
        # Axis equals to 1, because col direction in 3D cube is 1 (and not 0).
        # It operates on the 0th dimension of each slice (which is 1st
        # dimension of the cube).
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_single_cat_row_margins(test):
        cube = CrunchCube(FIXT_CA_X_SINGLE_CAT)
        expected = np.array([[13, 12], [16, 12], [11, 12]])
        # Axis equals to 2, because col direction in 3D cube is 2 (and not 1).
        # It operates on the 1st dimension of each slice (which is 2nd
        # dimension of the cube).
        actual = cube.margin(axis=2)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_single_cat_cell_margins(test):
        cube = CrunchCube(FIXT_CA_X_SINGLE_CAT)
        expected = np.array([25, 28, 23])
        # Axis equals to (1, 2), because the total is calculated for each slice.
        actual = cube.margin(axis=(1, 2))
        np.testing.assert_array_equal(actual, expected)
