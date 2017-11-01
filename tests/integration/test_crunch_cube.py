from unittest import TestCase

import numpy as np

from .fixtures import (
    fixt_cat_x_cat,
    fixt_univariate_categorical,
    fixt_voter_registration,
    fixt_simple_datetime,
    fixt_simple_text,
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

    def test_as_array_bivariate_cat_exclude_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [5, 2],
            [5, 3]
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_bivariate_cat_include_missing(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [5, 3, 2, 0],
            [5, 2, 3, 0],
            [0, 0, 0, 0],
        ])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_margins_univariate_cat_dim_none(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_numeric(self):
        cube = CrunchCube(fixt_voter_registration)
        expected = np.array([1000])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_datetime(self):
        cube = CrunchCube(fixt_simple_datetime)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_text(self):
        cube = CrunchCube(fixt_simple_text)
        expected = np.array([6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([10, 5])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([7, 8])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_univariate_cat_din_none(self):
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


    def test_proportions_bivariate_cat_din_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.3333333, 0.1333333],
            [0.3333333, 0.2000000],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_bivariate_cat_din_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.5, 0.4],
            [0.5, 0.6],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_bivariate_cat_din_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.7142857, 0.2857143],
            [0.6250000, 0.3750000],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_univariate_cat_din_none(self):
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

    def test_percentages_bivariate_cat_din_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [33.3333333, 13.3333333],
            [33.3333333, 20.],
        ])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_bivariate_cat_din_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [50, 40],
            [50, 60],
        ])
        actual = cube.percentages(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_bivariate_cat_din_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [71.4285714, 28.5714286],
            [62.50000, 37.50000],
        ])
        actual = cube.percentages(axis=1)
        np.testing.assert_almost_equal(actual, expected)
