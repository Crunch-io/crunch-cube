from unittest import TestCase

import numpy as np

from .fixtures import (
    fixt_cat_x_cat,
    fixt_univariate_categorical
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
        actual = cube.margins()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_none(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([15])
        actual = cube.margins()
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_0(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([10, 5])
        actual = cube.margins(dimension=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margins_bivariate_cat_dim_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([7, 8])
        actual = cube.margins(dimension=1)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_univariate_cat_din_none(self):
        cube = CrunchCube(fixt_univariate_categorical)
        expected = np.array([0.6666667, 0.3333333])
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
        actual = cube.proportions(dimension=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_bivariate_cat_din_1(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([
            [0.7142857, 0.2857143],
            [0.6250000, 0.3750000],
        ])
        actual = cube.proportions(dimension=1)
        np.testing.assert_almost_equal(actual, expected)
