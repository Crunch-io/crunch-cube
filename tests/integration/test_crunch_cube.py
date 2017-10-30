from unittest import TestCase

import numpy as np

from .fixtures import fixt_cat_x_cat
from cr.cube.crunch_cube import CrunchCube


class TestCrunchCube(TestCase):
    def test_crunch_cube_loads_data(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = fixt_cat_x_cat['value']
        actual = cube._cube
        self.assertEqual(actual, expected)

    def test_crunch_cube_as_array_categoricals(self):
        cube = CrunchCube(fixt_cat_x_cat)
        expected = np.array([[5, 2], [5, 3]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)
