from unittest import TestCase

from .fixtures import fixt_cat_x_cat

from cr.cube.crunch_cube import CrunchCube


class TestCrunchCube(TestCase):
    def test_crunch_cube_loads_data(self):
        cube = CrunchCube(fixt_cat_x_cat)
        self.assertEqual(cube._cube, fixt_cat_x_cat)
