from unittest import TestCase
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube

from .fixtures import MISSING_CAT_HS


class TestCrunchCube(TestCase):
    def test_missing_cat_hs_labels(self):
        cube = CrunchCube(MISSING_CAT_HS)

        # Don't expect the missing category "Non voters"
        expected = [[
            'Whites',
            'White college women voters',
            'White non-college women voters',
            'White college men voters',
            'White non-college men voters',
            'Black voters',
            'Latino and other voters',
        ]]
        actual = cube.labels(include_transforms_for_dims=[0])
        assert actual == expected
