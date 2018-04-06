from unittest import TestCase

from .fixtures import ECON_BLAME_WITH_HS
from .fixtures import ECON_BLAME_WITH_HS_MISSING
from .fixtures import ECON_BLAME_X_IDEOLOGY_ROW_HS
from .fixtures import CA_WITH_NETS

from cr.cube.crunch_cube import CrunchCube


class TestDimension(TestCase):
    def test_subtotals_indices_single_subtotal(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS).dimensions[0]
        expected = [{
            'anchor_ind': 1,
            'inds': [0, 1]
        }]
        actual = dimension.hs_indices
        self.assertEqual(actual, expected)

    def test_inserted_hs_indices_single_subtotal(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS).dimensions[0]
        # It can be verified what the inserted indices are, by comparing
        # labels with/without transforms.
        expected = [2]
        actual = dimension.inserted_hs_indices
        assert actual == expected

    def test_subtotals_indices_two_subtotals(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS_MISSING).dimensions[0]
        expected = [{
            'anchor_ind': 1,
            'inds': [0, 1]
        }, {
            'anchor_ind': 'bottom',
            'inds': [3, 4, 5]
        }]
        actual = dimension.hs_indices
        self.assertEqual(actual, expected)

    def test_inserted_hs_indices_two_subtotals(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS_MISSING).dimensions[0]
        # It can be verified what the inserted indices are, by comparing
        # labels with/without transforms.
        expected = [2, 6]
        actual = dimension.inserted_hs_indices
        self.assertEqual(actual, expected)

    def test_has_transforms_false(self):
        dimension = CrunchCube(
            ECON_BLAME_X_IDEOLOGY_ROW_HS
        ).dimensions[1]
        expected = False
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_has_transforms_true(self):
        dimension = CrunchCube(
            ECON_BLAME_X_IDEOLOGY_ROW_HS
        ).dimensions[0]
        expected = True
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_hs_indices_with_bad_data(self):
        cube = CrunchCube(CA_WITH_NETS)
        expected = ['bottom', 'bottom']

        ca_dim = cube.dimensions[0]
        actual = [entry['anchor_ind'] for entry in ca_dim.hs_indices]
        assert actual == expected

        cat_dim = cube.dimensions[1]
        actual = [entry['anchor_ind'] for entry in cat_dim.hs_indices]
        assert actual == expected
