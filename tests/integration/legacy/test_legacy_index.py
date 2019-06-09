# encoding: utf-8

import numpy as np

from unittest import TestCase

from cr.cube.crunch_cube import CrunchCube
from cr.cube.measures.index import Index

from ...fixtures import CR


class TestIndex(TestCase):
    def test_mr_x_cat_index(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [0.95865152585539, 1.0385904443566],
                [1.02305106635277, 0.9784320457270],
                [0.97603114632311, 1.0224274029149],
                [0.98102944430498, 1.0182906820384],
                [1.14466510106092, 0.8606566846476],
                [0.99292572005336, 1.0068293374540],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index(self):
        cube = CrunchCube(CR.CAT_X_MR)
        expected = np.array(
            [[0.8571429, 1.1152941, 0.9610984], [1.0769231, 0.9466231, 1.019037]]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_mr_index(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        expected = np.array(
            [
                [1.0000000000000000, 1.1724137931034484, 0.894736842105263],
                [0.8529411764705883, 1.0000000000000000, 0.763157894736842],
                [1.1176470588235294, 1.310344827586207, 1.0000000000000000],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_items_x_ca_cat_x_cat_index(self):
        cube = CrunchCube(CR.CA_ITEMS_X_CA_CAT_X_CAT)

        # Test index values
        expected = np.array(
            [
                [
                    [1.0, 1.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [2.0, 0.0, np.nan, np.nan, np.nan],
                    [0.0, 2.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [2.0, 0.0, np.nan, np.nan, np.nan],
                    [0.0, 2.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

        # Test pruning mask
        expected = np.array(
            [
                [
                    [False, False, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ],
                [
                    [False, False, True, True, True],
                    [False, False, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ],
                [
                    [True, True, True, True, True],
                    [False, False, True, True, True],
                    [False, False, True, True, True],
                    [True, True, True, True, True],
                ],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=True).mask
        np.testing.assert_array_equal(actual, expected)

    def test_econ_x_ideology_index_by_col(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [0.91861761, 0.96399471, 0.9394101, 1.06629132, 1.30027051, 0.67660435],
                [1.08138239, 1.03600529, 1.0605899, 0.93370868, 0.69972949, 1.32339565],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_econ_x_ideology_index_by_row(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [0.91861761, 0.96399471, 0.9394101, 1.06629132, 1.30027051, 0.67660435],
                [1.08138239, 1.03600529, 1.0605899, 0.93370868, 0.69972949, 1.32339565],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_index_by_col_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [2.8, 0.7, np.nan, 0.7],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.58333333, np.nan, 2.33333333],
                [0.0, 1.75, np.nan, 0.0],
                [0.0, 1.16666667, np.nan, 1.16666667],
                [0.0, 1.75, np.nan, 0.0],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        # Assert index without pruning
        np.testing.assert_almost_equal(actual, expected)

        expected = np.array(
            [
                [2.8, 0.7, 0.7],
                [0.0, 0.58333333, 2.33333333],
                [0.0, 1.75, 0.0],
                [0.0, 1.16666667, 1.16666667],
                [0.0, 1.75, 0.0],
            ]
        )
        table = Index.data(cube, weighted=True, prune=True)
        # Assert index witih pruning
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_almost_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(actual, pruned_expected[i])
