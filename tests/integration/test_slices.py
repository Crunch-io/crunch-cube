# encoding: utf-8

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.slices import FrozenSlice

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---


class DescribeFrozenSlice:
    def it_loads_from_cat_x_cat_cube(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        slice_ = FrozenSlice(cube)
        expected = np.array([[0.71428571, 0.28571429], [0.625, 0.375]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_row_proportions_with_and_without_hs(self):
        cube = CrunchCube(CR.CAT_X_CAT_PRUNING_HS)

        # Without insertions
        slice_ = FrozenSlice(cube, use_insertions=False)
        expected = np.array(
            [
                [0.47457627, 0.33898305, 0.16949153, 0.0, 0.01694915],
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.27272727, 0.36363636, 0.18181818, 0.0, 0.18181818],
                [0.1875, 0.5, 0.3125, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

        # With insertions (only row for now)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [0.47457627, 0.81355932, 0.33898305, 0.16949153, 0.0, 0.01694915],
                [0.24137931, 0.65517241, 0.4137931, 0.27586207, 0.0, 0.06896552],
                [0.5, 0.5, 0.0, 0.5, 0.0, 0.0],
                [0.27272727, 0.63636364, 0.36363636, 0.18181818, 0.0, 0.18181818],
                [0.1875, 0.6875, 0.5, 0.3125, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.33333333, 0.66666667, 0.33333333, 0.33333333, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_cat_x_cat_column_proportions(self):
        cube = CrunchCube(CR.CAT_X_CAT_PRUNING_HS)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [0.77777778, 0.69565217, 0.60606061, 0.52631579, np.nan, 0.33333333],
                [0.19444444, 0.27536232, 0.36363636, 0.42105263, np.nan, 0.66666667],
                [0.02777778, 0.01449275, 0.0, 0.05263158, np.nan, 0.0],
                [0.08333333, 0.10144928, 0.12121212, 0.10526316, np.nan, 0.66666667],
                [0.08333333, 0.15942029, 0.24242424, 0.26315789, np.nan, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
                [0.02777778, 0.02898551, 0.03030303, 0.05263158, np.nan, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_calculates_mr_x_cat_row_proportions(self):
        cube = CrunchCube(CR.MR_X_CAT_HS)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [
                    0.30769231,
                    0.26923077,
                    0.57692308,
                    0.0,
                    0.23076923,
                    0.19230769,
                    0.0,
                    0.42307692,
                ],
                [
                    0.09210526,
                    0.21052632,
                    0.30263158,
                    0.0,
                    0.34210526,
                    0.35526316,
                    0.0,
                    0.69736842,
                ],
                [
                    0.03389831,
                    0.1779661,
                    0.21186441,
                    0.0,
                    0.33050847,
                    0.45762712,
                    0.0,
                    0.78813559,
                ],
                [
                    0.03523035,
                    0.09756098,
                    0.13279133,
                    0.0,
                    0.35230352,
                    0.51490515,
                    0.0,
                    0.86720867,
                ],
                [
                    0.07012987,
                    0.15064935,
                    0.22077922,
                    0.0,
                    0.34805195,
                    0.43116883,
                    0.0,
                    0.77922078,
                ],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_mr_x_cat_column_proportions(self):
        cube = CrunchCube(CR.MR_X_CAT_HS)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [
                    0.53333333,
                    0.29166667,
                    0.38461538,
                    np.nan,
                    0.10526316,
                    0.07246377,
                    np.nan,
                    0.08730159,
                ],
                [
                    0.46666667,
                    0.47058824,
                    0.46938776,
                    np.nan,
                    0.34666667,
                    0.31395349,
                    np.nan,
                    0.32919255,
                ],
                [
                    0.30769231,
                    0.56756757,
                    0.5,
                    np.nan,
                    0.48148148,
                    0.48648649,
                    np.nan,
                    0.484375,
                ],
                [0.65, 0.72, 0.7, np.nan, 0.81761006, 0.85972851, np.nan, 0.84210526],
                [
                    0.84375,
                    0.84057971,
                    0.84158416,
                    np.nan,
                    0.80239521,
                    0.79807692,
                    np.nan,
                    0.8,
                ],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_calculates_cat_x_mr_row_proportions(self):
        cube = CrunchCube(CR.CAT_X_MR_HS)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [0.53333333, 0.46666667, 0.30769231, 0.65, 0.84375],
                [0.29166667, 0.47058824, 0.56756757, 0.72, 0.84057971],
                [0.38461538, 0.46938776, 0.5, 0.7, 0.84158416],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.10526316, 0.34666667, 0.48148148, 0.81761006, 0.80239521],
                [0.07246377, 0.31395349, 0.48648649, 0.85972851, 0.79807692],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.08730159, 0.32919255, 0.484375, 0.84210526, 0.8],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_cat_x_mr_column_proportions(self):
        cube = CrunchCube(CR.CAT_X_MR_HS)
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [0.30769231, 0.09210526, 0.03389831, 0.03523035, 0.07012987],
                [0.26923077, 0.21052632, 0.1779661, 0.09756098, 0.15064935],
                [0.57692308, 0.30263158, 0.21186441, 0.13279133, 0.22077922],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.23076923, 0.34210526, 0.33050847, 0.35230352, 0.34805195],
                [0.19230769, 0.35526316, 0.45762712, 0.51490515, 0.43116883],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.42307692, 0.69736842, 0.78813559, 0.86720867, 0.77922078],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_calculates_mr_x_mr_row_proportions(self):
        cube = CrunchCube(CR.MR_X_MR)
        slice_ = FrozenSlice(cube)
        expected = np.array(
            [
                [1.0, 0.42857143, 0.3, 1.0],
                [0.16666667, 1.0, 0.36363636, 1.0],
                [0.11538462, 0.4, 1.0, 1.0],
                [0.27272727, 0.64444444, 0.64150943, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_mr_x_mr_column_proportions(self):
        cube = CrunchCube(CR.MR_X_MR)
        slice_ = FrozenSlice(cube)
        expected = np.array(
            [
                [1.0, 0.16666667, 0.11538462, 0.27272727],
                [0.42857143, 1.0, 0.4, 0.64444444],
                [0.3, 0.36363636, 1.0, 0.64150943],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_reorders_cat_x_cat(self):
        cube = CrunchCube(CR.CAT_X_CAT_PRUNING_HS)
        reordered_ids = ([6, 1, 2, 5, 4, 3], [5, 1, 2, 4, 3])
        slice_ = FrozenSlice(cube, reordered_ids=reordered_ids)
        expected = np.array(
            [
                [0, 1, 1, 0, 1],
                [1, 28, 20, 0, 10],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 3, 8, 0, 5],
                [2, 3, 4, 0, 2],
            ]
        )
        np.testing.assert_equal(slice_.counts, expected)

    def it_prunes_cat_x_cat_with_hs(self):
        cube = CrunchCube(CR.CAT_X_CAT_PRUNING_HS)

        # Pruned - without insertions
        slice_ = FrozenSlice(cube, pruning=True)
        expected = np.array(
            [[28, 20, 10, 1], [1, 0, 1, 0], [3, 4, 2, 2], [3, 8, 5, 0], [1, 1, 1, 0]]
        )
        np.testing.assert_equal(slice_.counts, expected)

        # Pruned - with insertions
        slice_ = FrozenSlice(cube, use_insertions=True, pruning=True)
        expected = np.array(
            [
                [28, 48, 20, 10, 1],
                [7, 19, 12, 8, 2],
                [1, 1, 0, 1, 0],
                [3, 7, 4, 2, 2],
                [3, 11, 8, 5, 0],
                [1, 2, 1, 1, 0],
            ]
        )
        np.testing.assert_equal(slice_.counts, expected)

        # Not pruned - with insertions
        slice_ = FrozenSlice(cube, use_insertions=True)
        expected = np.array(
            [
                [28, 48, 20, 10, 0, 1],
                [7, 19, 12, 8, 0, 2],
                [1, 1, 0, 1, 0, 0],
                [3, 7, 4, 2, 0, 2],
                [3, 11, 8, 5, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 2, 1, 1, 0, 0],
            ]
        )
        np.testing.assert_equal(slice_.counts, expected)