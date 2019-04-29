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

    def it_calculates_column_proportions(self):
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
