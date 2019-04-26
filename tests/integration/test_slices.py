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

    def it_calculates_row_percentages_with_and_without_hs(self):
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
                [0.47457627, 0.33898305, 0.16949153, 0.0, 0.01694915],
                [0.24137931, 0.4137931, 0.27586207, 0.0, 0.06896552],
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.27272727, 0.36363636, 0.18181818, 0.0, 0.18181818],
                [0.1875, 0.5, 0.3125, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
