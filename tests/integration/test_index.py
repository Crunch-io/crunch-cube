# pylint: disable=missing-docstring, no-self-use, invalid-name

from unittest import TestCase
import numpy as np

from cr.cube.crunch_cube import CrunchCube

from .fixtures import CAT_X_CAT
from .fixtures import CAT_X_MR_SIMPLE
from .fixtures import PETS_X_PETS
from .fixtures import SELECTED_CROSSTAB_4
from .fixtures import CAT_X_MR_X_MR
from .fixtures import CA_ITEMS_X_CA_CAT_X_CAT
from .fixtures import FULL_CUBE as MR_X_MR
from .fixtures import NATREP as MR_ALONE


class TestIndex(TestCase):
    def test_mr_x_cat_index(self):
        cube = CrunchCube(SELECTED_CROSSTAB_4)
        expected = np.array([
            [0.95865152585539, 1.0385904443566],
            [1.02305106635277, 0.9784320457270],
            [0.97603114632311, 1.0224274029149],
            [0.98102944430498, 1.0182906820384],
            [1.14466510106092, 0.8606566846476],
            [0.99292572005336, 1.0068293374540]
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [0.8571429, 1.1152941, 0.9610984],
            [1.0769231, 0.9466231, 1.019037],
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_mr_index(self):
        cube = CrunchCube(PETS_X_PETS)
        expected = np.array([
            [1.0000000000000000, 1.1724137931034484, 0.894736842105263],
            [0.8529411764705883, 1.0000000000000000, 0.763157894736842],
            [1.1176470588235294, 1.310344827586207, 1.0000000000000000]
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_mr_x_mr_index(self):
        self.skipTest('not yet implemented')
        cube = CrunchCube(CAT_X_MR_X_MR)
        expected = np.array([])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_items_x_ca_cat_x_cat_index(self):
        cube = CrunchCube(CA_ITEMS_X_CA_CAT_X_CAT)

        # Test index values
        expected = np.array([
            [[1., 1., np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan]],

            [[2., 0., np.nan, np.nan, np.nan],
             [0., 2., np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan]],

            [[np.nan, np.nan, np.nan, np.nan, np.nan],
             [2., 0., np.nan, np.nan, np.nan],
             [0., 2., np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan]],
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

        # Test pruning mask
        expected = np.array([
            [[False, False, True, True, True],
             [True, True, True, True, True],
             [True, True, True, True, True],
             [True, True, True, True, True]],

            [[False, False, True, True, True],
             [False, False, True, True, True],
             [True, True, True, True, True],
             [True, True, True, True, True]],

            [[True, True, True, True, True],
             [False, False, True, True, True],
             [False, False, True, True, True],
             [True, True, True, True, True]],
        ])
        actual = cube.index(prune=True).mask
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_cat_slice_column_index(self):
        cube = CrunchCube(CAT_X_CAT)
        expected = np.array([
            [100, 80],
            [100, 120],
        ])
        actual = cube.slices[0].index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_slice_row_index(self):
        cube = CrunchCube(CAT_X_CAT)
        expected = np.array([
            [107.142857142857, 85.7142857142857],
            [93.75, 112.5],
        ])
        actual = cube.slices[0].index(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_slice_row_index_with_baseline(self):
        cube = CrunchCube(CAT_X_CAT)
        expected = np.array([
            [119.047619047619, 71.4285714285714],
            [104.16666666666667, 93.75],
        ])
        actual = cube.slices[0].index(axis=1, base=np.array([0.6, 0.4]))
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_slice_column_index_with_baseline(self):
        cube = CrunchCube(CAT_X_CAT)
        expected = np.array([
            [83.3333333333333, 66.6666666666667],
            [125, 150],
        ])
        actual = cube.slices[0].index(axis=0, base=np.array([0.6, 0.4]))
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_cat_slice_row_index(self):
        cube = CrunchCube(SELECTED_CROSSTAB_4)
        expected = np.array([
            [95.3416155822363, 104.394053238208],
            [101.879419344372, 98.2272247381305],
            [97.1985863211465, 102.642452778304],
            [99.2098729346168, 100.745292805163],
            [115.700063998356, 85.1908063256891],
            [100.477252947149, 99.5498278652431],
        ])
        actual = cube.slices[0].index(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_cat_slice_column_index(self):
        cube = CrunchCube(SELECTED_CROSSTAB_4)
        expected = np.array([
            [95.8651525855387, 103.859044435659],
            [102.305106635277, 97.8432045727022],
            [97.6031146323114, 102.24274029149],
            [98.1029444304978, 101.829068203842],
            [114.466510106092, 86.0656684647625],
            [99.2925720053358, 100.682933745397],
        ])
        actual = cube.slices[0].index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_mr_slice_column_index(self):
        mr_x_mr = CrunchCube(MR_X_MR)
        mr_alone = CrunchCube(MR_ALONE)
        expected = (
            mr_x_mr.proportions(axis=0) /
            mr_alone.proportions()[:, None] * 100
        )
        actual = mr_x_mr.slices[0].index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_mr_slice_row_index(self):
        mr_x_mr = CrunchCube(MR_X_MR)
        expected = np.array([
            96.5399786, 101.08725891, 89.22034634, 89.31908705,
            97.03625304, 99.96205366, 79.74421663, 94.32481007,
            98.95581177, 101.14422043, 95.50739913, 91.82091585,
            92.66856944, 102.10571127, 94.88279498, 84.22596655,
            92.62629722, 86.77934972, 99.31115914, 98.72846269,
            99.54678433, 94.13302782, 101.99733805, 102.24392708,
            97.87112979, 95.08750269, 100.61288629,
        ])
        actual = mr_x_mr.slices[0].index(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_slice_col_index(self):
        cat_x_mr = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [90.9090909, 106.9518717, 95.6937799],
            [104.4776119, 96.5759438, 102.1209741],
        ])
        actual = cat_x_mr.slices[0].index(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_slice_row_index(self):
        cat_x_mr = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [85.7142857, 111.5294118, 96.1098398],
            [107.6923077, 94.6623094, 101.9036954],
        ])
        actual = cat_x_mr.slices[0].index(axis=1)
        np.testing.assert_almost_equal(actual, expected)
