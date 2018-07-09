from unittest import TestCase
import numpy as np

from cr.cube.crunch_cube import CrunchCube

from .fixtures.cubes.scale_means import CA_CAT_X_ITEMS
from .fixtures.cubes.scale_means import CA_ITEMS_X_CAT
from .fixtures.cubes.scale_means import CA_X_MR
from .fixtures.cubes.scale_means import CAT_X_CA_CAT_X_ITEMS
from .fixtures.cubes.scale_means import CAT_X_CAT
from .fixtures.cubes.scale_means import CAT_X_MR
from .fixtures.cubes.scale_means import MR_X_CAT
from .fixtures.cubes.scale_means import UNIVARIATE_CAT


def test_ca_cat_x_items():
    cube = CrunchCube(CA_CAT_X_ITEMS)
    expected = np.array([1.50454821, 3.11233766, 3.35788192, 3.33271833])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_ca_items_x_cat():
    cube = CrunchCube(CA_ITEMS_X_CAT)
    expected = np.array([1.50454821, 3.11233766, 3.35788192, 3.33271833])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_ca_x_mr():
    cube = CrunchCube(CA_X_MR)
    expected = np.array([
        [1.29787234, 1.8       , 1.48730964, np.nan],  # noqa
        [3.31746032, 3.10743802, 3.09976976, np.nan],
        [3.31205674, 3.23913043, 3.37745455, np.nan],
        [3.53676471, 3.34814815, 3.3147877 , np.nan],  # noqa
    ])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_ca_cat_x_items():
    cube = CrunchCube(CAT_X_CA_CAT_X_ITEMS)
    expected = np.array([
        [1.34545455, 2.46938776, 2.7037037 , 2.65454545],  # noqa
        [1.41935484, 3.25663717, 3.48      , 3.58536585],  # noqa
        [1.49429038, 3.44905009, 3.59344262, 3.53630363],
        [1.43365696, 3.02816901, 3.37987013, 3.32107023],
        [1.22670025, 2.49473684, 2.79848866, 2.78987342],
        [2.53061224, 3.68421053, 3.9862069 , 4.03472222],  # noqa
    ])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_cat():
    cube = CrunchCube(CAT_X_CAT)
    expected = np.array([2.61411765, 2.34285714, 2.33793103, 3.38461538])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_mr():
    cube = CrunchCube(CAT_X_MR)
    expected = np.array([2.45070423, 2.54471545, 2.54263006, np.nan])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_cat():
    cube = CrunchCube(MR_X_CAT)
    expected = np.array([2.45070423, 2.54471545, 2.54263006, np.nan])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)


def test_univariate_cat():
    cube = CrunchCube(UNIVARIATE_CAT)
    expected = np.array([2.6865854])
    actual = cube.scale_means()
    np.testing.assert_almost_equal(actual, expected)
