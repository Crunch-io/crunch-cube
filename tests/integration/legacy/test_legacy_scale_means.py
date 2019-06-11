# encoding: utf-8

import numpy as np
import pytest

from cr.cube.crunch_cube import CrunchCube

from ...fixtures import CR, SM  # ---mnemonic: SM = 'scale means'---

from . import assert_scale_means_equal


def test_ca_cat_x_items():
    cube = CrunchCube(SM.CA_CAT_X_ITEMS)
    expected = [[np.array([1.50454821, 3.11233766, 3.35788192, 3.33271833]), None]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(0)
    assert actual is None
    with pytest.raises(ValueError):
        # Items dimension doesn't have means
        actual = slice_.scale_means_margin(1)


def test_ca_items_x_cat():
    cube = CrunchCube(SM.CA_ITEMS_X_CAT)
    expected = [[None, np.array([1.50454821, 3.11233766, 3.35788192, 3.33271833])]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    with pytest.raises(ValueError):
        # Items dimension doesn't have means
        actual = slice_.scale_means_margin(0)
    actual = slice_.scale_means_margin(1)
    assert actual is None


def test_ca_x_mr():
    cube = CrunchCube(SM.CA_X_MR)
    expected = [
        [np.array([1.29787234, 1.8, 1.48730964, np.nan]), None],
        [np.array([3.31746032, 3.10743802, 3.09976976, np.nan]), None],
        [np.array([3.31205674, 3.23913043, 3.37745455, np.nan]), None],
        [np.array([3.53676471, 3.34814815, 3.3147877, np.nan]), None],
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(axis=0)
    assert actual is None
    actual = slice_.scale_means_margin(axis=1)
    assert actual == 1.504548211036992


def test_cat_x_ca_cat_x_items():
    cube = CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS)
    expected = [
        [np.array([1.34545455, 2.46938776, 2.7037037, 2.65454545]), None],
        [np.array([1.41935484, 3.25663717, 3.48, 3.58536585]), None],
        [np.array([1.49429038, 3.44905009, 3.59344262, 3.53630363]), None],
        [np.array([1.43365696, 3.02816901, 3.37987013, 3.32107023]), None],
        [np.array([1.22670025, 2.49473684, 2.79848866, 2.78987342]), None],
        [np.array([2.53061224, 3.68421053, 3.9862069, 4.03472222]), None],
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    actual = cube.scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])
    actual = slice_.scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(axis=0)
    assert actual is None
    with pytest.raises(ValueError):
        slice_.scale_means_margin(axis=1)


def test_cat_x_cat():
    cube = CrunchCube(SM.CAT_X_CAT)
    expected = [
        [
            np.array([2.6009281, 2.3522267, 2.3197279, 3.3949192]),
            np.array(
                [1.43636364, 2.45238095, 2.4730832, 2.68387097, 2.8375, 2.15540541]
            ),
        ]
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(0)
    assert actual == 2.536319612590799
    actual = slice_.scale_means_margin(1)
    assert actual == 2.6846246973365617


def test_cat_x_mr():
    cube = CrunchCube(SM.CAT_X_MR)
    expected = [[np.array([2.45070423, 2.54471545, 2.54263006, np.nan]), None]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(0)
    assert actual is None
    actual = slice_.scale_means_margin(1)
    assert actual == 2.5323565323565322


def test_mr_x_cat():
    cube = CrunchCube(SM.MR_X_CAT)
    expected = [[None, np.array([2.45070423, 2.54471545, 2.54263006, np.nan])]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    actual = slice_.scale_means_margin(0)
    assert actual == 2.5323565323565322
    actual = slice_.scale_means_margin(1)
    assert actual is None


def test_univariate_cat():
    cube = CrunchCube(SM.UNIVARIATE_CAT)
    expected = [[np.array([2.6865854])]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test ScaleMeans marginal
    with pytest.raises(ValueError):
        actual = slice_.scale_means_margin(0)


def test_cat_x_cat_with_hs():
    cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)

    # Test without H&S
    expected = [
        [
            np.array(
                [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85]
            ),
            np.array([3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294]),
        ]
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    slice_ = cube.slices[0]
    actual = slice_.scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test with H&S
    expected = [
        [
            np.array(
                [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85]
            ),
            np.array(
                [3.87368421, 2.51767677, np.nan, 3.38429752, 3.66666667, 4.13235294]
            ),
        ]
    ]
    actual = cube.scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    actual = cube.slices[0].scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected[0])


def test_univariate_with_hs():
    cube = CrunchCube(CR.ECON_BLAME_WITH_HS)

    # Test without H&S
    expected = [[np.array([2.17352056])]]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    actual = cube.slices[0].scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test with H&S
    expected = [[np.array([2.17352056])]]
    actual = cube.scale_means(hs_dims=[0])
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    actual = cube.slices[0].scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected[0])


def test_cat_x_cat_with_hs_on_both_dims():
    cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)

    # Test without H&S
    expected = [
        [
            np.array(
                [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85]
            ),
            np.array([3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294]),
        ]
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    actual = cube.slices[0].scale_means()
    assert_scale_means_equal(actual, expected[0])

    # Test with H&S
    expected = [
        [
            np.array(
                [
                    2.19444444,
                    2.19230769,
                    2.26666667,
                    np.nan,
                    1.88990826,
                    1.76363636,
                    3.85,
                ]
            ),
            np.array(
                [3.87368421, 2.51767677, np.nan, 3.38429752, 3.66666667, 4.13235294]
            ),
        ]
    ]
    actual = cube.scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected)

    # Test for cube slices
    actual = cube.slices[0].scale_means(hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected[0])


def test_ca_x_mr_with_hs_and_pruning():
    cube = CrunchCube(CR.CA_X_MR_HS)
    expected = [
        [np.array([2.50818336, 2.56844883, 2.90251939, np.nan]), None],
        [np.array([2.78385708, 2.69292009, 3.11594714, np.nan]), None],
        [np.array([np.nan, np.nan, np.nan, np.nan]), None],
    ]
    actual = cube.scale_means()
    assert_scale_means_equal(actual, expected)
    expected = [
        [np.array([2.50818336, 2.56844883, 2.90251939]), None],
        [np.array([2.78385708, 2.69292009, 3.11594714]), None],
        [np.array([]), None],
    ]
    actual = cube.scale_means(prune=True)
    assert_scale_means_equal(actual, expected)
    actual = cube.scale_means(prune=True, hs_dims=[0, 1])
    assert_scale_means_equal(actual, expected)


def test_cat_x_cat_pruning_and_hs():
    cube = CrunchCube(CR.CAT_X_CAT_PRUNING_HS)
    expected = [
        [
            np.array([1.57933884, 2.10618401, 2.30460074, np.nan, 2.34680135]),
            np.array([1.74213625, 1.97, 2.45356177, 2.11838791, np.nan, 2.0]),
        ]
    ]
    actual = cube.scale_means()

    # Just H&S
    assert_scale_means_equal(actual, expected)
    expected = [
        [
            np.array([1.57933884, np.nan, 2.10618401, 2.30460074, np.nan, 2.34680135]),
            np.array([1.74213625, np.nan, 1.97, 2.45356177, 2.11838791, np.nan, 2.0]),
        ]
    ]
    actual = cube.scale_means(hs_dims=[0, 1])

    # Just pruning
    assert_scale_means_equal(actual, expected)
    expected = [
        [
            np.array([1.57933884, 2.10618401, 2.30460074, 2.34680135]),
            np.array([1.74213625, 1.97, 2.45356177, 2.11838791, 2.0]),
        ]
    ]
    actual = cube.scale_means(prune=True)
    assert_scale_means_equal(actual, expected)

    # Pruning and H&S
    assert_scale_means_equal(actual, expected)
    expected = [
        [
            np.array([1.57933884, np.nan, 2.10618401, 2.30460074, 2.34680135]),
            np.array([1.74213625, np.nan, 1.97, 2.45356177, 2.11838791, 2.0]),
        ]
    ]
    actual = cube.scale_means(hs_dims=[0, 1], prune=True)
    assert_scale_means_equal(actual, expected)


def test_cat_x_cat_scale_means_margin():
    cs = CrunchCube(SM.CAT_X_CAT_SM_MARGIN).slices[0]
    expected = 2.6846246973365617
    assert cs.scale_means_margin(1) == expected

    expected = 2.536319612590799
    assert cs.scale_means_margin(0) == expected


def test_cat_single_element_x_cat():
    cs = CrunchCube(SM.CAT_SINGLE_ELEMENT_X_CAT).slices[0]
    scale_means = cs.scale_means()
    np.testing.assert_equal(scale_means[0], np.array([np.nan, np.nan, np.nan, np.nan]))
    np.testing.assert_equal(scale_means[1], np.array([np.nan]))
