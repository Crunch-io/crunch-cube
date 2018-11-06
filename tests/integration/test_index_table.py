'''This module contains tests for the correct index functionality.'''

# pylint: disable=missing-docstring, invalid-name

import numpy as np

from cr.cube.crunch_cube import CrunchCube

from ..fixtures import CR


def test_cat_x_cat_slice_column_index():
    cube = CrunchCube(CR.CAT_X_CAT)
    expected = np.array([
        [100, 80],
        [100, 120],
    ])
    actual = cube.slices[0].index_table(axis=0)
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_cat_slice_row_index():
    cube = CrunchCube(CR.CAT_X_CAT)
    expected = np.array([
        [107.142857142857, 85.7142857142857],
        [93.75, 112.5],
    ])
    actual = cube.slices[0].index_table(axis=1)
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_cat_slice_row_index_with_baseline():
    cube = CrunchCube(CR.CAT_X_CAT)
    expected = np.array([
        [119.047619047619, 71.4285714285714],
        [104.16666666666667, 93.75],
    ])
    actual = cube.slices[0].index_table(axis=1, baseline=np.array([0.6, 0.4]))
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_cat_slice_column_index_with_baseline():
    cube = CrunchCube(CR.CAT_X_CAT)
    expected = np.array([
        [83.3333333333333, 66.6666666666667],
        [125, 150],
    ])
    actual = cube.slices[0].index_table(axis=0, baseline=np.array([0.6, 0.4]))
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_cat_slice_column_index():
    cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
    expected = np.array([
        [95.8651525855387, 103.859044435659],
        [102.305106635277, 97.8432045727022],
        [97.6031146323114, 102.24274029149],
        [98.1029444304978, 101.829068203842],
        [114.466510106092, 86.0656684647625],
        [99.2925720053358, 100.682933745397],
    ])
    actual = cube.slices[0].index_table(axis=0)
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_cat_slice_row_index():
    cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
    expected = np.array([
        [95.3416155822363, 104.394053238208],
        [101.879419344372, 98.2272247381305],
        [97.1985863211465, 102.642452778304],
        [99.2098729346168, 100.745292805163],
        [115.700063998356, 85.1908063256891],
        [100.477252947149, 99.5498278652431],
    ])
    actual = cube.slices[0].index_table(axis=1)
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_mr_slice_column_index():
    mr_x_mr = CrunchCube(CR.FULL_CUBE)
    mr_alone = CrunchCube(CR.MR_WGTD)
    expected = (
        mr_x_mr.proportions(axis=0) /
        mr_alone.proportions()[:, None] * 100
    )
    actual = mr_x_mr.slices[0].index_table(axis=0)
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_mr_slice_row_index():
    mr_x_mr = CrunchCube(CR.FULL_CUBE)
    expected = np.array([
        96.5399786, 101.08725891, 89.22034634, 89.31908705,
        97.03625304, 99.96205366, 79.74421663, 94.32481007,
        98.95581177, 101.14422043, 95.50739913, 91.82091585,
        92.66856944, 102.10571127, 94.88279498, 84.22596655,
        92.62629722, 86.77934972, 99.31115914, 98.72846269,
        99.54678433, 94.13302782, 101.99733805, 102.24392708,
        97.87112979, 95.08750269, 100.61288629,
    ])
    actual = mr_x_mr.slices[0].index_table(axis=1)[0]
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_mr_slice_column_index():
    cat_x_mr = CrunchCube(CR.CAT_X_MR)
    expected = np.array([
        [90.9090909, 106.9518717, 95.6937799],
        [104.4776119, 96.5759438, 102.1209741],
    ])
    actual = cat_x_mr.slices[0].index_table(axis=0)
    np.testing.assert_almost_equal(actual, expected)


def test_cat_x_mr_slice_row_index():
    cat_x_mr = CrunchCube(CR.CAT_X_MR)
    expected = np.array([
        [85.7142857, 111.5294118, 96.1098398],
        [107.6923077, 94.6623094, 101.9036954],
    ])
    actual = cat_x_mr.slices[0].index_table(axis=1)
    np.testing.assert_almost_equal(actual, expected)


def test_mr_x_mr_index_tables_parity_with_whaam_and_r():
    cat_x_mr = CrunchCube(CR.MR_X_MR_INDEX_TABLE)
    # Test column direction
    expected = np.array([
        [192.05298013, 97.23165321, 89.68799602],
        [99.22588537, 239.38592924, 95.99096915],
        [93.52597694, 98.08689727, 182.31556654],
    ])
    actual = cat_x_mr.slices[0].index_table(axis=0)
    np.testing.assert_almost_equal(actual, expected)
    # Test row direction
    expected = np.array([
        [192.0529801, 99.2258854, 93.5259769],
        [97.2316532, 239.3859292, 98.0868973],
        [89.687996, 95.9909692, 182.3155665],
    ])
    actual = cat_x_mr.slices[0].index_table(axis=1)
    np.testing.assert_almost_equal(actual, expected)
