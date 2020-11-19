# encoding: utf-8

"""Integration-test suite for column/row index measure behaviors."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from cr.cube.cube import Cube

from ..fixtures import CR


def test_cat_x_cat_slice_column_index():
    slice_ = Cube(CR.CAT_X_CAT).partitions[0]
    expected = np.array([[100, 80], [100, 120]])
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_x_cat_slice_column_index():
    slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
    expected = np.array(
        [
            [95.8651525855387, 103.859044435659],
            [102.305106635277, 97.8432045727022],
            [97.6031146323114, 102.24274029149],
            [98.1029444304978, 101.829068203842],
            [114.466510106092, 86.0656684647625],
            [99.2925720053358, 100.682933745397],
        ]
    )
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_x_mr_slice_column_index():
    mr_x_mr = Cube(CR.FULL_CUBE).partitions[0]
    mr_alone = Cube(CR.MR_WGTD).partitions[0]
    expected = (
        mr_x_mr.column_proportions / mr_alone._table_proportions_as_array[:, None] * 100
    )
    np.testing.assert_almost_equal(mr_x_mr.column_index, expected)


def test_cat_x_mr_slice_column_index():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = np.array(
        [[90.9090909, 106.9518717, 95.6937799], [104.4776119, 96.5759438, 102.1209741]]
    )
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_x_mr_index_tables_parity_with_whaam_and_r():
    slice_ = Cube(CR.MR_X_MR_INDEX_TABLE).partitions[0]
    # Test column direction
    expected = np.array(
        [
            [192.05298013, 97.23165321, 89.68799602],
            [99.22588537, 239.38592924, 95.99096915],
            [93.52597694, 98.08689727, 182.31556654],
        ]
    )
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_x_3vl_index_tables_parity_with_nssat():
    slice_ = Cube(CR.NSSAT_MR_X_3vl).partitions[0]
    # Test column direction
    expected = np.array(
        [
            [179.793686976007, 90.4924459426829],
            [182.343952039497, 88.5838171105893],
            [84.6174957937067, 101.993855386627],
            [np.nan, np.nan],
        ]
    )
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_x_mr_index_tables_parity_with_nssat():
    slice_ = Cube(CR.NSSAT_MR_X_MR).partitions[0]

    # Test column direction
    expected = np.array(
        [
            [
                114.917891097666,
                94.6007480891202,
                75.7981149285497,
                41.5084915084915,
                64.5687645687646,
                581.118881118881,
                np.nan,
                0,
                np.nan,
            ],
            [
                90.0597657183839,
                95.9426026719446,
                102.497687326549,
                84.1945288753799,
                261.93853427896,
                0,
                np.nan,
                0,
                np.nan,
            ],
            [
                99.4879510762734,
                101.567130443518,
                101.446145177951,
                106.834310398025,
                86.4170866330693,
                59.8272138228942,
                np.nan,
                119.654427645788,
                np.nan,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    np.testing.assert_almost_equal(slice_.column_index, expected)


def test_mr_single_cat_x_mr():
    # No pruning
    slice_ = Cube(CR.MR_1_X_MR_3).partitions[0]
    expected = [[100, 100, np.nan]]
    np.testing.assert_array_equal(slice_.column_index, expected)

    # With pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_1_X_MR_3, transforms=transforms).partitions[0]
    expected = [[100, 100]]
    np.testing.assert_array_equal(slice_.column_index, expected)


def test_mr_x_mr_single_cat():
    slice_ = Cube(CR.MR_X_MR_SINGLE_CAT).partitions[0]
    expected = [[100], [100], [np.nan]]
    np.testing.assert_array_equal(slice_.column_index, expected)

    # With pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_MR_SINGLE_CAT, transforms=transforms).partitions[0]
    expected = [[100], [100]]
    np.testing.assert_array_equal(slice_.column_index, expected)
