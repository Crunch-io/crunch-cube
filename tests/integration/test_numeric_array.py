# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import NA


class TestNumericArrays:
    """Test-suite for numeric-array behaviors."""

    @pytest.mark.parametrize(
        "cube_idx, expected_means, expected_col_base",
        (
            (
                None,
                [
                    #  --------Movies------------
                    #    S1           S2      S3
                    [87.6666667, 93.3333333, 1.0],  # Gender: Male
                    [52.5000000, 50.000000, 45.0],  # Gender: Female
                ],
                [5, 4, 2],
            ),
            (
                1,
                [  # -------Gender-----
                    #     M        F
                    [87.6666667, 52.5],  # S1: Dark Night
                    [93.3333333, 50.0],  # S2: Fight Club
                    [1.00000000, 45.0],  # S3: Meet the parents
                ],
                [[3, 2], [3, 1], [1, 1]],
            ),
        ),
    )
    def test_num_arr_grouped_by_cat(self, cube_idx, expected_means, expected_col_base):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT, cube_idx=cube_idx).partitions[0]

        np.testing.assert_almost_equal(slice_.means, expected_means)
        np.testing.assert_almost_equal(slice_.columns_base, expected_col_base)

    @pytest.mark.parametrize(
        "cube_idx, expected_means, expected_col_base",
        (
            (
                None,
                [
                    #  --------Movies------------
                    # S1    S2       S3
                    [50.0, 90.0, 46.6666667],  # Wave: 2014-12
                    [100.0, 81.0, 80.0],  # Wave: 2015-01
                ],
                [19, 18, 19],
            ),
            (
                1,
                [  # -------Wave-----
                    # 2014-12   2015-01
                    [50.000000, 100.0000],  # S1: Mov A
                    [90.000000, 81.00000],  # S2: Mov B
                    [46.6666667, 80.0000],  # S3: Mov C
                ],
                [[10, 9], [8, 10], [9, 10]],
            ),
        ),
    )
    def test_num_arr_grouped_by_date(self, cube_idx, expected_means, expected_col_base):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_DATE, cube_idx=cube_idx).partitions[0]

        np.testing.assert_almost_equal(slice_.means, expected_means)
        np.testing.assert_almost_equal(slice_.columns_base, expected_col_base)

    def test_num_arr_grouped_by_cat_weighted(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_WEIGHTED).partitions[0]

        np.testing.assert_almost_equal(
            slice_.means,
            [
                #  Movies:
                #  S1, S2, S3
                [91.625, 94.375, 1.0],  # Gender: Male
                [50.71428571, 50.0, 45.0],  # Gender: Female
            ],
        )
        np.testing.assert_almost_equal(slice_.columns_base, [5, 4, 2])

    @pytest.mark.parametrize(
        "cube_idx, expected_means, expected_col_base",
        (
            (
                None,
                [
                    # ----Num array---
                    # S1    S2    S3
                    [62.5, 90.0, 42.5],  # MR S1: Cat
                    [87.5, 81.0, 70.0],  # MR S2: Dog
                    [87.5, 81.0, 80.0],  # MR S3: Bird
                ],
                [[4, 3, 4], [4, 5, 5], [4, 5, 5]],
            ),
            (
                1,
                [
                    # -------MR-------
                    # S1    S2    S3
                    [62.5, 87.5, 87.5],  # S1 (num array)
                    [90.0, 81.0, 81.0],  # S2 (num array)
                    [42.5, 70.0, 80.0],  # S3 (num array)
                ],
                [[4, 4, 4], [3, 5, 5], [4, 5, 5]],
            ),
        ),
    )
    def test_num_arr_x_mr(self, cube_idx, expected_means, expected_col_base):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR, cube_idx=cube_idx).partitions[0]

        np.testing.assert_almost_equal(slice_.means, expected_means)
        # ---The columns_base is 2D because a NUM_ARR_X_MR matrix has a distinct
        # ---column-base for each cell.
        np.testing.assert_almost_equal(slice_.columns_base, expected_col_base)

    def test_num_arr_means_no_grouping(self):
        """Test means on no-dimensions measure of numeric array."""
        strand_ = Cube(NA.NUM_ARR_MEANS_NO_GROUPING).partitions[0]

        np.testing.assert_almost_equal(strand_.means, (2.5, 25.0))
        np.testing.assert_almost_equal(strand_.table_base, [6, 6])
