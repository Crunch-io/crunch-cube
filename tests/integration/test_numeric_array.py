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
                [7, 4],
            ),
        ),
    )
    def test_num_arr_grouped_by_cat(self, cube_idx, expected_means, expected_col_base):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT, cube_idx=cube_idx).partitions[0]

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

    def test_num_arr_means_no_grouping(self):
        """Test means on no-dimensions measure of numeric array."""
        strand_ = Cube(NA.NUM_ARR_MEANS_NO_GROUPING).partitions[0]

        np.testing.assert_almost_equal(strand_.means, (2.5, 25.0))
