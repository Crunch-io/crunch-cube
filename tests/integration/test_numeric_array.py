# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from cr.cube.cube import Cube

from ..fixtures import NA


# pylint: disable=no-self-use
class TestNumericArrays:
    """Test-suite for numeric-arrays behaviors."""

    def test_num_arr_grouped_by_cat(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT).partitions[0]

        np.testing.assert_almost_equal(
            slice_.means,
            [
                #  Movies:
                #  S1, S2, S3
                [87.66666667, 93.33333333, 1.0],  # Gender: Male
                [52.5, 50.0, 45.0],  # Gender: Female
            ],
        )
        np.testing.assert_almost_equal(slice_.column_base, [5, 4, 2])

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
        np.testing.assert_almost_equal(slice_.column_base, [5, 4, 2])

    def test_num_arr_means_no_grouping(self):
        """Test means on no-dimensions measure of numeric array."""
        strand_ = Cube(NA.NUM_ARR_MEANS_NO_GROUPING).partitions[0]

        np.testing.assert_almost_equal(strand_.means, (2.5, 25.0))
