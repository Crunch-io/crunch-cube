# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from cr.cube.cube import Cube

from ..fixtures import NA

NAN = np.nan


# pylint: disable=no-self-use
class TestNumericArrays:
    """Test-suite for numeric-arrays behaviors."""

    def test_means_without_grouping_by_dimensions(self):
        """Test means on no-dimensions measure of numeric array."""
        slice_ = Cube(NA.NUM_ARR_MEANS_NO_DIMS).partitions[0]
        np.testing.assert_equal(slice_.means, [2.5, 25])

    def test_means_with_grouping_by_single_categorical_dimension(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.MEANS_GROUPED_BY_SINGLE_CATEGORICAL).partitions[0]
        np.testing.assert_equal(
            slice_.means,
            [
                # subvars:
                # 0001, 0002
                [0, 0],  # cats: -- red
                [1, 10],  # ------- green
                [2, 20],  # ------- blue
                [3, 30],  # ------- 4
                [5, 50],  # ------- 9
            ],
        )
