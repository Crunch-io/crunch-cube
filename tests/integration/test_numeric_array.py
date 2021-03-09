# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import NA


class DescribeNumericArrays(object):
    """Test-suite for numeric-array behaviors."""

    def it_provides_means_scale_measures(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_SCALE_MEASURES).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.means,
            [
                # ------------- CAT -----------------
                # C1      C2    C3       C4      C5
                [np.nan, 70.00000, 53.2000, 62.752044, 70.0000],  # S1 (num array)
                [np.nan, 66.666667, 70.0, 65.585284, 67.692308],  # S2 (num array)
                [np.nan, 65.00000, 63.400, 62.38806, 49.230769],  # S3 (num array)
                [np.nan, 63.333333, 26.8, 57.029973, 39.230769],  # S4 (num array)
            ],
        )
        np.testing.assert_array_almost_equal(
            slice_.rows_scale_median, [3.0, 3.0, 3.0, 3.0]
        )

    def it_provides_means_for_num_array_grouped_by_cat(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT).partitions[0]

        np.testing.assert_almost_equal(
            slice_.means,
            [  # -------Gender-----
                #     M        F
                [87.6666667, 52.5],  # S1: Dark Night
                [93.3333333, 50.0],  # S2: Fight Club
                [1.00000000, 45.0],  # S3: Meet the parents
            ],
        )
        np.testing.assert_almost_equal(slice_.columns_base, [[3, 2], [3, 1], [1, 1]])

    def it_provides_means_for_num_array_grouped_by_date(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_DATE).partitions[0]

        np.testing.assert_almost_equal(
            slice_.means,
            [  # -------Wave-----
                # 2014-12   2015-01
                [50.000000, 100.0000],  # S1: Mov A
                [90.000000, 81.00000],  # S2: Mov B
                [46.6666667, 80.0000],  # S3: Mov C
            ],
        )
        np.testing.assert_almost_equal(slice_.columns_base, [[10, 9], [8, 10], [9, 10]])

    def it_provides_means_for_num_array_grouped_by_cat_weighted(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_WEIGHTED).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.means,
            [
                #  Gender:
                #  M  F
                [91.62499999, 50.71428571],  # S1 (num array)
                [94.37500000, 50.00000],  # S2 (num array)
                [1.000000000, 45.00000],  # S3 (num array)
            ],
        )
        np.testing.assert_almost_equal(slice_.columns_base, [[3, 2], [3, 1], [1, 1]])

    def it_provides_means_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR).partitions[0]

        np.testing.assert_almost_equal(
            slice_.means,
            [
                # -------------------------MR--------------------------
                #     S1         S2         S3         S4        S5
                [4.5526316, 4.7857143, 4.3333333, 4.4444444, 4.5526316],  # S1 (num arr)
                [3.7105263, 3.8571429, 3.8333333, 3.5555556, 3.7105263],  # S2 (num arr)
            ],
        )
        # ---The columns_base is 2D because a NUM_ARR_X_MR matrix has a distinct
        # ---column-base for each cell.
        np.testing.assert_almost_equal(
            slice_.columns_base, [[38, 14, 6, 18, 38], [38, 14, 6, 18, 38]]
        )

    def it_provides_means_for_numeric_array_with_no_grouping(self):
        """Test means on no-dimensions measure of numeric array."""
        strand = Cube(NA.NUM_ARR_MEANS_NO_GROUPING).partitions[0]

        assert strand.means == pytest.approx([2.5, 25.0])
        assert strand.unweighted_counts.tolist() == [6, 6]
        assert strand.unweighted_bases.tolist() == [6, 6]
        assert strand.table_base_range.tolist() == [6, 6]

    def it_provides_sum_for_num_array_grouped_by_cat(self):
        """Test sum on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]

        np.testing.assert_almost_equal(
            slice_.sums,
            [
                #  --------Gender------------
                # M    F
                [4.0, 3.0],  # S1 (Ticket Sold)
                [3.0, 0.0],  # S2 (Ticket Sold)
                [2.0, 3.0],  # S3 (Ticket Sold)
            ],
        )
        np.testing.assert_almost_equal(slice_.columns_base, [[3, 2], [3, 2], [3, 2]])

    def it_provides_sum_for_num_array_grouped_by_cat_hs(self):
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT_HS).partitions[0]

        np.testing.assert_almost_equal(
            slice_.sums,
            [
                #  --------Gender------------
                # M    F   M+F
                [4.0, 3.0, 7.0],  # S1 (Ticket Sold)
                [3.0, 0.0, 3.0],  # S2 (Ticket Sold)
                [2.0, 3.0, 5.0],  # S3 (Ticket Sold)
            ],
        )
        np.testing.assert_almost_equal(
            slice_.columns_base,
            [
                [3, 2, 5],
                [3, 2, 5],
                [3, 2, 5],
            ],
        )
