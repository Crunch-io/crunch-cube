# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import NA


class DescribeNumericArrays:
    """Test-suite for numeric-array behaviors."""

    def it_provides_means_scale_measures(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_SCALE_MEASURES).partitions[0]

        assert slice_.means == pytest.approx(
            np.array(
                [
                    # ------------- CAT -----------------
                    # C1      C2    C3       C4      C5
                    [np.nan, 70.00000, 53.2000, 62.752044, 70.0000],  # S1 (num array)
                    [np.nan, 66.666667, 70.0, 65.585284, 67.692308],  # S2 (num array)
                    [np.nan, 65.00000, 63.400, 62.38806, 49.230769],  # S3 (num array)
                    [np.nan, 63.333333, 26.8, 57.029973, 39.230769],  # S4 (num array)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.rows_scale_median == pytest.approx([3.0, 3.0, 3.0, 3.0])

    def it_provides_means_for_num_array_grouped_by_cat(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT).partitions[0]

        assert slice_.means == pytest.approx(
            np.array(
                [  # -------Gender-----
                    #     M        F
                    [87.6666667, 52.5],  # S1: Dark Night
                    [93.3333333, 50.0],  # S2: Fight Club
                    [1.00000000, 45.0],  # S3: Meet the parents
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [3, 1], [1, 1]]))

    @pytest.mark.parametrize(
        "element_transform",
        (
            {"S2": {"hide": True}, "key": "subvar_id"},
            {"S2": {"hide": True}},
            {"Fight Club": {"hide": True}, "key": "alias"},
            {"1": {"hide": True}},
        ),
    )
    def it_provides_means_for_num_array_hiding_transforms_grouped_by_cat(
        self, element_transform
    ):
        transforms = {"rows_dimension": {"elements": element_transform}}
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT, transforms=transforms
        ).partitions[0]

        assert slice_.means == pytest.approx(
            np.array(
                [  # -------Gender-----
                    #     M        F
                    [87.6666667, 52.5],  # S1: Dark Night
                    #     --      --       S2: Fight Club HIDDEN
                    [1.00000000, 45.0],  # S3: Meet the parents
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [1, 1]]))

    def it_provides_means_for_num_array_grouped_by_date(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_DATE).partitions[0]

        assert slice_.means == pytest.approx(
            np.array(
                [  # -------Wave-----
                    # 2014-12   2015-01
                    [50.000000, 100.0000],  # S1: Mov A
                    [90.000000, 81.00000],  # S2: Mov B
                    [46.6666667, 80.0000],  # S3: Mov C
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(
            np.array([[10, 9], [8, 10], [9, 10]])
        )

    def it_provides_means_for_num_array_grouped_by_cat_weighted(self):
        """Test means on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_WEIGHTED).partitions[0]

        assert slice_.means == pytest.approx(
            np.array(
                [
                    #  Gender:
                    #  M  F
                    [91.62499999, 50.71428571],  # S1 (num array)
                    [94.37500000, 50.00000],  # S2 (num array)
                    [1.000000000, 45.00000],  # S3 (num array)
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [3, 1], [1, 1]]))

    def it_provides_means_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR).partitions[0]
        expected_means = [
            # -------------------------MR--------------------------
            #     S1         S2         S3         S4        S5
            [4.5526316, 4.7857143, 4.3333333, 4.4444444, 4.5526316],  # S1 (num arr)
            [3.7105263, 3.8571429, 3.8333333, 3.5555556, 3.7105263],  # S2 (num arr)
        ]

        assert slice_.means == pytest.approx(np.array(expected_means))
        # ---The columns_base is 2D because a NUM_ARR_X_MR matrix has a distinct
        # ---column-base for each cell.
        assert slice_.columns_base == pytest.approx(
            np.array([[38, 14, 6, 18, 38], [38, 14, 6, 18, 38]])
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

        assert slice_.sums == pytest.approx(
            np.array(
                [
                    #  --------Gender------------
                    # M    F
                    [4.0, 3.0],  # S1 (Ticket Sold)
                    [3.0, 0.0],  # S2 (Ticket Sold)
                    [2.0, 3.0],  # S3 (Ticket Sold)
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [3, 2], [3, 2]]))

    def it_provides_sum_for_num_array_grouped_by_cat_hs(self):
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT_HS).partitions[0]

        assert slice_.sums == pytest.approx(
            np.array(
                [
                    #  --------Gender------------
                    # M    F   M+F
                    [4.0, 3.0, 7.0],  # S1 (Ticket Sold)
                    [3.0, 0.0, 3.0],  # S2 (Ticket Sold)
                    [2.0, 3.0, 5.0],  # S3 (Ticket Sold)
                ],
            )
        )
        assert slice_.columns_base == pytest.approx(
            np.array(
                [
                    [3, 2, 5],
                    [3, 2, 5],
                    [3, 2, 5],
                ],
            )
        )

    def it_provides_stddev_for_numeric_array_with_no_grouping(self):
        """Test stddev on no-dimensions measure of numeric array."""
        strand = Cube(NA.NUM_ARR_STDDEV_NO_GROUPING).partitions[0]

        assert strand.stddev == pytest.approx([2.5819889, 3.51188458, 2.12132034])
        assert strand.unweighted_counts.tolist() == [4, 3, 2]
        assert strand.unweighted_bases.tolist() == [4, 3, 2]
        assert strand.table_base_range.tolist() == [2, 4]

    def it_provides_stddev_for_num_array_grouped_by_cat(self):
        """Test stddev on numeric array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_STDDEV_GROUPED_BY_CAT).partitions[0]

        assert slice_.stddev == pytest.approx(
            np.array(
                [
                    #  --------Gender------------
                    #     M         F
                    [8.7368949, 17.6776695],  # S1 (Ticket Sold)
                    [2.8867513, np.nan],  # S2 (Ticket Sold)
                    [np.nan, np.nan],  # S3 (Ticket Sold)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [3, 1], [1, 1]]))

    def it_provides_stddev_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_STDDEV_X_MR).partitions[0]

        assert slice_.stddev == pytest.approx(
            np.array(
                [
                    # -------------------------MR----------------
                    #     S1      S2       S3
                    [1.4142136, np.nan, np.nan],  # S1 (num arr)
                    [np.nan, np.nan, np.nan],  # S2 (num arr)
                    [np.nan, np.nan, np.nan],  # S3 (num arr)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.columns_base == pytest.approx(
            np.array([[2, 1, 1], [1, 0, 0], [1, 1, 1]])
        )

    def it_provides_share_of_sum_for_numeric_array_with_no_grouping(self):
        strand = Cube(NA.NUM_ARR_SUM_NO_GROUPING).partitions[0]

        assert strand.sums == pytest.approx([25, 44])
        assert strand.share_sum == pytest.approx([0.3623188, 0.6376811])
        assert strand.unweighted_counts.tolist() == [6, 6]
        assert strand.unweighted_bases.tolist() == [6, 6]
        assert strand.table_base_range.tolist() == [6, 6]

    def it_provides_share_of_sum_for_num_array_grouped_by_cat(self):
        """Test share of sum on num array, grouped by single categorical dimension."""
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]

        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    #  --------Gender------------
                    #    M        F
                    [0.4444444, 0.5],  # S1 (Ticket Sold)
                    [0.3333333, 0.0],  # S2 (Ticket Sold)
                    [0.2222222, 0.5],  # S3 (Ticket Sold)
                ],
            )
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    # --------------Gender----------------
                    #     M        F
                    [0.5714285, 0.4285714],  # S1 (Ticket Sold)
                    [1.00, 0.0],  # S2 (Ticket Sold)
                    [0.4000, 0.6000],  # S3 (Ticket Sold)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    # --------------Gender----------------
                    #     M        F
                    [0.26666667, 0.2],  # S1 (Ticket Sold)
                    [0.2, 0.0],  # S2 (Ticket Sold)
                    [0.13333333, 0.2],  # S3 (Ticket Sold)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.columns_base == pytest.approx(np.array([[3, 2], [3, 2], [3, 2]]))

    def it_provides_share_of_sum_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_SUM_X_MR).partitions[0]

        assert slice_.sums == pytest.approx(
            np.array(
                [
                    # -------------MR----------------
                    # S1   S2    S3
                    [12.0, 5.0, 5.0],  # S1 (num arr)
                    [9.0, 0.0, 0.0],  # S2 (num arr)
                    [4.0, 4.0, 4.0],  # S3 (num arr)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    # --------------MR----------------
                    #     S1      S2       S3
                    [0.48, 0.5555555, 0.5555555],  # S1 (num arr)
                    [0.36, 0.0, 0.0],  # S2 (num arr)
                    [0.16, 0.4444444, 0.4444444],  # S3 (num arr)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    # --------------MR----------------
                    #     S1      S2       S3
                    [0.5454545, 0.2272727, 0.2272727],  # S1 (num arr)
                    [1.0, 0.0, 0.0],  # S2 (num arr)
                    [0.3333333, 0.3333333, 0.3333333],  # S3 (num arr)
                ],
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    # --------------MR----------------
                    #     S1      S2       S3
                    [0.27906977, 0.11627907, 0.11627907],  # S1 (num arr)
                    [0.20930233, 0.0, 0.0],  # S2 (num arr)
                    [0.09302326, 0.09302326, 0.09302326],  # S3 (num arr)
                ],
            ),
            nan_ok=True,
        )

    def it_provides_unweighted_valid_counts_for_num_array_grouped_by_cat(self):
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]

        # unweighted valid counts
        assert slice_.unweighted_counts.tolist() == [
            [3.0, 2.0],
            [3.0, 2.0],
            [3.0, 2.0],
        ]

    def it_provides_valid_counts_for_num_array_grouped_by_cat_hs(self):
        transforms = {
            "columns_dimension": {
                "insertions": [
                    {
                        # Valid counts will be nan
                        # SubDiff doesn't be applied to valid counts
                        "function": "subtotal",
                        "name": "DIFF B-A",
                        "args": [1, 3],
                        "anchor": "top",
                        "kwargs": {"negative": [1, 2]},
                    },
                    {
                        "function": "subtotal",
                        "anchor": 1,
                        "args": [1, 2],
                        "name": '"A" countries',
                    },
                    {
                        # Valid counts will be nan
                        # SubDiff doesn't be applied to valid counts
                        "function": "subtotal",
                        "name": "DIFF A-B",
                        "args": [1, 2],
                        "anchor": "bottom",
                        "kwargs": {"negative": [1, 2]},
                    },
                ]
            }
        }
        cube = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_WEIGHTED, transforms=transforms)
        slice_ = cube.partitions[0]

        # Numeric array renders valid counts weighted and unweighted
        assert slice_.unweighted_counts == pytest.approx(
            np.array(
                [
                    [np.nan, 3, 5, 2, np.nan],
                    [np.nan, 3, 4, 1, np.nan],
                    [np.nan, 1, 2, 1, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.counts == pytest.approx(
            np.array(
                [
                    [np.nan, 0.8, 1.5, 0.7, np.nan],
                    [np.nan, 0.8, 1.2, 0.4, np.nan],
                    [np.nan, 0.2, 0.5, 0.3, np.nan],
                ]
            ),
            nan_ok=True,
        )

    def it_provides_unweighted_valid_counts_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR).partitions[0]

        # unweighted valid counts
        assert slice_.unweighted_counts.tolist() == [
            [38.0, 14.0, 6.0, 18.0, 38.0],
            [38.0, 14.0, 6.0, 18.0, 38.0],
        ]

    def it_provides_weighted_and_unweighted_valid_counbts_for_num_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_WEIGHTED).partitions[0]

        # unweighted valid counts
        assert slice_.unweighted_counts.tolist() == [
            [3.0, 2.0],
            [3.0, 1.0],
            [1.0, 1.0],
        ]
        # weighted valid counts
        assert slice_.counts.tolist() == [
            [0.8, 0.7],
            [0.8, 0.4],
            [0.2, 0.3],
        ]

    def it_provides_unweighted_valid_counts_for_numeric_array_with_no_grouping(self):
        strand = Cube(NA.NUM_ARR_MEANS_NO_GROUPING).partitions[0]

        # unweighted valid counts
        assert strand.unweighted_counts.tolist() == [6.0, 6.0]

    def it_provides_weighted_and_unweighted_valid_counts_for_num_arr_no_grouping_wgtd(
        self,
    ):
        strand = Cube(NA.NUM_ARR_MEANS_NO_GROUPING_WEIGHTED).partitions[0]

        # unweighted valid counts
        assert strand.unweighted_counts.tolist() == [19.0, 16.0, 17.0, 20.0]
        # weighted valid counts
        assert strand.counts.tolist() == [10.33, 11.22, 18.99, 14.55]

    def it_provides_weighted_and_unweighted_valid_counts_for_num_arr_x_mr_wgtd(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR_WEIGHTED).partitions[0]

        # unweighted valid counts
        assert slice_.unweighted_counts.tolist() == [
            [4.0, 7.0],
            [4.0, 7.0],
            [4.0, 6.0],
            [2.0, 6.0],
        ]
        # weighted valid counts
        assert slice_.counts.tolist() == [
            [2.33, 7.009],
            [4.123, 6.777],
            [4.67, 6.4],
            [1.898, 6.1],
        ]

    def it_has_bases_that_dont_sum_across_num_array_subvars_x_cat(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT).partitions[0]

        assert slice_.row_weighted_bases.tolist() == [
            [5.0, 5.0],
            [4.0, 4.0],
            [2.0, 2.0],
        ]
        assert slice_.table_weighted_bases == pytest.approx(slice_.row_weighted_bases)
        # --- Can't sum across subvariables so column bases are the same as counts ---
        assert slice_.column_weighted_bases == pytest.approx(slice_.counts)

    def it_has_bases_that_dont_sum_across_num_array_subvars_x_cat_with_insertion(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS).partitions[0]

        assert slice_.row_weighted_bases.tolist() == [
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [19.0, 19.0, 19.0, 19.0, 19.0],
            [16.0, 16.0, 16.0, 16.0, 16.0],
            [17.0, 17.0, 17.0, 17.0, 17.0],
        ]
        assert (
            slice_.table_weighted_bases.tolist() == slice_.row_weighted_bases.tolist()
        )
        # --- Can't sum across subvariables so column bases are the same as counts ---
        assert slice_.column_weighted_bases.tolist() == slice_.counts.tolist()

    def it_has_bases_that_dont_sum_across_num_array_subvars_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_X_MR_WEIGHTED).partitions[0]
        row_bases = np.array(
            [[4.66, 8.9992], [8.243, 9.011], [5.781, 7.523445], [6.228, 8.553]]
        )

        # --- row bases add along the selection axis (and aren't the same as the counts)
        assert slice_.row_weighted_bases == pytest.approx(row_bases)
        # --- table bases are the same as the row bases ---
        assert slice_.table_weighted_bases == pytest.approx(row_bases)
        # --- but the column weighted bases are the same as the counts ---
        assert slice_.column_weighted_bases == pytest.approx(
            np.array([[2.33, 7.009], [4.123, 6.777], [4.67, 6.4], [1.898, 6.1]])
        )
