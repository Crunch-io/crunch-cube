# encoding: utf-8

"""Integrations tests for headings and subtotals behaviors."""

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.cube import Cube

from ..fixtures import CR


class TestHeadersAndSubtotals(object):
    """Legacy unit-test suite for inserted rows and columns."""

    def test_headings_econ_blame_one_subtotal(self):
        slice_ = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.row_labels == expected

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.ECON_BLAME_WITH_HS, transforms=transforms).partitions[0]
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.row_labels == expected

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        slice_ = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
            "Test Heading with Skipped",
        )
        assert slice_.row_labels == expected

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.ECON_BLAME_WITH_HS_MISSING, transforms=transforms).partitions[
            0
        ]
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.row_labels == expected

    def test_1D_one_subtotal(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
        counts = strand.counts
        np.testing.assert_array_equal(counts, (285, 396, 681, 242, 6, 68))

    def test_1D_one_subtotal_suppressed(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        strand = Cube(CR.ECON_BLAME_WITH_HS, transforms=transforms).partitions[0]

        counts = strand.counts

        np.testing.assert_array_equal(counts, (285, 396, 242, 6, 68))

    def test_1D_subtotals_counts_missing_excluded(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        counts = strand.counts
        np.testing.assert_array_equal(counts, (285, 396, 681, 242, 6, 68, 74))

    def test_1D_subtotals_proportions(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
        table_proportions = strand.table_proportions
        np.testing.assert_almost_equal(
            table_proportions,
            (0.2858576, 0.3971916, 0.6830491, 0.2427282, 0.0060181, 0.0682046),
        )

    def test_1D_subtotals_proportions_one_transform_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        strand = Cube(CR.ECON_BLAME_WITH_HS, transforms=transforms).partitions[0]

        table_proportions = strand.table_proportions

        np.testing.assert_almost_equal(
            table_proportions, (0.2858576, 0.3971916, 0.2427282, 0.0060181, 0.0682046)
        )

    def test_1D_subtotals_proportions_missing_excluded(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        table_proportions = strand.table_proportions
        np.testing.assert_almost_equal(
            table_proportions,
            (
                0.2858576,
                0.3971916,
                0.6830491,
                0.2427282,
                0.0060181,
                0.0682046,
                0.0742227,
            ),
        )

    def test_1D_subtotals_row_base(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        np.testing.assert_equal(strand.row_base, [285, 396, 681, 242, 6, 68, 74])

    def test_1D_subtotals_rows_dimension_fills(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        assert strand.rows_dimension_fills == (None,) * 7

    def test_1D_subtotals_inserted_row_idxs(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS_MISSING).partitions[0]
        assert strand.inserted_row_idxs == (2, 6)

    def test_1D_means_mr_subtotals_hidden(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        strand = Cube(CR.MR_MEAN_FILT_WGTD, transforms=transforms).partitions[0]
        assert strand.inserted_row_idxs == ()

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        assert slice_.row_labels == (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.column_labels == (
            "Very liberal",
            "Liberal",
            "Moderate",
            "Conservative",
            "Very Conservative",
            "Not sure",
        )

    def test_labels_on_2d_cube_with_hs_on_both_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        assert slice_.row_labels == (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.column_labels == (
            "Very liberal",
            "Liberal",
            "Moderate",
            "Test 2nd dim Heading",
            "Conservative",
            "Very Conservative",
            "Not sure",
        )

    def test_labels_on_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, transforms=transforms
        ).partitions[0]
        assert slice_.row_labels == (
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        )
        assert slice_.column_labels == (
            "Very liberal",
            "Liberal",
            "Moderate",
            "Conservative",
            "Very Conservative",
            "Not sure",
        )

    def test_subtotals_as_array_2d_cube_with_hs_on_row(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = np.array(
            [
                [3, 14, 80, 114, 67, 7],
                [59, 132, 162, 29, 12, 2],
                [62, 146, 242, 143, 79, 9],
                [6, 29, 109, 67, 26, 5],
                [1, 1, 1, 1, 0, 2],
                [3, 6, 23, 7, 5, 24],
            ]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_col(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_COL_HS).partitions[0]
        expected = np.array(
            [
                [3, 14, 80, 94, 114, 67, 7],
                [59, 132, 162, 294, 29, 12, 2],
                [6, 29, 109, 138, 67, 26, 5],
                [1, 1, 1, 2, 1, 0, 2],
                [3, 6, 23, 29, 7, 5, 24],
            ]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        expected = np.array(
            [
                [3, 14, 80, 94, 114, 67, 7],
                [59, 132, 162, 294, 29, 12, 2],
                [62, 146, 242, 388, 143, 79, 9],
                [6, 29, 109, 138, 67, 26, 5],
                [1, 1, 1, 2, 1, 0, 2],
                [3, 6, 23, 29, 7, 5, 24],
            ]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, transforms=transforms
        ).partitions[0]
        expected = np.array(
            [
                [3, 14, 80, 114, 67, 7],
                [59, 132, 162, 29, 12, 2],
                [6, 29, 109, 67, 26, 5],
                [1, 1, 1, 1, 0, 2],
                [3, 6, 23, 7, 5, 24],
            ]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_col(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = np.array([72, 182, 375, 218, 110, 40])
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_row(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = np.array([285, 396, 681, 242, 6, 68])
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_col(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        expected = np.array([72, 182, 375, 557, 218, 110, 40])
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_row(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        expected = np.array([285, 396, 681, 242, 6, 68])
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_cell(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = [
            [0.00300903, 0.01404213, 0.08024072, 0.11434303, 0.0672016, 0.00702106],
            [0.05917753, 0.13239719, 0.16248746, 0.02908726, 0.01203611, 0.00200602],
            [0.06218656, 0.14643932, 0.24272818, 0.14343029, 0.07923771, 0.00902708],
            [0.00601805, 0.02908726, 0.10932798, 0.0672016, 0.02607823, 0.00501505],
            [0.00100301, 0.00100301, 0.00100301, 0.00100301, 0, 0.00200602],
            [0.00300903, 0.00601805, 0.02306921, 0.00702106, 0.00501505, 0.02407222],
        ]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_col(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = [
            [0.04166667, 0.07692308, 0.21333333, 0.52293578, 0.60909091, 0.175],
            [0.81944444, 0.72527473, 0.432, 0.13302752, 0.10909091, 0.05],
            [0.86111111, 0.8021978, 0.64533333, 0.6559633, 0.71818182, 0.225],
            [0.08333333, 0.15934066, 0.29066667, 0.30733945, 0.23636364, 0.125],
            [0.01388889, 0.00549451, 0.00266667, 0.00458716, 0, 0.05],
            [0.04166667, 0.03296703, 0.06133333, 0.03211009, 0.04545455, 0.6],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_row(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = [
            [0.01052632, 0.04912281, 0.28070175, 0.4, 0.23508772, 0.0245614],
            [0.1489899, 0.33333333, 0.40909091, 0.07323232, 0.03030303, 0.00505051],
            [0.09104258, 0.2143906, 0.35535977, 0.20998532, 0.11600587, 0.01321586],
            [0.02479339, 0.11983471, 0.45041322, 0.2768595, 0.10743802, 0.02066116],
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0, 0.33333333],
            [0.04411765, 0.08823529, 0.33823529, 0.10294118, 0.07352941, 0.35294118],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_two_dim_by_cell(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        expected = [
            [
                0.00300903,
                0.01404213,
                0.08024072,
                0.09428285,
                0.11434303,
                0.0672016,
                0.00702106,
            ],
            [
                0.05917753,
                0.13239719,
                0.16248746,
                0.29488465,
                0.02908726,
                0.01203611,
                0.00200602,
            ],
            [
                0.06218656,
                0.14643932,
                0.24272818,
                0.3891675,
                0.14343029,
                0.07923771,
                0.00902708,
            ],
            [
                0.00601805,
                0.02908726,
                0.10932798,
                0.13841525,
                0.0672016,
                0.02607823,
                0.00501505,
            ],
            [0.00100301, 0.00100301, 0.00100301, 0.00200602, 0.00100301, 0, 0.00200602],
            [
                0.00300903,
                0.00601805,
                0.02306921,
                0.02908726,
                0.00702106,
                0.00501505,
                0.02407222,
            ],
        ]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_ca_labels_with_hs(self):
        slice_ = Cube(CR.SIMPLE_CA_HS).partitions[0]
        assert slice_.row_labels == ("ca_subvar_1", "ca_subvar_2", "ca_subvar_3")
        assert slice_.column_labels == ("a", "b", "Test A and B combined", "c", "d")

    def test_ca_as_array_with_hs(self):
        slice_ = Cube(CR.SIMPLE_CA_HS).partitions[0]
        expected = [[3, 3, 6, 0, 0], [1, 3, 4, 2, 0], [0, 2, 2, 1, 3]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_ca_proportions_with_hs(self):
        slice_ = Cube(CR.SIMPLE_CA_HS).partitions[0]
        expected = [
            [0.5, 0.5, 1, 0, 0],
            [0.16666667, 0.5, 0.66666667, 0.33333333, 0],
            [0, 0.33333333, 0.33333333, 0.16666667, 0.5],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_ca_margin_with_hs(self):
        slice_ = Cube(CR.SIMPLE_CA_HS).partitions[0]
        expected = [6, 6, 6]
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_hs_with_anchor_on_zero_position_labels(self):
        slice_ = Cube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM).partitions[0]
        assert slice_.row_labels == (
            "Serious net",
            "Very serious",
            "Somewhat serious",
            "Not very serious",
            "Not at all serious",
            "Not sure",
        )
        assert slice_.column_labels == (
            "Sexual assaults that go unreported or unpunished",
            "False accusations of sexual assault",
        )

    def test_hs_with_anchor_on_zero_position_as_props_by_col(self):
        slice_ = Cube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM).partitions[0]
        expected = [
            [0.93244626, 0.66023166],
            [0.63664278, 0.23166023],
            [0.29580348, 0.42857143],
            [0.04401228, 0.21428571],
            [0.00307062, 0.06177606],
            [0.02047083, 0.06370656],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_row(self):
        slice_ = Cube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM).partitions[0]
        expected = [
            [0.72705507, 0.27294493],
            [0.83827493, 0.16172507],
            [0.56555773, 0.43444227],
            [0.27922078, 0.72077922],
            [0.08571429, 0.91428571],
            [0.37735849, 0.62264151],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_cell(self):
        slice_ = Cube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM).partitions[0]
        expected = [
            [0.60936455, 0.22876254],
            [0.41605351, 0.08026756],
            [0.19331104, 0.14849498],
            [0.02876254, 0.07424749],
            [0.00200669, 0.02140468],
            [0.01337793, 0.02207358],
        ]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_subtotals_pvals_2d_cube_with_hs_on_row(self):
        """Ensure that pvals shape is the same as table shape with H%S"""
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        expected = [
            [
                1.92562832e-06,
                5.20117283e-12,
                8.30737469e-05,
                0.00000000e00,
                1.77635684e-15,
                1.13223165e-01,
            ],
            [
                2.90878432e-14,
                0.00000000e00,
                8.11477145e-02,
                0.00000000e00,
                5.87376814e-11,
                4.64141147e-06,
            ],
            [
                7.48253449e-04,
                1.32937094e-04,
                4.68860522e-02,
                3.30870975e-01,
                4.01133883e-01,
                2.08502104e-10,
            ],
            [
                1.05605732e-03,
                3.70613426e-03,
                6.11851617e-03,
                1.18269053e-02,
                8.68890220e-01,
                7.62914197e-02,
            ],
            [
                3.69990005e-01,
                9.19546240e-01,
                2.88068221e-01,
                7.57299844e-01,
                3.86924216e-01,
                2.41648361e-04,
            ],
            [
                3.53745446e-01,
                3.70094812e-02,
                5.03974440e-01,
                1.67769523e-02,
                3.15641644e-01,
                0.00000000e00,
            ],
        ]

        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_fruit_hs_top_bottom_labels(self):
        slice_ = Cube(CR.FRUIT_HS_TOP_BOTTOM).partitions[0]
        assert slice_.row_labels == ("TOP", "rambutan", "MIDDLE", "satsuma", "BOTTOM")

    def test_fruit_hs_top_bottom_inserted_indices(self):
        # TODO: Figure how to do with new slice
        cube = CrunchCube(CR.FRUIT_HS_TOP_BOTTOM)
        expected = [[0, 2, 4]]
        actual = cube.inserted_hs_indices(prune=True)
        assert actual == expected

    def test_fruit_hs_top_bottom_counts(self):
        strand = Cube(CR.FRUIT_HS_TOP_BOTTOM).partitions[0]
        counts = strand.counts
        np.testing.assert_array_equal(counts, (100, 33, 100, 67, 100))

    def test_fruit_x_pets_hs_top_bottom_middle_props(self):
        slice_ = Cube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM).partitions[0]
        expected = [
            [1.0, 1.0, 1.0],
            [0.3, 0.35294118, 0.31578947],
            [1.0, 1.0, 1.0],
            [0.7, 0.64705882, 0.68421053],
            [1.0, 1.0, 1.0],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_counts(self):
        slice_ = Cube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM).partitions[0]
        expected = [
            [40, 34, 38],
            [12, 12, 12],
            [40, 34, 38],
            [28, 22, 26],
            [40, 34, 38],
        ]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_hs_indices_pruned_cat_x_date(self):
        # TODO: Figure in frozen slice
        cube = CrunchCube(CR.CAT_X_DATE_HS_PRUNE)
        expected = [0, 3, 6]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_hs_indices_pruned_cat_x_num(self):
        cube = CrunchCube(CR.CAT_X_NUM_HS_PRUNE)
        expected = [0, 1, 3]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

        # Ensure cached properties are not updated
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_cat_x_num_counts_pruned_with_hs(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE, transforms=transforms).partitions[0]
        expected = [[0], [1], [1], [0]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_cat_x_num_counts_pruned_without_hs(self):
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE, transforms=transforms).partitions[0]
        expected = [[1]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_mr_x_cat_hs_counts(self):
        slice_ = Cube(CR.PETS_X_FRUIT_HS).partitions[0]
        expected = [[12, 28, 40], [12, 22, 34], [12, 26, 38]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_mr_x_cat_proportions_with_insertions(self):
        slice_ = Cube(CR.PETS_X_FRUIT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_proportions,
            [
                [0.13483146, 0.31460674, 0.4494382],
                [0.13483146, 0.24719101, 0.38202247],
                [0.13483146, 0.29213483, 0.42696629],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.4, 0.47457627, 0.4494382],
                [0.4, 0.37288136, 0.38202247],
                [0.4, 0.44067797, 0.42696629],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.3, 0.7, 1.0],
                [0.35294118, 0.64705882, 1.0],
                [0.31578947, 0.68421053, 1.0],
            ],
        )

    def test_missing_cat_hs_labels(self):
        slice_ = Cube(CR.MISSING_CAT_HS).partitions[0]
        assert slice_.row_labels == (
            "Whites",
            "White college women voters",
            "White non-college women voters",
            "White college men voters",
            "White non-college men voters",
            "Black voters",
            "Latino and other voters",
        )

    def test_ca_x_cat_counts_with_hs(self):
        # Assert counts without H&S
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts,
            [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        )
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[1]
        np.testing.assert_array_equal(
            slice_.counts,
            [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]],
        )
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[2]
        np.testing.assert_array_equal(
            slice_.counts,
            [[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1]],
        )

        # Assert counts with H&S
        slice_ = Cube(CR.CA_X_CAT_HS).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [1, 1, 0, 2, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        slice_ = Cube(CR.CA_X_CAT_HS).partitions[1]
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        slice_ = Cube(CR.CA_X_CAT_HS).partitions[2]
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 0, 1],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1],
            ],
        )

    def test_ca_x_cat_margin_with_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 1, 1, 1, 1])
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 1, 1, 1, 1])
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 1, 1, 1, 1])

    def test_cat_x_items_x_cats_margin_with_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CAT_X_ITEMS_X_CATS_HS, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            [
                1287.9364594075469,
                2050.0571926339885,
                782.9403891997617,
                225.4066607421201,
                2622.8036855384603,
                974.5889537143403,
                490.5036709315041,
                373.8221357520375,
            ],
        )
        slice_ = Cube(CR.CAT_X_ITEMS_X_CATS_HS, transforms=transforms).partitions[1]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            [
                1147.3697583254452,
                2557.8859179678857,
                1096.7841912034742,
                374.0411471364339,
                1876.3400274431515,
                1002.2399030962134,
                457.92228898229905,
                419.5110527202654,
            ],
        )
        slice_ = Cube(CR.CAT_X_ITEMS_X_CATS_HS, transforms=transforms).partitions[2]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            [
                1053.855075581148,
                2699.612841209989,
                1427.7399174151794,
                380.8205091587366,
                1027.7782011616534,
                606.7100283028576,
                218.42735718966821,
                265.29362712412535,
            ],
        )

    def test_cat_x_mr_weighted_with_hs(self):
        slice_ = Cube(CR.CAT_X_MR_WEIGHTED_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.05865163, 0.087823, 0.07486857, 0.0735683, 0.0, 0.08148267],
                [
                    0.20246563,
                    0.33500382,
                    0.33176765,
                    0.27870974,
                    0.36309359,
                    0.33341993,
                ],
                [
                    0.54592009,
                    0.45988528,
                    0.49802406,
                    0.48137697,
                    0.51250032,
                    0.47855168,
                ],
                [0.1051054, 0.06727875, 0.0733213, 0.12392602, 0.12440609, 0.07023618],
                [0.05508246, 0.04566041, 0.01379632, 0.02729368, 0.0, 0.03006505],
                [0.03277479, 0.00434874, 0.0082221, 0.01512529, 0.0, 0.00624449],
                [0.26111726, 0.42282682, 0.40663622, 0.35227804, 0.36309359, 0.4149026],
                [0.80703735, 0.8827121, 0.90466028, 0.83365501, 0.87559391, 0.89345428],
            ],
        )
        slice_ = Cube(CR.CAT_X_MR_WEIGHTED_HS).partitions[1]
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.00235883, 0.01361576, 0.01328221, 0.01212187, 0.0, 0.01345251],
                [0.13002321, 0.0591588, 0.09857174, 0.05056353, 0.0, 0.07844882],
                [0.65429951, 0.77915194, 0.74437239, 0.61537442, 1.0, 0.76212966],
                [0.13730378, 0.11171429, 0.11961331, 0.26739934, 0.0, 0.11558033],
                [0.04323988, 0.02601641, 0.01593825, 0.02729368, 0.0, 0.02108382],
                [0.03277479, 0.0103428, 0.0082221, 0.02724716, 0.0, 0.00930486],
                [0.13238204, 0.07277456, 0.11185395, 0.06268541, 0.0, 0.09190133],
                [0.78668155, 0.85192649, 0.85622634, 0.67805982, 1.0, 0.85403098],
            ],
        )
        slice_ = Cube(CR.CAT_X_MR_WEIGHTED_HS).partitions[2]
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.003676, 0.00486795, 0.0082221, 0.01212187, 0.0, 0.00650959],
                [
                    0.03884185,
                    0.00625322,
                    0.02928964,
                    0.02729368,
                    0.12440609,
                    0.01752802,
                ],
                [0.39625335, 0.4717925, 0.44918748, 0.46124151, 0.40543568, 0.46072884],
                [0.4255265, 0.44274565, 0.42191512, 0.43268073, 0.47015822, 0.43255049],
                [0.1054366, 0.07434068, 0.08316356, 0.05153692, 0.0, 0.07865889],
                [0.03026569, 0.0, 0.0082221, 0.01512529, 0.0, 0.00402417],
                [
                    0.04251785,
                    0.01112118,
                    0.03751174,
                    0.03941555,
                    0.12440609,
                    0.02403761,
                ],
                [0.4387712, 0.48291368, 0.48669922, 0.50065705, 0.52984178, 0.48476645],
            ],
        )

    def test_mr_x_ca_props_by_row_without_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.MR_X_CA_HS, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.66666667, 0.33333333, 0.00000000, 0.00000000],
                [0.33333333, 0.33333333, 0.33333333, 0.00000000],
                [0.00000000, 0.33333333, 0.33333333, 0.33333333],
            ],
        )
        slice_ = Cube(CR.MR_X_CA_HS, transforms=transforms).partitions[1]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.50000000, 0.50000000, 0.00000000, 0.00000000],
                [0.25000000, 0.25000000, 0.50000000, 0.00000000],
                [0.00000000, 0.25000000, 0.00000000, 0.75000000],
            ],
        )
        slice_ = Cube(CR.MR_X_CA_HS, transforms=transforms).partitions[2]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        )

    def test_mr_x_ca_props_by_row_with_hs(self):
        slice_ = Cube(CR.MR_X_CA_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [
                    0.66666667,
                    0.33333333,
                    1.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    1.00000000,
                ],
                [
                    0.33333333,
                    0.33333333,
                    0.66666667,
                    0.33333333,
                    0.00000000,
                    0.33333333,
                    1.00000000,
                ],
                [
                    0.00000000,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.33333333,
                    0.66666667,
                    1.00000000,
                ],
            ],
        )
        slice_ = Cube(CR.MR_X_CA_HS).partitions[1]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [
                    0.50000000,
                    0.50000000,
                    1.00000000,
                    0.00000000,
                    0.00000000,
                    0.00000000,
                    1.00000000,
                ],
                [
                    0.25000000,
                    0.25000000,
                    0.50000000,
                    0.50000000,
                    0.00000000,
                    0.50000000,
                    1.00000000,
                ],
                [
                    0.00000000,
                    0.25000000,
                    0.25000000,
                    0.00000000,
                    0.75000000,
                    0.75000000,
                    1.00000000,
                ],
            ],
        )
        slice_ = Cube(CR.MR_X_CA_HS).partitions[2]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

    def test_ca_cat_x_mr_x_ca_subvar_counts_pruning(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS, transforms=transforms).partitions[
            0
        ]
        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [22.2609925, 15.10645152],
                [21.46707647, 21.0230097],
                [219.18232448, 135.91751542],
            ],
        )

    def test_ca_cat_x_mr_x_ca_subvar_proportions_pruning(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS, transforms=transforms).partitions[
            0
        ]
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.56722442, 0.50350004],
                [0.53361631, 0.61850703],
                [0.98561362, 0.97332814],
            ],
        )

    def test_ca_x_mr_counts_pruning(self):
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [22.2609925, 21.46707647, 219.18232448],
                [34.4602057, 44.01337663, 314.36524891],
                [25.56308888, 25.335068, 217.58401251],
                [32.40171814, 33.841068, 438.90916486],
                [4.9911294, 8.54509709, 198.80859163],
            ],
        )

    def test_ca_x_mr_proportions_pruning(self):
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_proportions,
            [
                [0.08234797, 0.07602389, 0.15449432],
                [0.12747537, 0.15586976, 0.22158559],
                [0.09456311, 0.08972207, 0.15336772],
                [0.1198606, 0.11984537, 0.30937245],
                [0.01846321, 0.03026176, 0.14013355],
            ],
        )

    def test_col_labels_with_top_hs(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_MISSING).partitions[0]
        assert slice_.column_labels == (
            "Whites",
            "White college women voters",
            "White non-college women voters",
            "White college men voters",
            "White non-college men voters",
            "Black voters",
            "Latino and other voters",
        )

    def it_calculate_col_residuals_for_subtotals(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]

        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
            [
                [
                    0.49326036,
                    0.43967108,
                    0.43739495,
                    0.4093598,
                    0.42242603,
                    0.41688475,
                    0.47060217,
                ],
                [
                    0.35255854,
                    0.47915742,
                    0.47870319,
                    0.46986171,
                    0.3799671,
                    0.42844691,
                    0.4752359,
                ],
                [
                    0.44536177,
                    0.48562091,
                    0.48697607,
                    0.49823831,
                    0.49136926,
                    0.49885606,
                    0.46812184,
                ],
                [
                    0.22709084,
                    0.4103259,
                    0.41105414,
                    0.39507899,
                    0.32201514,
                    0.35776034,
                    0.37468029,
                ],
                [
                    0.4,
                    0.49487166,
                    0.49493871,
                    0.49948985,
                    0.45491071,
                    0.48840757,
                    0.49981735,
                ],
                [
                    0.41301152,
                    0.372678,
                    0.37676108,
                    0.44107522,
                    0.49937461,
                    0.48614202,
                    0.36229072,
                ],
            ],
        )

        np.testing.assert_almost_equal(
            slice_.columns_std_err,
            [
                [
                    0.06651121,
                    0.03916901,
                    0.01766622,
                    0.02325007,
                    0.0211213,
                    0.01564541,
                    0.03868326,
                ],
                [
                    0.04753898,
                    0.04268674,
                    0.01933464,
                    0.02668635,
                    0.01899836,
                    0.01607933,
                    0.03906415,
                ],
                [
                    0.06005257,
                    0.04326255,
                    0.01966878,
                    0.02829803,
                    0.02456846,
                    0.01872173,
                    0.03847938,
                ],
                [
                    0.03062092,
                    0.03655474,
                    0.01660232,
                    0.02243898,
                    0.01610076,
                    0.01342651,
                    0.03079853,
                ],
                [
                    0.05393599,
                    0.04408667,
                    0.01999039,
                    0.02836912,
                    0.02274554,
                    0.01832961,
                    0.04108473,
                ],
                [
                    0.05569046,
                    0.03320079,
                    0.01521724,
                    0.02505139,
                    0.02496873,
                    0.01824458,
                    0.02978011,
                ],
            ],
        )

    def it_calculate_residuals_for_subtotals_1col_2rows(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]

        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    5.51248445,
                    0.02683276,
                    -0.22374462,
                    -2.13500054,
                    -1.48559975,
                    -2.96936015,
                    2.0378446,
                ],
                [
                    -2.53018445,
                    1.48240375,
                    3.85931276,
                    1.28007219,
                    -6.22382047,
                    -4.37603499,
                    1.26887162,
                ],
                [
                    2.54284314,
                    1.39098139,
                    3.3615757,
                    -0.70804042,
                    -7.05452463,
                    -6.66285184,
                    2.97302533,
                ],
                [
                    -2.43393077,
                    1.10902345,
                    3.05012432,
                    0.79584274,
                    -3.63187854,
                    -2.51507523,
                    -0.30158645,
                ],
                [
                    -4.18311635,
                    2.20809445,
                    5.87331384,
                    1.7828024,
                    -8.48620633,
                    -5.93723152,
                    0.93214709,
                ],
                [
                    -0.75336796,
                    -2.53458647,
                    -6.44707428,
                    0.1070329,
                    11.12139895,
                    9.70800153,
                    -3.09346517,
                ],
            ],
        )

        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    3.53803455e-08,
                    9.78593126e-01,
                    8.22956022e-01,
                    3.27609659e-02,
                    1.37385020e-01,
                    2.98420609e-03,
                    4.15654768e-02,
                ],
                [
                    1.14002576e-02,
                    1.38232896e-01,
                    1.13706338e-04,
                    2.00519750e-01,
                    4.85192331e-10,
                    1.20857620e-05,
                    2.04486857e-01,
                ],
                [
                    1.09954577e-02,
                    1.64231069e-01,
                    7.74991104e-04,
                    4.78920155e-01,
                    1.73194792e-12,
                    2.68565170e-11,
                    2.94880115e-03,
                ],
                [
                    1.49358517e-02,
                    2.67420068e-01,
                    2.28746655e-03,
                    4.26123447e-01,
                    2.81365556e-04,
                    1.19006985e-02,
                    7.62967342e-01,
                ],
                [
                    2.87540141e-05,
                    2.72376900e-02,
                    4.27168678e-09,
                    7.46184742e-02,
                    0.00000000e00,
                    2.89875191e-09,
                    3.51260516e-01,
                ],
                [
                    4.51228831e-01,
                    1.12580138e-02,
                    1.14029897e-10,
                    9.14762883e-01,
                    0.00000000e00,
                    0.00000000e00,
                    1.97833774e-03,
                ],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [
                    0.13782324,
                    0.13991696,
                    0.29409917,
                    0.19584546,
                    0.23049127,
                    0.29492964,
                    0.16965035,
                ],
                [
                    0.0694202,
                    0.16278116,
                    0.33844864,
                    0.24068881,
                    0.20143838,
                    0.30541126,
                    0.17296997,
                ],
                [
                    0.15371012,
                    0.21209949,
                    0.41928489,
                    0.30224669,
                    0.29821611,
                    0.40027214,
                    0.23847113,
                ],
                [
                    0.04257561,
                    0.12679395,
                    0.2711432,
                    0.1870842,
                    0.16625565,
                    0.24611954,
                    0.12208259,
                ],
                [
                    0.08132809,
                    0.20416662,
                    0.40862927,
                    0.29739975,
                    0.25652936,
                    0.37465137,
                    0.20949558,
                ],
                [
                    0.08491854,
                    0.112028,
                    0.24396626,
                    0.21719344,
                    0.31903655,
                    0.37086338,
                    0.11716946,
                ],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [
                    0.00339092,
                    0.00344243,
                    0.00723584,
                    0.00481846,
                    0.00567087,
                    0.00725627,
                    0.00417397,
                ],
                [
                    0.00170797,
                    0.00400497,
                    0.00832698,
                    0.00592176,
                    0.00495607,
                    0.00751415,
                    0.00425565,
                ],
                [
                    0.00378179,
                    0.00521837,
                    0.01031583,
                    0.00743629,
                    0.00733713,
                    0.00984805,
                    0.0058672,
                ],
                [
                    0.0010475,
                    0.00311956,
                    0.00667104,
                    0.00460291,
                    0.00409045,
                    0.00605538,
                    0.00300365,
                ],
                [
                    0.00200095,
                    0.00502319,
                    0.01005367,
                    0.00731704,
                    0.00631149,
                    0.00921769,
                    0.0051543,
                ],
                [
                    0.00208928,
                    0.00275627,
                    0.0060024,
                    0.0053437,
                    0.00784938,
                    0.0091245,
                    0.00288277,
                ],
            ],
        )

    def it_calculates_residuals_for_multiple_insertions(self):
        slice_ = Cube(CR.FOOD_GROUP_X_SHAPE_PASTA_2ROWS1COL_INSERTION).partitions[0]

        assert slice_.inserted_column_idxs == (3,)
        assert len(slice_.inserted_column_idxs) == 1
        assert slice_.inserted_row_idxs == (2, 5)
        assert len(slice_.inserted_row_idxs) == 2
        assert slice_.row_proportions.shape == slice_.zscore.shape
        # Test zscore for 1 col and 2 rows insertions
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    4.52186866,
                    -1.14595453,
                    -1.15583579,
                    -1.75776465,
                    -2.40088226,
                    -0.35476272,
                    2.97099107,
                    3.02077733,
                ],
                [
                    -1.17059976,
                    2.81759967,
                    3.02086262,
                    4.49354286,
                    0.56410083,
                    -6.35971905,
                    1.38849055,
                    -0.57237387,
                ],
                [
                    2.73069348,
                    1.60707866,
                    1.78415497,
                    2.62163462,
                    -1.50223393,
                    -6.09807527,
                    3.76188322,
                    2.01538897,
                ],
                [
                    -2.48155861,
                    1.06725936,
                    3.90153206,
                    4.34683015,
                    2.00604609,
                    -4.79806048,
                    -1.31477798,
                    -1.02860297,
                ],
                [
                    -0.85351687,
                    -2.70945332,
                    -5.38012857,
                    -6.69526684,
                    -0.09004579,
                    10.93447341,
                    -3.01289341,
                    -1.33102084,
                ],
                [
                    -2.73069348,
                    -1.60707866,
                    -1.78415497,
                    -2.62163462,
                    1.50223393,
                    6.09807527,
                    -3.76188322,
                    -2.01538897,
                ],
            ],
        )

        # Test pvals for 1 col and 2 rows insertions
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    6.12960904e-06,
                    2.51813966e-01,
                    2.47748332e-01,
                    7.87875627e-02,
                    1.63555979e-02,
                    7.22767363e-01,
                    2.96840408e-03,
                    2.52126696e-03,
                ],
                [
                    2.41759696e-01,
                    4.83840956e-03,
                    2.52055696e-03,
                    7.00479138e-06,
                    5.72685511e-01,
                    2.02123207e-10,
                    1.64987721e-01,
                    5.67068716e-01,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    8.75091931e-03,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                ],
                [
                    1.30809196e-02,
                    2.85854735e-01,
                    9.55857972e-05,
                    1.38119093e-05,
                    4.48513304e-02,
                    1.60209410e-06,
                    1.88584530e-01,
                    3.03666283e-01,
                ],
                [
                    3.93372735e-01,
                    6.73941888e-03,
                    7.44326611e-08,
                    2.15278906e-11,
                    9.28250831e-01,
                    0.00000000e00,
                    2.58769793e-03,
                    1.83182153e-01,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    8.75091931e-03,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                ],
            ],
        )
        # Test std deviation for 1 col and 2 rows insertions
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [
                    0.12872965,
                    0.12660834,
                    0.25945758,
                    0.28457383,
                    0.18020547,
                    0.21664973,
                    0.18906989,
                    0.04708727,
                ],
                [
                    0.08962822,
                    0.18156083,
                    0.32133111,
                    0.35810181,
                    0.2343443,
                    0.18609117,
                    0.19615242,
                    0.02133826,
                ],
                [
                    0.15598565,
                    0.21881893,
                    0.39191503,
                    0.42704456,
                    0.28891505,
                    0.27932045,
                    0.26692138,
                    0.05167696,
                ],
                [
                    0.04922247,
                    0.13815009,
                    0.2774646,
                    0.30463101,
                    0.20558719,
                    0.15180204,
                    0.13590952,
                    0.0,
                ],
                [
                    0.09164028,
                    0.12587344,
                    0.25438636,
                    0.27985097,
                    0.22567786,
                    0.31815893,
                    0.15046527,
                    0.0,
                ],
                [
                    0.1038251,
                    0.18520985,
                    0.36056563,
                    0.39164243,
                    0.29738169,
                    0.34478173,
                    0.20059548,
                    0.0,
                ],
            ],
        )
        # Test std error for 1 col and 2 rows insertions
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [
                    0.00316713,
                    0.00311494,
                    0.00638342,
                    0.00700135,
                    0.00443358,
                    0.00533022,
                    0.00465168,
                    0.00115849,
                ],
                [
                    0.00220512,
                    0.00446693,
                    0.00790569,
                    0.00881036,
                    0.00576556,
                    0.00457839,
                    0.00482593,
                    0.00052498,
                ],
                [
                    0.00383771,
                    0.00538359,
                    0.00964226,
                    0.01050655,
                    0.00710816,
                    0.0068721,
                    0.00656705,
                    0.00127141,
                ],
                [
                    0.00121102,
                    0.0033989,
                    0.00682645,
                    0.00749482,
                    0.00505805,
                    0.00373478,
                    0.00334377,
                    0.0,
                ],
                [
                    0.00225462,
                    0.00309686,
                    0.00625865,
                    0.00688516,
                    0.00555234,
                    0.00782765,
                    0.00370189,
                    0.0,
                ],
                [
                    0.0025544,
                    0.00455671,
                    0.00887098,
                    0.00963556,
                    0.00731646,
                    0.00848265,
                    0.00493524,
                    0.0,
                ],
            ],
        )

        slice_ = Cube(CR.CAT_X_CAT_HS_TOTAL_BOTTOM).partitions[0]

        # Test zscore for 2 rows and 2 cols insertions (1 col at bottom)
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    4.52186866,
                    -1.14595453,
                    -1.15583579,
                    -1.75776465,
                    -2.40088226,
                    -0.35476272,
                    2.97099107,
                    3.02077733,
                    2.04273365,
                ],
                [
                    -1.17059976,
                    2.81759967,
                    3.02086262,
                    4.49354286,
                    0.56410083,
                    -6.35971905,
                    1.38849055,
                    -0.57237387,
                    -4.68606473,
                ],
                [
                    2.73069348,
                    1.60707866,
                    1.78415497,
                    2.62163462,
                    -1.50223393,
                    -6.09807527,
                    3.76188322,
                    2.01538897,
                    -2.55784655,
                ],
                [
                    -2.48155861,
                    1.06725936,
                    3.90153206,
                    4.34683015,
                    2.00604609,
                    -4.79806048,
                    -1.31477798,
                    -1.02860297,
                    -5.21045616,
                ],
                [
                    -0.85351687,
                    -2.70945332,
                    -5.38012857,
                    -6.69526684,
                    -0.09004579,
                    10.93447341,
                    -3.01289341,
                    -1.33102084,
                    7.37880896,
                ],
                [
                    -2.73069348,
                    -1.60707866,
                    -1.78415497,
                    -2.62163462,
                    1.50223393,
                    6.09807527,
                    -3.76188322,
                    -2.01538897,
                    2.55784655,
                ],
            ],
        )

        # Test pvals for 2 rows and 2 cols insertions (1 col at bottom)
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    6.12960904e-06,
                    2.51813966e-01,
                    2.47748332e-01,
                    7.87875627e-02,
                    1.63555979e-02,
                    7.22767363e-01,
                    2.96840408e-03,
                    2.52126696e-03,
                    4.10788114e-02,
                ],
                [
                    2.41759696e-01,
                    4.83840956e-03,
                    2.52055696e-03,
                    7.00479138e-06,
                    5.72685511e-01,
                    2.02123207e-10,
                    1.64987721e-01,
                    5.67068716e-01,
                    2.78508208e-06,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    8.75091931e-03,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                    1.05322556e-02,
                ],
                [
                    1.30809196e-02,
                    2.85854735e-01,
                    9.55857972e-05,
                    1.38119093e-05,
                    4.48513304e-02,
                    1.60209410e-06,
                    1.88584530e-01,
                    3.03666283e-01,
                    1.88376891e-07,
                ],
                [
                    3.93372735e-01,
                    6.73941888e-03,
                    7.44326611e-08,
                    2.15278906e-11,
                    9.28250831e-01,
                    0.00000000e00,
                    2.58769793e-03,
                    1.83182153e-01,
                    1.59650071e-13,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    8.75091931e-03,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                    1.05322556e-02,
                ],
            ],
        )
        # Test standard deviation for 2 rows and 2 cols insertions (1 col at bottom)
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [
                    0.12872965,
                    0.12660834,
                    0.25945758,
                    0.28457383,
                    0.18020547,
                    0.21664973,
                    0.18906989,
                    0.04708727,
                    0.28434335,
                ],
                [
                    0.08962822,
                    0.18156083,
                    0.32133111,
                    0.35810181,
                    0.2343443,
                    0.18609117,
                    0.19615242,
                    0.02133826,
                    0.26572964,
                ],
                [
                    0.15598565,
                    0.21881893,
                    0.39191503,
                    0.42704456,
                    0.28891505,
                    0.27932045,
                    0.26692138,
                    0.05167696,
                    0.3713438,
                ],
                [
                    0.04922247,
                    0.13815009,
                    0.2774646,
                    0.30463101,
                    0.20558719,
                    0.15180204,
                    0.13590952,
                    0.0,
                    0.20156053,
                ],
                [
                    0.09164028,
                    0.12587344,
                    0.25438636,
                    0.27985097,
                    0.22567786,
                    0.31815893,
                    0.15046527,
                    0.0,
                    0.34433599,
                ],
                [
                    0.1038251,
                    0.18520985,
                    0.36056563,
                    0.39164243,
                    0.29738169,
                    0.34478173,
                    0.20059548,
                    0.0,
                    0.38409593,
                ],
            ],
        )
        # Test std error for 2 rows and 2 cols insertions (1 col at bottom)
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [
                    0.00316713,
                    0.00311494,
                    0.00638342,
                    0.00700135,
                    0.00443358,
                    0.00533022,
                    0.00465168,
                    0.00115849,
                    0.00699568,
                ],
                [
                    0.00220512,
                    0.00446693,
                    0.00790569,
                    0.00881036,
                    0.00576556,
                    0.00457839,
                    0.00482593,
                    0.00052498,
                    0.00653773,
                ],
                [
                    0.00383771,
                    0.00538359,
                    0.00964226,
                    0.01050655,
                    0.00710816,
                    0.0068721,
                    0.00656705,
                    0.00127141,
                    0.00913615,
                ],
                [
                    0.00121102,
                    0.0033989,
                    0.00682645,
                    0.00749482,
                    0.00505805,
                    0.00373478,
                    0.00334377,
                    0.0,
                    0.00495898,
                ],
                [
                    0.00225462,
                    0.00309686,
                    0.00625865,
                    0.00688516,
                    0.00555234,
                    0.00782765,
                    0.00370189,
                    0.0,
                    0.00847168,
                ],
                [
                    0.0025544,
                    0.00455671,
                    0.00887098,
                    0.00963556,
                    0.00731646,
                    0.00848265,
                    0.00493524,
                    0.0,
                    0.00944989,
                ],
            ],
        )

    def it_calculates_residuals_for_columns_insertion(self):
        slice_ = Cube(CR.CA_SUBVAR_X_CAT_HS).partitions[0]

        # Test zscores for 1 column insertion
        assert slice_.inserted_column_idxs == (4,)
        assert len(slice_.inserted_column_idxs) == 1
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [2.00445931, 0.3354102, -1.34164079, -1.34164079, 2.12132034],
                [-0.40089186, 0.3354102, 1.34164079, -1.34164079, 0.0],
                [-1.60356745, -0.67082039, 0.0, 2.68328157, -2.12132034],
            ],
        )

        # Test pvals for 1 column insertion
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [0.04502088, 0.73731568, 0.17971249, 0.17971249, 0.03389485],
                [0.68849974, 0.73731568, 0.17971249, 0.17971249, 1.0],
                [0.10880943, 0.50233495, 1.0, 0.00729036, 0.03389485],
            ],
        )

        # Test std deviation for 1 column insertion
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [0.372678, 0.372678, 0.0, 0.0, 0.47140452],
                [0.22906142, 0.372678, 0.31426968, 0.0, 0.41573971],
                [0.0, 0.31426968, 0.22906142, 0.372678, 0.31426968],
            ],
        )

        # Test std error for 1 column insertion
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [0.08784105, 0.08784105, 0.0, 0.0, 0.11111111],
                [0.0539903, 0.08784105, 0.07407407, 0.0, 0.09799079],
                [0.0, 0.07407407, 0.0539903, 0.08784105, 0.07407407],
            ],
        )

        # Test col std dev
        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
            [
                [0.4330127, 0.48412292, 0.0, 0.0, 0.5],
                [0.4330127, 0.48412292, 0.47140452, 0.0, 0.47140452],
                [0.0, 0.4330127, 0.47140452, 0.0, 0.372678],
            ],
        )

        # Test col std err
        np.testing.assert_almost_equal(
            slice_.columns_std_err,
            [
                [0.21650635, 0.1711633, 0.0, 0.0, 0.14433757],
                [0.21650635, 0.1711633, 0.27216553, 0.0, 0.13608276],
                [0.0, 0.15309311, 0.27216553, 0.0, 0.10758287],
            ],
        )

        slice_ = Cube(CR.CA_X_CAT_HS).partitions[0]

        # Test zscores for 2 columns insertion bottom and interleaved
        assert slice_.inserted_column_idxs == (3, 6)
        assert len(slice_.inserted_column_idxs) == 2
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    1.36930639,
                    1.36930639,
                    -0.91287093,
                    1.49071198,
                    -0.91287093,
                    -0.91287093,
                    -1.49071198,
                ],
                [
                    -1.36930639,
                    -1.36930639,
                    0.91287093,
                    -1.49071198,
                    0.91287093,
                    0.91287093,
                    1.49071198,
                ],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

        # Test pvals for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    0.17090352,
                    0.17090352,
                    0.36131043,
                    0.13603713,
                    0.36131043,
                    0.36131043,
                    0.13603713,
                ],
                [
                    0.17090352,
                    0.17090352,
                    0.36131043,
                    0.13603713,
                    0.36131043,
                    0.36131043,
                    0.13603713,
                ],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

        # Test standard_deviation for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [0.4, 0.4, 0.0, 0.48989795, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.48989795],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        # Test standard error for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [0.17888544, 0.17888544, 0.0, 0.21908902, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.17888544, 0.17888544, 0.17888544, 0.17888544, 0.21908902],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        # Test col standard dev for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
            [
                [0.0, 0.0, 0.0, 0.47140452, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.47140452, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        # Test col standard error for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.columns_std_err,
            [
                [0.0, 0.0, 0.0, 0.27216553, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.27216553, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

    def it_calculates_residuals_for_rows_insertion(self):
        transforms = {"columns_dimension": {"insertions": {}}}
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]

        # Test zscores for 1 row insertion
        assert slice_.inserted_row_idxs == (1,)
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [2.06930398, -0.61133797, -1.25160615, np.nan, -1.19268916],
                [-2.03371753, 0.66650907, 1.07795469, np.nan, 1.34162721],
                [0.3436098, -1.079875, 0.98134469, np.nan, -0.26228228],
                [-0.90239493, -0.01688425, -0.18683508, np.nan, 2.962256],
                [-1.85225802, 1.24997148, 1.10571507, np.nan, -0.8041707],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [-0.22728508, -0.10690048, 0.5405717, np.nan, -0.31799761],
            ],
        )

        # Test pvals for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [0.03851757, 0.54097586, 0.21071341, np.nan, 0.23299113],
                [0.04198008, 0.50508577, 0.28105398, np.nan, 0.1797169],
                [0.73113976, 0.28019785, 0.32642279, np.nan, 0.79310382],
                [0.36684711, 0.98652895, 0.85178994, np.nan, 0.00305394],
                [0.06398878, 0.21130996, 0.26884987, np.nan, 0.4212984],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.82020207, 0.91486794, 0.58880283, np.nan, 0.75048675],
            ],
        )

        # Test std deviation for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [0.46216723, 0.41533263, 0.31225682, 0.0, 0.10250865],
                [0.26758936, 0.33710998, 0.28174342, 0.0, 0.14635252],
                [0.10559638, 0.0, 0.10250865, 0.0, 0.0],
                [0.17909696, 0.20464365, 0.1484817, 0.0, 0.14635252],
                [0.17909696, 0.28174342, 0.22554563, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.10250865, 0.10250865, 0.10250865, 0.0, 0.0],
            ],
        )

        # Test std error for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [0.04833892, 0.0434404, 0.03265951, 0.0, 0.01072157],
                [0.02798766, 0.03525895, 0.02946806, 0.0, 0.01530728],
                [0.01104452, 0.0, 0.01072157, 0.0, 0.0],
                [0.01873208, 0.02140405, 0.01552997, 0.0, 0.01530728],
                [0.01873208, 0.02946806, 0.02359023, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01072157, 0.01072157, 0.01072157, 0.0, 0.0],
            ],
        )

        # Test col std deviation for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
            [
                [0.41561694, 0.48762374, 0.49916867, np.nan, 0.4689693],
                [0.39644438, 0.48005275, 0.49353964, np.nan, 0.4689693],
                [0.16604076, 0.0, 0.22060003, np.nan, 0.0],
                [0.27659294, 0.32573599, 0.31156024, np.nan, 0.4689693],
                [0.27659294, 0.42678893, 0.4384431, np.nan, 0.0],
                [0.0, 0.0, 0.0, np.nan, 0.0],
                [0.16126906, 0.16853704, 0.22060003, np.nan, 0.0],
            ],
        )

        # Test col std error for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.columns_std_err,
            [
                [0.06895161, 0.08465401, 0.11473767, np.nan, 0.27200111],
                [0.06577085, 0.08333965, 0.1134438, np.nan, 0.27200111],
                [0.02754647, 0.0, 0.05070657, np.nan, 0.0],
                [0.04588727, 0.05654946, 0.07161446, np.nan, 0.27200111],
                [0.04588727, 0.07409277, 0.10077944, np.nan, 0.0],
                [0.0, 0.0, 0.0, np.nan, 0.0],
                [0.02675483, 0.0292589, 0.05070657, np.nan, 0.0],
            ],
        )

        slice_ = Cube(CR.FOOD_GROUP_X_SHAPE_OF_PASTA_2ROWS_INSERTION).partitions[0]

        # Test zscores for 2 rows insertions (interleaved and bottom)
        assert slice_.inserted_row_idxs == (2, 5)
        assert len(slice_.inserted_row_idxs) == 2
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    4.52186866,
                    -1.14595453,
                    -1.15583579,
                    -2.40088226,
                    -0.35476272,
                    2.97099107,
                    3.02077733,
                ],
                [
                    -1.17059976,
                    2.81759967,
                    3.02086262,
                    0.56410083,
                    -6.35971905,
                    1.38849055,
                    -0.57237387,
                ],
                [
                    2.73069348,
                    1.60707866,
                    1.78415497,
                    -1.50223393,
                    -6.09807527,
                    3.76188322,
                    2.01538897,
                ],
                [
                    -2.48155861,
                    1.06725936,
                    3.90153206,
                    2.00604609,
                    -4.79806048,
                    -1.31477798,
                    -1.02860297,
                ],
                [
                    -0.85351687,
                    -2.70945332,
                    -5.38012857,
                    -0.09004579,
                    10.93447341,
                    -3.01289341,
                    -1.33102084,
                ],
                [
                    -2.73069348,
                    -1.60707866,
                    -1.78415497,
                    1.50223393,
                    6.09807527,
                    -3.76188322,
                    -2.01538897,
                ],
            ],
        )

        # Test pvals for 2 rows insertions (interleaved and bottom)
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    6.12960904e-06,
                    2.51813966e-01,
                    2.47748332e-01,
                    1.63555979e-02,
                    7.22767363e-01,
                    2.96840408e-03,
                    2.52126696e-03,
                ],
                [
                    2.41759696e-01,
                    4.83840956e-03,
                    2.52055696e-03,
                    5.72685511e-01,
                    2.02123207e-10,
                    1.64987721e-01,
                    5.67068716e-01,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                ],
                [
                    1.30809196e-02,
                    2.85854735e-01,
                    9.55857972e-05,
                    4.48513304e-02,
                    1.60209410e-06,
                    1.88584530e-01,
                    3.03666283e-01,
                ],
                [
                    3.93372735e-01,
                    6.73941888e-03,
                    7.44326611e-08,
                    9.28250831e-01,
                    0.00000000e00,
                    2.58769793e-03,
                    1.83182153e-01,
                ],
                [
                    6.32012234e-03,
                    1.08037114e-01,
                    7.43984876e-02,
                    1.33036705e-01,
                    1.07353193e-09,
                    1.68638788e-04,
                    4.38639092e-02,
                ],
            ],
        )

        # Test std deviation for 2 rows insertions (interleaved and bottom)
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [
                    0.12872965,
                    0.12660834,
                    0.25945758,
                    0.18020547,
                    0.21664973,
                    0.18906989,
                    0.04708727,
                ],
                [
                    0.08962822,
                    0.18156083,
                    0.32133111,
                    0.2343443,
                    0.18609117,
                    0.19615242,
                    0.02133826,
                ],
                [
                    0.15598565,
                    0.21881893,
                    0.39191503,
                    0.28891505,
                    0.27932045,
                    0.26692138,
                    0.05167696,
                ],
                [
                    0.04922247,
                    0.13815009,
                    0.2774646,
                    0.20558719,
                    0.15180204,
                    0.13590952,
                    0.0,
                ],
                [
                    0.09164028,
                    0.12587344,
                    0.25438636,
                    0.22567786,
                    0.31815893,
                    0.15046527,
                    0.0,
                ],
                [
                    0.1038251,
                    0.18520985,
                    0.36056563,
                    0.29738169,
                    0.34478173,
                    0.20059548,
                    0.0,
                ],
            ],
        )

        # Test std error for 2 rows insertions (interleaved and bottom)
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [
                    0.00316713,
                    0.00311494,
                    0.00638342,
                    0.00443358,
                    0.00533022,
                    0.00465168,
                    0.00115849,
                ],
                [
                    0.00220512,
                    0.00446693,
                    0.00790569,
                    0.00576556,
                    0.00457839,
                    0.00482593,
                    0.00052498,
                ],
                [
                    0.00383771,
                    0.00538359,
                    0.00964226,
                    0.00710816,
                    0.0068721,
                    0.00656705,
                    0.00127141,
                ],
                [
                    0.00121102,
                    0.0033989,
                    0.00682645,
                    0.00505805,
                    0.00373478,
                    0.00334377,
                    0.0,
                ],
                [
                    0.00225462,
                    0.00309686,
                    0.00625865,
                    0.00555234,
                    0.00782765,
                    0.00370189,
                    0.0,
                ],
                [
                    0.0025544,
                    0.00455671,
                    0.00887098,
                    0.00731646,
                    0.00848265,
                    0.00493524,
                    0.0,
                ],
            ],
        )

    def it_calculates_residuals_for_cat_x_cat_with_missing_1_col_insertion(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_MISSING).partitions[0]

        assert slice_.inserted_column_idxs == (0,)
        assert slice_.inserted_row_idxs == ()

        # Test szcores for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    0.42390758,
                    -13.96932047,
                    -18.09470408,
                    14.81067365,
                    18.70362483,
                    -1.57589011,
                    0.89218751,
                ],
                [
                    -0.42390758,
                    13.96932047,
                    18.09470408,
                    -14.81067365,
                    -18.70362483,
                    1.57589011,
                    -0.89218751,
                ],
            ],
        )

        # Test pvals for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [0.67163322, 0.0, 0.0, 0.0, 0.0, 0.11505113, 0.37229243],
                [0.67163322, 0.0, 0.0, 0.0, 0.0, 0.11505113, 0.37229243],
            ],
        )

        # Test std deviation for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [0.47966119, 0.0, 0.0, 0.35251189, 0.40973249, 0.20930808, 0.25894917],
                [0.48723822, 0.36086396, 0.42327383, 0.0, 0.0, 0.24736422, 0.25468787],
            ],
        )

        # Test std error for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [0.0139872, 0.0, 0.0, 0.01027946, 0.01194804, 0.00610355, 0.00755111],
                [0.01420816, 0.01052301, 0.01234292, 0.0, 0.0, 0.00721329, 0.00742685],
            ],
        )

        # Test col std dev for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
            [
                [0.49962497, 0.0, 0.0, 0.0, 0.0, 0.49223325, 0.49991932],
                [0.49962497, 0.0, 0.0, 0.0, 0.0, 0.49223325, 0.49991932],
            ],
        )

        # Test col std err for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.columns_std_err,
            [
                [0.01686153, 0.0, 0.0, 0.0, 0.0, 0.04300662, 0.03868492],
                [0.01686153, 0.0, 0.0, 0.0, 0.0, 0.04300662, 0.03868492],
            ],
        )

    def it_calculates_residuals_for_cat_x_num_hs_pruned_with_3_rows_insertions(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE, transforms=transforms).partitions[0]

        # Test zscore for 3 rows insertions
        assert slice_.inserted_row_idxs == (0, 1, 3)
        np.testing.assert_almost_equal(
            slice_.zscore, np.tile(np.nan, slice_.row_proportions.shape)
        )

        # Test pvals for 3 rows insertions
        np.testing.assert_almost_equal(
            slice_.pvals, [[np.nan], [np.nan], [np.nan], [np.nan]]
        )

        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE).partitions[0]

        # Test zscore for 3 rows insertions (1 at left)
        assert slice_.inserted_row_idxs == (0, 3, 7)
        np.testing.assert_almost_equal(
            slice_.zscore, np.tile(np.nan, slice_.row_proportions.shape)
        )

        # Test pvals for 3 rows insertions (1 at left)
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

    def it_calculates_residuals_for_cat_x_items_x_cats_with_col_insertion(self):
        slice_ = Cube(CR.CAT_X_ITEMS_X_CATS_HS).partitions[2]

        # Test zscores
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [
                    -1.41314157,
                    0.7194691,
                    3.32108915,
                    0.59216028,
                    1.9336029,
                    1.82082302,
                    3.22592648,
                    0.33151107,
                    -3.85652314,
                    -2.07626205,
                    -4.98161407,
                ],
                [
                    -0.43808855,
                    -0.47996405,
                    -1.1006254,
                    0.78673655,
                    -2.19826999,
                    -2.26731388,
                    -3.26371775,
                    2.17216147,
                    1.70425587,
                    1.49053686,
                    4.94790819,
                ],
                [
                    0.05348129,
                    0.45491103,
                    0.71843222,
                    1.2463471,
                    0.72367382,
                    1.00097907,
                    0.45574899,
                    -0.30156157,
                    -1.8988183,
                    0.08307639,
                    -1.789589,
                ],
                [
                    -0.53082169,
                    1.13935624,
                    -0.40216969,
                    -1.24950071,
                    1.88766548,
                    0.60975054,
                    1.97668314,
                    -0.9757069,
                    -0.37995499,
                    -1.15590753,
                    -2.38363265,
                ],
                [
                    0.54705913,
                    -0.79251309,
                    -1.87065586,
                    -2.89476439,
                    -1.78545168,
                    -0.42363372,
                    -0.38265964,
                    -2.48501892,
                    2.36851523,
                    2.86400302,
                    2.49835292,
                ],
                [
                    1.31184119,
                    -1.03522621,
                    -0.87149941,
                    0.03842022,
                    -1.21587579,
                    0.39147125,
                    -0.58168782,
                    -0.42238394,
                    2.61921944,
                    -0.96180943,
                    0.83888703,
                ],
                [
                    1.4497716,
                    0.38201864,
                    -0.02478521,
                    1.04601785,
                    2.47224023,
                    0.92685756,
                    0.13699958,
                    0.5161739,
                    -1.44745069,
                    -1.70416866,
                    -2.42196891,
                ],
                [
                    0.04778726,
                    0.23461714,
                    0.36507151,
                    0.04563621,
                    1.19270052,
                    -0.9922388,
                    0.58190295,
                    0.28565055,
                    0.34036671,
                    -1.15545471,
                    -0.59631166,
                ],
            ],
        )

        # Test pvals
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [
                    1.57614104e-01,
                    4.71851932e-01,
                    8.96668906e-04,
                    5.53743267e-01,
                    5.31619710e-02,
                    6.86337632e-02,
                    1.25565606e-03,
                    7.40258478e-01,
                    1.15011214e-04,
                    3.78697212e-02,
                    6.30560949e-07,
                ],
                [
                    6.61322091e-01,
                    6.31252954e-01,
                    2.71059728e-01,
                    4.31436104e-01,
                    2.79298719e-02,
                    2.33710584e-02,
                    1.09960650e-03,
                    2.98434855e-02,
                    8.83332958e-02,
                    1.36083131e-01,
                    7.50152757e-07,
                ],
                [
                    9.57348438e-01,
                    6.49173248e-01,
                    4.72490824e-01,
                    2.12636993e-01,
                    4.69266012e-01,
                    3.16836929e-01,
                    6.48570490e-01,
                    7.62986307e-01,
                    5.75883707e-02,
                    9.33790796e-01,
                    7.35200075e-02,
                ],
                [
                    5.95542346e-01,
                    2.54554600e-01,
                    6.87559148e-01,
                    2.11481996e-01,
                    5.90708730e-02,
                    5.42027073e-01,
                    4.80774576e-02,
                    3.29209733e-01,
                    7.03978826e-01,
                    2.47718983e-01,
                    1.71427052e-02,
                ],
                [
                    5.84338110e-01,
                    4.28061562e-01,
                    6.13927960e-02,
                    3.79443218e-03,
                    7.41880816e-02,
                    6.71832962e-01,
                    7.01972145e-01,
                    1.29544548e-02,
                    1.78596452e-02,
                    4.18324018e-03,
                    1.24771908e-02,
                ],
                [
                    1.89573723e-01,
                    3.00563283e-01,
                    3.83481527e-01,
                    9.69352639e-01,
                    2.24032243e-01,
                    6.95448936e-01,
                    5.60776974e-01,
                    6.72744802e-01,
                    8.81312357e-03,
                    3.36145342e-01,
                    4.01532705e-01,
                ],
                [
                    1.47122221e-01,
                    7.02447540e-01,
                    9.80226288e-01,
                    2.95552789e-01,
                    1.34269254e-02,
                    3.54000494e-01,
                    8.91031123e-01,
                    6.05732961e-01,
                    1.47770738e-01,
                    8.83495822e-02,
                    1.54366710e-02,
                ],
                [
                    9.61885788e-01,
                    8.14505905e-01,
                    7.15058023e-01,
                    9.63600206e-01,
                    2.32986683e-01,
                    3.21081049e-01,
                    5.60632052e-01,
                    7.75145783e-01,
                    7.33580388e-01,
                    2.47904267e-01,
                    5.50967041e-01,
                ],
            ],
        )
        # Test std deviation
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [
                    0.03018845,
                    0.0319175,
                    0.04857593,
                    0.04751389,
                    0.08953993,
                    0.10328323,
                    0.1524741,
                    0.18022756,
                    0.13456855,
                    0.18607541,
                    0.28325803,
                ],
                [
                    0.0595667,
                    0.04309204,
                    0.04933763,
                    0.07460264,
                    0.11766836,
                    0.14009466,
                    0.20399558,
                    0.2858036,
                    0.24756061,
                    0.30673482,
                    0.43888846,
                ],
                [
                    0.0451034,
                    0.03524264,
                    0.04325743,
                    0.05778187,
                    0.09675494,
                    0.11460302,
                    0.16205685,
                    0.20556123,
                    0.16948566,
                    0.2256755,
                    0.33368666,
                ],
                [
                    0.02002223,
                    0.02248904,
                    0.01817414,
                    0.01864496,
                    0.05761441,
                    0.0601484,
                    0.09295886,
                    0.10388546,
                    0.09123329,
                    0.11333915,
                    0.17684956,
                ],
                [
                    0.04082826,
                    0.02342972,
                    0.02170135,
                    0.02434758,
                    0.06941901,
                    0.09116899,
                    0.13440853,
                    0.1652839,
                    0.1624643,
                    0.2043584,
                    0.29853975,
                ],
                [
                    0.03566708,
                    0.01482241,
                    0.02063053,
                    0.03399419,
                    0.05407998,
                    0.07387197,
                    0.10215211,
                    0.13465807,
                    0.12933925,
                    0.14491506,
                    0.23175336,
                ],
                [
                    0.02433953,
                    0.0148974,
                    0.01541497,
                    0.02545434,
                    0.04827421,
                    0.04793991,
                    0.06399285,
                    0.08495292,
                    0.06291114,
                    0.08184793,
                    0.13290769,
                ],
                [
                    0.01960535,
                    0.01548208,
                    0.01906989,
                    0.02260114,
                    0.04638769,
                    0.04187538,
                    0.07265425,
                    0.09220725,
                    0.07954493,
                    0.09381268,
                    0.15249847,
                ],
            ],
        )
        # Test standard error
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [
                    0.00034447,
                    0.0003642,
                    0.00055429,
                    0.00054217,
                    0.00102171,
                    0.00117853,
                    0.00173984,
                    0.00205652,
                    0.00153552,
                    0.00212325,
                    0.00323218,
                ],
                [
                    0.0006797,
                    0.00049171,
                    0.00056298,
                    0.00085127,
                    0.00134268,
                    0.00159858,
                    0.00232773,
                    0.00326122,
                    0.00282484,
                    0.00350006,
                    0.00500803,
                ],
                [
                    0.00051466,
                    0.00040214,
                    0.0004936,
                    0.00065933,
                    0.00110404,
                    0.0013077,
                    0.00184918,
                    0.0023456,
                    0.00193395,
                    0.00257512,
                    0.0038076,
                ],
                [
                    0.00022847,
                    0.00025662,
                    0.00020738,
                    0.00021275,
                    0.00065742,
                    0.00068634,
                    0.00106073,
                    0.00118541,
                    0.00104104,
                    0.00129328,
                    0.00201798,
                ],
                [
                    0.00046588,
                    0.00026735,
                    0.00024763,
                    0.00027782,
                    0.00079212,
                    0.0010403,
                    0.0015337,
                    0.00188601,
                    0.00185383,
                    0.00233187,
                    0.00340655,
                ],
                [
                    0.00040699,
                    0.00016913,
                    0.00023541,
                    0.0003879,
                    0.00061709,
                    0.00084293,
                    0.00116563,
                    0.00153654,
                    0.00147585,
                    0.00165358,
                    0.00264447,
                ],
                [
                    0.00027773,
                    0.00016999,
                    0.0001759,
                    0.00029045,
                    0.00055084,
                    0.00054703,
                    0.0007302,
                    0.00096937,
                    0.00071786,
                    0.00093394,
                    0.00151657,
                ],
                [
                    0.00022371,
                    0.00017666,
                    0.0002176,
                    0.0002579,
                    0.00052932,
                    0.00047783,
                    0.00082904,
                    0.00105215,
                    0.00090766,
                    0.00107047,
                    0.00174012,
                ],
            ],
        )

        transforms = {"columns_dimension": {"insertions": {}}}
        slice_no_col_insertion_ = Cube(
            CR.CAT_X_ITEMS_X_CATS_HS, transforms=transforms
        ).partitions[2]

        np.testing.assert_almost_equal(
            slice_no_col_insertion_.zscore,
            slice_.zscore[:, : slice_.zscore.shape[1] - 1],
        )
        np.testing.assert_almost_equal(
            slice_no_col_insertion_.pvals, slice_.pvals[:, : slice_.pvals.shape[1] - 1]
        )
        np.testing.assert_almost_equal(
            slice_no_col_insertion_.table_std_dev,
            slice_.table_std_dev[:, : slice_.table_std_dev.shape[1] - 1],
        )
        np.testing.assert_almost_equal(
            slice_no_col_insertion_.table_std_err,
            slice_.table_std_err[:, : slice_.table_std_err.shape[1] - 1],
        )

    def it_calculates_residuals_for_ca_as_0th_with_1_col_insertion(self):
        # Test for multi-cube when first cube represents a categorical-array
        slice_ = Cube(CR.CA_AS_0TH).partitions[0]

        # Test zscore for subtotal as 5 col (idx=4)
        assert slice_.inserted_column_idxs == (4,)
        np.testing.assert_almost_equal(
            slice_.zscore,
            [
                [1.4384278496e02, np.nan, -8.4743683791e01, np.nan, -1.4384278496e02],
                [-3.3776223679e02, np.nan, 1.7103706996e02, np.nan, 3.3776223679e02],
                [-1.2814934144e03, np.nan, 5.9098301224e02, np.nan, 1.2814934144e03],
                [-7.3548887670e02, np.nan, 3.9190486507e02, np.nan, 7.3548887670e02],
                [7.0872989672e02, np.nan, -3.9004636470e02, np.nan, -7.0872989672e02],
                [1.3410335200e02, np.nan, -7.1620373095e01, np.nan, -1.3410335200e02],
                [4.5252318065e02, np.nan, -2.6995569135e02, np.nan, -4.5252318065e02],
                [7.8922929161e02, np.nan, -4.0539816212e02, np.nan, -7.8922929161e02],
                [-1.3248239528e03, np.nan, 8.0070837374e02, np.nan, 1.3248239528e03],
                [-1.2133173836e03, np.nan, 7.2716831191e02, np.nan, 1.2133173836e03],
                [5.7039019588e02, np.nan, -2.5955608665e02, np.nan, -5.7039019588e02],
                [1.5099828673e02, np.nan, -1.0405723538e02, np.nan, -1.5099828673e02],
                [np.nan, np.nan, 3.2420688167e02, 5.3796991420e02, np.nan],
                [-1.7354768750e01, np.nan, 6.5307755488e01, np.nan, 1.7354768750e01],
                [6.2425522308e02, np.nan, -3.6280930476e02, np.nan, -6.2425522308e02],
                [3.1239460437e02, np.nan, -1.8799690191e02, np.nan, -3.1239460437e02],
                [5.9781961932e02, np.nan, -3.0957323307e02, np.nan, -5.9781961932e02],
                [8.7966688339e02, np.nan, -4.5120594474e02, np.nan, -8.7966688339e02],
                [6.0517388503e02, np.nan, -3.1155411016e02, np.nan, -6.0517388503e02],
                [np.nan, -2.2443369826e01, 1.0998128581e01, np.nan, np.nan],
                [np.nan, -4.8981413860e01, 3.1343267444e00, np.nan, np.nan],
                [np.nan, -1.2279863600e01, -1.7734993515e01, np.nan, np.nan],
                [np.nan, -3.5994711460e01, 5.5539580789e00, np.nan, np.nan],
                [np.nan, -4.3899660747e01, 1.9245110772e00, np.nan, np.nan],
                [-6.8023067787e01, np.nan, np.nan, np.nan, 6.8023067787e01],
                [-3.2651193534e01, np.nan, np.nan, np.nan, 3.2651193534e01],
                [
                    -2.5660773147e02,
                    2.1418205760e02,
                    np.nan,
                    -5.6733555096e01,
                    2.5660773147e02,
                ],
                [
                    3.2812124740e02,
                    4.5735057025e01,
                    np.nan,
                    -1.1799439341e02,
                    -3.2812124740e02,
                ],
                [
                    -4.4233316630e02,
                    2.7180866305e02,
                    np.nan,
                    -5.1712174907e01,
                    4.4233316630e02,
                ],
                [
                    -9.0766722278e02,
                    3.3652777380e02,
                    np.nan,
                    2.5617080011e01,
                    9.0766722278e02,
                ],
                [np.nan, -1.0085258765e02, np.nan, np.nan, np.nan],
                [np.nan, -4.2467074626e01, -2.9815026356e02, np.nan, np.nan],
                [np.nan, 4.5477279604e01, -1.0708633335e02, np.nan, np.nan],
                [np.nan, 2.6481299091e01, -1.6558491939e02, np.nan, np.nan],
                [
                    -7.1488546585e01,
                    np.nan,
                    2.1628267512e01,
                    1.2596810374e01,
                    7.1488546585e01,
                ],
                [
                    -4.0614755952e00,
                    np.nan,
                    2.4771762782e00,
                    -1.5116867265e00,
                    4.0614755952e00,
                ],
                [
                    -7.4115575325e01,
                    np.nan,
                    1.8927472595e01,
                    1.5482639326e01,
                    7.4115575325e01,
                ],
                [
                    -8.3748014038e01,
                    np.nan,
                    2.0155106648e01,
                    1.7406525294e01,
                    8.3748014038e01,
                ],
                [np.nan, np.nan, 7.4577100516e01, 3.6726993476e01, np.nan],
                [np.nan, np.nan, 1.4206276890e01, -1.6908070609e01, np.nan],
                [
                    -2.3488280771e02,
                    2.0900124008e02,
                    1.3915693119e02,
                    -9.2635755452e00,
                    2.3488280771e02,
                ],
                [np.nan, np.nan, 2.2479315683e01, -9.7225117840e00, np.nan],
                [np.nan, np.nan, 5.9819788074e01, 1.4913689901e01, np.nan],
                [np.nan, 2.5811339746e01, np.nan, -2.3046606416e01, np.nan],
                [np.nan, 3.3919529385e01, np.nan, 5.3421672113e01, np.nan],
                [np.nan, 5.1294221470e01, np.nan, 8.3465407102e-01, np.nan],
                [np.nan, 5.1873377873e01, np.nan, 3.9671855469e01, np.nan],
                [-9.1108917087e01, np.nan, np.nan, np.nan, 9.1108917087e01],
                [-8.2656990544e01, np.nan, np.nan, np.nan, 8.2656990544e01],
                [
                    1.4267773881e02,
                    -8.3353302495e01,
                    np.nan,
                    4.8349904938e00,
                    -1.4267773881e02,
                ],
                [
                    1.8313465940e02,
                    -7.4554570247e01,
                    np.nan,
                    -1.2007539044e02,
                    -1.8313465940e02,
                ],
                [
                    -1.4327127898e02,
                    np.nan,
                    3.8997895966e01,
                    3.2873649739e01,
                    1.4327127898e02,
                ],
                [
                    -5.4027797535e01,
                    np.nan,
                    2.1541918342e01,
                    7.7287663457e00,
                    5.4027797535e01,
                ],
                [
                    8.9319544542e01,
                    np.nan,
                    -6.4430616588e00,
                    -2.3624730231e01,
                    -8.9319544542e01,
                ],
                [
                    -3.7294644763e01,
                    np.nan,
                    9.3504419061e00,
                    3.1011065784e01,
                    3.7294644763e01,
                ],
                [
                    4.0557613193e02,
                    np.nan,
                    -7.4604498097e01,
                    -4.3802723077e01,
                    -4.0557613193e02,
                ],
                [np.nan, np.nan, 7.5419970636e-01, -2.2727871868e01, np.nan],
                [np.nan, np.nan, np.nan, -7.1657702639e01, np.nan],
                [
                    6.2248620699e02,
                    -1.8416144299e02,
                    np.nan,
                    -1.7558752705e02,
                    -6.2248620699e02,
                ],
                [np.nan, -1.0646207700e01, np.nan, np.nan, np.nan],
                [np.nan, -3.9963319461e01, np.nan, np.nan, np.nan],
                [np.nan, -3.2226859413e01, np.nan, np.nan, np.nan],
                [np.nan, -2.0011396180e01, np.nan, np.nan, np.nan],
                [np.nan, -1.8789849856e01, np.nan, np.nan, np.nan],
                [np.nan, -5.2178782695e01, np.nan, np.nan, np.nan],
                [
                    -8.6951552411e01,
                    2.6820253146e00,
                    1.1988726047e01,
                    np.nan,
                    8.6951552411e01,
                ],
                [
                    -9.0848015208e01,
                    4.5442739110e00,
                    1.4191957607e01,
                    np.nan,
                    9.0848015208e01,
                ],
                [
                    -2.3435847246e01,
                    -3.6990146410e-01,
                    np.nan,
                    1.1825276005e01,
                    2.3435847246e01,
                ],
            ],
        )

        # Test pvals
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [np.nan, np.nan, 0.0, 0.0, np.nan],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [0.0, np.nan, 0.0, np.nan, 0.0],
                [np.nan, 0.0, 0.0, np.nan, np.nan],
                [np.nan, 0.0, 1.72248938e-03, np.nan, np.nan],
                [np.nan, 0.0, 0.0, np.nan, np.nan],
                [np.nan, 0.0, 2.79272485e-08, np.nan, np.nan],
                [np.nan, 0.0, 5.42905536e-02, np.nan, np.nan],
                [0.0, np.nan, np.nan, np.nan, 0.0],
                [0.0, np.nan, np.nan, np.nan, 0.0],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, 0.0, np.nan, np.nan],
                [np.nan, 0.0, 0.0, np.nan, np.nan],
                [np.nan, 0.0, 0.0, np.nan, np.nan],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [
                    4.87635166e-05,
                    np.nan,
                    1.32426479e-02,
                    1.30613577e-01,
                    4.87635166e-05,
                ],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0, np.nan],
                [np.nan, np.nan, 0.0, 0.0, np.nan],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [np.nan, np.nan, 0.0, 0.0, np.nan],
                [np.nan, np.nan, 0.0, 0.0, np.nan],
                [np.nan, 0.0, np.nan, 0.0, np.nan],
                [np.nan, 0.0, np.nan, 0.0, np.nan],
                [np.nan, 0.0, np.nan, 4.03912509e-01, np.nan],
                [np.nan, 0.0, np.nan, 0.0, np.nan],
                [0.0, np.nan, np.nan, np.nan, 0.0],
                [0.0, np.nan, np.nan, np.nan, 0.0],
                [0.0, 0.0, np.nan, 1.33151998e-06, 0.0],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [0.0, np.nan, 0.0, 1.08801856e-14, 0.0],
                [0.0, np.nan, 1.17087007e-10, 0.0, 0.0],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [0.0, np.nan, 0.0, 0.0, 0.0],
                [np.nan, np.nan, 4.50729314e-01, 0.0, np.nan],
                [np.nan, np.nan, np.nan, 0.0, np.nan],
                [0.0, 0.0, np.nan, 0.0, 0.0],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, np.nan, np.nan],
                [0.0, 7.31779221e-03, 0.0, np.nan, 0.0],
                [0.0, 5.51249381e-06, 0.0, np.nan, 0.0],
                [0.0, 7.11455911e-01, np.nan, 0.0, 0.0],
            ],
        )

    def it_calculates_std_dev_and_std_error_for_ca_as_0th_with_1_col_insertion(self):
        # Test for multi-cube when first cube represents a categorical-array
        slice_ = Cube(CR.CA_AS_0TH).partitions[0]

        # Test zscore for subtotal as 5 col (idx=4)
        assert slice_.inserted_column_idxs == (4,)

        expected_std_deviation = [
            [0.1438138, 0.0971119, 0.07778819, 0.04429081, 0.13140055],
            [0.13183193, 0.09031914, 0.08657918, 0.07214947, 0.14342128],
            [0.10411178, 0.08888663, 0.0992925, 0.09861821, 0.16423975],
            [0.12097241, 0.10277723, 0.09348917, 0.06583866, 0.15257805],
            [0.15660843, 0.08832746, 0.06573181, 0.03660272, 0.11559057],
            [0.14358222, 0.09374957, 0.07826393, 0.0510568, 0.1316554],
            [0.15095317, 0.09136303, 0.070725, 0.04385813, 0.12302966],
            [0.15833925, 0.08609381, 0.06506537, 0.03534449, 0.1131462],
            [0.10264886, 0.10381825, 0.10504664, 0.0772097, 0.16512601],
            [0.10637098, 0.1080826, 0.10306713, 0.06838287, 0.16283436],
            [0.15358343, 0.08689318, 0.0711406, 0.0428149, 0.11966825],
            [0.14398369, 0.09892083, 0.07708256, 0.04075179, 0.13121297],
            [0.10275022, 0.08522117, 0.08622352, 0.07652609, 0.14237039],
            [0.13992705, 0.09222109, 0.08306137, 0.05647197, 0.13555244],
            [0.15476904, 0.08871376, 0.06689759, 0.04142257, 0.11809825],
            [0.14775902, 0.09605302, 0.07393571, 0.03950121, 0.12690291],
            [0.15418843, 0.08797115, 0.0691184, 0.04162677, 0.11887152],
            [0.16025899, 0.08445515, 0.06303418, 0.03391463, 0.11033191],
            [0.1543502, 0.08985461, 0.06903707, 0.03677618, 0.11865693],
            [0.02599989, 0.01840707, 0.01844552, 0.01856041, 0.03198198],
            [0.03074091, 0.01649874, 0.01741808, 0.01336613, 0.02745688],
            [0.0326823, 0.01908739, 0.01433833, 0.00780873, 0.02511321],
            [0.03011286, 0.01745871, 0.01774056, 0.0131523, 0.02814446],
            [0.02978198, 0.01688089, 0.01725457, 0.01515538, 0.02849443],
            [0.09607475, 0.07294246, 0.05652185, 0.02821982, 0.09624841],
            [0.09707608, 0.07238248, 0.05509512, 0.02901145, 0.09523804],
            [0.05402286, 0.04253346, 0.03530445, 0.01829122, 0.05816783],
            [0.06103806, 0.03924989, 0.02961515, 0.01271383, 0.05075195],
            [0.05159276, 0.04359968, 0.0373864, 0.01867459, 0.06033057],
            [0.04492411, 0.04476663, 0.04149075, 0.02381036, 0.06543169],
            [0.09648667, 0.06681351, 0.06301186, 0.05578153, 0.10703966],
            [0.08265628, 0.04686263, 0.02891369, 0.01692282, 0.05755924],
            [0.07484997, 0.05048717, 0.0375374, 0.02454273, 0.06744155],
            [0.07762593, 0.04972686, 0.03512367, 0.02065617, 0.06421694],
            [0.02013512, 0.01832991, 0.01825243, 0.01231738, 0.02864311],
            [0.02497181, 0.01704798, 0.01491972, 0.0094517, 0.02454273],
            [0.01992288, 0.01863661, 0.01782027, 0.01282486, 0.02879101],
            [0.01912448, 0.01905024, 0.01801799, 0.0131523, 0.02932693],
            [0.03490145, 0.03809827, 0.03504303, 0.02348087, 0.056783],
            [0.04236703, 0.0393937, 0.02908456, 0.01597497, 0.05146958],
            [0.06818743, 0.0601436, 0.05228524, 0.02954336, 0.08481022],
            [0.0404559, 0.04022815, 0.0299715, 0.01717224, 0.05298188],
            [0.03685302, 0.03905127, 0.03368439, 0.02075881, 0.05554099],
            [0.04251685, 0.03046338, 0.02401768, 0.01160625, 0.04047337],
            [0.03013636, 0.03110707, 0.03144647, 0.02416467, 0.05036125],
            [0.03580178, 0.03244328, 0.02896261, 0.01658442, 0.04651485],
            [0.02583596, 0.03248687, 0.0349622, 0.0224318, 0.05268828],
            [0.03892435, 0.03459611, 0.02992424, 0.01995841, 0.04986854],
            [0.03959058, 0.03383109, 0.02987689, 0.02002928, 0.04934226],
            [0.06832121, 0.03395632, 0.03008935, 0.02416467, 0.05135983],
            [0.0701069, 0.03479997, 0.02930278, 0.01797862, 0.04888249],
            [0.03683382, 0.02801841, 0.02761116, 0.0202404, 0.04421101],
            [0.03978649, 0.02761116, 0.02594536, 0.01717224, 0.04157581],
            [0.04411505, 0.02616279, 0.02302395, 0.01231738, 0.03694883],
            [0.0403159, 0.02599989, 0.02471526, 0.02002928, 0.04106275],
            [0.05240648, 0.01793917, 0.01347177, 0.0077174, 0.02372096],
            [0.03378924, 0.02745688, 0.02017027, 0.00780873, 0.03494196],
            [0.10787742, 0.06604119, 0.05439993, 0.03477964, 0.09212279],
            [0.11358481, 0.06021379, 0.04467187, 0.02363121, 0.07847839],
            [0.03813537, 0.02561575, 0.02407659, 0.01704798, 0.03905127],
            [0.04528421, 0.02354112, 0.01733652, 0.00858707, 0.03046338],
            [0.04376139, 0.02410598, 0.01901301, 0.01104287, 0.03261729],
            [0.04256667, 0.02497181, 0.02044934, 0.0112338, 0.03416402],
            [0.04259985, 0.02505674, 0.02037994, 0.01104287, 0.03412258],
            [0.04511235, 0.02262047, 0.01817462, 0.01010423, 0.03071788],
            [0.01368061, 0.01260183, 0.01220173, 0.01044915, 0.02041467],
            [0.01347177, 0.01309829, 0.01254546, 0.00967411, 0.02055302],
            [0.00798825, 0.00996292, 0.01243194, 0.01058397, 0.01912448],
        ]
        expected_standard_error = [
            [
                1.71261529e-04,
                1.15646292e-04,
                9.26345266e-05,
                5.27439770e-05,
                1.56479129e-04,
            ],
            [
                1.56992845e-04,
                1.07557084e-04,
                1.03103339e-04,
                8.59196251e-05,
                1.70794095e-04,
            ],
            [
                1.23982136e-04,
                1.05851179e-04,
                1.18243064e-04,
                1.17440087e-04,
                1.95585895e-04,
            ],
            [
                1.44060721e-04,
                1.22392880e-04,
                1.11332140e-04,
                7.84043616e-05,
                1.81698479e-04,
            ],
            [
                1.86498090e-04,
                1.05185279e-04,
                7.82771205e-05,
                4.35885625e-05,
                1.37651717e-04,
            ],
            [
                1.70985742e-04,
                1.11642239e-04,
                9.32010664e-05,
                6.08013010e-05,
                1.56782626e-04,
            ],
            [
                1.79763492e-04,
                1.08800211e-04,
                8.42232880e-05,
                5.22287122e-05,
                1.46510613e-04,
            ],
            [
                1.88559241e-04,
                1.02525334e-04,
                7.74834799e-05,
                4.20902015e-05,
                1.34740825e-04,
            ],
            [
                1.22240014e-04,
                1.23632580e-04,
                1.25095424e-04,
                9.19456343e-05,
                1.96641304e-04,
            ],
            [
                1.26672519e-04,
                1.28710809e-04,
                1.22738108e-04,
                8.14341528e-05,
                1.93912277e-04,
            ],
            [
                1.82895742e-04,
                1.03477258e-04,
                8.47182081e-05,
                5.09863847e-05,
                1.42507656e-04,
            ],
            [
                1.71463840e-04,
                1.17800458e-04,
                9.17942330e-05,
                4.85295178e-05,
                1.56255748e-04,
            ],
            [
                1.22360715e-04,
                1.01486138e-04,
                1.02679797e-04,
                9.11315487e-05,
                1.69542632e-04,
            ],
            [
                1.66632961e-04,
                1.09822035e-04,
                9.89141272e-05,
                6.72499863e-05,
                1.61423438e-04,
            ],
            [
                1.84307642e-04,
                1.05645307e-04,
                7.96653989e-05,
                4.93283115e-05,
                1.40638011e-04,
            ],
            [
                1.75959708e-04,
                1.14385309e-04,
                8.80467820e-05,
                4.70402549e-05,
                1.51123085e-04,
            ],
            [
                1.83616216e-04,
                1.04760967e-04,
                8.23100610e-05,
                4.95714870e-05,
                1.41558856e-04,
            ],
            [
                1.90845380e-04,
                1.00573926e-04,
                7.50646346e-05,
                4.03874384e-05,
                1.31389417e-04,
            ],
            [
                1.83808856e-04,
                1.07003898e-04,
                8.22132075e-05,
                4.37951370e-05,
                1.41303320e-04,
            ],
            [
                3.09621236e-05,
                2.19201655e-05,
                2.19659601e-05,
                2.21027740e-05,
                3.80859365e-05,
            ],
            [
                3.66079922e-05,
                1.96476209e-05,
                2.07424220e-05,
                1.59171371e-05,
                3.26971890e-05,
            ],
            [
                3.89199119e-05,
                2.27303381e-05,
                1.70748868e-05,
                9.29906911e-06,
                2.99062126e-05,
            ],
            [
                3.58600840e-05,
                2.07908144e-05,
                2.11264537e-05,
                1.56624909e-05,
                3.35160002e-05,
            ],
            [
                3.54660480e-05,
                2.01027102e-05,
                2.05477115e-05,
                1.80478762e-05,
                3.39327671e-05,
            ],
            [
                1.14411188e-04,
                8.68639644e-05,
                6.73093872e-05,
                3.36057439e-05,
                1.14617998e-04,
            ],
            [
                1.15603636e-04,
                8.61971110e-05,
                6.56103487e-05,
                3.45484618e-05,
                1.13414785e-04,
            ],
            [
                6.43334447e-05,
                5.06512312e-05,
                4.20425128e-05,
                2.17822031e-05,
                6.92695057e-05,
            ],
            [
                7.26875419e-05,
                4.67409607e-05,
                3.52673731e-05,
                1.51403462e-05,
                6.04382624e-05,
            ],
            [
                6.14395424e-05,
                5.19209383e-05,
                4.45218202e-05,
                2.22387450e-05,
                7.18450251e-05,
            ],
            [
                5.34981395e-05,
                5.33106071e-05,
                4.94095039e-05,
                2.83547141e-05,
                7.79197206e-05,
            ],
            [
                1.14901732e-04,
                7.95652646e-05,
                7.50380518e-05,
                6.64277687e-05,
                1.27468819e-04,
            ],
            [
                9.84317316e-05,
                5.58066420e-05,
                3.44320383e-05,
                2.01526406e-05,
                6.85447613e-05,
            ],
            [
                8.91355294e-05,
                6.01229465e-05,
                4.47016351e-05,
                2.92268519e-05,
                8.03131733e-05,
            ],
            [
                9.24413073e-05,
                5.92175245e-05,
                4.18272374e-05,
                2.45985227e-05,
                7.64731272e-05,
            ],
            [
                2.39780254e-05,
                2.18282876e-05,
                2.17360208e-05,
                1.46682204e-05,
                3.41098177e-05,
            ],
            [
                2.97378311e-05,
                2.03016942e-05,
                1.77672398e-05,
                1.12556185e-05,
                2.92268519e-05,
            ],
            [
                2.37252755e-05,
                2.21935140e-05,
                2.12213744e-05,
                1.52725572e-05,
                3.42859510e-05,
            ],
            [
                2.27745014e-05,
                2.26860886e-05,
                2.14568369e-05,
                1.56624909e-05,
                3.49241490e-05,
            ],
            [
                4.15626013e-05,
                4.53695581e-05,
                4.17312012e-05,
                2.79623371e-05,
                6.76203745e-05,
            ],
            [
                5.04530344e-05,
                4.69122212e-05,
                3.46355218e-05,
                1.90238891e-05,
                6.12928521e-05,
            ],
            [
                8.12014044e-05,
                7.16223719e-05,
                6.22641899e-05,
                3.51818821e-05,
                1.00996756e-04,
            ],
            [
                4.81771528e-05,
                4.79059369e-05,
                3.56917460e-05,
                2.04496602e-05,
                6.30937896e-05,
            ],
            [
                4.38866341e-05,
                4.65044413e-05,
                4.01132586e-05,
                2.47207502e-05,
                6.61413242e-05,
            ],
            [
                5.06314467e-05,
                3.62774993e-05,
                2.86015997e-05,
                1.38213682e-05,
                4.81979520e-05,
            ],
            [
                3.58880633e-05,
                3.70440459e-05,
                3.74482174e-05,
                2.87766469e-05,
                5.99730017e-05,
            ],
            [
                4.26347686e-05,
                3.86352719e-05,
                3.44902993e-05,
                1.97496592e-05,
                5.53924882e-05,
            ],
            [
                3.07669080e-05,
                3.86871811e-05,
                4.16349424e-05,
                2.67130402e-05,
                6.27441588e-05,
            ],
            [
                4.63532967e-05,
                4.11989810e-05,
                3.56354560e-05,
                2.37675874e-05,
                5.93862473e-05,
            ],
            [
                4.71466824e-05,
                4.02879536e-05,
                3.55790766e-05,
                2.38519858e-05,
                5.87595288e-05,
            ],
            [
                8.13607266e-05,
                4.04370887e-05,
                3.58320828e-05,
                2.87766469e-05,
                6.11621622e-05,
            ],
            [
                8.34872196e-05,
                4.14417508e-05,
                3.48953941e-05,
                2.14099518e-05,
                5.82120069e-05,
            ],
            [
                4.38637778e-05,
                3.33658892e-05,
                3.28809147e-05,
                2.41034049e-05,
                5.26489410e-05,
            ],
            [
                4.73799760e-05,
                3.28809147e-05,
                3.08971891e-05,
                2.04496602e-05,
                4.95108059e-05,
            ],
            [
                5.25346691e-05,
                3.11561128e-05,
                2.74182094e-05,
                1.46682204e-05,
                4.40007364e-05,
            ],
            [
                4.80104333e-05,
                3.09621236e-05,
                2.94323107e-05,
                2.38519858e-05,
                4.88998242e-05,
            ],
            [
                6.24085725e-05,
                2.13629638e-05,
                1.60429435e-05,
                9.19031085e-06,
                2.82482439e-05,
            ],
            [
                4.02381185e-05,
                3.26971890e-05,
                2.40198914e-05,
                9.29906911e-06,
                4.16108428e-05,
            ],
            [
                1.28466476e-04,
                7.86455426e-05,
                6.47824871e-05,
                4.14175381e-05,
                1.09704982e-04,
            ],
            [
                1.35263148e-04,
                7.17059491e-05,
                5.31977678e-05,
                2.81413700e-05,
                9.34564548e-05,
            ],
            [
                4.54137350e-05,
                3.05046720e-05,
                2.86717471e-05,
                2.03016942e-05,
                4.65044413e-05,
            ],
            [
                5.39269690e-05,
                2.80340878e-05,
                2.06452966e-05,
                1.02259614e-05,
                3.62774993e-05,
            ],
            [
                5.21135116e-05,
                2.87067564e-05,
                2.26417526e-05,
                1.31504687e-05,
                3.88424905e-05,
            ],
            [
                5.06907769e-05,
                2.97378311e-05,
                2.43522243e-05,
                1.33778427e-05,
                4.06844267e-05,
            ],
            [
                5.07302915e-05,
                2.98389743e-05,
                2.42695684e-05,
                1.31504687e-05,
                4.06350801e-05,
            ],
            [
                5.37223072e-05,
                2.69377237e-05,
                2.16433602e-05,
                1.20326855e-05,
                3.65805656e-05,
            ],
            [
                1.62916399e-05,
                1.50069698e-05,
                1.45305082e-05,
                1.24434309e-05,
                2.43109315e-05,
            ],
            [
                1.60429435e-05,
                1.55981794e-05,
                1.49398348e-05,
                1.15204683e-05,
                2.44756839e-05,
            ],
            [
                9.51285525e-06,
                1.18644047e-05,
                1.48046509e-05,
                1.26039800e-05,
                2.27745014e-05,
            ],
        ]

        np.testing.assert_almost_equal(slice_.table_std_dev, expected_std_deviation)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_standard_error)

    def it_provide_residual_test_stats_including_hs(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]
        np.testing.assert_array_equal(slice_.pvals, slice_.residual_test_stats[0])
        np.testing.assert_array_equal(slice_.zscore, slice_.residual_test_stats[1])

        assert slice_.residual_test_stats.shape == (2, 6, 7)
