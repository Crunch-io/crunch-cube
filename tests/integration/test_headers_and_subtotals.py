# encoding: utf-8

"""Integrations tests for headings and subtotals behaviors."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR, NA
from ..util import load_python_expression


class TestHeadersAndSubtotals(object):
    """Legacy unit-test suite for inserted rows and columns."""

    def test_headings_econ_blame_one_subtotal(self):
        strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
        assert strand.row_labels.tolist() == [
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        ]

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        strand = Cube(CR.ECON_BLAME_WITH_HS, transforms=transforms).partitions[0]

        assert strand.row_labels.tolist() == [
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        ]

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        strand = Cube(CR.CAT_HS_MT).partitions[0]
        assert strand.row_labels.tolist() == [
            "President O",
            "Republicans",
            "Test New He",
            "Both",
            "Neither",
            "Not sure",
            "Test Headin",
        ]

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        strand = Cube(CR.CAT_HS_MT, transforms=transforms).partitions[0]
        assert strand.row_labels.tolist() == [
            "President O",
            "Republicans",
            "Both",
            "Neither",
            "Not sure",
        ]

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
        strand = Cube(CR.CAT_HS_MT).partitions[0]
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
        strand = Cube(CR.CAT_HS_MT).partitions[0]
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
        strand = Cube(CR.CAT_HS_MT).partitions[0]
        assert strand.rows_base.tolist() == [285, 396, 681, 242, 6, 68, 74]

    def test_1D_subtotals_rows_dimension_fills(self):
        strand = Cube(CR.CAT_HS_MT).partitions[0]
        assert strand.rows_dimension_fills == (
            None,
            None,
            "#32a852",
            None,
            None,
            None,
            "#7532a8",
        )

    def test_1D_subtotals_inserted_row_idxs(self):
        strand = Cube(CR.CAT_HS_MT).partitions[0]
        assert strand.inserted_row_idxs == (2, 6)
        assert strand.diff_row_idxs == ()

    def test_1D_means_mr_subtotals_hidden(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        strand = Cube(CR.MR_MEAN_FILT_WGTD, transforms=transforms).partitions[0]
        assert strand.inserted_row_idxs == ()

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
        assert slice_.row_labels.tolist() == [
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        ]
        assert slice_.column_labels.tolist() == [
            "Very liberal",
            "Liberal",
            "Moderate",
            "Conservative",
            "Very Conservative",
            "Not sure",
        ]

    def test_labels_on_2d_cube_with_hs_on_both_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
        assert slice_.row_labels.tolist() == [
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        ]
        assert slice_.column_labels.tolist() == [
            "Very liberal",
            "Liberal",
            "Moderate",
            "Test 2nd dim Heading",
            "Conservative",
            "Very Conservative",
            "Not sure",
        ]

    def test_aliases_on_2d_cube_with_hs_on_both_dim(self):
        slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]

        assert slice_.column_aliases.tolist() == ["", "", "", "", "", "", ""]
        assert slice_.row_aliases.tolist() == ["", "", "", "", "", ""]

    def test_labels_on_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, transforms=transforms
        ).partitions[0]
        assert slice_.row_labels.tolist() == [
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        ]
        assert slice_.column_labels.tolist() == [
            "Very liberal",
            "Liberal",
            "Moderate",
            "Conservative",
            "Very Conservative",
            "Not sure",
        ]

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
        assert slice_.row_labels.tolist() == [
            "ca_subvar_1",
            "ca_subvar_2",
            "ca_subvar_3",
        ]
        assert slice_.column_labels.tolist() == [
            "a",
            "b",
            "Test A and B combined",
            "c",
            "d",
        ]

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
        assert slice_.row_labels.tolist() == [
            "Serious net",
            "Very serious",
            "Somewhat serious",
            "Not very serious",
            "Not at all serious",
            "Not sure",
        ]
        assert slice_.column_labels.tolist() == [
            "Sexual assaults that go unreported or unpunished",
            "False accusations of sexual assault",
        ]

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
        strand = Cube(CR.FRUIT_HS_TOP_BOTTOM).partitions[0]
        assert strand.row_labels.tolist() == [
            "TOP",
            "rambutan",
            "MIDDLE",
            "satsuma",
            "BOTTOM",
        ]

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
        strand = Cube(CR.MISSING_CAT_HS).partitions[0]
        assert strand.row_labels.tolist() == [
            "Whites",
            "White college women voters",
            "White non-college women voters",
            "White college men voters",
            "White non-college men voters",
            "Black voters",
            "Latino and other voters",
        ]

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
        assert slice_.column_labels.tolist() == [
            "Whites",
            "White college women voters",
            "White non-college women voters",
            "White college men voters",
            "White non-college men voters",
            "Black voters",
            "Latino and other voters",
        ]

    def it_calculate_col_residuals_for_subtotals(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_std_dev,
            load_python_expression("col-std-dev-cat-x-cat-hs-2rows-1col"),
        )
        np.testing.assert_almost_equal(
            slice_.column_std_err,
            load_python_expression("col-std-err-cat-x-cat-hs-2rows-1col"),
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("col-per-moe-cat-x-cat-hs-2rows-1col"),
        )

    def it_computes_residuals_for_subtotals_1col_2rows(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]

        np.testing.assert_almost_equal(
            slice_.zscores,
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
        assert slice_.diff_column_idxs == ()
        assert len(slice_.inserted_column_idxs) == 1
        assert slice_.inserted_row_idxs == (2, 5)
        assert len(slice_.inserted_row_idxs) == 2
        assert slice_.row_proportions.shape == slice_.zscores.shape
        # Test zscores for 1 col and 2 rows insertions
        np.testing.assert_almost_equal(
            slice_.zscores,
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

        # Test zscores for 2 rows and 2 cols insertions (1 col at bottom)
        np.testing.assert_almost_equal(
            slice_.zscores,
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
        transforms = {
            "columns_dimension": {
                "insertions": [
                    {
                        "anchor": 4,
                        "args": [0, 2],
                        "function": "subtotal",
                        "name": "A",
                    }
                ],
            },
        }
        slice_ = Cube(CR.CAT_4_X_CAT_4, transforms=transforms).partitions[0]

        # Test zscores for 1 column insertion
        assert slice_.inserted_column_idxs == (4,)
        assert slice_.diff_column_idxs == ()
        assert len(slice_.inserted_column_idxs) == 1
        assert slice_.zscores.tolist() == [
            pytest.approx([-0.12293906, 0.7330578, -1.284731, 0.79903932, 0.73305780]),
            pytest.approx([1.059370617, -0.529932, -0.8915899, 0.3826800, -0.5299318]),
            pytest.approx(
                [-0.93578132, 0.6564532, -0.27774127, 0.6211281, 0.656453177]
            ),
            pytest.approx(
                [-0.04138745, -0.7984189, 2.400749057, -1.7628186, -0.798419]
            ),
        ]

        # Test pvals for 1 column insertion
        assert slice_.pvals.tolist() == [
            pytest.approx([0.9021553, 0.4635231, 0.19888631, 0.42426760, 0.463523177]),
            pytest.approx([0.2894310, 0.5961592, 0.37261274, 0.70195702, 0.596159223]),
            pytest.approx([0.3493857, 0.5115325, 0.78121097, 0.53451530, 0.511532585]),
            pytest.approx([0.9669870, 0.4246274, 0.01636155, 0.07793109, 0.424627429]),
        ]

        # Test std deviation for 1 column insertion
        assert slice_.table_std_dev.tolist() == [
            pytest.approx([0.2232969, 0.2232969, 0.2156008, 0.2377652, 0.2232969]),
            pytest.approx([0.2754385, 0.2232969, 0.2575394, 0.2575394, 0.2232969]),
            pytest.approx([0.2232969, 0.2377652, 0.2575394, 0.2511773, 0.2377652]),
            pytest.approx([0.2445921, 0.2075515, 0.3068922, 0.1991062, 0.2075515]),
        ]

        # Test std error for 1 column insertion
        slice_.table_std_err.tolist() == [
            pytest.approx([0.0268343, 0.0268343, 0.0259094, 0.028573, 0.0268343]),
            pytest.approx([0.0331003, 0.0268343, 0.0309493, 0.0309493, 0.0268343]),
            pytest.approx([0.0268343, 0.028573, 0.0309493, 0.0301848, 0.028573]),
            pytest.approx([0.0293934, 0.0249421, 0.0368802, 0.0239272, 0.0249421]),
        ]

        # Test MoE for 1 column insertion
        assert slice_.table_proportions_moe.tolist() == [
            pytest.approx([0.02683428, 0.02683428, 0.02590941, 0.02857299, 0.02683428]),
            pytest.approx([0.0331003, 0.02683428, 0.03094931, 0.03094931, 0.02683428]),
            pytest.approx([0.02683428, 0.02857299, 0.03094931, 0.03018476, 0.02857299]),
            pytest.approx([0.0293934, 0.02494211, 0.03688019, 0.02392721, 0.02494211]),
        ]

        # Test row std dev
        assert slice_.row_std_dev.tolist() == [
            pytest.approx([0.43045067, 0.43045067, 0.4195881, 0.44934205, 0.43045067]),
            pytest.approx([0.4570685, 0.39165883, 0.43684405, 0.43684405, 0.39165883]),
            pytest.approx([0.40656234, 0.42635394, 0.45073638, 0.44326097, 0.42635394]),
            pytest.approx([0.4330127, 0.38122004, 0.49215296, 0.36823482, 0.38122004]),
        ]

        # Test row std err
        assert slice_.row_std_err.tolist() == [
            pytest.approx([0.05701458, 0.05701458, 0.0555758, 0.05951681, 0.05701458]),
            pytest.approx([0.05313313, 0.04552941, 0.05078209, 0.05078209, 0.04552941]),
            pytest.approx([0.0496695, 0.05208742, 0.05506621, 0.05415295, 0.05208742]),
            pytest.approx([0.0525105, 0.04622972, 0.05968231, 0.04465503, 0.04622972]),
        ]

        # Test row MoE
        assert slice_.row_proportions_moe.tolist() == [
            pytest.approx([0.11174653, 0.11174653, 0.10892657, 0.1166508, 0.11174653]),
            pytest.approx([0.10413903, 0.08923601, 0.09953107, 0.09953107, 0.08923601]),
            pytest.approx([0.09735042, 0.10208947, 0.1079278, 0.10613782, 0.10208947]),
            pytest.approx([0.1029187, 0.09060859, 0.11697518, 0.08752225, 0.09060859]),
        ]

        slice_ = Cube(CR.CA_X_CAT_HS).partitions[0]

        # Test zscores for 2 columns insertion bottom and interleaved
        assert slice_.inserted_column_idxs == (3, 6)
        assert slice_.diff_column_idxs == ()
        assert len(slice_.inserted_column_idxs) == 2
        np.testing.assert_almost_equal(
            slice_.zscores,
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
            slice_.column_std_dev,
            [
                [0.0, 0.0, 0.0, 0.47140452, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.47140452, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        # Test col standard error for 2 columns insertion bottom and interleaved
        np.testing.assert_almost_equal(
            slice_.column_std_err,
            [
                [0.0, 0.0, 0.0, 0.27216553, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.27216553, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

    def it_calculates_residuals_for_rows_insertion(self):
        transforms = {"columns_dimension": {"insertions": {}}}
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]

        # Test zscores for 1 row insertion
        assert slice_.inserted_row_idxs == (1,)
        np.testing.assert_almost_equal(
            slice_.zscores,
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

        # Test MoE for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.table_proportions_moe,
            [
                [0.09474253, 0.08514162, 0.06401147, 0.0, 0.02101389],
                [0.05485481, 0.06910627, 0.05775633, 0.0, 0.03000171],
                [0.02164686, 0.0, 0.02101389, 0.0, 0.0],
                [0.0367142, 0.04195117, 0.03043819, 0.0, 0.03000171],
                [0.0367142, 0.05775633, 0.046236, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02101389, 0.02101389, 0.02101389, 0.0, 0.0],
            ],
        )

        # Test col std deviation for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.column_std_dev,
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
            slice_.column_std_err,
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
        # Test col MoE for 1 row insertion
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            [
                [0.13514267, 0.16591882, 0.22488171, np.nan, 0.53311238],
                [0.12890849, 0.16334272, 0.22234576, np.nan, 0.53311238],
                [0.05399008, 0.0, 0.09938306, np.nan, 0.0],
                [0.0899374, 0.1108349, 0.14036177, np.nan, 0.53311238],
                [0.0899374, 0.14521917, 0.19752408, np.nan, 0.0],
                [0.0, 0.0, 0.0, np.nan, 0.0],
                [0.05243851, 0.0573464, 0.09938306, np.nan, 0.0],
            ],
        )

        slice_ = Cube(CR.FOOD_GROUP_X_SHAPE_OF_PASTA_2ROWS_INSERTION).partitions[0]

        # Test zscores for 2 rows insertions (interleaved and bottom)
        assert slice_.inserted_row_idxs == (2, 5)
        assert len(slice_.inserted_row_idxs) == 2
        np.testing.assert_almost_equal(
            slice_.zscores,
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
        assert slice_.diff_column_idxs == ()
        assert slice_.inserted_row_idxs == ()
        assert slice_.diff_row_idxs == ()

        # Test szcores for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.zscores,
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

        # Test MoE for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.table_proportions_moe,
            [
                [0.02741442, 0.0, 0.0, 0.02014736, 0.02341773, 0.01196273, 0.01479991],
                [0.02784747, 0.02062471, 0.02419167, 0.0, 0.0, 0.01413778, 0.01455636],
            ],
        )

        # Test col std dev for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.column_std_dev,
            [
                [0.49962497, 0.0, 0.0, 0.0, 0.0, 0.49223325, 0.49991932],
                [0.49962497, 0.0, 0.0, 0.0, 0.0, 0.49223325, 0.49991932],
            ],
        )

        # Test col std err for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.column_std_err,
            [
                [0.01686153, 0.0, 0.0, 0.0, 0.0, 0.04300662, 0.03868492],
                [0.01686153, 0.0, 0.0, 0.0, 0.0, 0.04300662, 0.03868492],
            ],
        )

        # Test MoE err for 1 column insertion at left
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            [
                [0.03304798, 0.0, 0.0, 0.0, 0.0, 0.08429142, 0.07582105],
                [0.03304798, 0.0, 0.0, 0.0, 0.0, 0.08429142, 0.07582105],
            ],
        )

    def it_calculates_residuals_for_cat_x_num_hs_pruned_with_3_rows_insertions(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE, transforms=transforms).partitions[0]

        # Test zscores for 3 rows insertions
        assert slice_.inserted_row_idxs == (0, 1, 3)
        np.testing.assert_almost_equal(
            slice_.zscores, np.full(slice_.row_proportions.shape, np.nan)
        )

        # Test pvals for 3 rows insertions
        np.testing.assert_almost_equal(
            slice_.pvals, [[np.nan], [np.nan], [np.nan], [np.nan]]
        )

        slice_ = Cube(CR.CAT_X_NUM_HS_PRUNE).partitions[0]

        # Test zscores for 3 rows insertions (1 at left)
        assert slice_.inserted_row_idxs == (0, 3, 7)
        np.testing.assert_almost_equal(
            slice_.zscores, np.tile(np.nan, slice_.row_proportions.shape)
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

        assert slice_.zscores == pytest.approx(np.full((8, 11), np.nan), nan_ok=True)
        assert slice_.pvals == pytest.approx(np.full((8, 11), np.nan), nan_ok=True)
        # Test std deviation
        assert slice_.table_std_dev == pytest.approx(
            np.array(
                [
                    [
                        0.081262,
                        0.08588701,
                        0.13015373,
                        0.12734995,
                        0.23544673,
                        0.2690963,
                        0.37873219,
                        0.43005992,
                        0.34113756,
                        0.43957718,
                        0.47966671,
                    ],
                    [
                        0.10013911,
                        0.07255806,
                        0.08302955,
                        0.12517688,
                        0.19584563,
                        0.2317994,
                        0.32932468,
                        0.43602658,
                        0.38958534,
                        0.45787225,
                        0.43808124,
                    ],
                    [
                        0.1041408,
                        0.0815162,
                        0.09991484,
                        0.1330253,
                        0.21966854,
                        0.25783111,
                        0.35229456,
                        0.42574735,
                        0.36589606,
                        0.45355357,
                        0.46387363,
                    ],
                    [
                        0.08957013,
                        0.10050334,
                        0.08135815,
                        0.08345183,
                        0.25031353,
                        0.2605118,
                        0.38065664,
                        0.41428257,
                        0.37500461,
                        0.43998456,
                        0.47639197,
                    ],
                    [
                        0.11100304,
                        0.06393387,
                        0.05923261,
                        0.06642898,
                        0.18675263,
                        0.2423074,
                        0.34440562,
                        0.40734247,
                        0.40203944,
                        0.46886829,
                        0.43911833,
                    ],
                    [
                        0.12595376,
                        0.05266939,
                        0.07321936,
                        0.12012933,
                        0.18908393,
                        0.25423682,
                        0.34011811,
                        0.42328438,
                        0.41112117,
                        0.44482086,
                        0.44852627,
                    ],
                    [
                        0.14285686,
                        0.08800165,
                        0.09103445,
                        0.14925503,
                        0.27456606,
                        0.272829,
                        0.35168085,
                        0.43620856,
                        0.34669406,
                        0.42533468,
                        0.48212798,
                    ],
                    [
                        0.10491832,
                        0.08302194,
                        0.10208267,
                        0.1207337,
                        0.24193425,
                        0.21970041,
                        0.3605953,
                        0.43200733,
                        0.38777077,
                        0.43701607,
                        0.4626228,
                    ],
                ]
            )
        )
        # Test standard error
        # TODO: When table_std_err is consolidated to not use the legacy cube matrix
        # this will change because it will no longer use table proportions that sum
        # across the cat array subvariables
        assert slice_.table_std_err.tolist() == [
            [
                0.002503208781014363,
                0.002645678243060755,
                0.0040092780892363455,
                0.003922909847106257,
                0.0072527419278186935,
                0.008289289005529856,
                0.011666531848262812,
                0.013247640073027834,
                0.01050846041311712,
                0.013540811540393714,
                0.01477573644938576,
            ],
            [
                0.0019273162337518283,
                0.0013964805514394677,
                0.001598018998385834,
                0.0024092028696857505,
                0.0037693210960351425,
                0.00446130138336647,
                0.006338310842225576,
                0.00839193712675601,
                0.007498110946252939,
                0.008812387426582706,
                0.008431481874316278,
            ],
            [
                0.002756108739438326,
                0.0021573439172278665,
                0.0026442678627658475,
                0.003520543248205185,
                0.005813575511407028,
                0.006823556245060651,
                0.00932355185980517,
                0.011267495761810778,
                0.009683518521260705,
                0.012003393250288858,
                0.012276516022862864,
            ],
            [
                0.004589898287231243,
                0.005150155566252456,
                0.004169086681230891,
                0.0042763742102681634,
                0.012826972262705693,
                0.013349568564020172,
                0.0195062259101231,
                0.021229340394633815,
                0.019216595225264612,
                0.02254640327800877,
                0.024412051063600258,
            ],
            [
                0.0034624631242383547,
                0.0019942578104980313,
                0.0018476137953101972,
                0.002072086496489134,
                0.005825282816207957,
                0.007558175427832714,
                0.010742875183553413,
                0.012706033247740327,
                0.012540618553206819,
                0.014625177929040122,
                0.013697202104911689,
            ],
            [
                0.0051135270848222765,
                0.0021382953353567386,
                0.002972592388901149,
                0.00487706395285152,
                0.007676513636638379,
                0.010321619788991376,
                0.013808266934404973,
                0.017184688016671,
                0.01669088084921974,
                0.018059035931767334,
                0.018209469612173172,
            ],
            [
                0.009666017803693546,
                0.005954390466730202,
                0.006159596479321062,
                0.010098932132962828,
                0.018577759763840025,
                0.018460226352068138,
                0.0237955199974899,
                0.02951485571393499,
                0.02345810290099581,
                0.02877910440134688,
                0.03262186790050417,
            ],
            [
                0.006441513689856468,
                0.005097174143472906,
                0.006267417394644421,
                0.007412506522596238,
                0.014853676131421963,
                0.013488618023323644,
                0.022138931497005063,
                0.026523308951121018,
                0.023807383110825458,
                0.0268308230591262,
                0.028402961127149476,
            ],
        ]

    def it_provide_residual_test_stats_including_hs(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]
        np.testing.assert_array_equal(slice_.pvals, slice_.residual_test_stats[0])
        np.testing.assert_array_equal(slice_.zscores, slice_.residual_test_stats[1])

        assert slice_.residual_test_stats.shape == (2, 6, 7)


class DescribeIntegrated_SubtotalDifferences(object):
    """TDD driver(s) for Subtotal Difference insertions."""

    def it_computes_measures_for_1D_cat_with_subdiffs(self):
        strand = Cube(
            CR.CAT,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [4, 5]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                }
            },
        ).partitions[0]

        assert strand.counts[0] == 81
        assert strand.table_proportions[0] == pytest.approx(0.1184210)
        assert strand.diff_row_idxs == (0,)

    def it_computes_measures_for_cat_x_cat_with_subdiffs_on_both(self):
        slice_ = Cube(
            CR.CAT_4_X_CAT_4,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == pytest.approx(
            [np.nan, -8, 0, -6, -3], nan_ok=True
        )
        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [np.nan, 0, 8, -2, 5], nan_ok=True
        )
        assert slice_.columns_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.columns_margin_proportion[0] == pytest.approx(11 / 266)
        assert slice_.rows_margin_proportion[0] == pytest.approx(-17 / 266)
        assert slice_.columns_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[0, 1] == slice_.column_weighted_bases[1, 1]
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_weighted_bases[1, 0] == slice_.row_weighted_bases[1, 1]
        assert slice_.column_proportions[0, :] == pytest.approx(
            [np.nan, -0.119403, 0, -0.0759494, -0.046875], nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions_moe[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions_moe[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.column_std_dev[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_std_dev[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.column_std_err[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_std_err[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.10810811, -0.02985075, 0.07352941], nan_ok=True
        )
        assert slice_.row_proportions_moe[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions_moe[:, 0] == pytest.approx(
            [np.nan, 0, 0.07074854, np.nan, 0.06203546], nan_ok=True
        )
        assert slice_.row_std_dev[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_std_dev[:, 0] == pytest.approx(
            [np.nan, 0, 0.3105169, np.nan, 0.26100352], nan_ok=True
        )
        assert slice_.row_std_err[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_std_err[:, 0] == pytest.approx(
            [np.nan, 0, 0.03609686, np.nan, 0.03165133], nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [np.nan, -0.03007519, 0, -0.02255639, -0.0112782], nan_ok=True
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.03007519, -0.0075188, 0.01879699], nan_ok=True
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.zscores[0, :] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(5, np.nan), nan_ok=True)

    def it_computes_measures_for_cat_x_cat_with_subdiffs_and_subtot_on_both(self):
        slice_ = Cube(
            CR.CAT_4_X_CAT_4,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                        {
                            "function": "subtotal",
                            "args": [3, 4],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                        {
                            "function": "subtotal",
                            "args": [3, 4],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == pytest.approx(
            [np.nan, -8, 0, -6, -3, -9], nan_ok=True
        )
        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [np.nan, 0, 8, -2, 5, 3], nan_ok=True
        )
        assert slice_.columns_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.columns_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[0, 1] == slice_.column_weighted_bases[1, 1]
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_weighted_bases[1, 0] == slice_.row_weighted_bases[1, 1]
        assert slice_.column_proportions[0, :] == pytest.approx(
            [np.nan, -0.119403, 0, -0.0759494, -0.046875, -0.06293706], nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.10810811, -0.02985075, 0.07352941, 0.02222222], nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [np.nan, -0.03007519, 0, -0.02255639, -0.0112782, -0.03383459], nan_ok=True
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.03007519, -0.0075188, 0.01879699, 0.0112782], nan_ok=True
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.zscores[0, :] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(6, np.nan), nan_ok=True)

    def it_computes_measures_for_ca_with_subdiff(self):
        slice_ = Cube(
            CR.CA_CAT_X_CA_SUBVAR,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [0],
                            "kwargs": {"negative": [4]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == [-178, -495, 0]
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(3, np.nan), nan_ok=True
        )
        assert slice_.column_proportions[0, :] == pytest.approx(
            [-0.10847044, -0.30201342, np.nan], nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(3, np.nan), nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [-0.10847044, -0.30201342, np.nan], nan_ok=True
        )
        assert slice_.zscores[0, :] == pytest.approx(np.full(3, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(3, np.nan), nan_ok=True)

    def it_computes_measures_for_mr_x_cat_subdiff(self):
        slice_ = Cube(
            CR.MR_X_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [2],
                            "kwargs": {"negative": [4]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [1.9215376, -12.3047603, -31.4956882, -88.6847375, -56.4466419]
        )
        assert slice_.columns_margin[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.columns_base[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [0.06074756, -0.17396625, -0.25044458, -0.2417213, -0.14981897]
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [0.0108952, -0.05820039, -0.12713165, -0.19403687, -0.11960763]
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)

    def it_computes_scale_median_for_cat_with_subdiff_x_cat_with_subdiff(self):
        slice_ = Cube(
            CR.CAT_HS_MT_X_CAT_HS_MT,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
            },
        ).partitions[0]

        assert slice_.columns_scale_median == pytest.approx(
            [np.nan, 1, 1, 1, np.nan, 3], nan_ok=True
        )

        assert slice_.rows_scale_median == pytest.approx(
            [np.nan, 2, 1, 2, 2, np.nan, 2], nan_ok=True
        )

    def it_computes_sum_for_numarray_with_subdiffs_and_subtot_on_columns(self):
        slice_ = Cube(
            NA.NUM_ARR_SUM_GROUPED_BY_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "subdiff",
                        },
                        {
                            "function": "subtotal",
                            "args": [1, 2],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]
        assert slice_.sums == pytest.approx(
            np.array(
                [
                    [np.nan, 4.0, 3.0, 7.0],
                    [np.nan, 3.0, 0.0, 3.0],
                    [np.nan, 2.0, 3.0, 5.0],
                ]
            ),
            nan_ok=True,
        )

        # pruning
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]
        assert slice_.sums == pytest.approx(
            np.array(
                [
                    [4.0, 3.0],
                    [3.0, 0.0],
                    [2.0, 3.0],
                ]
            )
        )

    def it_computes_share_of_sum_for_numarray_with_subdiffs_and_subtot_on_columns(self):
        slice_ = Cube(
            NA.NUM_ARR_SUM_GROUPED_BY_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "subdiff",
                        },
                        {
                            "function": "subtotal",
                            "args": [1, 2],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]
        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.44444444, 0.5, 0.46666667],
                    [np.nan, 0.33333333, 0.0, 0.2],
                    [np.nan, 0.22222222, 0.5, 0.33333333],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.57142857, 0.42857143, 1.0],
                    [np.nan, 1.0, 0.0, 1.0],
                    [np.nan, 0.4, 0.6, 1.0],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.26666667, 0.2, 0.46666667],
                    [np.nan, 0.2, 0.0, 0.2],
                    [np.nan, 0.13333333, 0.2, 0.33333333],
                ]
            ),
            nan_ok=True,
        )

        # pruning
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]
        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    [0.4444444, 0.5],
                    [0.3333333, 0.0],
                    [0.2222222, 0.5],
                ]
            )
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    [0.57142857, 0.42857143],
                    [1.0, 0.0],
                    [0.4, 0.6],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    [0.26666667, 0.2],
                    [0.2, 0.0],
                    [0.13333333, 0.2],
                ]
            ),
            nan_ok=True,
        )

    def it_computes_diff_indexes_for_cat_x_cat_with_subdiffs_on_both(self):
        slice_ = Cube(
            CR.CAT_4_X_CAT_4,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
            },
        ).partitions[0]

        assert slice_.diff_row_idxs == (0,)
        assert slice_.diff_column_idxs == (0,)
