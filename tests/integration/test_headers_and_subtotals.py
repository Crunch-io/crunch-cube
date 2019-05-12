# encoding: utf-8

"""Integrations tests for headings and subtotals behaviors."""

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.slices import FrozenSlice

from unittest import TestCase

from ..fixtures import CR


class TestHeadersAndSubtotals(TestCase):
    def test_headings_econ_blame_one_subtotal(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS))
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
        )
        self.assertEqual(slice_.row_labels, expected)

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS), transforms=transforms)
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        )
        self.assertEqual(slice_.row_labels, expected)

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING))
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Test New Heading (Obama and Republicans)",
            "Both",
            "Neither",
            "Not sure",
            "Test Heading with Skipped",
        )
        self.assertEqual(slice_.row_labels, expected)

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = FrozenSlice(
            CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING), transforms=transforms
        )
        expected = (
            "President Obama",
            "Republicans in Congress",
            "Both",
            "Neither",
            "Not sure",
        )
        self.assertEqual(slice_.row_labels, expected)

    def test_subtotals_as_array_one_transform(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS))
        expected = np.array([[285], [396], [681], [242], [6], [68]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_as_array_one_transform_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS), transforms=transforms)
        expected = np.array([[285], [396], [242], [6], [68]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_as_array_two_transforms_missing_excluded(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING))
        expected = np.array([[285], [396], [681], [242], [6], [68], [74]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_subtotals_proportions_one_transform(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS))
        expected = np.array(
            [
                [0.2858576],
                [0.3971916],
                [0.6830491],
                [0.2427282],
                [0.0060181],
                [0.0682046],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_subtotals_proportions_one_transform_do_not_fetch(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS), transforms=transforms)
        expected = np.array(
            [[0.2858576], [0.3971916], [0.2427282], [0.0060181], [0.0682046]]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_subtotals_proportions_two_transforms_missing_excluded(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING))
        expected = np.array(
            [
                [0.2858576],
                [0.3971916],
                [0.6830491],
                [0.2427282],
                [0.0060181],
                [0.0682046],
                [0.0742227],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
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
        slice_ = FrozenSlice(
            CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS), transforms=transforms
        )
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_COL_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
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
        slice_ = FrozenSlice(
            CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS), transforms=transforms
        )
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
        expected = np.array([72, 182, 375, 218, 110, 40])
        np.testing.assert_almost_equal(slice_.column_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_row(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
        expected = np.array([285, 396, 681, 242, 6, 68])
        np.testing.assert_almost_equal(slice_.row_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_col(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
        expected = np.array([72, 182, 375, 557, 218, 110, 40])
        np.testing.assert_almost_equal(slice_.column_margin, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_row(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
        expected = np.array([285, 396, 681, 242, 6, 68])
        np.testing.assert_almost_equal(slice_.row_margin, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_cell(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.SIMPLE_CA_HS))
        assert slice_.row_labels == ("ca_subvar_1", "ca_subvar_2", "ca_subvar_3")
        assert slice_.column_labels == ("a", "b", "Test A and B combined", "c", "d")

    def test_ca_as_array_with_hs(self):
        slice_ = FrozenSlice(CrunchCube(CR.SIMPLE_CA_HS))
        expected = [[3, 3, 6, 0, 0], [1, 3, 4, 2, 0], [0, 2, 2, 1, 3]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_ca_proportions_with_hs(self):
        slice_ = FrozenSlice(CrunchCube(CR.SIMPLE_CA_HS))
        expected = [
            [0.5, 0.5, 1, 0, 0],
            [0.16666667, 0.5, 0.66666667, 0.33333333, 0],
            [0, 0.33333333, 0.33333333, 0.16666667, 0.5],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_ca_margin_with_hs(self):
        slice_ = FrozenSlice(CrunchCube(CR.SIMPLE_CA_HS))
        expected = [6, 6, 6]
        np.testing.assert_almost_equal(slice_.row_margin, expected)

    def test_hs_with_anchor_on_zero_position_labels(self):
        slice_ = FrozenSlice(CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM))
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
        slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
        expected = [
            [
                1.92562832e-06,
                5.20117283e-12,
                8.30737469e-05,
                0,
                1.77635684e-15,
                1.13223165e-01,
            ],
            [2.90878432e-14, 0, 8.11477145e-02, 0, 5.87376814e-11, 4.64141147e-06],
            [np.nan] * 6,
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
                0,
            ],
        ]
        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_fruit_hs_top_bottom_labels(self):
        slice_ = FrozenSlice(CrunchCube(CR.FRUIT_HS_TOP_BOTTOM))
        assert slice_.row_labels == ("TOP", "rambutan", "MIDDLE", "satsuma", "BOTTOM")

    def test_fruit_hs_top_bottom_inserted_indices(self):
        # TODO: Figure how to do with new slice
        cube = CrunchCube(CR.FRUIT_HS_TOP_BOTTOM)
        expected = [[0, 2, 4]]
        actual = cube.inserted_hs_indices(prune=True)
        assert actual == expected

    def test_fruit_hs_top_bottom_counts(self):
        slice_ = FrozenSlice(CrunchCube(CR.FRUIT_HS_TOP_BOTTOM))
        expected = [[100], [33], [100], [67], [100]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_props(self):
        slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM))
        expected = [
            [1.0, 1.0, 1.0],
            [0.3, 0.35294118, 0.31578947],
            [1.0, 1.0, 1.0],
            [0.7, 0.64705882, 0.68421053],
            [1.0, 1.0, 1.0],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_counts(self):
        slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM))
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
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = FrozenSlice(CrunchCube(CR.CAT_X_NUM_HS_PRUNE), transforms=transforms)
        expected = [[0], [1], [1], [0]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_cat_x_num_counts_pruned_without_hs(self):
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = FrozenSlice(CrunchCube(CR.CAT_X_NUM_HS_PRUNE), transforms=transforms)
        expected = [[1]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_mr_x_cat_hs_counts(self):
        slice_ = FrozenSlice(CrunchCube(CR.PETS_X_FRUIT_HS))
        expected = [[12, 28, 40], [12, 22, 34], [12, 26, 38]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_mr_x_cat_proportions_with_insertions(self):
        slice_ = FrozenSlice(CrunchCube(CR.PETS_X_FRUIT_HS))
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
        slice_ = FrozenSlice(CrunchCube(CR.MISSING_CAT_HS))
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=0, transforms=transforms
        )
        np.testing.assert_array_equal(
            slice_.counts,
            [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        )
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=1, transforms=transforms
        )
        np.testing.assert_array_equal(
            slice_.counts,
            [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]],
        )
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=2, transforms=transforms
        )
        np.testing.assert_array_equal(
            slice_.counts,
            [[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1]],
        )

        # Assert counts with H&S
        slice_ = FrozenSlice(CrunchCube(CR.CA_X_CAT_HS), slice_idx=0)
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [1, 1, 0, 2, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        slice_ = FrozenSlice(CrunchCube(CR.CA_X_CAT_HS), slice_idx=1)
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1],
                [0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        )
        slice_ = FrozenSlice(CrunchCube(CR.CA_X_CAT_HS), slice_idx=2)
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=0, transforms=transforms
        )
        np.testing.assert_array_equal(slice_.column_margin, [1, 1, 1, 1, 1])
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=1, transforms=transforms
        )
        np.testing.assert_array_equal(slice_.column_margin, [1, 1, 1, 1, 1])
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_CAT_HS), slice_idx=2, transforms=transforms
        )
        np.testing.assert_array_equal(slice_.column_margin, [1, 1, 1, 1, 1])

    def test_cat_x_items_x_cats_margin_with_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = FrozenSlice(
            CrunchCube(CR.CAT_X_ITEMS_X_CATS_HS), slice_idx=0, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_margin,
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CAT_X_ITEMS_X_CATS_HS), slice_idx=1, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_margin,
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CAT_X_ITEMS_X_CATS_HS), slice_idx=2, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_margin,
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
        slice_ = FrozenSlice(CrunchCube(CR.CAT_X_MR_WEIGHTED_HS), slice_idx=0)
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
        slice_ = FrozenSlice(CrunchCube(CR.CAT_X_MR_WEIGHTED_HS), slice_idx=1)
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
        slice_ = FrozenSlice(CrunchCube(CR.CAT_X_MR_WEIGHTED_HS), slice_idx=2)
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
        slice_ = FrozenSlice(
            CrunchCube(CR.MR_X_CA_HS), slice_idx=0, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.66666667, 0.33333333, 0.00000000, 0.00000000],
                [0.33333333, 0.33333333, 0.33333333, 0.00000000],
                [0.00000000, 0.33333333, 0.33333333, 0.33333333],
            ],
        )
        slice_ = FrozenSlice(
            CrunchCube(CR.MR_X_CA_HS), slice_idx=1, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.50000000, 0.50000000, 0.00000000, 0.00000000],
                [0.25000000, 0.25000000, 0.50000000, 0.00000000],
                [0.00000000, 0.25000000, 0.00000000, 0.75000000],
            ],
        )
        slice_ = FrozenSlice(
            CrunchCube(CR.MR_X_CA_HS), slice_idx=2, transforms=transforms
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        )

    def test_mr_x_ca_props_by_row_with_hs(self):
        slice_ = FrozenSlice(CrunchCube(CR.MR_X_CA_HS), slice_idx=0)
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
        slice_ = FrozenSlice(CrunchCube(CR.MR_X_CA_HS), slice_idx=1)
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
        slice_ = FrozenSlice(CrunchCube(CR.MR_X_CA_HS), slice_idx=2)
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )

    def test_ca_cat_x_mr_x_ca_subvar_counts_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS),
            slice_idx=0,
            transforms=transforms,
        )
        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [22.2609925, 15.10645152],
                [21.46707647, 21.0230097],
                [219.18232448, 135.91751542],
            ],
        )

    def test_ca_cat_x_mr_x_ca_subvar_proportions_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS),
            slice_idx=0,
            transforms=transforms,
        )
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_MR_HS), slice_idx=0, transforms=transforms
        )
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
        slice_ = FrozenSlice(
            CrunchCube(CR.CA_X_MR_HS), slice_idx=0, transforms=transforms
        )
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
