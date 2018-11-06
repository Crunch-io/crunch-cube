# encoding: utf-8

"""Integrations tests for headings and subtotals behaviors."""

import numpy as np

from cr.cube.crunch_cube import CrunchCube

from unittest import TestCase

from ..fixtures import CR


class TestHeadersAndSubtotals(TestCase):

    def test_headings_econ_blame_one_subtotal(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=None)[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
            'Test Heading with Skipped',
        ]
        actual = cube.labels(include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels(include_transforms_for_dims=None)[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_with_missing(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
            'Skipped',
            'Not Asked',
            'No Data',
            'Test Heading with Skipped',
        ]
        actual = cube.labels(
            include_missing=True, include_transforms_for_dims=[0],
        )[0]
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_one_transform(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_one_transform_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=None)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_excluded(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([285, 396, 681, 242, 6, 68, 74])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_proportions_one_transform(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .6830491, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_one_transform_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_excluded(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([
            .2858576,
            .3971916,
            .6830491,
            .2427282,
            .0060181,
            .0682046,
            .0742227,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = [[
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
        ], [
            'Very liberal',
            'Liberal',
            'Moderate',
            'Conservative',
            'Very Conservative',
            'Not sure',
        ]]
        actual = cube.labels(include_transforms_for_dims=[0, 1])
        self.assertEqual(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_both_dim(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = [[
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
        ], [
            'Very liberal',
            'Liberal',
            'Moderate',
            'Test 2nd dim Heading',
            'Conservative',
            'Very Conservative',
            'Not sure',
        ]]
        actual = cube.labels(include_transforms_for_dims=[0, 1])
        self.assertEqual(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = [[
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ], [
            'Very liberal',
            'Liberal',
            'Moderate',
            'Conservative',
            'Very Conservative',
            'Not sure',
        ]]
        actual = cube.labels(include_transforms_for_dims=None)
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_row(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [3, 14, 80, 114, 67, 7],
            [59, 132, 162, 29, 12, 2],
            [62, 146, 242, 143, 79, 9],
            [6, 29, 109, 67, 26, 5],
            [1, 1, 1, 1, 0, 2],
            [3, 6, 23, 7, 5, 24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_col(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_COL_HS)
        expected = np.array([
            [3, 14, 80, 94, 114, 67, 7],
            [59, 132, 162, 294, 29, 12, 2],
            [6, 29, 109, 138, 67, 26, 5],
            [1, 1, 1, 2, 1, 0, 2],
            [3, 6, 23, 29, 7, 5, 24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [3, 14, 80, 94, 114, 67, 7],
            [59, 132, 162, 294, 29, 12, 2],
            [62, 146, 242, 388, 143, 79, 9],
            [6, 29, 109, 138, 67, 26, 5],
            [1, 1, 1, 2, 1, 0, 2],
            [3, 6, 23, 29, 7, 5, 24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [3, 14, 80, 114, 67, 7],
            [59, 132, 162, 29, 12, 2],
            [6, 29, 109, 67, 26, 5],
            [1, 1, 1, 1, 0, 2],
            [3, 6, 23, 7, 5, 24],
        ])
        actual = cube.as_array(include_transforms_for_dims=None)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([72, 182, 375, 218, 110, 40])
        actual = cube.margin(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_col(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([72, 182, 375, 557, 218, 110, 40])
        actual = cube.margin(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_row(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_cell(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.00300903, .01404213, .08024072, .11434303, .0672016, .00702106],
            [.05917753, .13239719, .16248746, .02908726, .01203611, .00200602],
            [.06218656, .14643932, .24272818, .14343029, .07923771, .00902708],
            [.00601805, .02908726, .10932798, .0672016, .02607823, .00501505],
            [.00100301, .00100301, .00100301, .00100301, 0, .00200602],
            [.00300903, .00601805, .02306921, .00702106, .00501505, .02407222]
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.04166667, .07692308, .21333333, .52293578, .60909091, .175],
            [.81944444, .72527473, .432, .13302752, .10909091, .05],
            [.86111111, .8021978, .64533333, .6559633, .71818182, .225],
            [.08333333, .15934066, .29066667, .30733945, .23636364, .125],
            [.01388889, .00549451, .00266667, .00458716, 0, .05],
            [.04166667, .03296703, .06133333, .03211009, .04545455, .6],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.01052632, .04912281, .28070175, .4, .23508772, .0245614],
            [.1489899, .33333333, .40909091, .07323232, .03030303, .00505051],
            [.09104258, .2143906, .35535977, .20998532, .11600587, .01321586],
            [.02479339, .11983471, .45041322, .2768595, .10743802, .02066116],
            [.16666667, .16666667, .16666667, .16666667, 0, .33333333],
            [.04411765, .08823529, .33823529, .10294118, .07352941, .35294118],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_two_dim_by_cell(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [
                .00300903,
                .01404213,
                .08024072,
                .09428285,
                .11434303,
                .0672016,
                .00702106
            ],
            [
                .05917753,
                .13239719,
                .16248746,
                .29488465,
                .02908726,
                .01203611,
                .00200602
            ],
            [
                .06218656,
                .14643932,
                .24272818,
                .3891675,
                .14343029,
                .07923771,
                .00902708
            ],
            [
                .00601805,
                .02908726,
                .10932798,
                .13841525,
                .0672016,
                .02607823,
                .00501505
            ],
            [
                .00100301,
                .00100301,
                .00100301,
                .00200602,
                .00100301,
                0,
                .00200602
            ],
            [
                .00300903,
                .00601805,
                .02306921,
                .02908726,
                .00702106,
                .00501505,
                .02407222
            ],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_labels_with_hs(self):
        cube = CrunchCube(CR.SIMPLE_CA_HS)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'Test A and B combined', 'c', 'd']
        ]
        actual = cube.labels(include_transforms_for_dims=[0])
        self.assertEqual(actual, expected)

    def test_ca_as_array_with_hs(self):
        cube = CrunchCube(CR.SIMPLE_CA_HS)
        expected = np.array([
            [3, 3, 6, 0, 0],
            [1, 3, 4, 2, 0],
            [0, 2, 2, 1, 3]
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_ca_proportions_with_hs(self):
        cube = CrunchCube(CR.SIMPLE_CA_HS)
        expected = np.array([
            [.5, .5, 1, 0, 0],
            [.16666667, .5, .66666667, .33333333, 0],
            [0, .33333333, .33333333, .16666667, .5]
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_margin_with_hs(self):
        cube = CrunchCube(CR.SIMPLE_CA_HS)
        expected = np.array([6, 6, 6])
        actual = cube.margin(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_with_anchor_on_zero_position_labels(self):
        cube = CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = [
            [
                'Serious net',
                'Very serious',
                'Somewhat serious',
                'Not very serious',
                'Not at all serious',
                'Not sure',
            ],
            [
                'Sexual assaults that go unreported or unpunished',
                'False accusations of sexual assault',
            ],
        ]
        actual = cube.labels(include_transforms_for_dims=[0, 1])
        self.assertEqual(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_col(self):
        cube = CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.93244626, 0.66023166],
            [0.63664278, 0.23166023],
            [0.29580348, 0.42857143],
            [0.04401228, 0.21428571],
            [0.00307062, 0.06177606],
            [0.02047083, 0.06370656],
        ])
        actual = cube.proportions(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_row(self):
        cube = CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.72705507, 0.27294493],
            [0.83827493, 0.16172507],
            [0.56555773, 0.43444227],
            [0.27922078, 0.72077922],
            [0.08571429, 0.91428571],
            [0.37735849, 0.62264151],
        ])
        actual = cube.proportions(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_cell(self):
        cube = CrunchCube(CR.ECON_US_PROBLEM_X_BIGGER_PROBLEM)
        expected = np.array([
            [0.60936455, 0.22876254],
            [0.41605351, 0.08026756],
            [0.19331104, 0.14849498],
            [0.02876254, 0.07424749],
            [0.00200669, 0.02140468],
            [0.01337793, 0.02207358],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_pvals_2d_cube_with_hs_on_row(self):
        """Ensure that pvals shape is the same as table shape with H%S"""
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [1.92562832e-06, 5.20117283e-12, 8.30737469e-05, 0.00000000e+00,
             1.77635684e-15, 1.13223165e-01],
            [2.90878432e-14, 0.00000000e+00, 8.11477145e-02, 0.00000000e+00,
             5.87376814e-11, 4.64141147e-06],
            [1.05605732e-03, 3.70613426e-03, 6.11851617e-03, 1.18269053e-02,
             8.68890220e-01, 7.62914197e-02],
            [3.69990005e-01, 9.19546240e-01, 2.88068221e-01, 7.57299844e-01,
             3.86924216e-01, 2.41648361e-04],
            [3.53745446e-01, 3.70094812e-02, 5.03974440e-01, 1.67769523e-02,
             3.15641644e-01, 0.00000000e+00],
        ])
        actual = cube.pvals()
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_hs_top_bottom_labels(self):
        cube = CrunchCube(CR.FRUIT_HS_TOP_BOTTOM)
        expected = [['TOP', 'rambutan', 'MIDDLE', 'satsuma', 'BOTTOM']]
        actual = cube.labels(include_transforms_for_dims=[0])
        assert actual == expected

    def test_fruit_hs_top_bottom_inserted_indices(self):
        cube = CrunchCube(CR.FRUIT_HS_TOP_BOTTOM)
        expected = [[0, 2, 4]]
        actual = cube.inserted_hs_indices(prune=True)
        assert actual == expected

    def test_fruit_hs_top_bottom_counts(self):
        cube = CrunchCube(CR.FRUIT_HS_TOP_BOTTOM)
        expected = np.array([100, 33, 100, 67, 100])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_props(self):
        cube = CrunchCube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM)
        expected = np.array([
            [1., 1., 1.],
            [0.3, 0.35294118, 0.31578947],
            [1., 1., 1.],
            [0.7, 0.64705882, 0.68421053],
            [1., 1., 1.],
        ])
        actual = cube.proportions(axis=0, include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_counts(self):
        cube = CrunchCube(CR.FRUIT_X_PETS_HS_TOP_BOTTOM)
        expected = np.array([
            [40, 34, 38],
            [12, 12, 12],
            [40, 34, 38],
            [28, 22, 26],
            [40, 34, 38],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_hs_indices_pruned_cat_x_date(self):
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
        cube = CrunchCube(CR.CAT_X_NUM_HS_PRUNE)
        expected = np.array([
            [0],
            [1],
            [1],
            [0],
        ])
        # Extract only non-masked (pruned) values
        table = cube.as_array(include_transforms_for_dims=[0], prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_num_counts_pruned_without_hs(self):
        cube = CrunchCube(CR.CAT_X_NUM_HS_PRUNE)
        expected = np.array([[1]])
        table = cube.as_array(prune=True)
        # Extract only non-masked (pruned) values
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_counts(self):
        cube = CrunchCube(CR.PETS_X_FRUIT_HS)
        expected = np.array([[12, 28, 40],
                             [12, 22, 34],
                             [12, 26, 38]])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_cell(self):
        cube = CrunchCube(CR.PETS_X_FRUIT_HS)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(
            axis=None, include_transforms_for_dims=[0, 1]
        ).shape
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_row(self):
        cube = CrunchCube(CR.PETS_X_FRUIT_HS)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(
            axis=0, include_transforms_for_dims=[0, 1]
        ).shape
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_col(self):
        cube = CrunchCube(CR.PETS_X_FRUIT_HS)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(
            axis=1, include_transforms_for_dims=[0, 1]
        ).shape
        np.testing.assert_array_equal(actual, expected)

    def test_missing_cat_hs_labels(self):
        cube = CrunchCube(CR.MISSING_CAT_HS)

        # Don't expect the missing category "Non voters"
        expected = [[
            'Whites',
            'White college women voters',
            'White non-college women voters',
            'White college men voters',
            'White non-college men voters',
            'Black voters',
            'Latino and other voters',
        ]]
        actual = cube.labels(include_transforms_for_dims=[0])
        assert actual == expected

    def test_ca_x_cat_counts_with_hs(self):
        cube = CrunchCube(CR.CA_X_CAT_HS)

        # Assert counts without H&S
        expected = np.array([
            [[1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],

            [[1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 1],
             [0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [1, 0, 0, 1, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1]]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

        # Assert counts with H&S
        expected = np.array([
            [[1, 1, 0, 2, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 2],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]],

            [[1, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 1, 1, 0, 1],
             [0, 0, 1, 1, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 1, 0, 1],
             [0, 1, 0, 1, 0, 0, 0],
             [0, 0, 1, 1, 0, 1, 1]]])

        # Include transforms for all CA and CAT dims (hence 0, 1 and 2)
        actual = cube.as_array(include_transforms_for_dims=[0, 1, 2])
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_cat_margin_with_hs(self):
        cube = CrunchCube(CR.CA_X_CAT_HS)

        # Assert counts without H&S
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])
        actual = cube.margin(axis=1, include_transforms_for_dims=[1])
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_items_x_cats_margin_with_hs(self):
        cube = CrunchCube(CR.CAT_X_ITEMS_X_CATS_HS)

        # Assert counts without H&S
        expected = np.array([
            [
                1287.9364594075469, 2050.0571926339885, 782.9403891997617,
                225.4066607421201, 2622.8036855384603, 974.5889537143403,
                490.5036709315041, 373.8221357520375,
            ],
            [
                1147.3697583254452, 2557.8859179678857, 1096.7841912034742,
                374.0411471364339, 1876.3400274431515, 1002.2399030962134,
                457.92228898229905, 419.5110527202654,
            ],
            [
                1053.855075581148, 2699.612841209989, 1427.7399174151794,
                380.8205091587366, 1027.7782011616534, 606.7100283028576,
                218.42735718966821, 265.29362712412535,
            ]
        ])
        actual = cube.margin(axis=2, include_transforms_for_dims=[1, 2])
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_mr_weighted_with_hs(self):
        cube = CrunchCube(CR.CAT_X_MR_WEIGHTED_HS)
        expected = np.array([
            [
                [0.05865163, 0.087823, 0.07486857, 0.0735683, 0., 0.08148267],
                [0.20246563, 0.33500382, 0.33176765, 0.27870974, 0.36309359,
                    0.33341993],
                [0.54592009, 0.45988528, 0.49802406, 0.48137697, 0.51250032,
                    0.47855168],
                [0.1051054, 0.06727875, 0.0733213, 0.12392602, 0.12440609,
                    0.07023618],
                [0.05508246, 0.04566041, 0.01379632, 0.02729368, 0.,
                    0.03006505],
                [0.03277479, 0.00434874, 0.0082221, 0.01512529, 0.,
                    0.00624449],
                [0.26111726, 0.42282682, 0.40663622, 0.35227804, 0.36309359,
                    0.4149026],
                [0.80703735, 0.8827121, 0.90466028, 0.83365501, 0.87559391,
                    0.89345428]
            ],
            [
                [0.00235883, 0.01361576, 0.01328221, 0.01212187, 0.,
                    0.01345251],
                [0.13002321, 0.0591588, 0.09857174, 0.05056353, 0., 0.07844882],
                [0.65429951, 0.77915194, 0.74437239, 0.61537442, 1.,
                    0.76212966],
                [0.13730378, 0.11171429, 0.11961331, 0.26739934, 0.,
                    0.11558033],
                [0.04323988, 0.02601641, 0.01593825, 0.02729368, 0.,
                    0.02108382],
                [0.03277479, 0.0103428, 0.0082221, 0.02724716, 0., 0.00930486],
                [0.13238204, 0.07277456, 0.11185395, 0.06268541, 0.,
                    0.09190133],
                [0.78668155, 0.85192649, 0.85622634, 0.67805982, 1., 0.85403098]
            ],
            [
                [0.003676, 0.00486795, 0.0082221, 0.01212187, 0., 0.00650959],
                [0.03884185, 0.00625322, 0.02928964, 0.02729368, 0.12440609,
                    0.01752802],
                [0.39625335, 0.4717925, 0.44918748, 0.46124151, 0.40543568,
                    0.46072884],
                [0.4255265, 0.44274565, 0.42191512, 0.43268073, 0.47015822,
                    0.43255049],
                [0.1054366, 0.07434068, 0.08316356, 0.05153692, 0., 0.07865889],
                [0.03026569, 0., 0.0082221, 0.01512529, 0., 0.00402417],
                [0.04251785, 0.01112118, 0.03751174, 0.03941555, 0.12440609,
                    0.02403761],
                [0.4387712, 0.48291368, 0.48669922, 0.50065705, 0.52984178,
                    0.48476645]
            ]
        ])
        actual = cube.proportions(axis=1, include_transforms_for_dims=[0, 1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_ca_props_by_row_without_hs(self):
        cube = CrunchCube(CR.MR_X_CA_HS)
        expected = np.array([
            [[0.66666667, 0.33333333, 0.00000000, 0.00000000],
             [0.33333333, 0.33333333, 0.33333333, 0.00000000],
             [0.00000000, 0.33333333, 0.33333333, 0.33333333]],

            [[0.50000000, 0.50000000, 0.00000000, 0.00000000],
             [0.25000000, 0.25000000, 0.50000000, 0.00000000],
             [0.00000000, 0.25000000, 0.00000000, 0.75000000]],

            [[np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan]],
        ])
        with self.assertRaises(ValueError):
            # "Table" direction not allowed, because cube's rows are CA dim
            actual = cube.proportions()
        actual = cube.proportions(axis=2)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_ca_props_by_row_with_hs(self):
        cube = CrunchCube(CR.MR_X_CA_HS)
        expected = np.array([
            [[0.66666667, 0.33333333, 1.00000000, 0.00000000, 0.00000000,
              0.00000000, 1.00000000],
             [0.33333333, 0.33333333, 0.66666667, 0.33333333, 0.00000000,
              0.33333333, 1.00000000],
             [0.00000000, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
              0.66666667, 1.00000000]],

            [[0.50000000, 0.50000000, 1.00000000, 0.00000000, 0.00000000,
              0.00000000, 1.00000000],
             [0.25000000, 0.25000000, 0.50000000, 0.50000000, 0.00000000,
              0.50000000, 1.00000000],
             [0.00000000, 0.25000000, 0.25000000, 0.00000000, 0.75000000,
              0.75000000, 1.00000000]],

            [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
        ])

        with self.assertRaises(ValueError):
            # "Table" direction not allowed, because cube's rows are CA dim
            actual = cube.proportions(include_transforms_for_dims=[0, 1, 2])

        actual = cube.proportions(axis=2, include_transforms_for_dims=[0, 1, 2])
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_cat_x_mr_x_ca_subvar_counts_pruning(self):
        cube = CrunchCube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS)
        expected = np.array([
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [True, True, True],
        ])
        actual = cube.as_array(prune=True)[0].mask
        np.testing.assert_array_equal(actual, expected)

    def test_ca_cat_x_mr_x_ca_subvar_proportions_pruning(self):
        cube = CrunchCube(CR.CA_CAT_X_MR_X_CA_SUBVAR_HS)
        expected = np.array([
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [True, True, True],
        ])

        with self.assertRaises(ValueError):
            # "Table" direction not allowed cuz CA items
            actual = cube.proportions(prune=True)[0].mask

        actual = cube.proportions(axis=1, prune=True)[0].mask
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_mr_counts_pruning(self):
        cube = CrunchCube(CR.CA_X_MR_HS)
        actual = cube.as_array(prune=True)[0].mask
        expected = np.array([
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True]
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_mr_proportions_pruning(self):
        cube = CrunchCube(CR.CA_X_MR_HS)
        expected = np.array([
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
        ])
        actual = cube.proportions(axis=None, prune=True)[0].mask
        np.testing.assert_array_equal(actual, expected)
