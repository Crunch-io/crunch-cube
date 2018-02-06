from unittest import TestCase

import numpy as np

from .fixtures import FIXT_ADMIT_X_GENDER_WEIGHTED
from .fixtures import FIXT_ECON_BLAME_WITH_HS
from .fixtures import FIXT_ECON_BLAME_WITH_HS_MISSING
from .fixtures import FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS
from .fixtures import FIXT_ECON_BLAME_X_IDEOLOGY_COL_HS
from .fixtures import FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS
from .fixtures import FIXT_SIMPLE_CA_HS
from .fixtures import FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM
from .fixtures import FIXT_FRUIT_HS_TOP_BOTTOM
from .fixtures import FIXT_FRUIT_X_PETS_HS_TOP_BOTTOM
from .fixtures import FIXT_CAT_X_DATE_HS_PRUNE
from .fixtures import FIXT_CAT_X_NUM_HS_PRUNE

from cr.cube.crunch_cube import CrunchCube


class TestHeadersAndSubtotals(TestCase):
    def test_headings_econ_blame_one_subtotal(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
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
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
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
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
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
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
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
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
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
        actual = cube.labels(include_missing=True, include_transforms_for_dims=[0])[0]
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_one_transform(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_one_transform_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 242, 6, 68])
        actual = cube.as_array(include_transforms_for_dims=None)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_excluded(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([285, 396, 681, 242, 6, 68, 77])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_included(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([285, 396, 681, 242, 6, 68, 3, 0, 0, 77])
        actual = cube.as_array(include_missing=True,
                               include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_proportions_one_transform(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .6830491, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_one_transform_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions(include_transforms_for_dims=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_excluded(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([
            .2858576,
            .3971916,
            .6830491,
            .2427282,
            .0060181,
            .0682046,
            .0772317,
        ])
        actual = cube.proportions(include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_included(self):
        cube = CrunchCube(FIXT_ECON_BLAME_WITH_HS_MISSING)
        expected = np.array([
            .28585757,
            .39719157,
            .68304915,
            .24272818,
            .00601805,
            .06820461,
            .00300903,
            0,
            0,
            .0772317,
        ])
        actual = cube.proportions(include_missing=True,
                                  include_transforms_for_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
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
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
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
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
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
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [3,  14,  80,  114, 67, 7],
            [59, 132, 162, 29,  12, 2],
            [62, 146, 242, 143, 79, 9],
            [6,  29,  109, 67,  26, 5],
            [1,  1,   1,   1,   0,  2],
            [3,  6,   23,  7,   5,  24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_col(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_COL_HS)
        expected = np.array([
            [3,  14,  80,  94,  114, 67, 7],
            [59, 132, 162, 294, 29,  12, 2],
            [6,  29,  109, 138, 67,  26, 5],
            [1,  1,   1,   2,   1,   0,  2],
            [3,  6,   23,  29,  7,   5,  24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [3,  14,  80,  94,  114, 67, 7],
            [59, 132, 162, 294, 29,  12, 2],
            [62, 146, 242, 388, 143, 79, 9],
            [6,  29,  109, 138, 67,  26, 5],
            [1,  1,   1,   2,   1,   0,  2],
            [3,  6,   23,  29,  7,   5,  24],
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [3,  14,  80,  114, 67, 7],
            [59, 132, 162, 29,  12, 2],
            [6,  29,  109, 67,  26, 5],
            [1,  1,   1,   1,   0,  2],
            [3,  6,   23,  7,   5,  24],
        ])
        actual = cube.as_array(include_transforms_for_dims=None)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([72, 182, 375, 218, 110, 40])
        actual = cube.margin(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_col(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([72, 182, 375, 557, 218, 110, 40])
        actual = cube.margin(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_row(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_cell(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.00300903, .01404213, .08024072, .11434303, .0672016,  .00702106],
            [.05917753, .13239719, .16248746, .02908726, .01203611, .00200602],
            [.06218656, .14643932, .24272818, .14343029, .07923771, .00902708],
            [.00601805, .02908726, .10932798, .0672016,  .02607823, .00501505],
            [.00100301, .00100301, .00100301, .00100301, 0,         .00200602],
            [.00300903, .00601805, .02306921, .00702106, .00501505, .02407222]
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.04166667, .07692308, .21333333, .52293578, .60909091, .175],
            [.81944444, .72527473, .432,      .13302752, .10909091, .05],
            [.86111111, .8021978,  .64533333, .6559633,  .71818182, .225],
            [.08333333, .15934066, .29066667, .30733945, .23636364, .125],
            [.01388889, .00549451, .00266667, .00458716, 0,         .05],
            [.04166667, .03296703, .06133333, .03211009, .04545455, .6],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [.01052632, .04912281, .28070175, .4,        .23508772, .0245614],
            [.1489899,  .33333333, .40909091, .07323232, .03030303, .00505051],
            [.09104258, .2143906,  .35535977, .20998532, .11600587, .01321586],
            [.02479339, .11983471, .45041322, .2768595,  .10743802, .02066116],
            [.16666667, .16666667, .16666667, .16666667, 0,         .33333333],
            [.04411765, .08823529, .33823529, .10294118, .07352941, .35294118],
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_two_dim_by_cell(self):
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
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
        cube = CrunchCube(FIXT_SIMPLE_CA_HS)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'Test A and B combined', 'c', 'd']
        ]
        actual = cube.labels(include_transforms_for_dims=[0])
        self.assertEqual(actual, expected)

    def test_ca_as_array_with_hs(self):
        cube = CrunchCube(FIXT_SIMPLE_CA_HS)
        expected = np.array([
            [3, 3, 6,  0, 0],
            [1, 3, 4,  2, 0],
            [0, 2, 2,  1, 3]
        ])
        actual = cube.as_array(include_transforms_for_dims=[0, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_ca_proportions_with_hs(self):
        cube = CrunchCube(FIXT_SIMPLE_CA_HS)
        expected = np.array([
            [.5,        .5,         1,        0,         0],
            [.16666667, .5,         .66666667, .33333333, 0],
            [0,         .33333333, .33333333, .16666667, .5]
        ])
        actual = cube.proportions(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_margin_with_hs(self):
        cube = CrunchCube(FIXT_SIMPLE_CA_HS)
        expected = np.array([6, 6, 6])
        actual = cube.margin(include_transforms_for_dims=[0, 1], axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_count_unweighted(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = 4526
        actual = cube.count(weighted=False)
        self.assertEqual(actual, expected)

    def test_count_weighted(self):
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = 4451.955438803242
        actual = cube.count(weighted=True)
        self.assertEqual(actual, expected)

    def test_hs_with_anchor_on_zero_position_labels(self):
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
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
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
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
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
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
        cube = CrunchCube(FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM)
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
        '''Ensure that pvals shape is the same as table shape with H%S'''
        cube = CrunchCube(FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = 6, 6
        actual = cube.pvals(axis=0, include_transforms_for_dims=[0, 1]).shape

        # Only assert the shape of the table, as the pvals are going to be
        # rewritten soon.
        # TODO: Change assertion after merging new Z-scores.
        np.testing.assert_array_equal(actual, expected)

    def test_fruit_hs_top_bottom_labels(self):
        cube = CrunchCube(FIXT_FRUIT_HS_TOP_BOTTOM)
        expected = [['TOP', 'rambutan', 'MIDDLE', 'satsuma', 'BOTTOM']]
        actual = cube.labels(include_transforms_for_dims=[0])
        assert actual == expected

    def test_fruit_hs_top_bottom_inserted_indices(self):
        cube = CrunchCube(FIXT_FRUIT_HS_TOP_BOTTOM)
        expected = [[0, 2, 4]]
        actual = cube.inserted_hs_indices(prune=True)
        assert actual == expected

    def test_fruit_hs_top_bottom_counts(self):
        cube = CrunchCube(FIXT_FRUIT_HS_TOP_BOTTOM)
        expected = np.array([100, 33, 100, 67, 100])
        actual = cube.as_array(include_transforms_for_dims=[0])
        np.testing.assert_array_equal(actual, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_counts(self):
        cube = CrunchCube(FIXT_FRUIT_X_PETS_HS_TOP_BOTTOM)
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
        cube = CrunchCube(FIXT_CAT_X_DATE_HS_PRUNE)
        expected = [0, 3, 6]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_hs_indices_pruned_cat_x_num(self):
        cube = CrunchCube(FIXT_CAT_X_NUM_HS_PRUNE)
        expected = [0, 1, 3]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_cat_x_num_hs_counts_pruned(self):
        cube = CrunchCube(FIXT_CAT_X_NUM_HS_PRUNE)
        expected = np.array([0, 1, 1, 0])
        actual = cube.as_array(include_transforms_for_dims=[0], prune=True)
        np.testing.assert_array_equal(actual, expected)
