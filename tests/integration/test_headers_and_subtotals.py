from unittest import TestCase

import numpy as np

from .fixtures import ADMIT_X_GENDER_WEIGHTED
from .fixtures import ECON_BLAME_WITH_HS
from .fixtures import ECON_BLAME_WITH_HS_MISSING
from .fixtures import ECON_BLAME_X_IDEOLOGY_ROW_HS
from .fixtures import ECON_BLAME_X_IDEOLOGY_COL_HS
from .fixtures import ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS
from .fixtures import SIMPLE_CA_HS
from .fixtures import ECON_US_PROBLEM_X_BIGGER_PROBLEM
from .fixtures import FRUIT_HS_TOP_BOTTOM
from .fixtures import FRUIT_X_PETS_HS_TOP_BOTTOM
from .fixtures import CAT_X_DATE_HS_PRUNE
from .fixtures import CAT_X_NUM_HS_PRUNE
from .fixtures import PETS_X_FRUIT_HS
from .fixtures import MISSING_CAT_HS
from .fixtures import CA_X_CAT_HS

from cr.cube.crunch_cube import CrunchCube


class TestHeadersAndSubtotals(TestCase):
    def test_headings_econ_blame_one_subtotal(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS, include_hs=True)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels()[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_one_subtotal_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels()[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_without_missing(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Test New Heading (Obama and Republicans)',
            'Both',
            'Neither',
            'Not sure',
            'Test Heading with Skipped',
        ]
        actual = cube.labels()[0]
        self.assertEqual(actual, expected)

    def test_headings_two_subtotal_without_missing_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING)
        expected = [
            'President Obama',
            'Republicans in Congress',
            'Both',
            'Neither',
            'Not sure',
        ]
        actual = cube.labels()[0]
        self.assertEqual(actual, expected)

    def test_headings_econ_blame_two_subtotal_with_missing(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
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
        actual = cube.labels(include_missing=True)[0]
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_one_transform(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS, include_hs=True)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_one_transform_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS)
        expected = np.array([285, 396, 242, 6, 68])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_excluded(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
        expected = np.array([285, 396, 681, 242, 6, 68, 77])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_two_transforms_missing_included(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
        expected = np.array([285, 396, 681, 242, 6, 68, 3, 0, 0, 77])
        actual = cube.as_array(include_missing=True)
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_proportions_one_transform(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS, include_hs=True)
        expected = np.array([
            .2858576, .3971916, .6830491, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_one_transform_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS)
        expected = np.array([
            .2858576, .3971916, .2427282, .0060181, .0682046,
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_excluded(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
        expected = np.array([
            .2858576,
            .3971916,
            .6830491,
            .2427282,
            .0060181,
            .0682046,
            .0772317,
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_two_transforms_missing_included(self):
        cube = CrunchCube(ECON_BLAME_WITH_HS_MISSING, include_hs=True)
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
        actual = cube.proportions(include_missing=True)
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_1st_dim(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
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
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_both_dim(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, include_hs=True)
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
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_on_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
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
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_row(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
        expected = np.array([
            [3,  14,  80,  114, 67, 7],  # noqa
            [59, 132, 162, 29,  12, 2],  # noqa
            [62, 146, 242, 143, 79, 9],  # noqa
            [6,  29,  109, 67,  26, 5],  # noqa
            [1,  1,   1,   1,   0,  2],  # noqa
            [3,  6,   23,  7,   5,  24],  # noqa
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_col(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_COL_HS, include_hs=True)
        expected = np.array([
            [3,  14,  80,  94,  114, 67, 7],  # noqa
            [59, 132, 162, 294, 29,  12, 2],  # noqa
            [6,  29,  109, 138, 67,  26, 5],  # noqa
            [1,  1,   1,   2,   1,   0,  2],  # noqa
            [3,  6,   23,  29,  7,   5,  24],  # noqa
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, include_hs=True)
        expected = np.array([
            [3,  14,  80,  94,  114, 67, 7],  # noqa
            [59, 132, 162, 294, 29,  12, 2],  # noqa
            [62, 146, 242, 388, 143, 79, 9],  # noqa
            [6,  29,  109, 138, 67,  26, 5],  # noqa
            [1,  1,   1,   2,   1,   0,  2],  # noqa
            [3,  6,   23,  29,  7,   5,  24],  # noqa
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_as_array_2d_cube_with_hs_on_both_dim_do_not_fetch(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS)
        expected = np.array([
            [3,  14,  80,  114, 67, 7],  # noqa
            [59, 132, 162, 29,  12, 2],  # noqa
            [6,  29,  109, 67,  26, 5],  # noqa
            [1,  1,   1,   1,   0,  2],  # noqa
            [3,  6,   23,  7,   5,  24],  # noqa
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([72, 182, 375, 218, 110, 40])
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_col(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, include_hs=True)
        expected = np.array([72, 182, 375, 557, 218, 110, 40])
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_margin_2d_cube_with_hs_on_two_dim_by_row(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, include_hs=True)
        expected = np.array([285, 396, 681, 242, 6, 68])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_cell(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
        expected = np.array([
            [.00300903, .01404213, .08024072, .11434303, .0672016,  .00702106],
            [.05917753, .13239719, .16248746, .02908726, .01203611, .00200602],
            [.06218656, .14643932, .24272818, .14343029, .07923771, .00902708],
            [.00601805, .02908726, .10932798, .0672016,  .02607823, .00501505],
            [.00100301, .00100301, .00100301, .00100301, 0,         .00200602],
            [.00300903, .00601805, .02306921, .00702106, .00501505, .02407222]
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_col(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
        expected = np.array([
            [.04166667, .07692308, .21333333, .52293578, .60909091, .175],
            [.81944444, .72527473, .432,      .13302752, .10909091, .05],
            [.86111111, .8021978,  .64533333, .6559633,  .71818182, .225],
            [.08333333, .15934066, .29066667, .30733945, .23636364, .125],
            [.01388889, .00549451, .00266667, .00458716, 0,         .05],
            [.04166667, .03296703, .06133333, .03211009, .04545455, .6],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_row_by_row(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS, include_hs=True)
        expected = np.array([
            [.01052632, .04912281, .28070175, .4,        .23508772, .0245614],
            [.1489899,  .33333333, .40909091, .07323232, .03030303, .00505051],
            [.09104258, .2143906,  .35535977, .20998532, .11600587, .01321586],
            [.02479339, .11983471, .45041322, .2768595,  .10743802, .02066116],
            [.16666667, .16666667, .16666667, .16666667, 0,         .33333333],
            [.04411765, .08823529, .33823529, .10294118, .07352941, .35294118],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_proportions_2d_cube_with_hs_on_two_dim_by_cell(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, include_hs=True)
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
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_labels_with_hs(self):
        cube = CrunchCube(SIMPLE_CA_HS, include_hs=True)
        expected = [
            ['ca_subvar_1', 'ca_subvar_2', 'ca_subvar_3'],
            ['a', 'b', 'Test A and B combined', 'c', 'd']
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_ca_as_array_with_hs(self):
        cube = CrunchCube(SIMPLE_CA_HS, include_hs=True)
        expected = np.array([
            [3, 3, 6,  0, 0],
            [1, 3, 4,  2, 0],
            [0, 2, 2,  1, 3]
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_ca_proportions_with_hs(self):
        cube = CrunchCube(SIMPLE_CA_HS, include_hs=True)
        expected = np.array([
            [.5,        .5,         1,        0,         0],
            [.16666667, .5,         .66666667, .33333333, 0],
            [0,         .33333333, .33333333, .16666667, .5]
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_margin_with_hs(self):
        cube = CrunchCube(SIMPLE_CA_HS, include_hs=True)
        expected = np.array([6, 6, 6])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_count_unweighted(self):
        cube = CrunchCube(ADMIT_X_GENDER_WEIGHTED)
        expected = 4526
        actual = cube.count(weighted=False)
        self.assertEqual(actual, expected)

    def test_count_weighted(self):
        cube = CrunchCube(ADMIT_X_GENDER_WEIGHTED)
        expected = 4451.955438803242
        actual = cube.count(weighted=True)
        self.assertEqual(actual, expected)

    def test_hs_with_anchor_on_zero_position_labels(self):
        cube = CrunchCube(ECON_US_PROBLEM_X_BIGGER_PROBLEM, include_hs=True)
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
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_col(self):
        cube = CrunchCube(ECON_US_PROBLEM_X_BIGGER_PROBLEM, include_hs=True)
        expected = np.array([
            [0.93244626, 0.66023166],
            [0.63664278, 0.23166023],
            [0.29580348, 0.42857143],
            [0.04401228, 0.21428571],
            [0.00307062, 0.06177606],
            [0.02047083, 0.06370656],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_row(self):
        cube = CrunchCube(ECON_US_PROBLEM_X_BIGGER_PROBLEM, include_hs=True)
        expected = np.array([
            [0.72705507, 0.27294493],
            [0.83827493, 0.16172507],
            [0.56555773, 0.43444227],
            [0.27922078, 0.72077922],
            [0.08571429, 0.91428571],
            [0.37735849, 0.62264151],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_hs_with_anchor_on_zero_position_as_props_by_cell(self):
        cube = CrunchCube(ECON_US_PROBLEM_X_BIGGER_PROBLEM, include_hs=True)
        expected = np.array([
            [0.60936455, 0.22876254],
            [0.41605351, 0.08026756],
            [0.19331104, 0.14849498],
            [0.02876254, 0.07424749],
            [0.00200669, 0.02140468],
            [0.01337793, 0.02207358],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_subtotals_pvals_2d_cube_with_hs_on_row(self):
        '''Ensure that pvals shape is the same as table shape with H%S'''
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = np.array([
            [1.92562832e-06, 5.20117283e-12, 8.30737469e-05, 0.00000000e+00, 1.77635684e-15, 1.13223165e-01],  # noqa
            [2.90878432e-14, 0.00000000e+00, 8.11477145e-02, 0.00000000e+00, 5.87376814e-11, 4.64141147e-06],  # noqa
            [1.05605732e-03, 3.70613426e-03, 6.11851617e-03, 1.18269053e-02, 8.68890220e-01, 7.62914197e-02],  # noqa
            [3.69990005e-01, 9.19546240e-01, 2.88068221e-01, 7.57299844e-01, 3.86924216e-01, 2.41648361e-04],  # noqa
            [3.53745446e-01, 3.70094812e-02, 5.03974440e-01, 1.67769523e-02, 3.15641644e-01, 0.00000000e+00],  # noqa
        ])
        actual = cube.pvals
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_hs_top_bottom_labels(self):
        cube = CrunchCube(FRUIT_HS_TOP_BOTTOM, include_hs=True)
        expected = [['TOP', 'rambutan', 'MIDDLE', 'satsuma', 'BOTTOM']]
        actual = cube.labels()
        assert actual == expected

    def test_fruit_hs_top_bottom_inserted_indices(self):
        cube = CrunchCube(FRUIT_HS_TOP_BOTTOM)
        expected = [[0, 2, 4]]
        actual = cube.inserted_hs_indices(prune=True)
        assert actual == expected

    def test_fruit_hs_top_bottom_counts(self):
        cube = CrunchCube(FRUIT_HS_TOP_BOTTOM, include_hs=True)
        expected = np.array([100, 33, 100, 67, 100])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_props(self):
        cube = CrunchCube(FRUIT_X_PETS_HS_TOP_BOTTOM, include_hs=True)
        expected = np.array([
            [1., 1., 1.],
            [0.3, 0.35294118, 0.31578947],
            [1., 1., 1.],
            [0.7, 0.64705882, 0.68421053],
            [1., 1., 1.],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_hs_top_bottom_middle_counts(self):
        cube = CrunchCube(FRUIT_X_PETS_HS_TOP_BOTTOM, include_hs=True)
        expected = np.array([
            [40, 34, 38],
            [12, 12, 12],
            [40, 34, 38],
            [28, 22, 26],
            [40, 34, 38],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_hs_indices_pruned_cat_x_date(self):
        cube = CrunchCube(CAT_X_DATE_HS_PRUNE)
        expected = [0, 3, 6]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_hs_indices_pruned_cat_x_num(self):
        cube = CrunchCube(CAT_X_NUM_HS_PRUNE, include_hs=True)
        expected = [0, 1, 3]
        actual = cube.inserted_hs_indices(prune=True)[0]
        assert actual == expected

    def test_cat_x_num_counts_pruned_with_hs(self):
        cube = CrunchCube(CAT_X_NUM_HS_PRUNE, include_hs=True)
        expected = np.array([
            [0],
            [1],
            [1],
            [0],
        ])
        # Extract only non-masked (pruned) values
        table = cube.as_array(prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_num_counts_pruned_without_hs(self):
        cube = CrunchCube(CAT_X_NUM_HS_PRUNE)
        expected = np.array([[1]])
        table = cube.as_array(prune=True)
        # Extract only non-masked (pruned) values
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_counts(self):
        cube = CrunchCube(PETS_X_FRUIT_HS, include_hs=True)
        expected = np.array([[12, 28, 40],
                             [12, 22, 34],
                             [12, 26, 38]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_cell(self):
        cube = CrunchCube(PETS_X_FRUIT_HS, include_hs=True)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(axis=None).shape
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_row(self):
        cube = CrunchCube(PETS_X_FRUIT_HS, include_hs=True)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(axis=0).shape
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_cat_hs_props_by_col(self):
        cube = CrunchCube(PETS_X_FRUIT_HS, include_hs=True)
        # TODO: Change expectation once the MR cell props are fixed.
        expected = (3, 3)
        actual = cube.proportions(axis=1).shape
        np.testing.assert_array_equal(actual, expected)

    def test_missing_cat_hs_labels(self):
        cube = CrunchCube(MISSING_CAT_HS, include_hs=True)

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
        actual = cube.labels()
        assert actual == expected

    def test_ca_x_cat_counts_with_hs(self):
        # Assert counts without H&S
        cube = CrunchCube(CA_X_CAT_HS)
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
        cube = CrunchCube(CA_X_CAT_HS, include_hs=True)
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
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_cat_margin_with_hs(self):
        cube = CrunchCube(CA_X_CAT_HS, include_hs=True)

        # Assert counts without H&S
        expected = np.array([
            [1, 1, 1, 3, 1, 1, 2],
            [1, 1, 1, 3, 1, 1, 2],
            [1, 1, 1, 3, 1, 1, 2],
        ])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)
