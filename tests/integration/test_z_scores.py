from unittest import TestCase

import numpy as np

from ..fixtures import CR

from cr.cube.crunch_cube import CrunchCube


# pylint: disable=missing-docstring, invalid-name, no-self-use
class TestStandardizedResiduals(TestCase):
    '''Test cr.cube implementation of Z-Scores.'''

    def test_standardized_residuals_admit_x_dept_unweighted(self):
        '''Z-Scores for normal unweighted crosstab.'''
        cube = CrunchCube(CR.ADMIT_X_DEPT_UNWEIGHTED)
        expected = np.array([
            [
                18.04029231,
                13.01839498,
                -2.57673984,
                -3.05952633,
                -7.23024453,
                -19.32141026,
            ],
            [
                -18.04029231,
                -13.01839498,
                2.57673984,
                3.05952633,
                7.23024453,
                19.32141026,
            ],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_admit_x_gender_weighted(self):
        '''Z-Scores for normal weighted crosstab.'''
        cube = CrunchCube(CR.ADMIT_X_GENDER_WEIGHTED)
        expected = np.array([
            [9.42561985, -9.42561985],
            [-9.42561985, 9.42561985],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_selected_crosstab(self):
        '''Residuals for MR x CAT unweighted.'''
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array([
            [-10.88317888, 10.88317888],
            [5.23577326, -5.23577326],
            [-8.89089261, 8.89089261],
            [-7.31367932, 7.31367932],
            [15.39360198, -15.39360198],
            [-1.15219648, 1.15219648],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_mr_x_cat(self):
        '''Residuals for MR x CAT from app.'''
        cube = CrunchCube(CR.SELECTED_CROSSTAB_ARRAY_FIRST)
        expected = np.array([
            [0.80134191, -0.80134191],
            [0.60455606, -0.60455606],
            [-0.30884247, 0.30884247],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_cat_x_mr(self):
        '''Residuals for CAT x MR from app.

        The results should be the exact transposition of the results from
        'test_standardized_residuals_mr_x_cat' test.
        '''
        cube = CrunchCube(CR.SELECTED_CROSSTAB_ARRAY_LAST)
        expected = np.array([
            [0.80134191, 0.60455606, -0.30884247],
            [-0.80134191, -0.60455606, 0.30884247],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_mr_x_mr(self):
        '''Residuals for MR x MR.'''
        cube = CrunchCube(CR.MR_X_MR)
        expected = np.array([
            [12.88418373, 0.1781302, -1.21901758, 4.15682487],
            [0.1781302, 11.910822, -2.70033782, 5.69476817],
            [-1.21901758, -2.70033782, 13.45338666, 9.29294984],
            [4.15682487, 5.69476817, 9.29294984, 15.37981857],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_mr_x_mr_heterogenous(self):
        '''Residuals for MR x MR (disparate MRs).'''
        cube = CrunchCube(CR.MR_X_OTHER_MR)
        expected = np.array([
            [-0.17238393, 38.51646532],
            [0.10271174, -39.11229693],
            [-0.26443564, -39.67503947],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_std_res_with_hs(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)

        # Don't include H&S dim (the one with transform)
        expected = np.array([
            [-4.76107671, -6.89997234, -3.93535518, 8.76458713, 7.95483156,
             -1.58387062],
            [7.6019656, 10.00457686, 1.74406524, -9.01760367, -6.54691501,
             -4.58037582],
            [-3.27515041, -2.90214798, 2.74136144, 2.51726734, -0.1650683,
             -1.77262166],
            [0.89649209, -0.10100532, -1.06236896, -0.3090285, -0.86520876,
             3.67095238],
            [-0.92734884, -2.08565946, -0.66824935, -2.39155976, -1.00345445,
             13.61755117],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

        actual = cube.zscore(hs_dims=[1])
        np.testing.assert_almost_equal(actual, expected)

        # Include H&S (expect additional row of all zeros)
        expected = np.array([
            [-4.76107671, -6.89997234, -3.93535518, 8.76458713, 7.95483156,
             -1.58387062],
            [7.6019656, 10.00457686, 1.74406524, -9.01760367, -6.54691501,
             -4.58037582],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [-3.27515041, -2.90214798, 2.74136144, 2.51726734, -0.1650683,
             -1.77262166],
            [0.89649209, -0.10100532, -1.06236896, -0.3090285, -0.86520876,
             3.67095238],
            [-0.92734884, -2.08565946, -0.66824935, -2.39155976, -1.00345445,
             13.61755117],
        ])
        actual = cube.zscore(hs_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

        actual = cube.zscore(hs_dims=[0])
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_z_score(self):
        cube = CrunchCube(CR.CA_X_MR_SIG_TESTING_SUBTOTALS)
        expected = np.array([
            [[-0.97184831, 0.48259011, -0.22320406, 1.90369568, 0.84276015],
             [5.73141167, -2.36476938, -3.08927542, -5.58099225, -0.43365468],
             [-0.19454649, 1.67287623, 3.52023931, 4.06157819, -1.46455273],
             [-3.51318583, -0.57798682, -1.48960709, -3.60500408, 0.73879457]],

            [[4.36886643, 0.4898096, -2.79475924, -1.67637129, 0.45868614],
             [-1.7697398, -1.64251181, -1.18172874, 1.13980645, 1.48562001],
             [0.74453913, 1.41129291, 4.51918293, 2.44542304, -4.65650512],
             [-2.1434799, 0.02197845, -2.83315628, -4.08730972, 3.11108133]],

            [[2.99534716, 0.18812169, -1.55495526, -3.8775711, -1.65892496],
             [0.78341231, -1.37933059, -1.36159789, 1.27468504, 2.41276199],
             [0.18754893, 2.15035285, 4.19074561, 2.7168425, -3.1400203],
             [-3.33614118, -0.7832844, -2.3400954, -2.8017063, 1.13412111]],

            [[5.49914879, -0.8250958, 0.47711643, -2.35340835, 0.75708154],
             [-0.37181993, -1.00626148, -2.51096895, 1.88779512, 0.99337053],
             [0.12820659, 1.15156133, 3.49017797, 1.11329826, -4.44156835],
             [-3.09836263, 0.31537683, -1.41949801, -2.77042193, 3.19138838]],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_z_score_with_mr_as_row(self):
        cube = CrunchCube(CR.STARTTIME_X_NORDIC_COUNTRIES_X_FOOD_GROOPS)
        expected = np.array([
            [[-0.59209185, -1.38186211, -0.61880093, 2.76977489],
             [-0.27226001, -0.90729991, 1.19984516, 0.18320768],
             [-0.98437766, -1.06099892, 0.00862638, 2.02551167],
             [-2.69271055, -0.70247291, 0.97731075, 2.35431028],
             [2.14156799, -0.26884, -0.28051055, -1.62988758]],

            [[0.44613504, 3.10078541, -1.85754165, -1.95161161],
             [1.47682121, -0.6300945, -1.37772587, 0.71037862],
             [-1.94913113, -1.85068411, -1.08041003, 4.04917418],
             [-1.06337882, -0.82946295, -0.13656886, 1.73737263],
             [3.08099203, 0.23633954, -2.13326401, -1.40411192]],

            [[-0.51632005, -0.60341324, 1.08349741, 0.19639231],
             [-1.10735911, -0.61498431, 0.48664175, 1.19660131],
             [-0.65940873, 1.51305134, 0.17963641, -1.19850121],
             [-1.89200746, 1.14283138, -1.54202798, 1.44133862],
             [2.41917846, -3.08798247, 1.53524725, -1.16466106]],

            [[-1.09404899, -1.27933801, np.nan, 2.80467961],
             [0.56389782, -1.94631489, 1.32101442, 0.54469167],
             [np.nan, np.nan, np.nan, np.nan],
             [0.9635808, -0.95595127, np.nan, 0.09616004],
             [-0.78237941, 1.48226192, np.nan, -0.70605851]],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_z_score_with_mr_as_col(self):
        cube = CrunchCube(CR.FOOD_GROOPS_X_STARTTIME_X_NORDIC_COUNTRIES)
        expected = np.array([
            [[0.01506113, -0.7584501, -0.02786318, -0.31878531, 0.47399192],
             [-0.11623796, 1.1295313, 0.42710872, 1.34964387, -1.01589615],
             [0.60343328, -1.01855773, 0.17035643, -1.87610542, 1.79957599],
             [-0.68273333, 0.46006786, -1.32308749, -0.13319154, -1.83163132]],

            [[-1.13060031, -0.164368, -0.7420848, 0.67712838, 0.21226447],
             [1.06695989, 0.44796674, 0.22470228, -0.03843031, 0.10441562],
             [0.40930073, 0.23774307, 1.63921143, 0.39261527, -1.31335519],
             [-0.86324397, -1.34517767, -1.6837454, -2.67317279, 1.25300357]],

            [[0.43001972, 1.3126165, -0.10717325, 1.50608056, 0.65586374],
             [-2.54392682, -1.99840816, -0.0500687, -0.53201525, -1.92602265],
             [3.30777899, 0.76234711, 0.24895137, -2.36576364, 2.17004946],
             [np.nan, 1.38291283, np.nan, np.nan, np.nan]],

            [[2.99977078, -0.32043235, -0.34165324, 1.89240508, -0.27546776],
             [-4.26735077, -0.58829315, 1.92992202, -1.05792001, 0.03639644],
             [1.15068951, 1.01695401, -2.17562799, -0.58087799, 0.4756435],
             [2.88459281, 0.44900385, -1.83002841, -1.48069796, -0.19880727]],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)
