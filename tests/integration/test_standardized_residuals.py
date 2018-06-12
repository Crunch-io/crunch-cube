from unittest import TestCase

import numpy as np

from .fixtures import ADMIT_X_DEPT_UNWEIGHTED
from .fixtures import ADMIT_X_GENDER_WEIGHTED
from .fixtures import SELECTED_CROSSTAB_4
from .fixtures import SEL_ARR_FIRST
from .fixtures import SEL_ARR_LAST
from .fixtures import MR_X_MR
from .fixtures import MR_X_MR_HETEROGENOUS
from .fixtures import ECON_BLAME_X_IDEOLOGY_ROW_HS

from cr.cube.crunch_cube import CrunchCube


class TestStandardizedResiduals(TestCase):
    '''Test cr.cube implementation of Z-Scores.'''

    def test_standardized_residuals_admit_x_dept_unweighted(self):
        '''Z-Scores for normal unweighted crosstab.'''
        cube = CrunchCube(ADMIT_X_DEPT_UNWEIGHTED)
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
        cube = CrunchCube(ADMIT_X_GENDER_WEIGHTED)
        expected = np.array([
            [9.42561985, -9.42561985],
            [-9.42561985, 9.42561985],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_selected_crosstab(self):
        '''Residuals for MR x CAT unweighted.'''
        cube = CrunchCube(SELECTED_CROSSTAB_4)
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
        cube = CrunchCube(SEL_ARR_FIRST)
        expected = np.array([
            [0.80134191, -0.80134191],
            [0.60455606, -0.60455606],
            [-0.30884247, 0.30884247],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_cat_x_mr(self):
        '''Residuals for CAT x MR from app.

        The results should be the exact transsposition of the results from
        'test_standardized_residuals_mr_x_cat' test.
        '''
        cube = CrunchCube(SEL_ARR_LAST)
        expected = np.array([
            [0.80134191, 0.60455606, -0.30884247],
            [-0.80134191, -0.60455606, 0.30884247],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_standardized_residuals_mr_x_mr(self):
        '''Residuals for MR x MR.'''
        cube = CrunchCube(MR_X_MR)
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
        cube = CrunchCube(MR_X_MR_HETEROGENOUS)
        expected = np.array([
            [-0.17238393, 38.51646532],
            [0.10271174, -39.11229693],
            [-0.26443564, -39.67503947],
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_std_res_with_hs(self):
        cube = CrunchCube(ECON_BLAME_X_IDEOLOGY_ROW_HS)

        # Don't include H&S dim (the one with transform)
        expected = np.array([
            [-4.76107671, -6.89997234, -3.93535518,  8.76458713, 7.95483156,  -1.58387062],  # noqa
            [ 7.6019656 , 10.00457686,  1.74406524, -9.01760367, -6.54691501, -4.58037582],  # noqa
            [-3.27515041, -2.90214798,  2.74136144,  2.51726734, -0.1650683 , -1.77262166],  # noqa
            [ 0.89649209, -0.10100532, -1.06236896, -0.3090285 , -0.86520876,  3.67095238],  # noqa
            [-0.92734884, -2.08565946, -0.66824935, -2.39155976, -1.00345445, 13.61755117],  # noqa
        ])
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

        actual = cube.zscore(hs_dims=[1])
        np.testing.assert_almost_equal(actual, expected)

        # Include H&S (expect additional row of all zeros)
        expected = np.array([
            [-4.76107671, -6.89997234, -3.93535518,  8.76458713, 7.95483156,  -1.58387062],  # noqa
            [ 7.6019656 , 10.00457686,  1.74406524, -9.01760367, -6.54691501, -4.58037582],  # noqa
            [     np.nan,      np.nan,      np.nan,      np.nan,      np.nan,      np.nan],  # noqa
            [-3.27515041, -2.90214798,  2.74136144,  2.51726734, -0.1650683 , -1.77262166],  # noqa
            [ 0.89649209, -0.10100532, -1.06236896, -0.3090285 , -0.86520876,  3.67095238],  # noqa
            [-0.92734884, -2.08565946, -0.66824935, -2.39155976, -1.00345445, 13.61755117],  # noqa
        ])
        actual = cube.zscore(hs_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

        actual = cube.zscore(hs_dims=[0])
        np.testing.assert_almost_equal(actual, expected)
