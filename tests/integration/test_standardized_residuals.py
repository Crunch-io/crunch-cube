from unittest import TestCase

import numpy as np

from .fixtures import ADMIT_X_DEPT_UNWEIGHTED
from .fixtures import ADMIT_X_GENDER_WEIGHTED
from .fixtures import SELECTED_CROSSTAB_4
from .fixtures import SEL_ARR_FIRST
from .fixtures import SEL_ARR_LAST
from .fixtures import MR_X_MR
from .fixtures import MR_X_MR_HETEROGENOUS

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
