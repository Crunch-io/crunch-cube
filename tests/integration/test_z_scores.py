from unittest import TestCase

import numpy as np

from .fixtures import FIXT_ADMIT_X_DEPT_UNWEIGHTED
from .fixtures import FIXT_ADMIT_X_GENDER_WEIGHTED
from .fixtures import FIXT_SELECTED_CROSSTAB_4
from .fixtures import FIXT_SEL_ARR_FIRST
from .fixtures import FIXT_SEL_ARR_LAST

from cr.cube.crunch_cube import CrunchCube


class TestZScores(TestCase):
    '''Test cr.cube implementation of Z-Scores.'''

    def test_z_scores_admit_x_dept_unweighted(self):
        '''Z-Scores for normal unweighted crosstab.'''
        cube = CrunchCube(FIXT_ADMIT_X_DEPT_UNWEIGHTED)
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
        actual = cube.z_scores
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_x_gender_weighted(self):
        '''Z-Scores for normal weighted crosstab.'''
        cube = CrunchCube(FIXT_ADMIT_X_GENDER_WEIGHTED)
        expected = np.array([
            [9.42561985, -9.42561985],
            [-9.42561985, 9.42561985],
        ])
        actual = cube.z_scores
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_selected_crosstab(self):
        '''Residuals for MR x CAT unweighted.'''
        cube = CrunchCube(FIXT_SELECTED_CROSSTAB_4)
        expected = np.array([
            [-10.88317888, 10.88317888],
            [5.23577326, -5.23577326],
            [-8.89089261, 8.89089261],
            [-7.31367932, 7.31367932],
            [15.39360198, -15.39360198],
            [-1.15219648, 1.15219648],
        ])
        actual = cube.z_scores
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_mr_x_cat(self):
        '''Residuals for MR x CAT from app.'''
        cube = CrunchCube(FIXT_SEL_ARR_FIRST)
        expected = np.array([
            [0.80134191, -0.80134191],
            [0.60455606, -0.60455606],
            [-0.30884247, 0.30884247],
        ])
        actual = cube.z_scores
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_cat_x_mr(self):
        '''Residuals for CAT x MR from app.

        The results should be the exact transsposition of the results from
        'test_z_scores_mr_x_cat' test.
        '''
        cube = CrunchCube(FIXT_SEL_ARR_LAST)
        expected = np.array([
            [0.80134191, 0.60455606, -0.30884247],
            [-0.80134191, -0.60455606, 0.30884247],
        ])
        actual = cube.z_scores
        np.testing.assert_almost_equal(actual, expected)
