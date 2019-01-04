from unittest import TestCase

import numpy as np

from ..fixtures import CR

from cr.cube.crunch_cube import CrunchCube


# pylint: disable=missing-docstring, invalid-name, no-self-use
class TestStandardizedResiduals(TestCase):
    '''Test cr.cube implementation of column family pairwise comparisons'''

    def test_same_col_counts(self):
        '''Test statistics for columns that are all the same.'''
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        expected = np.zeros([4,4])
        actual = cube.pairwise_column_chisq()
        np.testing.assert_equal(actual, expected)

    def test_same_col_pvals(self):
        '''P-values for columns that are all the same.'''
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        expected = np.ones([4,4])
        actual = cube.pairwise_column_pvals()
        np.testing.assert_equal(actual, expected)
