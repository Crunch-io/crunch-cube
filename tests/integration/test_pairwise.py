from unittest import TestCase

import numpy as np
import pytest

from ..fixtures import CR

from cr.cube.crunch_cube import CrunchCube


# pylint: disable=missing-docstring, invalid-name, no-self-use
class TestStandardizedResiduals(TestCase):
    """Test cr.cube implementation of column family pairwise comparisons"""

    @pytest.mark.xfail
    def test_same_col_counts(self):
        """Test statistics for columns that are all the same."""
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        expected = np.zeros([4, 4])
        actual = cube.pairwise_chisq(axis=0)
        np.testing.assert_equal(actual, expected)

    @pytest.mark.xfail
    def test_same_col_pvals(self):
        """P-values for columns that are all the same."""
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        expected = np.ones([4, 4])
        actual = cube.pairwise_pvals(axis=0)
        np.testing.assert_equal(actual, expected)
