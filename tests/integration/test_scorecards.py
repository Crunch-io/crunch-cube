# encoding: utf-8

"""Integration tests for scorecards."""

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.slices import FrozenSlice

from ..fixtures import SC  # ---mnemonic: SC = 'scorecards'---


class DescribeIntegratedScoreCard(object):
    def it_loads_from_simple_scorecard_json(self):
        slice_ = FrozenSlice(CrunchCube(SC.SANDWICHES_BY_MIKE))
        np.testing.assert_almost_equal(
            slice_.column_percentages,
            [
                [48.03921569, 37.0],
                [41.17647059, 40.0],
                [34.31372549, 45.0],
                [39.21568627, np.nan],
                [np.nan, 41.0],
            ],
        )
