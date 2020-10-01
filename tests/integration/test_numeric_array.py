# encoding: utf-8

"""Integration-test suite for numeric-arrays."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from cr.cube.cube import Cube

from ..fixtures import NA


class TestNumericArrays(object):
    def test_means_without_grouping_by_dimensions(self):
        slice_ = Cube(NA.NUM_ARR_MEANS_NO_DIMS).partitions[0]
        np.testing.assert_equal(slice_.means, [2.5, 25])
