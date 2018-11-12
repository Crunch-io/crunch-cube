# encoding: utf-8

"""Integration test suite for cr.cube.cube_slice module."""

import numpy as np
import pytest

from cr.cube.crunch_cube import CrunchCube

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---


class DescribeIntegratedCubeSlice(object):

    def it_provides_a_console_friendly_repr_for_a_slice(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        slice_ = cube.slices[0]

        repr_ = repr(slice_)

        assert repr_ == (
            "CubeSlice(name='v4', dim_types='CAT x CAT', dims='v4 x v7')\n"
            "      C    E\n"
            "--  ---  ---\n"
            "B     5    2\n"
            "C     5    3"
        )

    @pytest.mark.xfail(reason='WIP', strict=True)
    def it_knows_zscore_for_MR_X_MR_X_CAT(self):
        cube = CrunchCube(CR.MR_X_MR_X_CAT)
        slice_ = cube.slices[0]

        zscore = slice_.zscore()

        np.testing.assert_array_almost_equal(
            zscore,
            # ---fix this expectation once we have plausible result---
            np.array([
                [1, 2, 3],
                [4, 5, 6]
            ])
        )


def test_labels_with_hs_and_pruning():
    cs = CrunchCube(CR.CAT_X_CAT_PRUNING_HS).slices[0]

    # Withouut pruning or H&S
    expected = [
        [
            u'Married', u'Separated', u'Divorced', u'Widowed', u'Single',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Republicans in Congress', u'Both',
            u'Neither', u'Not sure',
        ],
    ]
    actual = cs.labels()
    assert actual == expected

    # Apply pruning
    expected = [
        [
            u'Married', u'Separated', u'Divorced', u'Widowed',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Republicans in Congress', u'Both',
            u'Not sure',
        ],
    ]
    actual = cs.labels(prune=True)
    assert actual == expected

    # Apply H&S
    expected = [
        [
            u'Married', u'left alone', u'Separated', u'Divorced', u'Widowed',
            u'Single', u'Domestic partnership',
        ],
        [
            u'President Obama', u'Obama + Republicans',
            u'Republicans in Congress', u'Both', u'Neither', u'Not sure',
        ],
    ]
    actual = cs.labels(hs_dims=[0, 1])
    assert actual == expected

    # Apply H&S and pruning
    expected = [
        [
            u'Married', u'left alone', u'Separated', u'Divorced', u'Widowed',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Obama + Republicans',
            u'Republicans in Congress', u'Both', u'Not sure',
        ],
    ]
    actual = cs.labels(prune=True, hs_dims=[0, 1])
    assert actual == expected
