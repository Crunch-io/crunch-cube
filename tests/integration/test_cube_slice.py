# encoding: utf-8

"""Integration test suite for cr.cube.cube_slice module."""

import numpy as np

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

    def it_knows_zscore_for_MR_X_MR_X_CAT(self):
        cube = CrunchCube(CR.MR_X_MR_X_CAT)
        slice_ = cube.slices[0]

        zscore = slice_.zscore()

        np.testing.assert_array_almost_equal(
            zscore,
            np.array([
                [-2.0890161, 2.0890161],
                [-0.31167807, 0.31167807],
                [-0.4574511, 0.4574511],
                [0.08713899, -0.08713899],
                [-1.88534457, 1.88534457],
                [-1.18201963, 1.18201963],
                [-0.14570511, 0.14570511],
                [-0.79331024, 0.79331024],
                [-0.10870154, 0.10870154],
                [0.39665764, -0.39665764],
                [-0.76214626, 0.76214626],
                [-1.59164242, 1.59164242],
                [np.nan, np.nan],
                [-1.97343777, 1.97343777],
                [-0.63278752, 0.63278752],
                [-0.4446455, 0.4446455],
                [-0.10084772, 0.10084772],
                [-0.42861964, 0.42861964],
            ])
        )


class DescribeCubeSliceAPI(object):

    def it_calculates_correct_index_tables_for_single_elements(self):
        cs = CrunchCube(CR.MR_X_CAT_BOTH_SINGLE_ELEMENT).slices[0]

        # Check for column direction (as in the exporter), backed by R
        expected = np.array([[0., 110.39006714]])
        index_table = cs.index_table(axis=0)
        np.testing.assert_array_almost_equal(index_table, expected)

        # Check for row direction, backed by R
        expected = np.array([[0., 116.472612]])
        index_table = cs.index_table(axis=1)
        np.testing.assert_array_almost_equal(index_table, expected)


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
