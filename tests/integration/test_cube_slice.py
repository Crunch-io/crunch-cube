# encoding: utf-8

"""Integration test suite for cr.cube.cube_slice module."""

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.util import compress_pruned

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

    def it_prunes_margin_with_only_empty_hs(self):
        cube = CrunchCube(CR.CA_SUBVAR_HS_X_CA_CAT_X_CAT_COLPCT)
        cube_slice = cube.slices[2]

        total_margin = cube_slice.margin(
            prune=True, include_transforms_for_dims=[0, 1])
        assert type(total_margin) is np.ma.core.MaskedConstant

        column_margin = cube_slice.margin(
            axis=0, prune=True, include_transforms_for_dims=[0, 1])
        assert all(column_margin.mask)

        row_margin = cube_slice.margin(
            axis=1, prune=True, include_transforms_for_dims=[0, 1])
        assert all(row_margin.mask)

    def it_prunes_mr_x_cat_pvals_correctly(self):
        cube = CrunchCube(CR.MR_X_CAT_PRUNE_WITH_SIG)
        cube_slice = cube.slices[0]

        col_margin = compress_pruned(cube_slice.margin(axis=0, prune=True))
        assert col_margin.shape == (7, 7)

        total_margin = compress_pruned(cube_slice.margin(prune=True))
        assert total_margin.shape == (7,)

    def it_prunes_mr_x_mr_margins_correctly(self):
        cube_slice = CrunchCube(CR.NETWORK_LIVE_TV).slices[0]
        row_margin = compress_pruned(cube_slice.margin(axis=1, prune=True))
        np.testing.assert_array_almost_equal(
            row_margin,
            np.array([
                [18665, 18665, 18665, 18665, 18665, 18665],
                [17323, 17323, 17323, 17323, 17323, 17323],
                [14974, 14974, 14974, 14974, 14974, 14974],
                [49439, 49439, 49439, 49439, 49439, 49439],
                [25600, 25600, 25600, 25600, 25600, 25600],
                [43920, 43920, 43920, 43920, 43920, 43920],
                [37415, 37415, 37415, 37415, 37415, 37415],
                [20934, 20934, 20934, 20934, 20934, 20934],
                [17059, 17059, 17059, 17059, 17059, 17059],
            ]),
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
