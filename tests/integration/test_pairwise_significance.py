# encoding: utf-8
# pylint: disable=protected-access

"""Integration tests for pairwise comparisons."""

from unittest import TestCase
import pytest
import json

import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.measures.wishart_pairwise_significance import WishartPairwiseSignificance

from ..fixtures import CR
from ..util import load_expectation


# pylint: disable=missing-docstring, invalid-name, no-self-use
class TestStandardizedResiduals(TestCase):
    """Test cr.cube implementation of column family pairwise comparisons"""

    def test_mr_x_cat_residuals(self):
        expected = np.array(json.loads(load_expectation("mr_x_cat_chi_squared_rows")))
        cube = CrunchCube(CR.MR_X_CAT_NOT_SO_SIMPLE_ALLTYPES)
        actual = WishartPairwiseSignificance._factory(
            cube.slices[0], axis=0, weighted=True
        )._pairwise_chisq
        for i in range(len(actual)):
            np.testing.assert_almost_equal(actual[i], expected[i])

    def test_mr_x_cat_pvals(self):
        cube = CrunchCube(CR.MR_X_CAT_NOT_SO_SIMPLE_ALLTYPES)
        expectation = json.loads(load_expectation("mr_x_cat_pairwise_pvals"))
        # Take only 0th slice, because it's the only one
        actual = cube.wishart_pairwise_pvals(axis=0)[0]
        for act, expected in zip(actual, expectation):
            np.testing.assert_almost_equal(act, np.array(expected))

    def test_same_col_counts(self):
        """Test statistics for columns that are all the same."""
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        pairwise_pvalues = WishartPairwiseSignificance._factory(
            cube.slices[0], axis=0, weighted=True
        )
        expected = np.zeros([4, 4])
        actual = pairwise_pvalues._pairwise_chisq
        np.testing.assert_equal(actual, expected)

    def test_hirotsu_chisq(self):
        """Test statistic for hirotsu data matches R"""
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION)
        pairwise_pvalues = WishartPairwiseSignificance._factory(
            cube.slices[0], axis=0, weighted=True
        )
        expected = np.array(
            [
                [
                    0.0,
                    2.821910158116655,
                    0.9259711818781733,
                    12.780855448128131,
                    16.79727869630099,
                    0.924655442873681,
                    0.8008976269312448,
                    9.616972398702428,
                    1.4496863124510315,
                    18.556098937181705,
                ],
                [
                    2.821910158116655,
                    0.0,
                    1.6831132737959318,
                    8.683471852181562,
                    13.451053159265136,
                    0.38467827774871005,
                    1.5094961530071807,
                    9.081312924348003,
                    0.25833985406056126,
                    16.3533306337074,
                ],
                [
                    0.9259711818781733,
                    1.6831132737959318,
                    0.0,
                    24.348935423464653,
                    46.689386077899826,
                    0.18470822825752797,
                    1.376598707986204,
                    22.063658540387774,
                    1.0102118795109807,
                    47.62124004565971,
                ],
                [
                    12.780855448128131,
                    8.683471852181562,
                    24.348935423464653,
                    0.0,
                    0.8073979083263744,
                    8.490641259215641,
                    5.141740694105387,
                    1.2536004848874829,
                    3.576241745092247,
                    2.1974561987876613,
                ],
                [
                    16.79727869630099,
                    13.451053159265136,
                    46.689386077899826,
                    0.8073979083263744,
                    0.0,
                    11.792012011326468,
                    6.847609367845222,
                    0.743555569450378,
                    5.218390456727495,
                    0.725476017865348,
                ],
                [
                    0.924655442873681,
                    0.38467827774871005,
                    0.18470822825752797,
                    8.490641259215641,
                    11.792012011326468,
                    0.0,
                    0.7072537831958036,
                    7.620018353425002,
                    0.3321969685319031,
                    14.087591553810693,
                ],
                [
                    0.8008976269312448,
                    1.5094961530071807,
                    1.376598707986204,
                    5.141740694105387,
                    6.847609367845222,
                    0.7072537831958036,
                    0.0,
                    3.6724354409467352,
                    0.39674326208673527,
                    8.546159019524978,
                ],
                [
                    9.616972398702428,
                    9.081312924348003,
                    22.063658540387774,
                    1.2536004848874829,
                    0.743555569450378,
                    7.620018353425002,
                    3.6724354409467352,
                    0.0,
                    3.4464292421171003,
                    1.5916695633869193,
                ],
                [
                    1.4496863124510315,
                    0.25833985406056126,
                    1.0102118795109807,
                    3.576241745092247,
                    5.218390456727495,
                    0.3321969685319031,
                    0.39674326208673527,
                    3.4464292421171003,
                    0.0,
                    6.85424450468994,
                ],
                [
                    18.556098937181705,
                    16.3533306337074,
                    47.62124004565971,
                    2.1974561987876613,
                    0.725476017865348,
                    14.087591553810693,
                    8.546159019524978,
                    1.5916695633869193,
                    6.85424450468994,
                    0.0,
                ],
            ]
        )
        actual = pairwise_pvalues._pairwise_chisq
        np.testing.assert_almost_equal(actual, expected)

    def test_same_col_pvals(self):
        """P-values for columns that are all the same."""
        cube = CrunchCube(CR.SAME_COUNTS_3x4)
        expected = [np.ones([4, 4])]
        actual = cube.wishart_pairwise_pvals(axis=0)
        np.testing.assert_equal(actual, expected)

        # Assert correct exception in case of not-implemented direction
        with pytest.raises(NotImplementedError):
            cube.wishart_pairwise_pvals(axis=1)

    def test_hirotsu_pvals(self):
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION)
        actual = cube.wishart_pairwise_pvals(axis=0)
        expected = [
            np.array(
                [
                    1,
                    0.999603716443816,
                    0.99999993076784,
                    0.435830186989942,
                    0.171365670494448,
                    0.999999931581745,
                    0.999999979427862,
                    0.726740806122402,
                    0.999997338047414,
                    0.105707739899106,
                    0.999603716443816,
                    1,
                    0.999991395396033,
                    0.806407150042716,
                    0.380296648898666,
                    0.999999999961875,
                    0.999996333717649,
                    0.773583582093158,
                    0.999999999998836,
                    0.192375246184738,
                    0.99999993076784,
                    0.999991395396033,
                    1,
                    0.017277623171216,
                    3.29012189337341e-06,
                    0.99999999999994,
                    0.999998237045896,
                    0.0365273119329589,
                    0.999999857555538,
                    2.23456306602809e-06,
                    0.435830186989942,
                    0.806407150042716,
                    0.017277623171216,
                    1,
                    0.999999977981595,
                    0.821586701043061,
                    0.982573114952466,
                    0.999999169027016,
                    0.998041030837588,
                    0.999934687968906,
                    0.171365670494448,
                    0.380296648898666,
                    3.29012189337341e-06,
                    0.999999977981595,
                    1,
                    0.52406354520284,
                    0.926322806048378,
                    0.99999998900118,
                    0.981100354607917,
                    0.999999991067971,
                    0.999999931581745,
                    0.999999999961875,
                    0.99999999999994,
                    0.821586701043061,
                    0.52406354520284,
                    1,
                    0.999999992799126,
                    0.883025655503086,
                    0.99999999998941,
                    0.33149560264078,
                    0.999999979427862,
                    0.999996333717649,
                    0.999998237045896,
                    0.982573114952466,
                    0.926322806048378,
                    0.999999992799126,
                    1,
                    0.997674862917282,
                    0.99999999995011,
                    0.81726901111794,
                    0.726740806122402,
                    0.773583582093158,
                    0.0365273119329589,
                    0.999999169027016,
                    0.99999998900118,
                    0.883025655503086,
                    0.997674862917282,
                    1,
                    0.998461227115608,
                    0.999994436499243,
                    0.999997338047414,
                    0.999999999998836,
                    0.999999857555538,
                    0.998041030837588,
                    0.981100354607917,
                    0.99999999998941,
                    0.99999999995011,
                    0.998461227115608,
                    1,
                    0.925999125959122,
                    0.105707739899106,
                    0.192375246184738,
                    2.23456306602809e-06,
                    0.999934687968906,
                    0.999999991067971,
                    0.33149560264078,
                    0.81726901111794,
                    0.999994436499243,
                    0.925999125959122,
                    1,
                ]
            ).reshape(10, 10)
        ]
        np.testing.assert_almost_equal(actual, expected)

    def test_hirotsu_pvals_with_hs(self):
        """The shape of the result should be 11 x 11, with H&S (at index 5)."""
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS)
        expected = [
            np.array(
                [
                    1,
                    0.999603716443816,
                    0.99999993076784,
                    0.435830186989942,
                    0.171365670494448,
                    np.nan,
                    0.999999931581745,
                    0.999999979427862,
                    0.726740806122402,
                    0.999997338047414,
                    0.105707739899106,
                    0.999603716443816,
                    1,
                    0.999991395396033,
                    0.806407150042716,
                    0.380296648898666,
                    np.nan,
                    0.999999999961875,
                    0.999996333717649,
                    0.773583582093158,
                    0.999999999998836,
                    0.192375246184738,
                    0.99999993076784,
                    0.999991395396033,
                    1,
                    0.017277623171216,
                    3.29012189337341e-06,
                    np.nan,
                    0.99999999999994,
                    0.999998237045896,
                    0.0365273119329589,
                    0.999999857555538,
                    2.23456306602809e-06,
                    0.435830186989942,
                    0.806407150042716,
                    0.017277623171216,
                    1,
                    0.999999977981595,
                    np.nan,
                    0.821586701043061,
                    0.982573114952466,
                    0.999999169027016,
                    0.998041030837588,
                    0.999934687968906,
                    0.171365670494448,
                    0.380296648898666,
                    3.29012189337341e-06,
                    0.999999977981595,
                    1,
                    np.nan,
                    0.52406354520284,
                    0.926322806048378,
                    0.99999998900118,
                    0.981100354607917,
                    0.999999991067971,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0.999999931581745,
                    0.999999999961875,
                    0.99999999999994,
                    0.821586701043061,
                    0.52406354520284,
                    np.nan,
                    1,
                    0.999999992799126,
                    0.883025655503086,
                    0.99999999998941,
                    0.33149560264078,
                    0.999999979427862,
                    0.999996333717649,
                    0.999998237045896,
                    0.982573114952466,
                    0.926322806048378,
                    np.nan,
                    0.999999992799126,
                    1,
                    0.997674862917282,
                    0.99999999995011,
                    0.81726901111794,
                    0.726740806122402,
                    0.773583582093158,
                    0.0365273119329589,
                    0.999999169027016,
                    0.99999998900118,
                    np.nan,
                    0.883025655503086,
                    0.997674862917282,
                    1,
                    0.998461227115608,
                    0.999994436499243,
                    0.999997338047414,
                    0.999999999998836,
                    0.999999857555538,
                    0.998041030837588,
                    0.981100354607917,
                    np.nan,
                    0.99999999998941,
                    0.99999999995011,
                    0.998461227115608,
                    1,
                    0.925999125959122,
                    0.105707739899106,
                    0.192375246184738,
                    2.23456306602809e-06,
                    0.999934687968906,
                    0.999999991067971,
                    np.nan,
                    0.33149560264078,
                    0.81726901111794,
                    0.999994436499243,
                    0.925999125959122,
                    1,
                ]
            ).reshape(11, 11)
        ]
        actual = cube.wishart_pairwise_pvals(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_odd_latent_dimensions(self):
        """Test code path for the pfaffian of an odd-dimensioned matrix
        Latent matrix size is (n_min - 1) so 3 for a 4x4 table.
        """
        cube = CrunchCube(CR.PAIRWISE_4X4)
        actual = cube.wishart_pairwise_pvals(axis=0)
        expected = [
            np.array(
                [
                    1,
                    0.949690252544668,
                    0.917559534718816,
                    0.97630069244232,
                    0.949690252544668,
                    1,
                    0.35511045473507,
                    0.999999119644564,
                    0.917559534718816,
                    0.35511045473507,
                    1,
                    0.313869429972919,
                    0.97630069244232,
                    0.999999119644564,
                    0.313869429972919,
                    1,
                ]
            ).reshape(4, 4)
        ]
        np.testing.assert_almost_equal(actual, expected)

    def test_pairwise_with_zero_margin(self):
        """Assert pairwise doesn't break when a 0 value is in any slice margin.

        We currently don't know the correct maths behind this. This test needs to be
        updated once we figure this out.
        """
        cube = CrunchCube(CR.PAIRWISE_WITH_ZERO_MARGIN)

        # TODO: Replace with updated expectation when maths is figured out
        expected = [np.ones((10, 10), dtype=float)]

        actual = cube.wishart_pairwise_pvals(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_compare_to_column(self):
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS)
        actual = cube.compare_to_column(slice=0, column=2)
        expected_tstats = np.array(
            [
                [0.926265419379764, -1.06821799614525, 0],
                [2.46608073666841, 1.45733791450049, 0],
                [4.47444356884568, -0.127565104025694, 0],
                [-0.825615066069789, 1.09455611515513, 0],
                [-3.08993537423754, 0.0491518359221346, 0],
                [1.72556180249472, 0.403257023461294, 0],
                [0.633715242008222, -0.215815666049458, 0],
                [-1.67730697176792, -0.472402692576022, 0],
                [1.17745694642709, 0.663805453182429, 0],
                [-3.74325301090516, -0.723631945942289, 0],
            ]
        ).reshape(10, 3)
        expected_pvals = np.array(
            [
                [0.3543704462583741, 0.2854486628456121, 1],
                [0.0137065536277976, 0.1450553643559893, 1],
                [0.0000078991077082, 0.898495781851322, 1],
                [0.4090774528587606, 0.2737382169902576, 1],
                [0.0020173609289613, 0.9607993020139403, 1],
                [0.0845124966799853, 0.6867680056298973, 1],
                [0.5263071263993964, 0.8291359554655602, 1],
                [0.093569693339443, 0.6366499668820378, 1],
                [0.2390913772747494, 0.5068305468138976, 1],
                [0.0001845038553001, 0.4693091017156237, 1],
            ]
        )
        np.testing.assert_almost_equal(actual.t_stats, expected_tstats)
        np.testing.assert_almost_equal(actual.p_vals, expected_pvals)

    def test_pairwise_indices_only_larger(self):
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS)
        expected_indices = np.array(
            [
                [(1,), (), ()],
                [(2,), (), ()],
                [(1, 2), (), ()],
                [(), (0,), ()],
                [(), (0,), (0,)],
                [(), (), ()],
                [(), (), ()],
                [(), (), ()],
                [(), (), ()],
                [(), (0,), (0,)],
            ]
        )
        pairwise_indices = cube.slices[0].pairwise_indices()
        np.testing.assert_array_equal(pairwise_indices, expected_indices)

    def test_pairwise_indices_larger_and_smaller(self):
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS)
        expected_indices = np.array(
            [
                [(1,), (0,), ()],
                [(2,), (), (0,)],
                [(1, 2), (0,), (0,)],
                [(1,), (0,), ()],
                [(1, 2), (0,), (0,)],
                [(), (), ()],
                [(), (), ()],
                [(), (), ()],
                [(), (), ()],
                [(1, 2), (0,), (0,)],
            ]
        )
        pairwise_indices = cube.slices[0].pairwise_indices(only_larger=False)
        np.testing.assert_array_equal(pairwise_indices, expected_indices)
