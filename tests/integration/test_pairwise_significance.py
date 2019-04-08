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

    def test_mr_x_cat_wishart_residuals(self):
        expected = np.array(json.loads(load_expectation("mr_x_cat_chi_squared_rows")))
        cube = CrunchCube(CR.MR_X_CAT_NOT_SO_SIMPLE_ALLTYPES)
        actual = WishartPairwiseSignificance._factory(
            cube.slices[0], axis=0, weighted=True
        )._pairwise_chisq
        for i in range(len(actual)):
            np.testing.assert_almost_equal(actual[i], expected[i])

    def test_mr_x_cat_wishart_pvals(self):
        cube = CrunchCube(CR.MR_X_CAT_NOT_SO_SIMPLE_ALLTYPES)
        expectation = json.loads(load_expectation("mr_x_cat_pairwise_pvals"))
        # Take only 0th slice, because it's the only one
        actual = cube.wishart_pairwise_pvals(axis=0)[0]
        for act, expected in zip(actual, expectation):
            np.testing.assert_almost_equal(act, np.array(expected))

    def test_mr_x_cat_wishart_pairwise_indices(self):
        slice_ = CrunchCube(CR.MR_X_CAT_NOT_SO_SIMPLE_ALLTYPES).slices[0]
        expected = np.array(
            [
                [(2, 4), (), (0,), (), (0,), ()],
                [(), (), (), (), (), ()],
                [(), (), (), (), (), ()],
                [(), (), (), (), (), ()],
                [(), (), (), (), (), ()],
            ]
        )
        actual_pairwise_indices = slice_.wishart_pairwise_indices(axis=0)
        np.testing.assert_array_equal(actual_pairwise_indices, expected)

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
            json.loads(load_expectation("cat_x_cat_hirotsu_chi_squared"))
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
        expected = [np.array(json.loads(load_expectation("cat_x_cat_hirotsu_pvals")))]
        np.testing.assert_almost_equal(actual, expected)

    def test_hirotsu_pairwise_indices(self):
        slice_ = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION).slices[0]
        actual_pairwise_indices = slice_.wishart_pairwise_indices(axis=0)
        expected = np.array([(), (), (3, 4, 7, 9), (2,), (2,), (), (), (2,), (), (2,)])
        np.testing.assert_array_equal(actual_pairwise_indices, expected)

    def test_pairwise_indices_with_hs(self):
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS)
        expected = [
            [
                (3, 4, 8, 10),
                (3, 4, 8, 10),
                (3, 4, 8, 10),
                (),
                (),
                np.nan,
                (3, 4, 8, 10),
                (3, 4, 10),
                (),
                (4, 10),
                (),
            ],
            [
                (),
                (),
                (),
                (0, 2, 6, 7),
                (0, 2, 6, 7),
                np.nan,
                (),
                (),
                (0, 2),
                (),
                (0, 2, 6, 7),
            ],
            [(), (), (), (), (), np.nan, (), (), (), (), (1,)],
        ]
        actual = cube.slices[0].pairwise_indices(hs_dims=[0, 1]).tolist()
        assert actual == expected

    def test_hirotsu_pvals_with_hs(self):
        """The shape of the result should be 11 x 11, with H&S (at index 5)."""
        cube = CrunchCube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS)
        expected = [
            np.array(json.loads(load_expectation("cat_x_cat_with_hs_hirotsu_pvals")))
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
        actual = cube.compare_to_column(slice_idx=0, column_idx=2)
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
