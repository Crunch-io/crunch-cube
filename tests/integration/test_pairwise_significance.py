# encoding: utf-8

"""Integration tests for pairwise comparisons."""

from unittest import TestCase

import numpy as np

from cr.cube.cube import Cube

from ..fixtures import CR


class TestStandardizedResiduals(TestCase):
    """Test cr.cube implementation of column family pairwise comparisons"""

    def test_ttests_scale_means_cat_x_cat_pruning_and_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 0.27797695, 0.36961717, np.nan, 0.50325269],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.78188608, 0.71314053, np.nan, 0.61776958]
        )

        # Just H&S
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 0.13853038, 0.27797695, 0.36961717, np.nan, 0.50325269],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0, 0.89009174, 0.78188608, 0.71314053, np.nan, 0.61776958],
        )

        # Just pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 0.13853038, 0.27797695, 0.36961717, 0.50325269],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0, 0.89009174, 0.78188608, 0.71314053, 0.61776958],
        )

        # Pruning and H&S
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, 0.27797695, 0.36961717, 0.50325269]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.78188608, 0.71314053, 0.61776958]
        )

    def test_pairwise_t_stats_scale_means_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        t_stats_scale_means = slice_.pairwise_significance_tests[0].t_stats_scale_means

        np.testing.assert_almost_equal(
            t_stats_scale_means,
            [
                0.0,
                -0.02920515,
                -0.00204203,
                -0.12378367,
                -0.12308137,
                -0.09268705,
                -0.0130318,
                -0.03529702,
                -0.10098701,
                -0.04275669,
                -0.13194208,
            ],
        )

    def test_pairwise_p_vals_scale_means_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        p_vals_scale_means = slice_.pairwise_significance_tests[0].p_vals_scale_means

        np.testing.assert_almost_equal(
            p_vals_scale_means,
            [
                1.0,
                0.97670588,
                0.99837081,
                0.90150093,
                0.9020504,
                0.92615595,
                0.98960476,
                0.97184896,
                0.91957141,
                0.96590434,
                0.89504075,
            ],
        )

    def test_ttests_scale_means_use_unweighted_n_for_variance(self):
        slice_ = Cube(CR.CAT_X_CAT_WEIGHTED_TTESTS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]
        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, -0.27209886, -0.23255587, -0.40411283]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.78602559, 0.81625976, 0.68699271]
        )

    def test_pairwise_t_stats_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        expected = [
            [
                0.0,
                -0.06178448,
                0.30342874,
                -3.18865018,
                -3.82130608,
                -2.99560531,
                0.07456344,
                -0.68932699,
                -2.95469238,
                -0.46970468,
                -4.14956044,
            ],
            [
                0.0,
                1.18922394,
                0.38254102,
                3.3306654,
                3.45013209,
                2.7520633,
                0.59216241,
                0.86352416,
                2.54171145,
                1.10130414,
                3.3839919,
            ],
            [
                0.0,
                -1.7080666,
                -0.931165,
                -0.87419923,
                -0.24915622,
                -0.2367748,
                -0.97198009,
                -0.38504801,
                -0.03910193,
                -1.02720423,
                0.19773989,
            ],
        ]
        t_stats = slice_.pairwise_significance_tests[0].t_stats
        np.testing.assert_almost_equal(t_stats, expected)

    def test_pairwise_p_vals_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        expected = [
            [
                1.00000000e00,
                9.50744855e-01,
                7.61580875e-01,
                1.45494511e-03,
                1.35271514e-04,
                2.75262668e-03,
                9.40575477e-01,
                4.90755141e-01,
                3.16822332e-03,
                6.38672254e-01,
                3.44275537e-05,
            ],
            [
                1.00000000e00,
                2.34589189e-01,
                7.02082945e-01,
                8.84605265e-04,
                5.67562813e-04,
                5.94375533e-03,
                5.53862215e-01,
                3.88027539e-01,
                1.11098015e-02,
                2.71039211e-01,
                7.25442977e-04,
            ],
            [
                1.00000000e00,
                8.78852262e-02,
                3.51831360e-01,
                3.82131035e-01,
                8.03255937e-01,
                8.12841336e-01,
                3.31271828e-01,
                7.00272311e-01,
                9.68813218e-01,
                3.04581978e-01,
                8.43264694e-01,
            ],
        ]
        p_vals = slice_.pairwise_significance_tests[0].p_vals
        np.testing.assert_almost_equal(p_vals, expected)

    def test_pairwise_indices_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        expected = [
            [
                (3, 4, 5, 8, 10),
                (3, 4, 5, 8, 10),
                (3, 4, 5, 8, 10),
                (),
                (),
                (10,),
                (3, 4, 5, 8, 10),
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
                (0, 2),
                (),
                (),
                (0, 2),
                (),
                (0, 2, 6, 7),
            ],
            [(), (), (), (), (), (1,), (), (), (), (), (1,)],
        ]
        assert slice_.pairwise_indices.tolist() == expected

    def test_compare_to_column(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS).partitions[0]
        actual = slice_.pairwise_significance_tests[2]
        expected_tstats = [
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
        expected_pvals = [
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
        np.testing.assert_almost_equal(actual.t_stats, expected_tstats)
        np.testing.assert_almost_equal(actual.p_vals, expected_pvals)

    def test_cat_x_cat_pairwise_indices_only_larger(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS).partitions[0]
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
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

    def test_mr_x_cat_pairwise_indices_only_larger(self):
        # Without H&S
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.MR_X_CAT_HS, transforms=transforms).partitions[0]
        expected_indices = [
            [(1, 3, 4), (), (), (), (), ()],
            [(), (), (), (), (), ()],
            [(), (), (), (0,), (0,), ()],
            [(), (), (), (), (1,), ()],
            [(), (), (), (), (), ()],
        ]
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

        # With H&S
        slice_ = Cube(CR.MR_X_CAT_HS).partitions[0]
        expected_indices = [
            [(1, 4, 5, 7), (), (4, 5, 7), (), (), (), (), ()],
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (0,), (0,), (), (0,)],
            [(), (), (), (), (), (1, 2), (), (1, 2)],
            [(), (), (), (), (), (), (), ()],
        ]
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

    def test_cat_x_mr_pairwise_indices_only_larger(self):
        # Without H&S
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CAT_X_MR_HS, transforms=transforms).partitions[0]
        expected_indices = [
            [(1, 2, 3, 4), (2, 3), (), (), (2,)],
            [(), (), (), (), (3,)],
            [(), (), (), (), ()],
            [(), (0,), (0,), (0,), (0,)],
            [(), (), (), (0, 1, 4), ()],
            [(), (), (), (), ()],
        ]
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

        # With H&S
        slice_ = Cube(CR.CAT_X_MR_HS).partitions[0]
        expected_indices = [
            [(1, 2, 3, 4), (2, 3), (), (), (2,)],
            [(), (), (), (), (3,)],
            [(1, 2, 3, 4), (2, 3), (), (), (3,)],
            [(), (), (), (), ()],
            [(), (0,), (0,), (0,), (0,)],
            [(), (), (), (0, 1, 4), ()],
            [(), (), (), (), ()],
            [(), (0,), (0, 1), (0, 1, 4), (0,)],
        ]
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

    def test_mr_x_mr_pairwise_indices_only_larger(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        expected_indices = np.array(
            [
                [(1, 2, 3), (), (), ()],
                [(), (0, 2, 3), (), (2,)],
                [(), (), (0, 1, 3), (1,)],
                [(), (), (), ()],
            ]
        )
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

    def test_pairwise_indices_larger_and_smaller(self):
        transforms = {"pairwise_indices": {"only_larger": False}}
        slice_ = Cube(
            CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS, transforms=transforms
        ).partitions[0]
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
        np.testing.assert_array_equal(slice_.pairwise_indices, expected_indices)

    def test_cat_x_cat_summary_pairwise_indices(self):
        # Only larger
        slice_ = Cube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS).partitions[0]
        pairwise_indices = slice_.summary_pairwise_indices
        expected_indices = np.array([(2,), (0, 2), ()])
        np.testing.assert_array_equal(pairwise_indices, expected_indices)

        # Larger and smaller
        transforms = {"pairwise_indices": {"only_larger": False}}
        slice_ = Cube(
            CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS, transforms=transforms
        ).partitions[0]
        pairwise_indices = slice_.summary_pairwise_indices
        expected_indices = np.array([(1, 2), (0, 2), (0, 1)], dtype="i,i")
        np.testing.assert_array_equal(pairwise_indices, expected_indices)

    def test_ttests_use_unweighted_n_for_variance(self):
        """The weights on this cube demonstrate much higher variance (less
        extreme t values, and higher associated p-values) than if weighted_n
        were used in the variance estimate of the test statistic.
        """
        slice_ = Cube(CR.CAT_X_CAT_WEIGHTED_TTESTS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]
        expected_tstats = np.array(
            [
                [0.0, 1.3892930788974391, 0.8869425734660505, 1.402945620973322],
                [0.0, 0.1903540333363253, 0.30894158244285624, 0.3994739596013725],
                [0.0, 0.03761142927757482, 1.2682277741610029, 0.36476016345069556],
                [0.0, -1.187392798652706, -1.0206496663686406, -1.35111583891054],
                [0.0, -1.742783579889951, -2.425391682127969, -3.0738474093706927],
            ]
        ).reshape(5, 4)
        expected_pvals = np.array(
            [
                [1.0, 0.1673820620286901, 0.37579738470724267, 0.16373028998420036],
                [1.0, 0.8493616019040273, 0.7575734897713429, 0.6903959137827367],
                [1.0, 0.9700615941125716, 0.20566822638024163, 0.7160606992310101],
                [1.0, 0.23747780923355655, 0.30821629616167123, 0.17970733830083074],
                [1.0, 0.0839987707197456, 0.015862691173528676, 0.002723927327002773],
            ]
        ).reshape(5, 4)
        np.testing.assert_almost_equal(actual.t_stats, expected_tstats)
        np.testing.assert_almost_equal(actual.p_vals, expected_pvals)
        pairwise_indices = slice_.pairwise_indices
        expected_indices = np.array(
            [
                [(), (), (), ()],
                [(), (), (), ()],
                [(), (), (), ()],
                [(), (), (), ()],
                [(2, 3), (), (), ()],
            ]
        )
        np.testing.assert_array_equal(pairwise_indices, expected_indices)
