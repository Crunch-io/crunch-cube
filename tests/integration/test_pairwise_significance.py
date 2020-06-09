# encoding: utf-8

"""Integration tests for pairwise comparisons."""

from unittest import TestCase

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR, SM
from ..util import load_python_expression


class Describe_Slice(object):
    """Integration-test suite for pairwise-t aspects of _Slice object."""

    @pytest.mark.parametrize(
        "fixture, pw_indices_dict, expectation",
        (
            (
                CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS,
                {},
                "cat-x-cat-hs-pw-idxs",
            ),
            (CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS, {}, "cat-x-cat-pw-idxs"),
            (
                CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS,
                {"only_larger": False},
                "cat-x-cat-pw-idxs-recip",
            ),
            (CR.MR_X_CAT, {}, "mr-x-cat-pw-idxs"),
            (CR.MR_X_CAT_HS, {}, "mr-x-cat-hs-pw-idxs"),
            (CR.EDU_FAV5_FAV5, {}, "cat-x-mr-aug-pw-idxs"),
            (CR.EDU_FAV5_FAV5, {"only_larger": False}, "cat-x-mr-aug-pw-idxs-recip"),
            (CR.CAT_HS_X_MR, {}, "cat-hs-x-mr-pw-idxs"),
            (CR.CAT_X_MR_2, {}, "cat-x-mr-pw-idxs"),
            (CR.MR_X_MR, {}, "mr-x-mr-pw-idxs"),
        ),
    )
    def it_provides_pairwise_indices(self, fixture, pw_indices_dict, expectation):
        slice_ = _Slice(
            Cube(fixture),
            slice_idx=0,
            transforms={"pairwise_indices": pw_indices_dict},
            population=None,
            mask_size=0,
        )

        pairwise_indices = slice_.pairwise_indices

        actual = pairwise_indices.tolist()
        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)

    @pytest.mark.parametrize(
        "fixture, pw_indices_dict, expectation",
        (
            (
                CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS,
                {"alpha": [0.175, 0.025], "only_larger": False},
                "cat-x-cat-hs-pw-idxs-alt",
            ),
            (CR.MR_X_CAT, {"alpha": [0.175, 0.01]}, "mr-x-cat-pw-idxs-alt"),
            (CR.EDU_FAV5_FAV5, {"alpha": [0.175, 0.01]}, "cat-x-mr-aug-pw-idxs-alt"),
        ),
    )
    def it_provides_pairwise_indices_alt(self, fixture, pw_indices_dict, expectation):
        """Provides indicies meeting secondary sig-test threshold, when specified."""
        slice_ = _Slice(
            Cube(fixture),
            slice_idx=0,
            transforms={"pairwise_indices": pw_indices_dict},
            population=None,
            mask_size=0,
        )

        pairwise_indices_alt = slice_.pairwise_indices_alt

        actual = pairwise_indices_alt.tolist()
        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (
                CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS,
                "cat-x-cat-hs-scale-mean-pw-idxs",
            ),
            (CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS, "cat-x-cat-scale-mean-pw-idxs"),
            (CR.CAT_X_MR_2, "cat-x-mr-scale-mean-pw-idxs"),
            (CR.EDU_FAV5_FAV5, "cat-x-mr-aug-scale-mean-pw-idxs"),
        ),
    )
    def it_provides_scale_mean_pairwise_indices(self, fixture, expectation):
        """Provides column-indicies meeting sig-test threshold on column scale means."""
        slice_ = _Slice(
            Cube(fixture), slice_idx=0, transforms={}, population=None, mask_size=0
        )

        actual = slice_.scale_mean_pairwise_indices

        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (
                CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS,
                "cat-x-cat-hs-scale-mean-pw-idxs-alt",
            ),
            (
                CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS,
                "cat-x-cat-scale-mean-pw-idxs-alt",
            ),
            (CR.CAT_X_MR_2, "cat-x-mr-scale-mean-pw-idxs-alt"),
            (CR.EDU_FAV5_FAV5, "cat-x-mr-aug-scale-mean-pw-idxs-alt"),
        ),
    )
    def it_provides_scale_mean_pairwise_indices_alt(self, fixture, expectation):
        """Provides col idxs meeting secondary sig-test threshold on scale mean."""
        slice_ = _Slice(
            Cube(fixture),
            slice_idx=0,
            transforms={"pairwise_indices": {"alpha": [0.01, 0.08]}},
            population=None,
            mask_size=0,
        )

        actual = slice_.scale_mean_pairwise_indices_alt

        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)


class TestStandardizedResiduals(TestCase):
    """Test cr.cube implementation of column family pairwise comparisons"""

    def test_pairwise_measures_scale_means_nps_type(self):
        slice_ = Cube(SM.FACEBOOK_APPS_X_AGE).partitions[0]
        # Testing col 0 with others
        actual = slice_.pairwise_significance_tests[0]

        assert actual.p_vals_scale_means[0] == 1
        assert actual.t_stats_scale_means[0] == 0
        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, -1.08966143, 1.27412884, 4.14629088]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0000000e00, 2.7638510e-01, 2.0302047e-01, 4.0226310e-05],
        )

        # Testing col 1 with others
        actual = slice_.pairwise_significance_tests[1]

        assert actual.p_vals_scale_means[1] == 1
        assert actual.t_stats_scale_means[1] == 0
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [2.7638510e-01, 1.0000000e00, 1.4344585e-03, 1.7905266e-09],
        )
        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [1.08966143, 0.0, 3.19741668, 6.10466696]
        )

    def test_ttests_scale_means_cat_x_cat_pruning_and_hs(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, -1.64461503, -1.92387847, np.nan, -1.06912069],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.1046981, 0.059721, np.nan, 0.2918845]
        )

        # Just H&S
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, -0.9387958, -1.644615, -1.9238785, np.nan, -1.0691207],
        )

        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0, 0.3500141, 0.1046981, 0.059721, np.nan, 0.2918845],
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
            [0.0, -0.93879579, -1.64461503, -1.92387847, -1.06912069],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.3500141, 0.1046981, 0.059721, 0.2918845]
        )

        # Pruning and H&S
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, -1.64461503, -1.92387847, -1.06912069]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.1046981, 0.059721, 0.2918845]
        )

    def test_pairwise_t_stats_scale_means_with_hs(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [
                0.0,
                0.70543134,
                0.06749456,
                3.59627589,
                4.15672996,
                3.26353764,
                0.30034779,
                0.8383222,
                3.02379687,
                0.85672124,
                4.26974185,
            ],
        )

        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [
                1.00000000e00,
                4.80680137e-01,
                9.46191784e-01,
                3.31862965e-04,
                3.31301800e-05,
                1.10780471e-03,
                7.63968024e-01,
                4.02022180e-01,
                2.52958600e-03,
                3.91811876e-01,
                2.03020376e-05,
            ],
        )

    def test_ttests_scale_means_use_unweighted_n_for_variance(self):
        slice_ = Cube(CR.CAT_X_CAT_WEIGHTED_TTESTS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, 4.38871748, 3.99008596, 5.15679647]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0000000e00, 1.3839564e-05, 7.4552516e-05, 4.0145665e-07],
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

    def test_cat_mr_x_itself_pairwise_compare_columns(self):
        slice_ = Cube(CR.EDU_FAV5_FAV5).partitions[0]

        actual = slice_.pairwise_significance_tests[2]
        expected_tstats = [
            [3.2386393, 2.09194923, 0.0, 2.8693425, -4.15518752],
            [6.88147131, 5.97625571, 0.0, 6.76454926, -0.48584994],
            [7.83715229, 3.62375238, 0.0, 4.59038147, -1.05390881],
            [6.47017555, 2.36733655, 0.0, 2.08742665, -0.35158859],
        ]
        expected_pvals = [
            [
                1.25043153e-03,
                3.67114360e-02,
                1.00000000e00,
                4.20783661e-03,
                3.49648630e-05,
            ],
            [
                1.19420029e-11,
                3.24026250e-09,
                1.00000000e00,
                2.38187248e-11,
                6.27168180e-01,
            ],
            [
                1.46549439e-14,
                3.05956626e-04,
                1.00000000e00,
                5.04126756e-06,
                2.92150962e-01,
            ],
            [
                1.70207404e-10,
                1.81187549e-02,
                1.00000000e00,
                3.71254466e-02,
                7.25212661e-01,
            ],
        ]

        assert slice_.cube_is_mr_by_itself is True
        np.testing.assert_array_almost_equal(actual.t_stats, expected_tstats)
        np.testing.assert_array_almost_equal(actual.p_vals, expected_pvals)

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
