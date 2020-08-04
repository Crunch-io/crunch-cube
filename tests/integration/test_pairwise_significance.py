# encoding: utf-8

"""Integration tests for pairwise comparisons."""

from unittest import TestCase

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice
from cr.cube.util import counts_with_subtotals

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

        actual = slice_.pairwise_indices

        expected = np.array(load_python_expression(expectation), dtype=tuple)
        assert (expected == actual).all(), "\n%s\n\n%s" % (expected, actual)

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

        actual = slice_.pairwise_indices_alt

        expected = np.array(load_python_expression(expectation), dtype=tuple)
        assert (expected == actual).all(), "\n%s\n\n%s" % (expected, actual)

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

    def test_cat_x_cat_summary_pairwise_indices(self):
        # Only larger
        slice_ = Cube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS).partitions[0]
        pairwise_indices = slice_.summary_pairwise_indices
        expected_indices = np.array([(2,), (0, 2), ()], dtype=tuple)
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
            ],
            dtype=tuple,
        )
        np.testing.assert_array_equal(pairwise_indices, expected_indices)

    def test_cat_x_mr_weighted_augmented(self):
        """Same behaviour of test_mr_subvar_x_mr_augmented_pairwise_t_test"""
        slice_ = Cube(CR.CAT_X_MR_WEIGHTED_AUGMENTED).partitions[0]
        actual = slice_.pairwise_significance_tests[1]
        overlap_margins = np.sum(slice_._cube.counts, axis=0)[:, 0, :, 0]
        shadow_proportions_tab1 = slice_._cube.counts[0][:, 0, :, 0] / overlap_margins
        shadow_proportions_tab2 = slice_._cube.counts[1][:, 0, :, 0] / overlap_margins

        assert slice_.cube_is_mr_aug is True
        np.testing.assert_array_almost_equal(
            overlap_margins,
            [
                [4188.13667426, 3046.38734874, 1047.76027958],
                [3046.38734874, 3628.42165249, 859.70982873],
                [1047.76027958, 859.70982873, 5133.76481161],
            ],
        )
        np.testing.assert_array_almost_equal(
            slice_.column_proportions,
            [
                [0.71392963, 0.74796797, 0.55475245],
                [0.28607037, 0.25203203, 0.44524755],
            ],
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1.diagonal(), slice_.column_proportions[0, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2.diagonal(), slice_.column_proportions[1, :]
        )
        # each diagonal of the shadow proportions tab is equal to the correspondent row
        # for the actual proportions
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            [[-10.2964264, 0.0, -20.09577285], [5.96350953, 0.0, 24.14335882]],
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            [[0.0, 1.00000000e00, 0.0], [2.55612775e-09, 1.00000000e00, 0.0]],
        )
        assert slice_._cube.counts.shape == (2, 3, 2, 3, 2)
        assert actual.t_stats.shape == (2, 3)

    def test_ca_subvar_hs_x_mr_augmented(self):
        slice_ = Cube(CR.CA_SUBVAR_HS_X_MR_AUGMENTED).partitions[0]
        actual = slice_.pairwise_significance_tests[1]
        overlap_margins = np.sum(slice_._cube.counts, axis=0)[:, 0, :, 0]
        addend_idxs = [s.addend_idxs for s in slice_._cube.dimensions[0].subtotals]
        counts_with_hs = counts_with_subtotals(
            addend_idxs, slice_.inserted_row_idxs, slice_._cube.counts
        )
        assert slice_.inserted_row_idxs == (0,)
        assert slice_.cube_is_mr_aug is True
        assert actual.t_stats.shape == (3, 4)
        assert slice_.counts.shape == (3, 4)
        assert counts_with_hs.shape == (3, 4, 2, 4, 2)
        np.testing.assert_array_almost_equal(
            overlap_margins,
            [[44, 3, 0, 0], [3, 34, 0, 0], [0, 0, 348, 0], [0, 0, 0, 0]],
        )

        slice_no_aug_ = Cube(CR.CA_SUBVAR_HS_X_MR).partitions[0]

        for i in range(slice_no_aug_.shape[0]):
            np.testing.assert_array_almost_equal(
                slice_no_aug_.counts[i], counts_with_hs[i][:, 0, :, 0].diagonal()
            )

    def test_cat_hs_x_mr_augmented_wgtd(self):
        slice_ = Cube(CR.CAT_HS_X_MR_AUGMENTED_WGTD).partitions[0]
        actual = slice_.pairwise_significance_tests[1]
        overlap_margins = np.sum(slice_._cube.counts, axis=0)[:, 0, :, 0]
        addend_idxs = [s.addend_idxs for s in slice_._cube.dimensions[0].subtotals]
        counts_with_hs = counts_with_subtotals(
            addend_idxs, slice_.inserted_row_idxs, slice_._cube.counts
        )

        # CATxMR (9, 3, 2, 3, 2) shape, 9 = (5 + 4subtot) tabs of shadow proportions
        assert slice_.inserted_row_idxs == (0, 1, 4, 8)
        assert slice_.cube_is_mr_aug is True
        assert actual.t_stats.shape == (9, 3)
        assert slice_.counts.shape == (9, 3)
        np.testing.assert_array_almost_equal(
            overlap_margins,
            np.array(
                [
                    [4188.13667426, 3046.38734874, 1047.76027958],
                    [3046.38734874, 3628.42165249, 859.70982873],
                    [1047.76027958, 859.70982873, 5133.76481161],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            np.array(
                [
                    [6.11927895, 0.0, -14.1303337],
                    [-7.69445366, 0.0, 12.49573404],
                    [5.69285527, 0.0, -11.74466913],
                    [1.38057613, 0.0, -5.07435061],
                    [-1.53449069, 0.0, 3.79025526],
                    [-3.55908594, 0.0, 9.3099453],
                    [-5.57805647, 0.0, 5.41192963],
                    [0.22264108, 0.0, 1.98230386],
                    [-5.55112327, 0.0, 5.63360373],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            np.array(
                [
                    [9.76839276e-10, 1.00000000e00, 0.00000000e00],
                    [1.55431223e-14, 1.00000000e00, 0.00000000e00],
                    [1.28612265e-08, 1.00000000e00, 0.00000000e00],
                    [1.67441552e-01, 1.00000000e00, 3.96622408e-07],
                    [1.24942021e-01, 1.00000000e00, 1.51478682e-04],
                    [3.73949987e-04, 1.00000000e00, 0.00000000e00],
                    [2.49833210e-08, 1.00000000e00, 6.39580235e-08],
                    [8.23819611e-01, 1.00000000e00, 4.74757203e-02],
                    [2.91411313e-08, 1.00000000e00, 1.81814095e-08],
                ]
            ),
        )
        actual = slice_.pairwise_significance_tests[1]

        slice_no_hs_ = Cube(CR.CAT_X_MR_AUGMENTED).partitions[0]
        actual_no_hs = slice_no_hs_.pairwise_significance_tests[1]

        # Same fixture without insertion has 5,3 shape with same values for t_stats
        # not considering the insertions in the fixture above
        np.testing.assert_array_almost_equal(
            actual_no_hs.t_stats,
            np.array(
                [
                    [5.69285527, 0.0, -11.74466913],
                    [1.38057613, 0.0, -5.07435061],
                    [-3.55908594, 0.0, 9.3099453],
                    [-5.57805647, 0.0, 5.41192963],
                    [0.22264108, 0.0, 1.98230386],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(actual_no_hs.t_stats[0], actual.t_stats[2])
        np.testing.assert_array_almost_equal(actual_no_hs.t_stats[1], actual.t_stats[3])
        np.testing.assert_array_almost_equal(actual_no_hs.t_stats[2], actual.t_stats[5])
        np.testing.assert_array_almost_equal(actual_no_hs.t_stats[3], actual.t_stats[6])
        np.testing.assert_array_almost_equal(actual_no_hs.t_stats[4], actual.t_stats[7])

        slice_no_aug_ = Cube(CR.CAT_HS_X_MR_WGTD).partitions[0]

        np.testing.assert_array_almost_equal(slice_.counts, slice_no_aug_.counts)
        for i in range(slice_no_aug_.shape[0]):
            np.testing.assert_array_almost_equal(
                slice_no_aug_.counts[i], counts_with_hs[i][:, 0, :, 0].diagonal()
            )

    def test_mr_x_mr_weighted_augmented_pairwise_t_test(self):
        """This test proofs the hypotesis testing for MR1_X_MR2 considering the
        overlaps. To calculate the overlap of this kind of cube we need to calculate
        the shadow proportion behind the MR1_X_MR2 table:
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Pol.Know          |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | Senate            |    71.39   |  74.80  |    55.48   |
        | Deficit           |    67.75   |  72.58  |    44.91   |
        | Unemp.            |    62.70   |  66.04  |    68.24   |
        | Tariffs           |    80.47   |  83.61  |    64.52   |
        | Elec.vot          |    43.52   |  44.56  |    33.81   |
        +-------------------+------------+---------+------------+

         ==================SENATE - SELECTED (COUNTS)==============
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Trust             |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | NYTimes           |    2990    |  2309   |    461     |
        | WaPo              |    2309    |  2714   |    428     |
        | FoxNews           |    461     |  428    |    2848    |
        +-------------------+------------+---------+------------+

        ==================SENATE - OTHER (COUNTS)================
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Trust             |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | NYTimes           |   1998     |  737    |    586     |
        | WaPo              |    737     |  914    |    431     |
        | FoxNews           |    586     |  431    |    2285    |
        +-------------------+------------+---------+------------+

        So to get the 71.39 we have to divide counts / margin. For example
        the first shadow proportion will be 2990 / (2990+1198) where 2990 is the count
        of the NYTimesXNYTimes in the Senate Selected tab and the denominator is the
        sum of the counts of the Selected and Other Senate tab in the position (0,0).

        This procedure, is repeated for each couple of (selected and other) tabs for
        each of the 5 Pol.Know variables.

        Given the shadow proportions and the overlap margins for all the tabs we use
        these figures to calculate the t-statistic considering the overlaps.

        In this test we are comparing the first column `NYTimes` with the other 2
        using the correct t_stats arrays."""
        slice_ = Cube(CR.MR_X_MR_WEIGHTED_AUGMENTED).partitions[0]
        actual = slice_.pairwise_significance_tests[1]
        overlap_margins = np.sum(slice_._cube.counts[0], axis=0)[:, 0, :, 0]
        # MRxMRxMR has (5, 2, 3, 2, 3, 2) shape, so 5 tabs of shadow proportions
        shadow_proportions_tab = [
            (slice_._cube.counts[i][0][:, 0, :, 0] / overlap_margins)
            for i in range(slice_._cube.counts.shape[0])
        ]

        assert slice_.cube_is_mr_aug is True
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            [
                [-10.2964264, 0.0, -20.09577285],
                [-16.38251357, 0.0, -28.57841019],
                [-7.42875194, 0.0, 2.33766663],
                [-10.88338158, 0.0, -22.45002834],
                [-2.12632668, 0.0, -11.07466431],
            ],
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            [
                [0.0, 1.00000000e00, 0.0],
                [0.0, 1.00000000e00, 0.0],
                [1.19015908e-13, 1.00000000e00, 1.94264286e-02],
                [0.0, 1.00000000e00, 0.0],
                [3.35015757e-02, 1.00000000e00, 0.0],
            ],
        )
        np.testing.assert_array_almost_equal(
            slice_.column_proportions,
            [
                [0.71392963, 0.74796797, 0.55475245],
                [0.67749583, 0.72581913, 0.44912163],
                [0.62702902, 0.6603541, 0.68236206],
                [0.80466553, 0.8360987, 0.64524993],
                [0.43519965, 0.44556699, 0.33807508],
            ],
        )
        # each diagonal of the shadow proportions tab is equal to the correspondent row
        # for the actual proportions
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab[0].diagonal(), slice_.column_proportions[0, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab[1].diagonal(), slice_.column_proportions[1, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab[2].diagonal(), slice_.column_proportions[2, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab[3].diagonal(), slice_.column_proportions[3, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab[4].diagonal(), slice_.column_proportions[4, :]
        )
        assert slice_._cube.counts.shape == (5, 2, 3, 2, 3, 2)
        assert actual.t_stats.shape == (5, 3)

    def test_mr_subvar_x_mr_augmented_pairwise_t_test(self):
        """This test proofs the hypotesis testing for MR_SUBVAR_X_MR considering the
        overlaps. To calculate the overlap of this kind of cube we need to calculate
        the shadow proportion behind the SUBVAR_X_MR table:
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Increasing Taxes  |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | Selected          |    82.71   |  85.96  |    63.21   |
        | Other             |    17.29   |  14.4   |    36.79   |
        +-------------------+------------+---------+------------+

         ==================TAB 1 - SELECTED (COUNTS)=============
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Trust             |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | NYTimes           |    3464    |  2647   |    590     |
        | WaPo              |    2647    |  3119   |    529     |
        | FoxNews           |    590     |  529    |    3245    |
        +-------------------+------------+---------+------------+

        ==================TAB 2 - OTHER (COUNTS)=================
                            +-----------------------------------+
                            |               Trust               |
        +-------------------+-----------------------------------+
        | Trust             |                                   |
        +-------------------+----------- +---------+------------+
        |                   |    NYTimes |  WaPo   |  FoxNews   |
        +===================+============+=========+============+
        | NYTimes           |    724     |  400    |    458     |
        | WaPo              |    400     |  509    |    331     |
        | FoxNews           |    458     |  331    |    1889    |
        +-------------------+------------+---------+------------+

        So to get the 82.71 we have to divide counts / margin. For example
        the first shadow proportion will be 3464 / (3464+724) where 3464 is the count
        of the NYTimesXNYTimes in the first tab and the denominator is the sum of the
        counts of the first tab and second tab in the position (0,0).

        Given the shadow proportions and the overlap margins for all the tabs we use
        these figures to calculate the t-statistic considering the overlaps.

        In this test we are comparing the first column `NYTimes` with the other 2
        using the correct t_stats arrays.
        """
        slice_ = Cube(CR.CAT_SUBVAR_X_MR_AUGMENTED).partitions[0]
        actual = slice_.pairwise_significance_tests[0]
        overlap_margins = np.sum(slice_._cube.counts, axis=0)[:, 0, :, 0]
        shadow_proportions_tab1 = slice_._cube.counts[0][:, 0, :, 0] / overlap_margins
        shadow_proportions_tab2 = slice_._cube.counts[1][:, 0, :, 0] / overlap_margins

        assert slice_.cube_is_mr_aug is True
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            [[0.0, 16.02651373, -22.43766743], [0.0, -6.56624439, 29.71952066]],
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            [
                [1.00000000e00, 0.00000000e00, 0.00000000e00],
                [1.00000000e00, 5.42792478e-11, 0.00000000e00],
            ],
        )
        np.testing.assert_array_almost_equal(
            slice_.column_proportions,
            [
                [0.82707248, 0.85960427, 0.63206628],
                [0.17292752, 0.14039573, 0.36793372],
            ],
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1,
            [
                [0.82707248, 0.86881078, 0.56294105],
                [0.86881078, 0.85960427, 0.61504991],
                [0.56294105, 0.61504991, 0.63206628],
            ],
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2,
            [
                [0.17292752, 0.13118922, 0.43705895],
                [0.13118922, 0.14039573, 0.38495009],
                [0.43705895, 0.38495009, 0.36793372],
            ],
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1.diagonal(), slice_.column_proportions[0, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2.diagonal(), slice_.column_proportions[1, :]
        )

    def test_mr_x_mr_pairwise_t_test(self):
        slice_ = Cube(CR.MR_X_MR_2).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        assert slice_.cube_is_mr_aug is False
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            [
                [-0.80066287, 0.0, -2.22179624],
                [-0.60726974, 0.0, -2.1554303],
                [1.203117, 0.0, 0.14490412],
                [1.59774062, 0.0, 0.43050695],
                [-1.71054946, 0.0, 1.53817055],
            ],
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            [
                [0.42416402, 1.0, 0.02645888],
                [0.54427888, 1.0, 0.03130077],
                [0.23018433, 1.0, 0.88480775],
                [0.11149176, 1.0, 0.66689419],
                [0.08853052, 1.0, 0.12423633],
            ],
        )
