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

    def test_ca_subvar_x_cat_hs_pairwise_t_tests(self):
        slice_ = Cube(CR.CA_SUBVAR_X_CAT_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        np.testing.assert_array_almost_equal(
            actual.t_stats, load_python_expression("ca-subvar-x-cat-hs-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("ca-subvar-x-cat-hs-pw-pvals")
        )

    def test_ca_subvar_hs_x_mr_augmented_pairwise_t_tests(self):
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

    def test_cat_nps_numval_x_cat_scale_means_pariwise_t_tests(self):
        slice_ = Cube(SM.CAT_NPS_NUMVAL_X_CAT).partitions[0]

        actual = slice_.pairwise_significance_tests[0]
        assert actual.p_vals_scale_means[0] == 1
        assert actual.t_stats_scale_means[0] == 0
        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, 1.08966143, -1.27412884, -4.14629088]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0000000e00, 2.7638510e-01, 2.0302047e-01, 4.0226310e-05],
        )

        actual = slice_.pairwise_significance_tests[1]
        assert actual.p_vals_scale_means[1] == 1
        assert actual.t_stats_scale_means[1] == 0
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [2.7638510e-01, 1.0000000e00, 1.4344585e-03, 1.7905266e-09],
        )
        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [-1.08966143, 0.0, -3.19741668, -6.10466696]
        )

    def test_cat_x_cat_pairwise_t_tests(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS).partitions[0]
        actual = slice_.pairwise_significance_tests[2]

        np.testing.assert_almost_equal(
            actual.t_stats, load_python_expression("cat-x-cat-pw-tstats")
        )
        np.testing.assert_almost_equal(
            actual.p_vals, load_python_expression("cat-x-cat-pw-pvals")
        )

    def test_cat_x_cat_hs_pairwise_t_tests(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        t_stats = slice_.pairwise_significance_tests[0].t_stats
        p_vals = slice_.pairwise_significance_tests[0].p_vals

        np.testing.assert_almost_equal(
            t_stats, load_python_expression("cat-x-cat-hs-pw-tstats")
        )
        np.testing.assert_almost_equal(
            p_vals, load_python_expression("cat-x-cat-hs-pw-pvals")
        )

    def test_cat_x_cat_hs_scale_means_pairwise_t_tests(self):
        slice_ = Cube(CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION_WITH_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            load_python_expression("cat-x-cat-hs-scale-means-pw-tstats"),
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            load_python_expression("cat-x-cat-hs-scale-means-pw-pvals"),
        )

    def test_cat_x_cat_pruning_and_hs_scale_means_pairwise_t_tests(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 1.64461503, 1.92387847, np.nan, 1.06912069],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.1046981, 0.059721, np.nan, 0.2918845]
        )

        # Just H&S
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 0.9387958, 1.644615, 1.9238785, np.nan, 1.0691207],
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
            [0.0, 0.93879579, 1.64461503, 1.92387847, 1.06912069],
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
            actual.t_stats_scale_means, [0.0, 1.64461503, 1.92387847, 1.06912069]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.1046981, 0.059721, 0.2918845]
        )

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

    def test_cat_x_cat_wgtd_pairwise_t_tests(self):
        """The weights on this cube demonstrate much higher variance (less
        extreme t values, and higher associated p-values) than if weighted_n
        were used in the variance estimate of the test statistic.
        """
        slice_ = Cube(CR.CAT_X_CAT_WEIGHTED_TTESTS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]
        pairwise_indices = slice_.pairwise_indices

        np.testing.assert_almost_equal(
            actual.t_stats, load_python_expression("cat-x-cat-wgtd-pw-tstats")
        )
        np.testing.assert_almost_equal(
            actual.p_vals, load_python_expression("cat-x-cat-wgtd-pw-pvals")
        )
        np.testing.assert_array_equal(
            pairwise_indices,
            np.array(load_python_expression("cat-x-cat-wgtd-pw-indices"), dtype=tuple),
        )

    def test_cat_x_cat_wgtd_scale_means_pariwise_t_tests(self):
        slice_ = Cube(CR.CAT_X_CAT_WEIGHTED_TTESTS).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means, [0.0, -4.38871748, -3.99008596, -5.15679647]
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means,
            [1.0000000e00, 1.3839564e-05, 7.4552516e-05, 4.0145665e-07],
        )

    def test_cat_hs_x_mr_wgtd_augmented_pairwise_t_tests(self):
        slice_ = Cube(CR.CAT_HS_X_MR_AUGMENTED_WGTD).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        assert slice_.cube_is_mr_aug is True
        np.testing.assert_array_almost_equal(
            actual.t_stats, load_python_expression("cat-hs-x-mr-wgtd-aug-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("cat-hs-x-mr-wgtd-aug-pw-pvals")
        )

    def test_cat_x_mr_weighted_augmented_pairwise_t_tests(self):
        slice_ = Cube(CR.CAT_X_MR_WEIGHTED_AUGMENTED).partitions[0]
        actual = slice_.pairwise_significance_tests[1]
        overlap_margins = np.sum(slice_._cube.counts, axis=0)[:, 0, :, 0]
        shadow_proportions_tab1 = slice_._cube.counts[0][:, 0, :, 0] / overlap_margins
        shadow_proportions_tab2 = slice_._cube.counts[1][:, 0, :, 0] / overlap_margins

        assert slice_.cube_is_mr_aug is True
        # --- each diagonal of the shadow proportions tab is equal to the correspondent
        # --- row for the actual proportions
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1.diagonal(), slice_.column_proportions[0, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2.diagonal(), slice_.column_proportions[1, :]
        )
        np.testing.assert_array_almost_equal(
            actual.t_stats,
            [[-10.2964264, 0.0, -20.09577285], [5.96350953, 0.0, 24.14335882]],
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals,
            [[0.0, 1.00000000e00, 0.0], [2.55612775e-09, 1.00000000e00, 0.0]],
        )

    def test_mr_x_mr_pairwise_t_tests(self):
        slice_ = Cube(CR.MR_X_MR_2).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        assert slice_.cube_is_mr_aug is False
        np.testing.assert_array_almost_equal(
            actual.t_stats, load_python_expression("mr-x-mr-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("mr-x-mr-pw-pvals")
        )

    def test_mr_x_mr_weighted_augmented_pairwise_t_tests(self):
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
            actual.t_stats, load_python_expression("mr-x-mr-wgtd-aug-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("mr-x-mr-wgtd-aug-pw-pvals")
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

    def test_mr_subvar_x_mr_augmented_pairwise_t_tests(self):
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
            actual.t_stats, load_python_expression("mr-subvar-x-mr-aug-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("mr-subvar-x-mr-aug-pw-pvals")
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1,
            load_python_expression("mr-subvar-x-mr-aug-pw-shadow-proportions-tab1"),
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2,
            load_python_expression("mr-subvar-x-mr-aug-pw-shadow-proportions-tab2"),
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab1.diagonal(), slice_.column_proportions[0, :]
        )
        np.testing.assert_array_almost_equal(
            shadow_proportions_tab2.diagonal(), slice_.column_proportions[1, :]
        )
