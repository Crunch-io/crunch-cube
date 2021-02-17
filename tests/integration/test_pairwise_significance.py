# encoding: utf-8

"""Integration tests for pairwise comparisons."""

from unittest import TestCase

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR, SM, OL
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
            (CR.MR_X_CAT_HS_MT, {}, "mr-x-cat-hs-pw-idxs"),
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
        ),
    )
    def it_provides_columns_scale_mean_pairwise_indices(self, fixture, expectation):
        """Provides column-indicies meeting sig-test threshold on column scale means."""
        slice_ = _Slice(
            Cube(fixture), slice_idx=0, transforms={}, population=None, mask_size=0
        )

        actual = slice_.columns_scale_mean_pairwise_indices

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
        ),
    )
    def it_provides_columns_scale_mean_pairwise_indices_alt(self, fixture, expectation):
        """Provides col idxs meeting secondary sig-test threshold on scale mean."""
        slice_ = _Slice(
            Cube(fixture),
            slice_idx=0,
            transforms={"pairwise_indices": {"alpha": [0.01, 0.08]}},
            population=None,
            mask_size=0,
        )

        actual = slice_.columns_scale_mean_pairwise_indices_alt

        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)


class TestStandardizedResiduals(TestCase):
    """Test cr.cube implementation of column family pairwise comparisons"""

    def test_ca_subvar_hs_x_cat_hs_pairwise_t_tests(self):
        slice_ = Cube(CR.CA_SUBVAR_HS_X_CAT_HS).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        np.testing.assert_array_almost_equal(
            actual.t_stats, load_python_expression("ca-subvar-hs-x-cat-hs-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("ca-subvar-hs-x-cat-hs-pw-pvals")
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
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        actual = slice_.pairwise_significance_tests[0]

        np.testing.assert_almost_equal(
            actual.t_stats_scale_means,
            [0.0, 1.64461503, 1.92387847, np.nan, 1.06912069],
        )
        np.testing.assert_almost_equal(
            actual.p_vals_scale_means, [1.0, 0.1046981, 0.059721, np.nan, 0.2918845]
        )

        # Just H&S
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
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
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
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
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
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

    def test_mr_x_mr_pairwise_t_tests(self):
        slice_ = Cube(CR.MR_X_MR_2).partitions[0]
        actual = slice_.pairwise_significance_tests[1]

        np.testing.assert_array_almost_equal(
            actual.t_stats, load_python_expression("mr-x-mr-pw-tstats")
        )
        np.testing.assert_array_almost_equal(
            actual.p_vals, load_python_expression("mr-x-mr-pw-pvals")
        )


class TestOverlaps(TestCase):
    def test_mr_simple_overlaps(self):
        strand_ = Cube(OL.SIMPLE_OVERLAPS).partitions[0]

        overlaps = strand_.overlaps

        assert overlaps.tolist() == [
            # Subvariables:
            # A, B, C
            [3, 1, 1],  # A
            [1, 2, 2],  # B
            [1, 2, 3],  # C
        ]

        valid_overlaps = strand_.valid_overlaps

        assert valid_overlaps.tolist() == [
            # Subvariables:
            # A, B, C
            [5, 5, 5],  # A
            [5, 7, 7],  # B
            [5, 7, 7],  # C
        ]

        sig_tests = strand_.pairwise_significance_tests[0]

        assert sig_tests.t_stats.tolist() == [0, 1.3367572072832183, 0.5859387772465644]
        assert sig_tests.p_vals.tolist() == [
            0.0,
            0.23889753532091706,
            0.5833667356783647,
        ]

        sig_tests = strand_.pairwise_significance_tests[1]

        assert sig_tests.t_stats.tolist() == [
            -1.3367572072832183,
            0.0,
            -1.0801234497346437,
        ]
        assert sig_tests.p_vals.tolist() == [
            0.23889753532091706,
            0.0,
            0.32941091674326217,
        ]

        sig_tests = strand_.pairwise_significance_tests[2]

        assert sig_tests.t_stats.tolist() == [
            -0.5859387772465644,
            1.0801234497346437,
            0.0,
        ]
        assert sig_tests.p_vals.tolist() == [
            0.5833667356783647,
            0.32941091674326217,
            0.0,
        ]

    def test_cat_x_mr_overlaps(self):
        slice_ = Cube(OL.CAT_X_MR_OVERLAPS).partitions[0]

        overlaps = slice_.overlaps

        assert overlaps.tolist() == [
            [
                # A, B, C
                [0, 0, 0],  # A
                [0, 1, 1],  # B
                [0, 1, 2],  # C
            ],  # G[0] == 2
            [
                # A, B, C
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],  # G[1] == 1
        ]

        valid_overlaps = slice_.valid_overlaps

        assert valid_overlaps.tolist() == [
            [
                # A, B, C
                [1, 1, 1],
                [1, 3, 3],
                [1, 3, 3],
            ],  # G[0] == 2
            [
                # A, B, C
                [2, 2, 2],  # A
                [2, 2, 2],  # B
                [2, 2, 2],  # C
            ],  # G[1] == 2
        ]

        sig_tests = slice_.pairwise_significance_tests[0]
        np.testing.assert_almost_equal(
            sig_tests.t_stats,
            [
                [0.0, -1.22474487, -2.44948974],
                [0.0, 1.41421356, 1.41421356],
            ],
        )
        np.testing.assert_almost_equal(
            sig_tests.p_vals,
            [
                [0.0, 0.43590578, 0.24675171],
                [0.0, np.nan, np.nan],
            ],
        )

        sig_tests = slice_.pairwise_significance_tests[1]
        np.testing.assert_almost_equal(
            sig_tests.t_stats,
            [
                [1.22474487, 0.0, -1.22474487],
                [-1.41421356, 0.0, np.nan],
            ],
        )
        np.testing.assert_almost_equal(
            sig_tests.p_vals,
            [
                [0.43590578, 0.0, 0.43590578],
                [np.nan, 0.0, np.nan],
            ],
        )

        sig_tests = slice_.pairwise_significance_tests[2]
        np.testing.assert_almost_equal(
            sig_tests.t_stats,
            [
                [2.44948974, 1.22474487, 0.0],
                [-1.41421356, np.nan, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            sig_tests.p_vals,
            [
                [0.24675171, 0.43590578, 0.0],
                [np.nan, np.nan, 0.0],
            ],
        )
