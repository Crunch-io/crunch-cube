# encoding: utf-8

"""Integration tests for pairwise comparisons."""

from unittest import TestCase

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR, OL, SM, NA
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
        # check new api also
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_t_stats(1),
            load_python_expression("ca-subvar-hs-x-cat-hs-pw-tstats"),
        )
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_p_vals(1),
            load_python_expression("ca-subvar-hs-x-cat-hs-pw-pvals"),
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
        # test new api also
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_t_stats(2),
            load_python_expression("cat-x-cat-pw-tstats"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_p_vals(2),
            load_python_expression("cat-x-cat-pw-pvals"),
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
        # test new api also
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_t_stats(0),
            load_python_expression("cat-x-cat-hs-pw-tstats"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_p_vals(0),
            load_python_expression("cat-x-cat-hs-pw-pvals"),
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
        # test new api also
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_t_stats(0),
            load_python_expression("cat-x-cat-wgtd-pw-tstats"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_p_vals(0),
            load_python_expression("cat-x-cat-wgtd-pw-pvals"),
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
        # test new api also
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_t_stats(1),
            load_python_expression("mr-x-mr-pw-tstats"),
        )
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_p_vals(1),
            load_python_expression("mr-x-mr-pw-pvals"),
        )


class TestOverlapsPairwiseSignificance(TestCase):
    def test_pairwise_significance_cat_x_mr_sub_x_mr_sel_0th_subvar(self):
        slice_ = Cube(OL.CAT_X_MR_SUB_X_MR_SEL).partitions[0]

        assert slice_.column_percentages.tolist() == [
            [0.0, 100.0, 100.0],
            [100.0, 0.0, 0.0],
        ]
        assert slice_.pairwise_significance_t_stats(0) == pytest.approx(
            np.array(
                [
                    [0.0, 3.11420549, 2.61911361],
                    [0.0, -3.11420549, -2.61911361],
                ]
            )
        )
        assert slice_.pairwise_significance_p_vals(0) == pytest.approx(
            np.array(
                [
                    [0.0, 0.05270861, 0.07906174],
                    [0.0, 0.05270861, 0.07906174],
                ]
            )
        )

    def test_pairwise_significance_cat_x_mr_sub_x_mr_sel_1st_subvar(self):
        slice_ = Cube(OL.CAT_X_MR_SUB_X_MR_SEL).partitions[0]

        assert slice_.column_percentages.tolist() == [
            [0.0, 100.0, 100.0],
            [100.0, 0.0, 0.0],
        ]
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [-3.11420549, 0.0, 0.0],
                    [3.11420549, 0.0, 0.0],
                ]
            )
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.05270861, 0.0, 1.0],
                    [0.05270861, 0.0, 1.0],
                ]
            )
        )

    def test_pairwise_significance_cat_x_mr_sub_x_mr_sel_2nd_subvar(self):
        slice_ = Cube(OL.CAT_X_MR_SUB_X_MR_SEL).partitions[0]

        assert slice_.column_percentages.tolist() == [
            [0.0, 100.0, 100.0],
            [100.0, 0.0, 0.0],
        ]
        assert slice_.pairwise_significance_t_stats(2) == pytest.approx(
            np.array(
                [
                    [-2.61911361, 0.0, 0.0],
                    [2.61911361, 0.0, 0.0],
                ],
            ),
        )
        assert slice_.pairwise_significance_p_vals(2) == pytest.approx(
            np.array(
                [
                    [0.07906174, 1.0, 0.0],
                    [0.07906174, 1.0, 0.0],
                ],
            ),
        )

    def test_pairwise_significance_cat_x_mr_realistic_example(self):
        slice_ = Cube(OL.CAT_X_MR_REALISTIC_EXAMPLE).partitions[0]

        assert slice_.column_percentages == pytest.approx(
            np.array(
                [
                    [52.7687, 52.5926, 51.5504, 47.6852, 51.3889, np.nan],
                    [47.2313, 47.4074, 48.4496, 52.3148, 48.6111, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_t_stats(4) == pytest.approx(
            np.array(
                [
                    [1.00337549, 0.64382181, 0.0773666, -1.3677023, 0.0, np.nan],
                    [-1.00337549, -0.64382181, -0.0773666, 1.3677023, 0.0, np.nan],
                ],
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(4) == pytest.approx(
            np.array(
                [
                    [0.31647509, 0.52017481, 0.93838264, 0.17241219, 0.0, np.nan],
                    [0.31647509, 0.52017481, 0.93838264, 0.17241219, 0.0, np.nan],
                ],
            ),
            nan_ok=True,
        )

    def test_pairwise_significance_mr_x_mr(self):
        slice_ = Cube(OL.MR_X_MR).partitions[0]

        assert slice_.column_percentages == pytest.approx(
            np.array(
                [
                    [100.0, 66.66667, np.nan],
                    [66.66667, 100.0, np.nan],
                    [0.0, 0.0, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [3.51874302, 0.0, np.nan],
                    [-3.51874302, 0.0, np.nan],
                    [0.0, 0.0, np.nan],
                ],
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.00310191, 0.0, np.nan],
                    [0.00310191, 0.0, np.nan],
                    [1.0, 0.0, np.nan],
                ],
            ),
            nan_ok=True,
        )

    def test_pairwise_cat_x_mr_gender_x_all_pets_owned_with_weighted_counts(self):
        slice_ = Cube(OL.CAT_X_MR_GENDER_X_ALL_PETS_OWNED).partitions[0]

        assert slice_.column_percentages.tolist() == pytest.approx(
            np.array(
                [
                    [66.6667, 14.28571, 50.0],
                    [33.33333, 85.714286, 50.0],
                ]
            )
        )

        # Assert for first column (subvariable)
        assert slice_.pairwise_significance_t_stats(0).tolist() == pytest.approx(
            np.array(
                [
                    [0.0, -2.6315597, -1.76353],
                    [0.0, 2.6315597, 1.76353],
                ]
            ),
        )
        assert slice_.pairwise_significance_p_vals(0) == pytest.approx(
            np.array(
                [
                    [0.0, 0.01410448, 0.0879948],
                    [0.0, 0.01410448, 0.0879948],
                ]
            ),
        )

        # Assert for second column (subvariable)
        assert slice_.pairwise_significance_t_stats(1).tolist() == pytest.approx(
            np.array(
                [
                    [2.63156, 0.0, 8.10444],
                    [-2.63156, 0.0, -8.10444],
                ]
            ),
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.01410448, 0, 0.025067e-06],
                    [0.01410448, 0, 0.025067e-06],
                ]
            ),
        )

        # Assert for third column (subvariable)
        assert slice_.pairwise_significance_t_stats(2).tolist() == pytest.approx(
            np.array(
                [
                    [1.763531, -8.104439, 0.0],
                    [-1.763531, 8.104439, 0.0],
                ]
            ),
        )
        assert slice_.pairwise_significance_p_vals(2) == pytest.approx(
            np.array(
                [
                    [0.0879948, 0.025067e-06, 0],
                    [0.0879948, 0.025067e-06, 0],
                ]
            ),
        )

    def test_pairwise_significance_indices(self):
        transforms = {"pairwise_indices": {"alpha": [0.05, 0.13]}}
        slice_ = Cube(
            OL.CAT_X_MR_GENDER_X_ALL_PETS_OWNED, transforms=transforms
        ).partitions[0]

        assert slice_.column_percentages.tolist() == pytest.approx(
            np.array(
                [
                    [66.6667, 14.28571, 50.0],
                    [33.33333, 85.714286, 50.0],
                ]
            )
        )

        assert slice_.pairwise_indices.tolist() == [
            [(1,), (), (1,)],
            [(), (0, 2), ()],
        ]

        assert slice_.pairwise_indices_alt.tolist() == [
            [(1, 2), (), (1,)],
            [(), (0, 2), (0,)],
        ]

    def test_alt_pairwise_indices_without_alt_alpha(self):
        slice_ = Cube(OL.CAT_X_MR_GENDER_X_ALL_PETS_OWNED).partitions[0]
        assert slice_.pairwise_indices_alt is None

    def test_pairwise_significance_all_empty(self):
        # ---Keep the alpha value this small to demo the error found in cr.server
        transforms = {"pairwise_indices": {"alpha": [0.0000000001]}}
        slice_ = Cube(
            OL.CAT_X_MR_GENDER_X_ALL_PETS_OWNED, transforms=transforms
        ).partitions[0]

        assert slice_.column_percentages.tolist() == pytest.approx(
            np.array(
                [
                    [66.6667, 14.28571, 50.0],
                    [33.33333, 85.714286, 50.0],
                ]
            )
        )

        assert slice_.pairwise_indices.tolist() == [
            [(), (), ()],
            [(), (), ()],
        ]

    def test_pairwise_sig_for_realistic_example_mr_x_mr(self):
        slice_ = Cube(OL.MR_X_MR_REALISTIC_EXAMPLE).partitions[0]
        assert slice_.pairwise_significance_t_stats(3) == pytest.approx(
            np.array(
                [
                    [61.55, 43.02, -2.26, 0.0, 9.79, -105.35],
                    [-2.67, 89.18, -21.60, 0.0, -12.43, -58.98],
                    [-2.43, 3.15, 103.35, 0.0, 44.51, -85.86],
                    [-86.29, -60.68, -90.81, 0.0, -56.66, -127.32],
                    [-25.01, -4.42, -12.87, 0.0, 93.79, -53.79],
                    [-3.52, 0.78, -15.23, 0.0, -4.65, 138.61],
                ]
            ),
            abs=10e-2,
        )


class TestMeanDifferenceSignificance(object):
    def test_mean_diff_significance_for_numeric_array_grouped_by_cat(self):
        slice_ = Cube(NA.NUM_ARR_MULTI_NUMERIC_MEASURES_GROUPED_BY_CAT).partitions[0]
        assert slice_.pairwise_significance_means_t_stats(0) == pytest.approx(
            np.array(
                [
                    [-0.0, -0.7482866, -4.61524529, -5.29463547],
                    [-0.0, -4.64371041, -7.14738681, -3.67511747],
                    [-0.0, 3.13387218, 6.35310465, 2.3932706],
                ]
            )
        )
        assert slice_.pairwise_significance_means_p_vals(0) == pytest.approx(
            np.array(
                [
                    [1.00000000e00, 4.60526868e-01, 5.38716457e-05, 7.11866628e-06],
                    [1.00000000e00, 7.34317022e-05, 2.90876736e-08, 8.12768987e-04],
                    [1.00000000e00, 4.02303806e-03, 3.00760739e-07, 2.23647892e-02],
                ]
            )
        )

    def test_mean_diff_significance_for_numeric_array_grouped_by_cat_hs(self):
        transforms = {
            "columns_dimension": {
                "insertions": [
                    {
                        "function": "subtotal",
                        "name": "DIFF B-A",
                        "args": [4, 5],
                        "anchor": "top",
                        "kwargs": {"negative": [1, 2, 3]},
                    },
                    {
                        "function": "subtotal",
                        "args": [1, 2, 3],
                        "anchor": 1,
                        "name": '"A" countries',
                    },
                    {
                        "function": "subtotal",
                        "args": [4, 5],
                        "anchor": 2,
                        "name": '"B" countries',
                    },
                    {
                        "function": "subtotal",
                        "name": "DIFF A-B",
                        "args": [1, 2, 3],
                        "anchor": "bottom",
                        "kwargs": {"negative": [4, 5]},
                    },
                ]
            }
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS, transforms=transforms
        ).partitions[0]

        # Column (0,3,5,8) are subtotals and the hypothesis testing for a subtotal
        # computable ATM.
        for col in slice_.inserted_column_idxs:
            assert slice_.pairwise_significance_means_t_stats(
                col
            ).tolist() == pytest.approx(
                np.full(slice_.means.shape, np.nan), nan_ok=True
            )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_means_t_stats(7),
            load_python_expression("num-arr-means-grouped-by-cat-hs-t-stats-col-7"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_means_p_vals(7),
            load_python_expression("num-arr-means-grouped-by-cat-hs-p-vals-col-7"),
        )
        assert slice_.pairwise_means_indices.tolist() == [
            [None, (), (7,), None, (), None, (), (), None],
            [None, (), (), None, (), None, (), (), None],
            [None, (), (1, 4, 7), None, (), None, (1, 4, 7), (), None],
            [None, (), (), None, (), None, (), (), None],
        ]

        # Test no subtotals
        slice_ = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (4,), (), (), ()],
            [(), (), (), (), ()],
            [(), (0, 2, 4), (), (0, 2, 4), ()],
            [(), (), (), (), ()],
        ]
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_means_t_stats(4),
            load_python_expression("num-arr-means-grouped-by-cat-t-stats-col-4"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_means_p_vals(4),
            load_python_expression("num-arr-means-grouped-by-cat-p-vals-col-4"),
        )

    def test_mean_diff_significance_for_numeric_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MULTI_NUMERIC_MEASURES_X_MR).partitions[0]

        assert slice_.pairwise_significance_means_t_stats(0) == pytest.approx(
            np.array(
                [
                    [-0.0, 1.3056921, -1.1842639],
                    [-0.0, 1.7117695, -1.6297401],
                    [np.nan, np.nan, np.nan],
                    [np.nan, -5.8872721, -4.4442801],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(0) == pytest.approx(
            np.array(
                [
                    [1.00000000e00, 1.96331347e-01, 2.40688158e-01],
                    [1.00000000e00, 9.17815019e-02, 1.08068956e-01],
                    [np.nan, np.nan, np.nan],
                    [np.nan, 1.57810759e-07, 3.57756031e-05],
                ]
            ),
            nan_ok=True,
        )

    def test_mean_diff_significance_indices_for_numeric_array_grouped_by_cat(self):
        transforms = {"pairwise_indices": {"alpha": [0.05, 0.08]}}
        slice_ = Cube(
            NA.NUM_ARR_MULTI_NUMERIC_MEASURES_GROUPED_BY_CAT, transforms=transforms
        ).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(2, 3), (2, 3), (), ()],
            [(1, 2, 3), (), (), ()],
            [(), (0,), (0, 1, 3), (0,)],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(2, 3), (2, 3), (), ()],
            [(1, 2, 3), (), (), (2,)],
            [(), (0,), (0, 1, 3), (0,)],
        ]

    def test_mean_diff_significance_is_not_available(self):
        transforms = {"pairwise_indices": {"alpha": [0.05, 0.08]}}
        slice_ = Cube(CR.CAT_X_CAT, transforms=transforms).partitions[0]

        with pytest.raises(ValueError) as e:
            slice_.pairwise_means_indices
        assert (
            str(e.value) == "`.pairwise_means_indices` is undefined for a cube-result"
            " without a mean measure"
        )
        with pytest.raises(ValueError) as e:
            slice_.pairwise_means_indices_alt
        assert (
            str(e.value)
            == "`.pairwise_means_indices_alt` is undefined for a cube-result"
            " without a mean measure"
        )
        with pytest.raises(ValueError) as e:
            slice_.pairwise_significance_means_p_vals(0)
        assert (
            str(e.value)
            == "`.pairwise_significance_means_p_vals` is undefined for a cube-result"
            " without a mean measure"
        )
        with pytest.raises(ValueError) as e:
            slice_.pairwise_significance_means_t_stats(0)
        assert (
            str(e.value)
            == "`.pairwise_significance_means_t_stats` is undefined for a cube-result"
            " without a mean measure"
        )

    def test_mean_diff_significance_for_cat_x_cat(self):
        slice_ = Cube(CR.MEAN_CAT_X_CAT).partitions[0]

        assert slice_.pairwise_significance_means_t_stats(1) == pytest.approx(
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [14.1421356, -0.0, 3.1066887, np.nan],
                    [1.75662013, -0.0, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(1) == pytest.approx(
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [8.21565038e-15, 1.00000000e00, 4.80968355e-03, np.nan],
                    [8.91869315e-02, 1.00000000e00, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
        )

    def test_mean_diff_significance_indices_for_cat_x_cat(self):
        transforms = {"pairwise_indices": {"alpha": [0.05, 0.01]}}
        slice_ = Cube(CR.MEAN_CAT_X_CAT, transforms=transforms).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), ()],
            [(1, 2), (), (1,), ()],
            [(), (), (), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(), (), (), (0, 2)],
            [(1, 2), (), (1,), ()],
            [(), (), (), ()],
        ]

    def test_mean_diff_significance_indices_num_array_grouped_by_cat_hs_weighted(self):
        insertions = [
            {
                "function": "subtotal",
                "args": [1, 2, 3],
                "anchor": "top",
                "name": '"A" countries',
            },
            {
                "function": "subtotal",
                "args": [4, 5],
                "anchor": "top",
                "name": '"B" countries',
            },
            {
                "function": "subtotal",
                "name": "DIFF A-B",
                "args": [1, 2, 3],
                "anchor": "top",
                "kwargs": {"negative": [4, 5]},
            },
            {
                "function": "subtotal",
                "name": "DIFF B-A",
                "args": [4, 5],
                "anchor": "top",
                "kwargs": {"negative": [1, 2, 3]},
            },
        ]
        transforms = {
            "pairwise_indices": {"alpha": [0.05, 0.01]},
            "columns_dimension": {"insertions": insertions},
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [None, None, None, None, (), (), (), (), ()],
            [None, None, None, None, (), (), (), (), ()],
            [None, None, None, None, (), (), (), (6,), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [None, None, None, None, (), (), (), (), ()],
            [None, None, None, None, (), (), (), (), ()],
            [None, None, None, None, (), (), (), (4, 6), (4, 6)],
        ]

        # Test no subtotals
        transforms = {"pairwise_indices": {"alpha": [0.05, 0.01]}}
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), (), ()],
            [(), (), (), (), ()],
            [(), (), (), (2,), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(), (), (), (), ()],
            [(), (), (), (), ()],
            [(), (), (), (0, 2), (0, 2)],
        ]

        # Test pruning
        transforms = {
            "columns_dimension": {
                "prune": True,
                "elements": {"1": {"hide": True}},
                "insertions": insertions,
            },
            "pairwise_indices": {"alpha": [0.05, 0.01]},
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [None, None, None, None, (), (), (), ()],
            [None, None, None, None, (), (), (), ()],
            [None, None, None, None, (), (), (5,), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [None, None, None, None, (), (), (), ()],
            [None, None, None, None, (), (), (), ()],
            [None, None, None, None, (), (), (4, 5), (4, 5)],
        ]
