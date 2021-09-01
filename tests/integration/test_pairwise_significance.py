# encoding: utf-8

"""Integration tests for pairwise comparisons."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR, OL, SM, NA
from ..util import load_python_expression


class Describe_Slice:
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


class TestStandardizedResiduals:
    """Test cr.cube implementation of column family pairwise comparisons"""

    def test_ca_subvar_hs_x_cat_hs_pairwise_t_tests(self):
        slice_ = Cube(CR.CA_SUBVAR_X_CA_CAT_HS).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_t_stats(1),
            np.full((3, 5), np.nan),
        )
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_p_vals(1),
            np.full((3, 5), np.nan),
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

        np.testing.assert_almost_equal(
            slice_.pairwise_significance_t_stats(0),
            load_python_expression("cat-x-cat-hs-pw-tstats"),
        )
        np.testing.assert_almost_equal(
            slice_.pairwise_significance_p_vals(0),
            load_python_expression("cat-x-cat-hs-pw-pvals"),
        )

    def test_cat_hs_x_cat_hs_pairwise_t_tests(self):
        slice_ = Cube(CR.CAT_HS_X_CAT_HS).partitions[0]

        # test pairwise with subtotal col selected
        assert slice_.pairwise_significance_t_stats(4) == pytest.approx(
            np.array(
                [
                    [1.90437719, 2.88752484, 0.38370528, -0.42342669, 0.0],
                    [4.53155077, 7.40874302, 0.8412411, -0.90992467, 0.0],
                    [5.27119847, 8.66574355, 0.97045666, -1.04616611, 0.0],
                    [0.42542565, 2.89651937, -1.38636076, 1.29486253, 0.0],
                    [-1.02040616, -4.66527973, -7.76319202, 5.75233621, 0.0],
                    [-4.7807473, -7.97016737, 5.1000019, -5.68266923, 0.0],
                    [-5.31255684, -10.85932713, 0.01312205, -0.01359356, 0.0],
                ]
            )
        )
        assert slice_.pairwise_significance_p_vals(4) == pytest.approx(
            np.array(
                [
                    [5.7038392e-02, 3.9323348e-03, 7.0124482e-01, 6.7203843e-01, 1.0],
                    [6.2851044e-06, 2.0050627e-13, 4.0033105e-01, 3.6299338e-01, 1.0],
                    [1.5379911e-07, 0.0000000e00, 3.3195680e-01, 2.9563556e-01, 1.0],
                    [6.7058302e-01, 3.8219005e-03, 1.6581834e-01, 1.9554643e-01, 1.0],
                    [3.0768851e-01, 3.3250146e-06, 1.4210854e-14, 1.0446054e-08, 1.0],
                    [1.9050300e-06, 2.8865798e-15, 3.7757066e-07, 1.5614610e-08, 1.0],
                    [1.2313795e-07, 0.0000000e00, 9.8953195e-01, 9.8915586e-01, 1.0],
                ]
            )
        )

    def test_cat_x_cat_pairwise_t_tests_with_subdiffs_and_subtot_on_both(self):
        transforms = {
            "rows_dimension": {
                "insertions": [
                    {
                        "function": "subtotal",
                        "args": [1],
                        "kwargs": {"negative": [2]},
                        "anchor": 2,
                        "name": "NPS",
                    },
                    {
                        "function": "subtotal",
                        "args": [3, 4],
                        "anchor": 1,
                        "name": "subtotal",
                    },
                ]
            },
            "columns_dimension": {
                "insertions": [
                    {
                        "function": "subtotal",
                        "args": [1],
                        "kwargs": {"negative": [2]},
                        "anchor": 3,
                        "name": "NPS",
                    },
                    {
                        "function": "subtotal",
                        "args": [3, 4],
                        "anchor": 2,
                        "name": "subtotal",
                    },
                    {
                        "function": "subtotal",
                        "args": [1, 2],
                        "anchor": 3,
                        "name": "subtotal",
                    },
                ]
            },
        }
        slice_ = Cube(CR.CAT_4_X_CAT_4, transforms=transforms).partitions[0]

        # Col idx 0 is subdiff and the pairwise sig test is nan for all the cells.
        assert slice_.pairwise_significance_t_stats(4) == pytest.approx(
            np.full((6, 7), np.nan), nan_ok=True
        )
        assert slice_.pairwise_significance_p_vals(4) == pytest.approx(
            np.full((6, 7), np.nan), nan_ok=True
        )
        # Pairwise sig test for subdiffs cols and rows is always nan
        assert slice_.pairwise_significance_t_stats(4)[0] == pytest.approx(
            np.full(7, np.nan), nan_ok=True
        )
        assert slice_.pairwise_significance_p_vals(4)[0] == pytest.approx(
            np.full(7, np.nan), nan_ok=True
        )
        assert slice_.pairwise_significance_t_stats(4)[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.pairwise_significance_p_vals(4)[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )

        # Testing select a base col (6)
        assert slice_.pairwise_significance_t_stats(6) == pytest.approx(
            np.array(
                [
                    [-0.5587170, 0, -0.7407835, -1.2503229, np.nan, -0.338631, 0],
                    [0.109805, 0.5134014, 1.0456736, 1.7044496, np.nan, 0.345617, 0],
                    [0.388913, -0.5765643, -0.457851, -0.755029, np.nan, -0.0596157, 0],
                    [-1.380126, 1.6928294, -0.457133, -0.6850696, np.nan, -0.498122, 0],
                    [-0.963876, 0.0541259, -0.3355484, -0.550853, np.nan, -0.547228, 0],
                    [1.1518654, 0.5864119, 1.6783064, 2.5512164, np.nan, 1.0520486, 0],
                ]
            ),
            nan_ok=True,
            rel=1e-4,
        )
        assert slice_.pairwise_significance_p_vals(6) == pytest.approx(
            np.array(
                [
                    [0.57732344, 1.0, 0.45967235, 0.21325265, np.nan, 0.73527115, 1.0],
                    [0.9127344, 0.6086304, 0.2969431, 0.0904996, np.nan, 0.7300230, 1],
                    [0.6979817, 0.5653320, 0.6475443, 0.4514907, np.nan, 0.9525260, 1],
                    [0.1699348, 0.0931265, 0.6480592, 0.4944246, np.nan, 0.6189893, 1],
                    [0.3369112, 0.9569263, 0.7375545, 0.5826056, np.nan, 0.5848814, 1],
                    [0.2515066, 0.5587188, 0.0948110, 0.0118010, np.nan, 0.2941494, 1],
                ]
            ),
            nan_ok=True,
            rel=1e-4,
        )
        # Testing select a insertion col (2)
        assert slice_.pairwise_significance_t_stats(2) == pytest.approx(
            np.array(
                [
                    [0.1026676, 0.7053229, 0, -0.7137068, np.nan, 0.4910434, 0.7407835],
                    [-0.931539, -0.399469, 0, 0.9168681, np.nan, -0.843487, -1.045673],
                    [0.9177602, -0.229198, 0, -0.416058, np.nan, 0.488158, 0.4578510],
                    [-1.137828, 2.9098290, 0.0, -0.335727, np.nan, -0.065513, 0.457133],
                    [-0.806757, 0.3820092, 0, -0.301685, np.nan, -0.278420, 0.3355484],
                    [-0.292654, -0.881680, 0, 1.2483777, np.nan, -0.691992, -1.678306],
                ]
            ),
            nan_ok=True,
            rel=1e-4,
        )
        assert slice_.pairwise_significance_p_vals(2) == pytest.approx(
            np.array(
                [
                    [0.918325, 0.481442, 1.0, 0.476164, np.nan, 0.623803, 0.459672],
                    [0.352654, 0.689979, 1.0, 0.360215, np.nan, 0.399719, 0.296943],
                    [0.359807, 0.818952, 1.0, 0.677772, np.nan, 0.625842, 0.647544],
                    [0.256501, 0.004032, 1.0, 0.7373960, np.nan, 0.94781462, 0.6480592],
                    [0.420727, 0.702866, 1.0, 0.763176, np.nan, 0.780907, 0.737554],
                    [0.770078, 0.379023, 1.0, 0.213219, np.nan, 0.489550, 0.094811],
                ]
            ),
            nan_ok=True,
            rel=1e-4,
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
        pairwise_indices = slice_.pairwise_indices

        np.testing.assert_array_equal(
            pairwise_indices,
            np.array(load_python_expression("cat-x-cat-wgtd-pw-indices"), dtype=tuple),
        )
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

        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_t_stats(1),
            load_python_expression("mr-x-mr-pw-tstats"),
        )
        np.testing.assert_array_almost_equal(
            slice_.pairwise_significance_p_vals(1),
            load_python_expression("mr-x-mr-pw-pvals"),
        )

    def test_cat_x_cat_with_hs_and_pruning_t_tests(self):
        # Pruned - without insertions
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        assert slice_.pairwise_significance_t_stats(3) == pytest.approx(
            np.array(
                [
                    [1.61500754, 1.000723, 0.68787199, 0.0],
                    [1.02533817, np.nan, 1.01356464, np.nan],
                    [-2.14788648, -1.99808531, -2.01568854, 0.0],
                    [1.81069861, 3.22389432, 2.58138944, np.nan],
                    [0.99418064, 0.99735512, 1.01356464, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(3) == pytest.approx(
            np.array(
                [
                    [0.11480579, 0.324030029, 0.49943424, 1.0],
                    [0.31186072, np.nan, 0.3228962, np.nan],
                    [0.03833944, 0.053756746, 0.057467762, 1.0],
                    [0.07831369, 0.002790704, 0.01782960, np.nan],
                    [0.32659255, 0.32563775, 0.322896241, np.nan],
                ]
            ),
            nan_ok=True,
        )

        # Pruned (just rows) - with insertions, col inserted id 1
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [0.90169345, 0.0, -0.86382097, -1.33091849, np.nan, -1.34418779],
                    [-0.92428183, 0.0, 0.86752477, 1.1640832, np.nan, 1.44689871],
                    [0.4331791, 0.0, -1.0191817, 0.6924624, np.nan, -1.0191817],
                    [-0.30252414, 0.0, 0.28870882, 0.0962085, np.nan, 2.09447737],
                    [-1.17001528, 0.0, 0.94512457, 0.92660979, np.nan, -3.59769363],
                    [-0.03628091, 0.0, 0.03747396, 0.42977612, np.nan, -1.40812549],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.36932341, 1.0, 0.38975346, 0.18673365, np.nan, 0.183227951],
                    [0.35749992, 1.0, 0.38773156, 0.24760958, np.nan, 0.15238873],
                    [0.665790452, 1.0, 0.310576130, 0.490512577, np.nan, 0.31162633],
                    [0.762862615, 1.0, 0.773401832, 0.923578823, np.nan, 0.0398388806],
                    [0.24469499, 1.0, 0.34687333, 0.35672228, np.nan, 5.94507747e-04],
                    [0.971128594, 1.0, 0.970181808, 0.668433443, np.nan, 0.163519781],
                ]
            ),
            nan_ok=True,
        )

        # Pruned (just columns) - with insertions
        transforms = {"columns_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [0.90169345, 0.0, -0.86382097, -1.33091849, -1.34418779],
                    [-0.92428183, 0.0, 0.86752477, 1.1640832, 1.44689871],
                    [0.4331791, 0.0, -1.0191817, 0.6924624, -1.0191817],
                    [-0.30252414, 0.0, 0.28870882, 0.0962085, 2.09447737],
                    [-1.17001528, 0.0, 0.94512457, 0.92660979, -3.59769363],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [-0.03628091, 0.0, 0.03747396, 0.42977612, -1.40812549],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.369323414, 1.0, 0.389753462, 0.186733655, 0.18322795143],
                    [0.357499924, 1.0, 0.38773156, 0.24760958, 0.1523887297],
                    [0.66579045, 1.0, 0.31057612, 0.49051257, 0.3116263295],
                    [0.76286261, 1.0, 0.77340183, 0.92357882, 0.0398388806],
                    [0.24469499, 1.0, 0.346873337, 0.356722284, 0.00059450774],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.97112859, 1.0, 0.97018180, 0.66843344, 0.1635197813],
                ]
            ),
            nan_ok=True,
        )

        # Pruned (rows and columns) - with insertions
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [0.90169345, 0.0, -0.86382097, -1.33091849, -1.34418779],
                    [-0.92428183, 0.0, 0.86752477, 1.1640832, 1.44689871],
                    [0.4331791, 0.0, -1.0191817, 0.6924624, -1.0191817],
                    [-0.30252414, 0.0, 0.28870882, 0.0962085, 2.09447737],
                    [-1.17001528, 0.0, 0.94512457, 0.92660979, -3.59769363],
                    [-0.03628091, 0.0, 0.03747396, 0.42977612, -1.40812549],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.36932341401, 1.0, 0.38975346251, 0.18673365581, 0.18322795143],
                    [0.35749992438, 1.0, 0.3877315607, 0.2476095888, 0.1523887297],
                    [0.6657904524, 1.0, 0.3105761295, 0.4905125765, 0.3116263295],
                    [0.7628626149, 1.0, 0.7734018321, 0.9235788234, 0.0398388806],
                    [0.2446949971, 1.0, 0.34687333799, 0.35672228487, 0.00059450774],
                    [0.9711285942, 1.0, 0.9701818076, 0.6684334428, 0.1635197813],
                ]
            ),
            nan_ok=True,
        )

        # Not pruned - with insertions
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
        assert slice_.pairwise_significance_t_stats(1) == pytest.approx(
            np.array(
                [
                    [0.90169345, 0.0, -0.86382097, -1.33091849, np.nan, -1.34418779],
                    [-0.92428183, 0.0, 0.86752477, 1.1640832, np.nan, 1.44689871],
                    [0.4331791, 0.0, -1.0191817, 0.6924624, np.nan, -1.0191817],
                    [-0.30252414, 0.0, 0.28870882, 0.0962085, np.nan, 2.09447737],
                    [-1.17001528, 0.0, 0.94512457, 0.92660979, np.nan, -3.59769363],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [-0.03628091, 0.0, 0.03747396, 0.42977612, np.nan, -1.40812549],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.369323414, 1.0, 0.389753462, 0.186733655, np.nan, 0.183227951],
                    [0.357499924, 1.0, 0.38773156, 0.24760958, np.nan, 0.15238872],
                    [0.66579045, 1.0, 0.31057612, 0.49051257, np.nan, 0.31162632],
                    [0.76286261, 1.0, 0.77340183, 0.92357882, np.nan, 0.03983888],
                    [0.24469499, 1.0, 0.346873337, 0.3567222848, np.nan, 0.0005945077],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.97112859, 1.0, 0.97018180, 0.66843344, np.nan, 0.16351978],
                ]
            ),
            nan_ok=True,
        )

    def test_cat_hs_x_cat_hs_hiding_and_pruning_t_tests(self):
        slice_ = Cube(
            CR.CAT_HS_X_CAT_HS_EMPTIES,
            transforms={
                "rows_dimension": {
                    "elements": {"2": {"hide": True}},
                    "prune": True,
                    "order": {"type": "explicit", "element_ids": [0, 5, 2, 1, 4]},
                },
                "columns_dimension": {
                    "elements": {"2": {"hide": True}},
                    "prune": True,
                    "order": {"type": "explicit", "element_ids": [4, 2, 5, 0]},
                },
            },
        ).partitions[0]

        assert slice_.pairwise_significance_t_stats(2) == pytest.approx(
            np.array(
                [
                    [0.7768486, -3.15797218, 0.0, 1.31529737, -3.15797218],
                    [1.3255319, 8.61476224, 0.0, 2.24083933, 8.61476224],
                    [-0.56515582, 5.75456407, 0.0, -0.97693327, 5.75456407],
                    [2.92481931, 0.52514588, 0.0, 4.02292482, 0.52514588],
                    [-1.81141912, -3.17763277, 0.0, -3.28290825, -3.17763277],
                    [-0.09211776, -4.20201154, 0.0, -0.16074485, -4.20201154],
                    [0.54642458, 2.89882543, 0.0, 0.95666343, 2.89882543],
                ]
            )
        )
        assert slice_.pairwise_indices.tolist() == [
            [(1, 4), (), (1, 4), (1, 4), ()],
            [(), (0, 2, 3), (), (2,), (0, 2, 3)],
            [(), (0, 2, 3), (), (), (0, 2, 3)],
            [(1, 2, 4), (), (), (0, 1, 2, 4), ()],
            [(3,), (), (1, 3, 4), (), ()],
            [(1, 4), (), (1, 4), (1, 4), ()],
            [(), (0, 2), (), (), (0, 2)],
        ]

    def test_cat_hs_subdiff_x_cat_pairwise_t_test(self):
        slice_ = Cube(CR.CAT_HS_SUBDIFF_X_CAT).partitions[0]
        expected_tstats = np.array(
            load_python_expression("cat-hs-subdiff-x-cat-pw-tstats")
        )
        expected_pvals = np.array(
            load_python_expression("cat-hs-subdiff-x-cat-pw-pvals")
        )

        assert slice_.pairwise_significance_t_stats(3) == pytest.approx(expected_tstats)
        assert slice_.pairwise_significance_p_vals(3) == pytest.approx(expected_pvals)


class TestOverlapsPairwiseSignificance:
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
                    [1.22474487, 0.0, np.nan],
                    [-1.22474487, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_p_vals(1) == pytest.approx(
            np.array(
                [
                    [0.28786413, 1.0, np.nan],
                    [0.27521973, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
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
                    [13.40, 9.62, -0.49, 0.0, 2.27, -24.08],
                    [-0.58, 19.94, -4.76, 0.0, -2.88, -13.48],
                    [-0.53, 0.70, 22.76, 0.0, 10.32, -19.63],
                    [-18.79, -13.57, -20.00, 0.0, -13.14, -29.10],
                    [-5.45, -0.99, -2.83, 0.0, 21.75, -12.29],
                    [-0.76, 0.17, -3.35, 0.0, -1.07, 31.68],
                ]
            ),
            abs=10e-2,
        )

    def test_pairwise_significance_indices_for_realistic_example_mr_x_mr(self):
        transforms = {"columns_dimension": {"elements": {"1": {"hide": True}}}}
        slice_ = Cube(OL.MR_X_MR_REALISTIC_EXAMPLE, transforms=transforms).partitions[0]

        assert slice_.pairwise_indices.tolist() == [
            [(0, 1, 2, 3, 4), (1, 2, 3, 4), (4,), (4,), (1, 2, 4)],
            [(1, 4), (1, 2, 3, 4), (4,), (1, 3, 4), (1, 4)],
            [(4,), (4,), (0, 2, 3, 4), (4,), (0, 2, 4)],
            [(4,), (1, 4), (4,), (0, 1, 3, 4), (1, 4)],
            [(4,), (1, 4), (4,), (1, 4), (0, 1, 2, 4)],
            [(1,), (1,), (), (1,), (1,)],
        ]

    def test_pairwise_sig_for_mr_x_mr_vs_mr_single_subvar_x_mr(self):
        mr_x_mr_slice = Cube(OL.MR_X_MR).partitions[0]
        t_stats_mr_x_mr = mr_x_mr_slice.pairwise_significance_t_stats(1)
        mr_subvar_x_mr_slice = Cube(OL.MR_SINGLE_SUBVAR_X_MR).partitions[0]
        t_stats_mr_subvar_x_mr = mr_subvar_x_mr_slice.pairwise_significance_t_stats(1)
        # Assert same row stats are the same in both cases (MR x MR and MR_SEL x MR)
        np.testing.assert_array_equal(t_stats_mr_x_mr[0], t_stats_mr_subvar_x_mr[0])


class TestMeanDifferenceSignificance:
    def test_mean_diff_significance_for_numeric_array_grouped_by_cat(self):
        slice_ = Cube(NA.NUM_ARR_MULTI_NUMERIC_MEASURES_GROUPED_BY_CAT).partitions[0]
        assert slice_.pairwise_significance_means_t_stats(0) == pytest.approx(
            np.array(
                [
                    [0.0, -0.32190273, -1.884166, -2.16152588],
                    [0.0, -1.91311986, -2.91790845, -1.50036042],
                    [0.0, 1.18770459, 2.59364411, 0.97704863],
                ]
            )
        )
        assert slice_.pairwise_significance_means_p_vals(0) == pytest.approx(
            np.array(
                [
                    [1.0, 0.76970151, 0.17305508, 0.15932937],
                    [1.0, 0.15766242, 0.07043505, 0.20868233],
                    [1.0, 0.40057107, 0.08440979, 0.4040728],
                ]
            )
        )

    def test_mean_diff_significance_for_numeric_array_grouped_by_cat_hs(self):
        transforms = {
            "pairwise_indices": {"alpha": [0.5]},
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
            },
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
        assert slice_.pairwise_significance_means_t_stats(7) == pytest.approx(
            np.array(
                load_python_expression("num-arr-means-grouped-by-cat-hs-t-stats-col-7")
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(7) == pytest.approx(
            np.array(
                load_python_expression("num-arr-means-grouped-by-cat-hs-p-vals-col-7")
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (7,), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), (), ()],
            [(), (), (1, 7), (), (), (), (1, 7), (), ()],
            [(), (), (), (), (), (), (), (), ()],
        ]

        # Test no subtotals
        transforms = {"pairwise_indices": {"alpha": [0.5]}}
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS, transforms=transforms
        ).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (4,), (), (), ()],
            [(), (), (), (), ()],
            [(), (0, 4), (), (0, 4), ()],
            [(), (), (), (), ()],
        ]
        assert slice_.pairwise_significance_means_t_stats(4) == pytest.approx(
            np.array(
                load_python_expression("num-arr-means-grouped-by-cat-t-stats-col-4")
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(4) == pytest.approx(
            np.array(
                load_python_expression("num-arr-means-grouped-by-cat-p-vals-col-4")
            ),
            nan_ok=True,
        )

    def test_mean_diff_significance_for_numeric_array_x_mr(self):
        slice_ = Cube(NA.NUM_ARR_MULTI_NUMERIC_MEASURES_X_MR).partitions[0]

        assert slice_.pairwise_significance_means_t_stats(0) == pytest.approx(
            np.array(
                [
                    [0, 0.461036, -0.41818914],
                    [0, 0.64326512, -0.5674027],
                    [np.nan, np.nan, np.nan],
                    [np.nan, -2.29161759, -1.72993372],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(0) == pytest.approx(
            np.array(
                [
                    [1.0, 0.67005873, 0.69845513],
                    [1.0, 0.54118287, 0.59229185],
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.08370781, 0.15869486],
                ]
            ),
            nan_ok=True,
        )

    def test_mean_diff_significance_indices_for_numeric_array_grouped_by_cat(self):
        transforms = {"pairwise_indices": {"alpha": [0.45, 0.40]}}
        slice_ = Cube(
            NA.NUM_ARR_MULTI_NUMERIC_MEASURES_GROUPED_BY_CAT, transforms=transforms
        ).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(2, 3), (2, 3), (), ()],
            [(1, 2, 3), (), (), ()],
            [(), (), (0, 3), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(2, 3), (2, 3), (), ()],
            [(1, 2, 3), (), (), ()],
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
                    [9.35414347, 0.0, 2.1529364, np.nan],
                    [1.26202085, 0, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(1) == pytest.approx(
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [8.46461576e-05, 1.00000000e00, 6.33217224e-02, np.nan],
                    [2.27836998e-01, 1.00000000e00, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), ()],
            [(1, 2), (), (), ()],
            [(), (), (), ()],
        ]

        # Hiding rows and columns
        slice_ = Cube(
            CR.MEAN_CAT_X_CAT,
            transforms={
                "pairwise_indices": {"alpha": [0.1]},
                "rows_dimension": {
                    "elements": {"1": {"hide": True}},
                    "prune": True,
                },
                "columns_dimension": {
                    "elements": {"1": {"hide": True}},
                    "prune": True,
                    "order": {"type": "explicit", "element_ids": [4, 2, 5, 0]},
                },
            },
        ).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (1,)],
            [(), (), ()],
        ]
        assert slice_.pairwise_significance_means_t_stats(1) == pytest.approx(
            np.array(
                [
                    [np.nan, 0, 2.1529364],
                    [np.nan, 0, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.pairwise_significance_means_p_vals(1) == pytest.approx(
            np.array(
                [
                    [np.nan, 1, 0.06332172],
                    [np.nan, 1.0, np.nan],
                ]
            ),
            nan_ok=True,
        )

    def test_mean_diff_significance_indices_for_cat_x_cat(self):
        transforms = {"pairwise_indices": {"alpha": [0.15, 0.05]}}
        slice_ = Cube(CR.MEAN_CAT_X_CAT, transforms=transforms).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), ()],
            [(1, 2), (), (), ()],
            [(), (), (), ()],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(), (), (), (0,)],
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
            "pairwise_indices": {"alpha": [0.45, 0.333]},
            "columns_dimension": {"insertions": insertions},
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), (6,), (6,)],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(), (), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), (4, 6), (4, 6)],
        ]

        # Test no subtotals
        transforms = {"pairwise_indices": {"alpha": [0.45, 0.333]}}
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]
        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), (), ()],
            [(), (), (), (), ()],
            [(), (), (), (2,), (2,)],
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
            "pairwise_indices": {"alpha": [0.45, 0.333]},
        }
        slice_ = Cube(
            NA.NUM_ARR_MEANS_GROUPED_BY_CAT_HS_WEIGHTED, transforms=transforms
        ).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (5,), (5,)],
        ]
        assert slice_.pairwise_means_indices_alt.tolist() == [
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (4, 5), (4, 5)],
        ]

    def test_mean_indices_cat_x_mr_subvar(self):
        slice_ = Cube(CR.CAT_X_MR_SUBVAR_MEANS).partitions[0]

        assert slice_.pairwise_means_indices.tolist() == [[(), (), (), ()]]
