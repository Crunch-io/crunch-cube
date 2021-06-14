# encoding: utf-8

"""Integrations tests for headings and subtotals behaviors."""

from __future__ import division

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR, NA


class DescribeIntegrated_SubtotalDifferences(object):
    """TDD driver(s) for Subtotal Difference insertions."""

    def it_computes_measures_for_1D_cat_with_subdiffs(self):
        strand = Cube(
            CR.CAT,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [4, 5]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                }
            },
        ).partitions[0]

        assert strand.counts[0] == 81
        assert strand.table_proportions[0] == pytest.approx(0.1184210)

    def it_computes_measures_for_cat_x_cat_with_subdiffs_on_both(self):
        slice_ = Cube(
            CR.CAT_4_X_CAT_4,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == pytest.approx(
            [np.nan, -8, 0, -6, -3], nan_ok=True
        )
        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [np.nan, 0, 8, -2, 5], nan_ok=True
        )
        assert slice_.columns_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.columns_margin_proportion[0] == pytest.approx(11 / 266)
        assert slice_.rows_margin_proportion[0] == pytest.approx(-17 / 266)
        assert slice_.columns_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[0, 1] == slice_.column_weighted_bases[1, 1]
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_weighted_bases[1, 0] == slice_.row_weighted_bases[1, 1]
        assert slice_.column_proportions[0, :] == pytest.approx(
            [np.nan, -0.119403, 0, -0.0759494, -0.046875], nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions_moe[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions_moe[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.column_std_dev[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_std_dev[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.column_std_err[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_std_err[0, :] == pytest.approx(
            [np.nan, np.nan, 0, np.nan, np.nan], nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.10810811, -0.02985075, 0.07352941], nan_ok=True
        )
        assert slice_.row_proportions_moe[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions_moe[:, 0] == pytest.approx(
            [np.nan, 0, 0.07074854, np.nan, 0.06203546], nan_ok=True
        )
        assert slice_.row_std_dev[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_std_dev[:, 0] == pytest.approx(
            [np.nan, 0, 0.3105169, np.nan, 0.26100352], nan_ok=True
        )
        assert slice_.row_std_err[0, :] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_std_err[:, 0] == pytest.approx(
            [np.nan, 0, 0.03609686, np.nan, 0.03165133], nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [np.nan, -0.03007519, 0, -0.02255639, -0.0112782], nan_ok=True
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.03007519, -0.0075188, 0.01879699], nan_ok=True
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.zscores[0, :] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(5, np.nan), nan_ok=True)

    def it_computes_measures_for_cat_x_cat_with_subdiffs_and_subtot_on_both(self):
        slice_ = Cube(
            CR.CAT_4_X_CAT_4,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                        {
                            "function": "subtotal",
                            "args": [3, 4],
                            "anchor": "bottom",
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
                            "anchor": "top",
                            "name": "NPS",
                        },
                        {
                            "function": "subtotal",
                            "args": [3, 4],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == pytest.approx(
            [np.nan, -8, 0, -6, -3, -9], nan_ok=True
        )
        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [np.nan, 0, 8, -2, 5, 3], nan_ok=True
        )
        assert slice_.columns_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_margin[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.columns_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[0, 1] == slice_.column_weighted_bases[1, 1]
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_weighted_bases[1, 0] == slice_.row_weighted_bases[1, 1]
        assert slice_.column_proportions[0, :] == pytest.approx(
            [np.nan, -0.119403, 0, -0.0759494, -0.046875, -0.06293706], nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(6, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.10810811, -0.02985075, 0.07352941, 0.02222222], nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [np.nan, -0.03007519, 0, -0.02255639, -0.0112782, -0.03383459], nan_ok=True
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [np.nan, 0, 0.03007519, -0.0075188, 0.01879699, 0.0112782], nan_ok=True
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.zscores[0, :] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(6, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(6, np.nan), nan_ok=True)

    def it_computes_measures_for_ca_with_subdiff(self):
        slice_ = Cube(
            CR.CA_CAT_X_CA_SUBVAR,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [0],
                            "kwargs": {"negative": [4]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[0, :].tolist() == [-178, -495, 0]
        assert slice_.rows_base[0] == pytest.approx(np.nan, nan_ok=True)
        assert slice_.row_weighted_bases[0, :] == pytest.approx(
            np.full(3, np.nan), nan_ok=True
        )
        assert slice_.column_proportions[0, :] == pytest.approx(
            [-0.10847044, -0.30201342, np.nan], nan_ok=True
        )
        assert slice_.row_proportions[0, :] == pytest.approx(
            np.full(3, np.nan), nan_ok=True
        )
        assert slice_.table_proportions[0, :] == pytest.approx(
            [-0.10847044, -0.30201342, np.nan], nan_ok=True
        )
        assert slice_.zscores[0, :] == pytest.approx(np.full(3, np.nan), nan_ok=True)
        assert slice_.pvals[0, :] == pytest.approx(np.full(3, np.nan), nan_ok=True)

    def it_computes_measures_for_mr_x_cat_subdiff(self):
        slice_ = Cube(
            CR.MR_X_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [2],
                            "kwargs": {"negative": [4]},
                            "anchor": "top",
                            "name": "NPS",
                        },
                    ]
                },
            },
        ).partitions[0]

        assert slice_.counts[:, 0].tolist() == pytest.approx(
            [1.9215376, -12.3047603, -31.4956882, -88.6847375, -56.4466419]
        )
        assert slice_.columns_margin[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.columns_base[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_weighted_bases[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.column_proportions[:, 0] == pytest.approx(
            np.full(5, np.nan), nan_ok=True
        )
        assert slice_.row_proportions[:, 0] == pytest.approx(
            [0.06074756, -0.17396625, -0.25044458, -0.2417213, -0.14981897]
        )
        assert slice_.table_proportions[:, 0] == pytest.approx(
            [0.0108952, -0.05820039, -0.12713165, -0.19403687, -0.11960763]
        )
        assert slice_.zscores[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)
        assert slice_.pvals[:, 0] == pytest.approx(np.full(5, np.nan), nan_ok=True)

    def it_computes_scale_median_for_cat_with_subdiff_x_cat_with_subdiff(self):
        slice_ = Cube(
            CR.CAT_HS_MT_X_CAT_HS_MT,
            transforms={
                "rows_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "NPS",
                        }
                    ]
                },
            },
        ).partitions[0]

        assert slice_.columns_scale_median == pytest.approx(
            [np.nan, 1, 1, 1, np.nan, 3], nan_ok=True
        )

        assert slice_.rows_scale_median == pytest.approx(
            [np.nan, 2, 1, 2, 2, np.nan, 2], nan_ok=True
        )

    def it_computes_sum_for_numarray_with_subdiffs_and_subtot_on_columns(self):
        slice_ = Cube(
            NA.NUM_ARR_SUM_GROUPED_BY_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "subdiff",
                        },
                        {
                            "function": "subtotal",
                            "args": [1, 2],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]
        assert slice_.sums == pytest.approx(
            np.array(
                [
                    [np.nan, 4.0, 3.0, 7.0],
                    [np.nan, 3.0, 0.0, 3.0],
                    [np.nan, 2.0, 3.0, 5.0],
                ]
            ),
            nan_ok=True,
        )

        # pruning
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]
        assert slice_.sums == pytest.approx(
            np.array(
                [
                    [4.0, 3.0],
                    [3.0, 0.0],
                    [2.0, 3.0],
                ]
            )
        )

    def it_computes_share_of_sum_for_numarray_with_subdiffs_and_subtot_on_columns(self):
        slice_ = Cube(
            NA.NUM_ARR_SUM_GROUPED_BY_CAT,
            transforms={
                "columns_dimension": {
                    "insertions": [
                        {
                            "function": "subtotal",
                            "args": [1],
                            "kwargs": {"negative": [2]},
                            "anchor": "top",
                            "name": "subdiff",
                        },
                        {
                            "function": "subtotal",
                            "args": [1, 2],
                            "anchor": "bottom",
                            "name": "subtotal",
                        },
                    ]
                },
            },
        ).partitions[0]
        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.44444444, 0.5, 0.46666667],
                    [np.nan, 0.33333333, 0.0, 0.2],
                    [np.nan, 0.22222222, 0.5, 0.33333333],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.5714285, 0.4285714, np.nan],
                    [np.nan, 1.0, 0.0, np.nan],
                    [np.nan, 0.4, 0.6, np.nan],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    [np.nan, 0.26666667, 0.2, np.nan],
                    [np.nan, 0.2, 0.0, np.nan],
                    [np.nan, 0.13333333, 0.2, np.nan],
                ]
            ),
            nan_ok=True,
        )

        # pruning
        slice_ = Cube(NA.NUM_ARR_SUM_GROUPED_BY_CAT).partitions[0]
        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    [0.4444444, 0.5],
                    [0.3333333, 0.0],
                    [0.2222222, 0.5],
                ]
            )
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    [0.57142857, 0.42857143],
                    [1.0, 0.0],
                    [0.4, 0.6],
                ]
            ),
            nan_ok=True,
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    [0.26666667, 0.2],
                    [0.2, 0.0],
                    [0.13333333, 0.2],
                ]
            ),
            nan_ok=True,
        )
