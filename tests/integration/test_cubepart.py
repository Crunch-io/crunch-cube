# encoding: utf-8

"""Integration-test suite for `cr.cube.cubepart` module."""

import numpy as np
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.cube import Cube
from cr.cube.enums import DIMENSION_TYPE as DT

# ---mnemonic: CR = 'cube-response'---
# ---mnemonic: TR = 'transforms'---
from ..fixtures import CR, TR
from ..util import load_python_expression


class Describe_Slice(object):
    """Integration-test suite for _Slice object."""

    def it_provides_values_for_cat_x_cat(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]

        assert slice_.column_labels.tolist() == ["C", "E"]
        assert slice_.columns_dimension_name == "v7"
        assert slice_.columns_dimension_type == DT.CAT
        assert slice_.description == "Pet Owners"
        np.testing.assert_almost_equal(
            slice_.row_proportions, np.array([[0.71428571, 0.28571429], [0.625, 0.375]])
        )
        assert slice_.inserted_column_idxs == ()
        assert slice_.inserted_row_idxs == ()
        assert slice_.is_empty is False
        assert slice_.name == "v4"
        np.testing.assert_array_almost_equal(
            slice_.residual_test_stats,
            [
                [[0.71439304, 0.71439304], [0.71439304, 0.71439304]],
                [[0.36596253, -0.36596253], [-0.36596253, 0.36596253]],
            ],
        )
        assert slice_.row_labels.tolist() == ["B", "C"]
        assert slice_.rows_dimension_description == "Pet Owners"
        assert slice_.rows_dimension_fills == (None, None)
        assert slice_.rows_dimension_name == "v4"
        assert slice_.rows_dimension_type == DT.CAT
        assert slice_.shape == (2, 2)
        assert slice_.table_name is None
        assert slice_.variable_name == "v7"

    def it_provides_values_for_cat_hs_mt_x_cat_hs_mt(self):
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, population=1000).partitions[0]

        np.testing.assert_array_equal(
            slice_.unweighted_counts,
            [
                [28, 48, 20, 10, 0, 1],
                [7, 19, 12, 8, 0, 2],
                [1, 1, 0, 1, 0, 0],
                [3, 7, 4, 2, 0, 2],
                [3, 11, 8, 5, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 2, 1, 1, 0, 0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.column_index,
            [
                [119.51424314, np.nan, 93.79691945, 81.24002871, np.nan, 50.17378808],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [129.57286183, np.nan, 0.0, 234.24141174, np.nan, 0.0],
                [68.74132837, np.nan, 99.37070393, 89.71346023, np.nan, 554.56882771],
                [48.00638105, np.nan, 137.75263905, 149.33201406, np.nan, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [83.86593194, np.nan, 91.83509328, 160.99065701, np.nan, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.77796143, 0.69805616, 0.61055807, 0.52882073, np.nan, 0.32659933],
                [0.1953168, 0.27401008, 0.360181, 0.41988366, np.nan, 0.67340067],
                [0.02837466, 0.01483081, 0.0, 0.05129561, np.nan, 0.0],
                [0.08347107, 0.1012239, 0.12066365, 0.10893707, np.nan, 0.67340067],
                [0.08347107, 0.15795536, 0.23951735, 0.25965098, np.nan, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
                [0.02672176, 0.02793377, 0.02926094, 0.05129561, np.nan, 0.0],
            ],
        )
        assert slice_.columns_dimension_name == "ShutdownBla"
        assert slice_.columns_dimension_type == DT.CAT
        np.testing.assert_almost_equal(
            slice_.column_std_dev,
            [
                [0.41561694, 0.45910103, 0.48762374, 0.49916867, np.nan, 0.4689693],
                [0.39644438, 0.44601408, 0.48005275, 0.49353964, np.nan, 0.4689693],
                [0.16604076, 0.12087539, 0.0, 0.22060003, np.nan, 0.0],
                [0.27659294, 0.30162497, 0.32573599, 0.31156024, np.nan, 0.4689693],
                [0.27659294, 0.36469915, 0.42678893, 0.4384431, np.nan, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
                [0.16126906, 0.1647831, 0.16853704, 0.22060003, np.nan, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.column_std_err,
            [
                [0.06895161, 0.05506512, 0.08465401, 0.11473767, np.nan, 0.27200111],
                [0.06577085, 0.05349546, 0.08333965, 0.1134438, np.nan, 0.27200111],
                [0.02754647, 0.01449794, 0.0, 0.05070657, np.nan, 0.0],
                [0.04588727, 0.03617726, 0.05654946, 0.07161446, np.nan, 0.27200111],
                [0.04588727, 0.04374245, 0.07409277, 0.10077944, np.nan, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
                [0.02675483, 0.01976428, 0.0292589, 0.05070657, np.nan, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("cat-x-cat-pruning-hs-col-prop-moe"),
        )
        assert slice_.dimension_types == (DT.CAT, DT.CAT)
        assert slice_.inserted_column_idxs == (1,)
        assert slice_.inserted_row_idxs == (1,)
        assert slice_.is_empty is False
        assert slice_.name == "MaritalStat"
        assert slice_.ndim == 2
        np.testing.assert_almost_equal(
            slice_.pvals,
            [
                [0.03851757, 0.0922145, 0.54097586, 0.21071341, np.nan, 0.23299113],
                [0.04198008, 0.11390712, 0.50508577, 0.28105398, np.nan, 0.1797169],
                [0.73113976, 0.41072494, 0.28019785, 0.32642279, np.nan, 0.79310382],
                [0.36684711, 0.29203707, 0.98652895, 0.85178994, np.nan, 0.00305394],
                [0.06398878, 0.47430453, 0.21130996, 0.26884987, np.nan, 0.4212984],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.82020207, 0.70318269, 0.91486794, 0.58880283, np.nan, 0.75048675],
            ],
        )
        assert slice_.rows_dimension_description == "What is you"
        assert slice_.rows_dimension_fills == (None, None, None, None, None, None, None)
        assert slice_.rows_dimension_type == DT.CAT
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.47502103, 0.81547519, 0.34045416, 0.16820858, 0.0, 0.01631623],
                [0.24473593, 0.65688643, 0.4121505, 0.27407663, 0.0, 0.06903693],
                [0.515, 0.515, 0.0, 0.485, 0.0, 0.0],
                [0.27321912, 0.63390442, 0.3606853, 0.18575293, 0.0, 0.18034265],
                [0.19080605, 0.69080605, 0.5, 0.30919395, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.33333333, 0.66666667, 0.33333333, 0.33333333, 0.0, 0.0],
            ],
        )
        assert slice_.table_base == 91
        assert slice_.table_name is None
        np.testing.assert_almost_equal(
            slice_.table_std_dev,
            [
                [0.46216723, 0.49904908, 0.41533263, 0.31225682, 0.0, 0.10250865],
                [0.26758936, 0.40613936, 0.33710998, 0.28174342, 0.0, 0.14635252],
                [0.10559638, 0.10559638, 0.0, 0.10250865, 0.0, 0.0],
                [0.17909696, 0.26654957, 0.20464365, 0.1484817, 0.0, 0.14635252],
                [0.17909696, 0.32509465, 0.28174342, 0.22554563, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.10250865, 0.14418892, 0.10250865, 0.10250865, 0.0, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            [
                [0.04833892, 0.05219646, 0.0434404, 0.03265951, 0.0, 0.01072157],
                [0.02798766, 0.04247886, 0.03525895, 0.02946806, 0.0, 0.01530728],
                [0.01104452, 0.01104452, 0.0, 0.01072157, 0.0, 0.0],
                [0.01873208, 0.02787891, 0.02140405, 0.01552997, 0.0, 0.01530728],
                [0.01873208, 0.03400224, 0.02946806, 0.02359023, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01072157, 0.01508098, 0.01072157, 0.01072157, 0.0, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.table_proportions_moe,
            load_python_expression("cat-x-cat-pruning-hs-table-prop-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions_moe,
            [
                [0.12688317, 0.09856198, 0.12040055, 0.09504033, 0.0, 0.03218951],
                [0.1564864, 0.17279947, 0.17915928, 0.16235278, 0.0, 0.09227527],
                [0.69232826, 0.69232826, 0.0, 0.69232826, 0.0, 0.0],
                [0.26214652, 0.28339745, 0.28249344, 0.22878774, 0.0, 0.22617896],
                [0.19317446, 0.22720657, 0.24580874, 0.22720657, 0.0, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.54137703, 0.54137703, 0.54137703, 0.54137703, 0.0, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.population_counts_moe,
            [
                [94.7425342, 102.30317352, 85.14161786, 64.01146595, 0.0, 21.01388583],
                [54.85480624, 83.25703295, 69.10626964, 57.75633502, 0.0, 30.00171283],
                [21.64685796, 21.64685796, 0.0, 21.01388583, 0.0, 0.0],
                [36.71419889, 54.64165401, 41.9511732, 30.43818609, 0.0, 30.00171283],
                [36.71419889, 66.6431728, 57.75633469, 46.23600106, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [21.01388583, 29.55818233, 21.01388583, 21.01388583, 0.0, 0.0],
            ],
        )
        np.testing.assert_almost_equal(
            slice_.zscores,
            [
                [2.06930398, 1.68383013, -0.61133797, -1.25160615, np.nan, -1.19268916],
                [-2.03371753, -1.58087282, 0.66650907, 1.07795469, np.nan, 1.34162721],
                [0.3436098, -0.82261857, -1.079875, 0.98134469, np.nan, -0.26228228],
                [-0.90239493, -1.05366336, -0.01688425, -0.18683508, np.nan, 2.962256],
                [-1.85225802, -0.7154929, 1.24997148, 1.10571507, np.nan, -0.8041707],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [-0.22728508, -0.3810277, -0.10690048, 0.5405717, np.nan, -0.31799761],
            ],
        )

    def it_provides_values_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]

        np.testing.assert_almost_equal(
            slice_.table_std_dev, load_python_expression("cat-hs-x-mr-tbl-stddev")
        )
        np.testing.assert_almost_equal(
            slice_.table_std_err, load_python_expression("cat-hs-x-mr-tbl-stderr")
        )
        np.testing.assert_almost_equal(
            slice_.column_std_dev, load_python_expression("cat-hs-x-mr-col-stddev")
        )
        np.testing.assert_almost_equal(
            slice_.column_std_err, load_python_expression("cat-hs-x-mr-col-stderr")
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("cat-hs-x-mr-col-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.zscores, load_python_expression("cat-hs-x-mr-zscores")
        )
        np.testing.assert_almost_equal(
            slice_.pvals, load_python_expression("cat-hs-x-mr-pvals")
        )

    def it_provides_values_for_mean_cat_x_cat_hs(self):
        slice_ = Cube(CR.MEANS_CAT_X_CAT_HS).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.means,
            np.array([[24.43935757, 37.32122746, np.nan, 55.48571956, 73.02427659]]),
        )
        np.testing.assert_array_almost_equal(slice_.rows_margin, np.array([np.nan]))
        np.testing.assert_array_almost_equal(
            slice_.columns_margin, np.array([np.nan] * len(slice_.counts[0, :]))
        )

    @pytest.mark.skip(reason="Needs change to Cube.counts (and add Cube.means)")
    # --- this fails because `Cube.counts` returns the _means_ measure (and not the
    # --- cube-count measure, as it should now). This needs to be fixed, including
    # --- resolving whether we should continue to return NaNs or raise an exception (as
    # --- would be the general behavior when requesting a measure on a cube that lacks
    # --- cube-measure its computation is based on.
    def but_it_has_no_counts_because_there_is_no_cube_count_measure(self):
        slice_ = Cube(CR.MEANS_CAT_X_CAT_HS).partitions[0]
        assert slice_.counts == pytest.approx(
            np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]]), nan_ok=True
        )

    def it_provides_values_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]

        np.testing.assert_almost_equal(
            slice_.table_std_dev, load_python_expression("mr-x-cat-hs-tbl-stddev")
        )
        np.testing.assert_almost_equal(
            slice_.table_std_err, load_python_expression("mr-x-cat-hs-tbl-stderr")
        )
        np.testing.assert_almost_equal(
            slice_.column_std_dev, load_python_expression("mr-x-cat-hs-col-stddev")
        )
        np.testing.assert_almost_equal(
            slice_.column_std_err, load_python_expression("mr-x-cat-hs-col-stderr")
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("mr-x-cat-hs-col-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.pvals, load_python_expression("mr-x-cat-hs-pvals")
        )
        np.testing.assert_almost_equal(
            slice_.zscores, load_python_expression("mr-x-cat-hs-zscores")
        )

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (CR.CAT_HS_X_MR, "cat-hs-x-mr-row-proportions"),
            (CR.MR_X_CAT_HS_MT, "mr-x-cat-hs-row-proportions"),
            (CR.MR_X_MR, "mr-x-mr-row-proportions"),
        ),
    )
    def it_knows_the_row_proportions(self, fixture, expectation):
        slice_ = _Slice(
            Cube(fixture), slice_idx=0, transforms={}, population=None, mask_size=0
        )

        row_proportions = slice_.row_proportions

        np.testing.assert_almost_equal(
            row_proportions, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, expectation",
        (
            (CR.CAT_HS_X_MR, "cat-hs-x-mr-column-proportions"),
            (CR.MR_X_CAT_HS_MT, "mr-x-cat-hs-column-proportions"),
            (CR.MR_X_MR, "mr-x-mr-column-proportions"),
        ),
    )
    def it_knows_the_column_proportions(self, fixture, expectation):
        slice_ = _Slice(
            Cube(fixture), slice_idx=0, transforms={}, population=None, mask_size=0
        )

        column_proportions = slice_.column_proportions

        np.testing.assert_almost_equal(
            column_proportions, load_python_expression(expectation)
        )

    @pytest.mark.parametrize(
        "fixture, row_order, col_order, expectation",
        (
            (CR.CA_X_CAT_HS, [3, 1, 2], [3, 1, 2], "ca-x-cat-hs-explicit-order"),
            (CR.CA_X_MR, [0, 1, 2, 5, 4], [3, 1, 2], "ca-x-mr-explicit-order"),
            (CR.CAT_X_CAT, [1, 2], [3, 1], "cat-x-cat-explicit-order"),
            (CR.CAT_X_MR, [2, 1], [3, 2, 1], "cat-x-mr-explicit-order"),
            (CR.MR_X_CAT, [2, 1, 3, 4, 5], [5, 1, 4, 3, 2], "mr-x-cat-explicit-order"),
            (
                CR.MR_X_CAT_HS_MT,
                [5, 1, 4, 2, 3],
                [5, 1, 4, 3, 2],
                "mr-x-cat-hs-explicit-order",
            ),
            (CR.MR_X_MR, [1, 2, 3, 0], [2, 1, 3, 0], "mr-x-mr-explicit-order"),
        ),
    )
    def it_respects_explicit_order_transform_for_dim_types(
        self, fixture, row_order, col_order, expectation
    ):
        transforms = {
            "rows_dimension": {"order": {"type": "explicit", "element_ids": row_order}},
            "columns_dimension": {
                "order": {"type": "explicit", "element_ids": col_order}
            },
        }
        slice_ = _Slice(
            Cube(fixture),
            slice_idx=0,
            transforms=transforms,
            population=None,
            mask_size=0,
        )

        actual = [[int(n) for n in row_counts] for row_counts in slice_.counts]

        expected = load_python_expression(expectation)
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)

    @pytest.mark.parametrize(
        "measure_propname, expectation",
        (
            ("column_index", "cat-x-cat-col-idx-explicit-order"),
            ("unweighted_counts", "cat-x-cat-ucounts-explicit-order"),
            ("zscores", "cat-x-cat-zscores-explicit-order"),
        ),
    )
    def and_it_respects_explicit_order_transform_for_measures(
        self, measure_propname, expectation
    ):
        transforms = {
            "rows_dimension": {
                "order": {"type": "explicit", "element_ids": [2, 4, 3, 1]}
            },
            "columns_dimension": {
                "order": {"type": "explicit", "element_ids": [3, 2, 4, 1]}
            },
        }
        slice_ = _Slice(
            Cube(CR.CAT_4_X_CAT_4),
            slice_idx=0,
            transforms=transforms,
            population=None,
            mask_size=0,
        )

        actual = getattr(slice_, measure_propname)

        expected = load_python_expression(expectation)
        np.testing.assert_almost_equal(actual, expected)

    def it_ignores_hidden_subtotals(self):
        """A subtotal with `"hide": True` does not appear.

        This behavior is added in the "interim", insertion-has-no-id state to allow
        display of a global (lives-on-variable) insertion to be suppressed without
        actually deleting it, which would require unnatural acts to restore it later if
        wanted again.
        """
        transforms = {
            "rows_dimension": {
                "insertions": [
                    {
                        "anchor": "top",
                        "args": [1, 2],
                        "function": "subtotal",
                        "hide": True,
                        "name": "Apple+Banana",
                    }
                ]
            },
            "columns_dimension": {
                "insertions": [
                    {
                        "anchor": 4,
                        "args": [1, 4],
                        "function": "subtotal",
                        "hide": True,
                        "name": "Asparagus+Daikon",
                    }
                ]
            },
        }
        slice_ = _Slice(Cube(CR.CAT_4_X_CAT_4), 0, transforms, None, 0)

        assert slice_.row_labels.tolist() == ["Apple", "Banana", "Cherry", "Date"]
        assert slice_.column_labels.tolist() == [
            "Asparagus",
            "Broccoli",
            "Cauliflower",
            "Daikon",
        ]
        np.testing.assert_equal(
            slice_.counts,
            [[14, 14, 13, 16], [22, 14, 19, 19], [14, 16, 19, 18], [17, 12, 28, 11]],
        )

    def it_places_insertions_on_a_reordered_dimension_in_the_right_position(self):
        """Subtotal anchors follow re-ordered rows.

        The key fixture characteristic is that an ordering transform is combined with
        subtotal insertions such that their subtotal position is changed by the
        ordering.
        """
        transforms = {
            "rows_dimension": {
                "insertions": [
                    {
                        "anchor": "top",
                        "args": [1, 2],
                        "function": "subtotal",
                        "name": "Apple+Banana",
                    },
                    {
                        "anchor": 4,
                        "args": [1, 4],
                        "function": "subtotal",
                        "name": "Apple+Date",
                    },
                    {
                        "anchor": "bottom",
                        "args": [3, 4],
                        "function": "subtotal",
                        "name": "Cherry+Date",
                    },
                ],
                "order": {"element_ids": [2, 4, 3, 1], "type": "explicit"},
            },
            "columns_dimension": {
                "insertions": [
                    {
                        "anchor": "top",
                        "args": [1, 2],
                        "function": "subtotal",
                        "name": "Asparagus+Broccoli",
                    },
                    {
                        "anchor": 4,
                        "args": [1, 4],
                        "function": "subtotal",
                        "name": "Asparagus+Daikon",
                    },
                    {
                        "anchor": "bottom",
                        "args": [3, 4],
                        "function": "subtotal",
                        "name": "Cauliflower+Daikon",
                    },
                ],
                "order": {"element_ids": [2, 4, 3, 1], "type": "explicit"},
            },
        }
        slice_ = Cube(CR.CAT_4_X_CAT_4, transforms=transforms).partitions[0]

        assert slice_.row_labels.tolist() == [
            "Apple+Banana",
            "Banana",
            "Date",
            "Apple+Date",
            "Cherry",
            "Apple",
            "Cherry+Date",
        ]
        assert slice_.column_labels.tolist() == [
            "Asparagus+Broccoli",
            "Broccoli",
            "Daikon",
            "Asparagus+Daikon",
            "Cauliflower",
            "Asparagus",
            "Cauliflower+Daikon",
        ]
        np.testing.assert_equal(
            slice_.counts,
            [
                #     2   4  1+4  3   1  3+4
                [64, 28, 35, 71, 32, 36, 67],
                [36, 14, 19, 41, 19, 22, 38],
                [29, 12, 11, 28, 28, 17, 39],
                [57, 26, 27, 58, 41, 31, 68],
                [30, 16, 18, 32, 19, 14, 37],
                [28, 14, 16, 30, 13, 14, 29],
                [59, 28, 29, 60, 47, 31, 76],
            ],
        )

    def it_provides_same_proportions_without_explicit_order(self):
        transforms = TR.TEST_DASHBOARD_TRANSFORM_SINGLE_EL_VISIBLE
        slice_ = Cube(CR.TEST_DASHBOARD_FIXTURE, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(slice_.column_proportions, [[0.48313902]])
        np.testing.assert_almost_equal(slice_.row_proportions, [[0.61110996]])

        # delete the explicit order
        transforms = TR.TEST_DASHBOARD_TRANSFORM_NO_ORDERING
        slice_wo_explicit_order_ = Cube(
            CR.TEST_DASHBOARD_FIXTURE, transforms=transforms
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_wo_explicit_order_.column_proportions, [[0.48313902]]
        )
        np.testing.assert_almost_equal(
            slice_wo_explicit_order_.row_proportions, [[0.61110996]]
        )
        np.testing.assert_almost_equal(
            slice_.columns_base, slice_wo_explicit_order_.columns_base
        )
        np.testing.assert_almost_equal(
            slice_.rows_base, slice_wo_explicit_order_.rows_base
        )

    def it_knows_when_it_is_empty(self):
        assert Cube(CR.OM_SGP8334215_VN_2019_SEP_19).partitions[0].is_empty is True

    def it_provides_unpruned_table_margin(self):
        slice_ = _Slice(Cube(CR.MR_X_CAT_HS_MT), 0, None, None, 0)
        np.testing.assert_array_equal(
            slice_.table_base_unpruned, [165, 210, 242, 450, 476]
        )
        np.testing.assert_almost_equal(
            slice_.table_margin_unpruned,
            [176.3655518, 211.4205877, 247.7407379, 457.0509557, 471.9317685],
        )

    def it_prunes_cat_x_cat_with_hs(self):
        # Pruned - without insertions
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        expected = np.array(
            [[28, 20, 10, 1], [1, 0, 1, 0], [3, 4, 2, 2], [3, 8, 5, 0], [1, 1, 1, 0]]
        )
        np.testing.assert_equal(slice_.unweighted_counts, expected)

        # Pruned (just rows) - with insertions
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [28, 48, 20, 10, 0, 1],
                [7, 19, 12, 8, 0, 2],
                [1, 1, 0, 1, 0, 0],
                [3, 7, 4, 2, 0, 2],
                [3, 11, 8, 5, 0, 0],
                [1, 2, 1, 1, 0, 0],
            ]
        )
        np.testing.assert_equal(slice_.unweighted_counts, expected)

        # Pruned (just columns) - with insertions
        transforms = {"columns_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [28, 48, 20, 10, 1],
                [7, 19, 12, 8, 2],
                [1, 1, 0, 1, 0],
                [3, 7, 4, 2, 2],
                [3, 11, 8, 5, 0],
                [0, 0, 0, 0, 0],
                [1, 2, 1, 1, 0],
            ]
        )
        np.testing.assert_equal(slice_.unweighted_counts, expected)

        # Pruned (rows and columns) - with insertions
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [28, 48, 20, 10, 1],
                [7, 19, 12, 8, 2],
                [1, 1, 0, 1, 0],
                [3, 7, 4, 2, 2],
                [3, 11, 8, 5, 0],
                [1, 2, 1, 1, 0],
            ]
        )
        np.testing.assert_equal(slice_.unweighted_counts, expected)

        # Not pruned - with insertions
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
        expected = np.array(
            [
                [28, 48, 20, 10, 0, 1],
                [7, 19, 12, 8, 0, 2],
                [1, 1, 0, 1, 0, 0],
                [3, 7, 4, 2, 0, 2],
                [3, 11, 8, 5, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 2, 1, 1, 0, 0],
            ]
        )
        np.testing.assert_equal(slice_.unweighted_counts, expected)

    def it_accommodates_an_all_missing_element_rows_dimension(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_ALL_MISSING_ROW_ELEMENTS), 0, None, None, 0)
        row_proportions = slice_.row_proportions
        np.testing.assert_almost_equal(row_proportions, np.array([]).reshape((0, 2)))

    def it_knows_means_with_subtotals_on_cat_x_cat(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_MEAN_SUBTOT), 0, None, None, 0)

        means = slice_.means

        np.testing.assert_almost_equal(
            means, np.array([[38.3333333, np.nan, 65.0, 55.0, 34.0]])
        )

    def it_knows_its_selected_categories(self):
        slice_ = Cube(CR.MR_X_MR_SELECTED_CATEGORIES).partitions[0]

        assert slice_.selected_category_labels == ("Very Favorable",)


class Describe_Strand(object):
    """Integration-test suite for `cr.cube.cubepart._Strand` object."""

    def it_provides_values_for_univariate_cat(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]

        assert strand.bases == (15, 15)
        assert strand.counts == (10, 5)
        assert strand.cube_index == 0
        assert strand.dimension_types == (DT.CAT,)
        assert strand.has_means is False
        assert strand.inserted_row_idxs == ()
        assert strand.is_empty is False
        assert strand.means == (np.nan, np.nan)
        np.testing.assert_equal(strand.min_base_size_mask, [False, False])
        assert strand.name == "v7"
        assert strand.ndim == 1
        assert strand.population_counts == (0.0, 0.0)
        assert strand.row_count == 2
        assert strand.row_labels == ("C", "E")
        np.testing.assert_equal(strand.rows_base, [10, 5])
        assert strand.rows_dimension_fills == (None, None)
        assert strand.rows_dimension_name == "v7"
        assert strand.rows_dimension_type == DT.CAT
        np.testing.assert_equal(strand.rows_margin, [10, 5])
        assert strand.scale_mean - 1.66667 < 0.0001
        assert strand.shape == (2,)
        assert strand.table_base == 15
        assert strand.table_base_unpruned == 15
        assert strand.table_margin == 15
        assert strand.table_margin_unpruned == 15
        assert strand.table_name == "v7: C"
        assert pytest.approx(strand.table_percentages) == (66.66667, 33.33333)
        assert pytest.approx(strand.table_proportions) == (0.666667, 0.333333)
        assert strand.title == "Registered Voters"
        assert strand.unweighted_bases == (15, 15)
        assert strand.unweighted_counts == (10, 5)
        assert pytest.approx(strand.var_scale_mean) == 0.8888888
        assert strand.variable_name == "v7"

    def it_provides_values_for_cat_with_means_and_insertions(self):
        strand = Cube(CR.CAT_WITH_MEANS_AND_INSERTIONS).partitions[0]

        assert strand.is_empty is False
        np.testing.assert_almost_equal(
            strand.means, [19.85555556, 13.85416667, 52.78947368, np.nan, np.nan]
        )
        assert strand.title == "Untitled"
        assert strand.unweighted_counts == (409, 113, 139, 409, 252)

    def it_provides_std_dev_err_univ_mr_with_hs(self):
        strand = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]

        np.testing.assert_almost_equal(
            strand.standard_deviation,
            [
                0.46426724,
                0.3584419,
                0.2351762,
                0.32431855,
                0.2891897,
                0.24800318,
                0.15104855,
                0.49700725,
                0.14466968,
            ],
        )
        np.testing.assert_almost_equal(
            strand.standard_error,
            [
                0.0025417,
                0.0019624,
                0.0012875,
                0.0017755,
                0.0015832,
                0.0013577,
                0.0008269,
                0.002721,
                0.000792,
            ],
        )

    def it_places_insertions_on_a_reordered_dimension_in_the_right_position(self):
        """Subtotal anchors follow re-ordered rows.

        The key fixture characteristic is that an ordering transform is combined with
        subtotal insertions such that their subtotal position is changed by the
        ordering.
        """
        transforms = {
            "rows_dimension": {
                "insertions": [
                    {
                        "anchor": "top",
                        "args": [1, 2],
                        "function": "subtotal",
                        "name": "Sum A-C",
                    },
                    {
                        "anchor": 4,
                        "args": [1, 2],
                        "function": "subtotal",
                        "name": "Total A-C",
                    },
                    {
                        "anchor": "bottom",
                        "args": [4, 5],
                        "function": "subtotal",
                        "name": "Total D-E",
                    },
                ],
                "order": {"element_ids": [2, 4, 5, 1], "type": "explicit"},
            }
        }
        strand = Cube(CR.CAT_SUBTOT_ORDER, transforms=transforms).partitions[0]

        assert strand.row_labels == (
            "Sum A-C",
            "C1 & C2",
            "D",
            "Total A-C",
            "E",
            "AB",
            "Total D-E",
        )
        assert strand.counts == (31506, 16275, 3480, 31506, 4262, 15231, 7742)

    def it_knows_when_it_is_empty(self):
        strand = Cube(CR.OM_SGP8334215_VN_2019_SEP_19_STRAND).partitions[0]
        assert strand.is_empty is True


class Describe_Nub(object):
    """Integration-test suite for `cr.cube.cubepart._Nub` object."""

    def it_is_not_empty(self):
        cube = Cube(CR.ECON_MEAN_NO_DIMS)
        nub = cube.partitions[0]
        assert nub.is_empty is False

    def it_is_empty(self):
        cube = Cube(CR.ECON_NODATA_NO_DIMS)
        nub = cube.partitions[0]
        assert nub.is_empty is True
