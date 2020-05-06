# encoding: utf-8

import numpy as np
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.cube import Cube
from cr.cube.enum import DIMENSION_TYPE as DT

# ---mnemonic: CR = 'cube-response'---
# ---mnemonic: TR = 'transforms'---
from ..fixtures import CR, TR


class Describe_Slice(object):
    """Integration-test suite for _Slice object."""

    def it_provides_values_for_cat_x_cat(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]

        assert slice_.column_labels == ("C", "E")
        assert slice_.columns_dimension_name == "v7"
        assert slice_.columns_dimension_type == DT.CAT
        assert slice_.cube_is_mr_by_itself is False
        assert slice_.description == "Pet Owners"
        np.testing.assert_almost_equal(
            slice_.row_proportions, np.array([[0.71428571, 0.28571429], [0.625, 0.375]])
        )
        assert slice_.inserted_column_idxs == ()
        assert slice_.inserted_row_idxs == ()
        assert slice_.insertions == []
        assert slice_.is_empty is False
        assert slice_.name == "v4"
        np.testing.assert_array_almost_equal(
            slice_.residual_test_stats,
            [
                [[0.71439304, 0.71439304], [0.71439304, 0.71439304]],
                [[0.36596253, -0.36596253], [-0.36596253, 0.36596253]],
            ],
        )
        assert slice_.row_labels == ("B", "C")
        assert slice_.rows_dimension_description == "Pet Owners"
        assert slice_.rows_dimension_fills == (None, None)
        assert slice_.rows_dimension_name == "v4"
        assert slice_.rows_dimension_type == DT.CAT
        assert slice_.shape == (2, 2)
        assert slice_.table_name is None
        assert slice_.variable_name == "v7"

    def it_provides_values_for_cat_x_cat_pruning_hs(self):
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]

        np.testing.assert_array_equal(
            slice_.base_counts,
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
                [119.51424328, np.nan, 93.79691922, 81.24002902, np.nan, 50.17378721],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [129.57286502, np.nan, 0.0, 234.24140666, np.nan, 0.0],
                [68.74132753, np.nan, 99.37070479, 89.71345927, np.nan, 554.5688323],
                [48.0063805, np.nan, 137.75263952, 149.33201417, np.nan, 0.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [83.86593205, np.nan, 91.83509301, 160.9906575, np.nan, 0.0],
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
        assert slice_.columns_dimension_name == "ShutdownBlame"
        assert slice_.columns_dimension_type == DT.CAT
        np.testing.assert_almost_equal(
            slice_.columns_std_dev,
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
            slice_.columns_std_err,
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
        assert slice_.dimension_types == (DT.CAT, DT.CAT)
        assert slice_.inserted_column_idxs == (1,)
        assert slice_.inserted_row_idxs == (1,)
        assert slice_.is_empty is False
        assert slice_.name == "MaritalStatus"
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
        assert slice_.rows_dimension_description == "What is your marital status?"
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
            slice_.zscore,
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
        slice_ = Cube(CR.CAT_X_CAT_4X4, transforms=transforms).partitions[0]

        print("slice_.row_labels == %s" % (slice_.row_labels,))
        assert slice_.row_labels == (
            "Apple+Banana",
            "Banana",
            "Date",
            "Apple+Date",
            "Cherry",
            "Apple",
            "Cherry+Date",
        )
        assert slice_.column_labels == (
            "Asparagus+Broccoli",
            "Broccoli",
            "Daikon",
            "Asparagus+Daikon",
            "Cauliflower",
            "Asparagus",
            "Cauliflower+Daikon",
        )
        print("slice_.counts == \n%s" % (slice_.counts,))
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
            slice_.column_base, slice_wo_explicit_order_.column_base
        )
        np.testing.assert_almost_equal(
            slice_.row_base, slice_wo_explicit_order_.row_base
        )

    def it_respect_proportions_with_ordering_transform_cat_x_cat(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]

        np.testing.assert_almost_equal(
            slice_.column_proportions, [[0.5, 0.4], [0.5, 0.6]]
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions, [[0.71428571, 0.28571429], [0.625, 0.375]]
        )

        transforms = {
            "columns_dimension": {"order": {"element_ids": [3, 1], "type": "explicit"}},
            "rows_dimension": {"order": {"element_ids": [1, 2], "type": "explicit"}},
        }
        slice_with_ordering_ = Cube(CR.CAT_X_CAT, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_with_ordering_.column_proportions, [[0.4, 0.5], [0.6, 0.5]]
        )
        np.testing.assert_almost_equal(
            slice_with_ordering_.row_proportions,
            [[0.28571429, 0.71428571], [0.375, 0.625]],
        )
        np.testing.assert_almost_equal(slice_.row_base, slice_with_ordering_.row_base)
        np.testing.assert_almost_equal(
            np.flip(slice_.column_base), slice_with_ordering_.column_base
        )

    def it_respect_proportions_with_ordering_transform_cat_x_mr(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]

        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [[0.3, 0.35294118, 0.31578947], [0.7, 0.64705882, 0.6842105]],
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [[0.4285714, 0.48, 0.5217391], [0.5384615, 0.4074074, 0.5531915]],
        )

        transforms = {
            "columns_dimension": {
                "order": {"element_ids": [3, 2, 1], "type": "explicit"}
            },
            "rows_dimension": {"order": {"element_ids": [2, 1], "type": "explicit"}},
        }
        slice_with_ordering_ = Cube(CR.CAT_X_MR, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_with_ordering_.column_proportions,
            [[0.6842105, 0.64705882, 0.7], [0.31578947, 0.35294118, 0.3]],
        )
        np.testing.assert_almost_equal(
            slice_with_ordering_.row_proportions,
            [[0.5531915, 0.4074074, 0.5384615], [0.5217391, 0.48, 0.4285714]],
        )
        np.testing.assert_almost_equal(
            np.flip(np.flip(slice_.row_base, 0), 1), slice_with_ordering_.row_base
        )
        np.testing.assert_almost_equal(
            np.flip(slice_.column_base), slice_with_ordering_.column_base
        )

    def it_respect_row_proportions_with_ordering_transform_mr_x_cat(self):
        slice_ = Cube(
            CR.MR_X_CAT_HS,
            transforms=TR.GENERIC_TRANSFORMS_DICTS["both_order_mr_x_cat"],
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.25883849, 0.06423004, 0.39673409, 0.0, 0.74116151],
                [0.63354565, 0.44079255, 0.2344488, 0.0, 0.36645435],
            ],
        )
        np.testing.assert_equal(
            slice_.column_base, [[101, 32, 208, 0, 375], [39, 15, 69, 0, 126]]
        )
        np.testing.assert_equal(slice_.row_base, [385, 26])

        # --- rows flip after removing the rows order ---
        slice_ = Cube(
            CR.MR_X_CAT_HS,
            transforms=TR.GENERIC_TRANSFORMS_DICTS["no_row_order_mr_x_cat"],
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.63354565, 0.44079255, 0.2344488, 0.0, 0.36645435],
                [0.25883849, 0.06423004, 0.39673409, 0.0, 0.74116151],
            ],
        )
        np.testing.assert_equal(
            slice_.column_base, [[39, 15, 69, 0, 126], [101, 32, 208, 0, 375]]
        )
        np.testing.assert_equal(slice_.row_base, [26, 385])

    def it_respect_col_proportions_with_ordering_transform_mr_x_cat(self):
        slice_ = Cube(
            CR.MR_X_CAT_HS,
            transforms=TR.GENERIC_TRANSFORMS_DICTS["both_order_mr_x_cat"],
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.81825272, 0.78206694, 0.79964474, np.nan, 0.79162243],
                [0.36700322, 0.63991606, 0.11791067, np.nan, 0.09519879],
            ],
        )
        np.testing.assert_equal(
            slice_.column_base, [[101, 32, 208, 0, 375], [39, 15, 69, 0, 126]]
        )
        np.testing.assert_equal(slice_.row_base, [385, 26])

        # --- remove the columns order ---
        slice_ = Cube(
            CR.MR_X_CAT_HS,
            transforms=TR.GENERIC_TRANSFORMS_DICTS["no_col_order_mr_x_cat"],
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [0.78206694, 0.81825272, 0.79964474, np.nan, 0.79162243],
                [0.63991606, 0.36700322, 0.11791067, np.nan, 0.09519879],
            ],
        )
        np.testing.assert_equal(
            slice_.column_base, [[32, 101, 208, 0, 375], [15, 39, 69, 0, 126]]
        )
        np.testing.assert_equal(slice_.row_base, [385, 26])

    def it_respect_proportions_with_ordering_transform_ca_x_cat(self):
        transforms = TR.GENERIC_TRANSFORMS_DICTS["both_order_ca_x_cat"]
        slice_ = Cube(CR.CA_X_CAT_HS, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [[0.33333333, 0.33333333, 0.0, 0.0, 0.33333333, 0.33333333, 0.66666667]],
        )
        np.testing.assert_almost_equal(
            slice_.column_proportions, [[1.0, 0.33333333, 0.0, 0.0, 1.0, 1.0, 1.0]]
        )

        # remove the order
        transforms_wo_ordering = TR.GENERIC_TRANSFORMS_DICTS["no_order_ca_x_cat"]
        slice_wo_ordering_ = Cube(
            CR.CA_X_CAT_HS, transforms=transforms_wo_ordering
        ).partitions[0]

        np.testing.assert_almost_equal(
            slice_wo_ordering_.row_proportions,
            [[0.0, 0.0, 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.66666667]],
        )
        np.testing.assert_almost_equal(
            slice_wo_ordering_.column_proportions,
            [[0.0, 0.0, 1.0, 0.33333333, 1.0, 1.0, 1.0]],
        )

    def it_respect_proportions_with_ordering_transform_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]

        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [1.0, 0.28566937, 0.43456698, 1.0],
                [0.13302403, 1.0, 0.34959546, 1.0],
                [0.12391245, 0.23498805, 1.0, 1.0],
                [0.22804396, 0.47751837, 0.72838875, 1.0],
            ],
        )

        transforms = {
            "columns_dimension": {
                "order": {"element_ids": [2, 1, 3, 0], "type": "explicit"}
            },
            "rows_dimension": {
                "order": {"element_ids": [1, 2, 3, 0], "type": "explicit"}
            },
        }
        slice_with_row_ordering_ = Cube(CR.MR_X_MR, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_with_row_ordering_.row_proportions,
            [
                [0.28566937, 1.0, 0.43456698, 1.0],
                [1.0, 0.13302403, 0.34959546, 1.0],
                [0.23498805, 0.12391245, 1.0, 1.0],
                [0.47751837, 0.22804396, 0.72838875, 1.0],
            ],
        )
        # assert that the first column is flipped
        np.testing.assert_almost_equal(
            slice_.row_proportions[:, 0], slice_with_row_ordering_.row_proportions[:, 1]
        )
        np.testing.assert_almost_equal(
            slice_.column_base,
            [[12, 18, 26, 44], [7, 29, 20, 45], [10, 22, 34, 53], [12, 29, 34, 61]],
        )
        np.testing.assert_almost_equal(
            slice_with_row_ordering_.column_base,
            [[18, 12, 26, 44], [29, 7, 20, 45], [22, 10, 34, 53], [29, 12, 34, 61]],
        )
        np.testing.assert_almost_equal(
            slice_.row_base,
            [[12, 7, 10, 12], [18, 29, 22, 29], [26, 20, 34, 34], [44, 45, 53, 61]],
        )
        np.testing.assert_almost_equal(
            slice_with_row_ordering_.row_base,
            [[7, 12, 10, 12], [29, 18, 22, 29], [20, 26, 34, 34], [45, 44, 53, 61]],
        )

    def it_respect_proportions_with_ordering_transform_ca_x_mr(self):
        slice_ = Cube(CR.CA_X_MR).partitions[0]

        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0.56722442, 0.53361631, 0.98561362, np.nan],
                [0.46968779, 0.52205975, 0.96608066, np.nan],
                [0.41712503, 0.45425752, 0.96050869, np.nan],
                [0.51077015, 0.50379377, 0.98869263, np.nan],
                [0.1512758, 0.24490666, 0.9917377, np.nan],
            ],
        )

        transforms = {
            "rows_dimension": {
                "order": {"element_ids": [0, 1, 2, 5, 4], "type": "explicit"}
            }
        }
        slice_with_row_ordering_ = Cube(CR.CA_X_MR, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_with_row_ordering_.row_proportions,
            [
                [0.56722442, 0.53361631, 0.98561362, np.nan],
                [0.46968779, 0.52205975, 0.96608066, np.nan],
                [0.1512758, 0.24490666, 0.9917377, np.nan],
                [0.51077015, 0.50379377, 0.98869263, np.nan],
                [0.41712503, 0.45425752, 0.96050869, np.nan],
            ],
        )
        # assert that columns are flipped
        np.testing.assert_almost_equal(
            slice_.row_proportions[4, :], slice_with_row_ordering_.row_proportions[2, :]
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions[2, :],
            slice_with_row_ordering_.row_proportions[-1, :],
        )
        np.testing.assert_almost_equal(
            slice_with_row_ordering_.row_base[4, :], slice_.row_base[2, :]
        )
        np.testing.assert_almost_equal(
            slice_with_row_ordering_.row_base[2, :], slice_.row_base[-1, :]
        )

    @pytest.mark.parametrize(
        "fixture, table_name, expected",
        [(CR.EDU_FAV5_FAV5, "Education", True), (CR.AGE_FAVMR, None, False)],
    )
    def it_knows_when_cube_is_mr_by_itself(self, fixture, table_name, expected):
        cube = Cube(fixture)
        slice_ = cube.partitions[0]

        assert slice_.table_name == table_name
        assert slice_.cube_is_mr_by_itself is expected

    def it_knows_when_it_is_empty(self):
        assert Cube(CR.OM_SGP8334215_VN_2019_SEP_19).partitions[0].is_empty is True

    def it_provides_nan_margin_when_has_weighted_mean_without_weighted_counts(self):
        slice_ = Cube(CR.AGE_AGE_GENDER).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.counts,
            np.array(
                [[24.43935757, 37.32122746, 61.76058503, 55.48571956, 73.02427659]]
            ),
        )
        np.testing.assert_array_almost_equal(
            slice_.means,
            np.array([[24.43935757, 37.32122746, np.nan, 55.48571956, 73.02427659]]),
        )
        np.testing.assert_array_almost_equal(slice_.rows_margin, np.array([np.nan]))
        np.testing.assert_array_almost_equal(
            slice_.columns_margin, np.array([np.nan] * len(slice_.counts[0, :]))
        )

    def it_calculates_various_measures(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }

        # Without insertions
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        expected = [
            [0.47502103, 0.34045416, 0.16820858, 0.0, 0.01631623],
            [0.515, 0.0, 0.485, 0.0, 0.0],
            [0.27321912, 0.3606853, 0.18575293, 0.0, 0.18034265],
            [0.19080605, 0.5, 0.30919395, 0.0, 0.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.33333333, 0.33333333, 0.33333333, 0.0, 0.0],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_provides_unpruned_table_margin(self):
        slice_ = _Slice(Cube(CR.MR_X_CAT_HS), 0, None, None, 0)
        np.testing.assert_array_equal(
            slice_.table_base_unpruned, [165, 210, 242, 450, 476]
        )
        np.testing.assert_almost_equal(
            slice_.table_margin_unpruned,
            [176.3655518, 211.4205877, 247.7407379, 457.0509557, 471.9317685],
        )

    def it_reorders_various_measures_on_mr_x_cat(self):
        transforms = {
            "rows_dimension": {
                "order": {"type": "explicit", "element_ids": [5, 1, 4, 2, 3]}
            }
        }
        slice_ = Cube(CR.MR_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.base_counts,
            [
                [27, 58, 85, 0, 134, 166, 0, 300],
                [8, 7, 15, 0, 6, 5, 0, 11],
                [13, 36, 49, 0, 130, 190, 0, 320],
                [7, 16, 23, 0, 26, 27, 0, 53],
                [4, 21, 25, 0, 39, 54, 0, 93],
            ],
        )
        np.testing.assert_almost_equal(slice_.row_base, [385, 26, 369, 76, 118])
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            [376.76564059, 31.63152104, 366.88839144, 70.73073413, 125.75911351],
        )
        np.testing.assert_almost_equal(slice_.table_base, [476, 165, 450, 210, 242])
        np.testing.assert_almost_equal(
            slice_.table_margin,
            [471.93176847, 176.36555176, 457.05095566, 211.42058767, 247.74073787],
        )

    def it_calculates_mr_x_cat_row_proportions(self):
        slice_ = Cube(CR.MR_X_CAT_HS).partitions[0]
        expected = [
            [
                0.44079255,
                0.1927531,
                0.63354565,
                0.0,
                0.13200555,
                0.2344488,
                0.0,
                0.36645435,
            ],
            [
                0.12706997,
                0.17758354,
                0.30465351,
                0.0,
                0.35154979,
                0.3437967,
                0.0,
                0.69534649,
            ],
            [
                0.02245085,
                0.15543673,
                0.17788758,
                0.0,
                0.40588131,
                0.41623111,
                0.0,
                0.82211242,
            ],
            [
                0.03842827,
                0.11799739,
                0.15642566,
                0.0,
                0.35971868,
                0.48385566,
                0.0,
                0.84357434,
            ],
            [
                0.06423004,
                0.19460845,
                0.25883849,
                0.0,
                0.34442742,
                0.39673409,
                0.0,
                0.74116151,
            ],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_mr_x_cat_column_proportions(self):
        slice_ = Cube(CR.MR_X_CAT_HS).partitions[0]
        expected = [
            [
                0.63991606,
                0.18579712,
                0.36700322,
                np.nan,
                0.0709326,
                0.11791067,
                np.nan,
                0.09519879,
            ],
            [
                0.57106291,
                0.30796582,
                0.38122252,
                np.nan,
                0.32298183,
                0.31211929,
                np.nan,
                0.31751821,
            ],
            [
                0.23101896,
                0.47698573,
                0.42048358,
                np.nan,
                0.55509399,
                0.51026605,
                np.nan,
                0.53145537,
            ],
            [
                0.6728815,
                0.6856928,
                0.68250053,
                np.nan,
                0.79661367,
                0.85638988,
                np.nan,
                0.82983691,
            ],
            [
                0.78206694,
                0.83094212,
                0.81825272,
                np.nan,
                0.78257903,
                0.79964474,
                np.nan,
                0.79162243,
            ],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_calculates_cat_x_mr_row_proportions(self):
        slice_ = Cube(CR.CAT_X_MR_HS).partitions[0]
        expected = [
            [0.63991606, 0.57106291, 0.23101896, 0.6728815, 0.78206694],
            [0.18579712, 0.30796582, 0.47698573, 0.6856928, 0.83094212],
            [0.36700322, 0.38122252, 0.42048358, 0.68250053, 0.81825272],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.0709326, 0.32298183, 0.55509399, 0.79661367, 0.78257903],
            [0.11791067, 0.31211929, 0.51026605, 0.85638988, 0.79964474],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.09519879, 0.31751821, 0.53145537, 0.82983691, 0.79162243],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_calculates_cat_x_mr_column_proportions(self):
        slice_ = Cube(CR.CAT_X_MR_HS).partitions[0]
        expected = [
            [0.44079255, 0.12706997, 0.02245085, 0.03842827, 0.06423004],
            [0.1927531, 0.17758354, 0.15543673, 0.11799739, 0.19460845],
            [0.63354565, 0.30465351, 0.17788758, 0.15642566, 0.25883849],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.13200555, 0.35154979, 0.40588131, 0.35971868, 0.34442742],
            [0.2344488, 0.3437967, 0.41623111, 0.48385566, 0.39673409],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.36645435, 0.69534649, 0.82211242, 0.84357434, 0.74116151],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def it_calculates_mr_x_cat_various_measures(self):
        slice_ = Cube(CR.MR_X_CAT_HS).partitions[0]
        expected_zscore = [
            [
                5.98561407,
                0.10667037,
                np.nan,
                np.nan,
                -2.65642772,
                -1.58344064,
                np.nan,
                np.nan,
            ],
            [
                2.06703905,
                -0.40052278,
                np.nan,
                np.nan,
                -0.26977425,
                -0.52803412,
                np.nan,
                np.nan,
            ],
            [
                -1.98375582,
                -0.4294405,
                np.nan,
                np.nan,
                1.14820815,
                0.06992955,
                np.nan,
                np.nan,
            ],
            [
                -1.52909314,
                -2.51725298,
                np.nan,
                np.nan,
                -0.24776194,
                2.62630708,
                np.nan,
                np.nan,
            ],
            [
                -0.23349936,
                0.84630085,
                np.nan,
                np.nan,
                -0.62837448,
                0.05687326,
                np.nan,
                np.nan,
            ],
        ]
        expected_pvals = [
            [
                2.15574980e-09,
                9.15050488e-01,
                np.nan,
                np.nan,
                7.89733774e-03,
                1.13321065e-01,
                np.nan,
                np.nan,
            ],
            [
                3.87304749e-02,
                6.88771512e-01,
                np.nan,
                np.nan,
                7.87333931e-01,
                5.97475651e-01,
                np.nan,
                np.nan,
            ],
            [
                4.72830679e-02,
                6.67602683e-01,
                np.nan,
                np.nan,
                2.50882647e-01,
                9.44249733e-01,
                np.nan,
                np.nan,
            ],
            [
                1.26241357e-01,
                1.18273874e-02,
                np.nan,
                np.nan,
                8.04318606e-01,
                8.63169073e-03,
                np.nan,
                np.nan,
            ],
            [
                8.15373669e-01,
                3.97384931e-01,
                np.nan,
                np.nan,
                5.29758649e-01,
                9.54646158e-01,
                np.nan,
                np.nan,
            ],
        ]
        expected_table_std_dev = [
            [
                0.26982777,
                0.18268971,
                0.31735855,
                0.0,
                0.15203597,
                0.20070068,
                0.0,
                0.24779961,
            ],
            [
                0.20175242,
                0.2363915,
                0.30254544,
                0.0,
                0.32214688,
                0.31904263,
                0.0,
                0.42250711,
            ],
            [
                0.10614473,
                0.26958793,
                0.28661105,
                0.0,
                0.40445594,
                0.40822282,
                0.0,
                0.49311729,
            ],
            [
                0.17290444,
                0.29282782,
                0.33136132,
                0.0,
                0.45318467,
                0.48738756,
                0.0,
                0.46756128,
            ],
            [
                0.22056401,
                0.36225248,
                0.40489719,
                0.0,
                0.44650059,
                0.46520183,
                0.0,
                0.49151833,
            ],
        ]
        expected_table_std_err = [
            [
                0.02031794,
                0.01375648,
                0.023897,
                0.0,
                0.01144826,
                0.0151127,
                0.0,
                0.01865923,
            ],
            [
                0.01387539,
                0.01625767,
                0.02080736,
                0.0,
                0.02215544,
                0.02194194,
                0.0,
                0.02905764,
            ],
            [
                0.00674372,
                0.01712781,
                0.01820934,
                0.0,
                0.02569641,
                0.02593574,
                0.0,
                0.03132936,
            ],
            [
                0.00808768,
                0.01369714,
                0.01549956,
                0.0,
                0.0211979,
                0.02279776,
                0.0,
                0.02187038,
            ],
            [
                0.01015302,
                0.01667523,
                0.01863825,
                0.0,
                0.02055334,
                0.0214142,
                0.0,
                0.0226256,
            ],
        ]
        expected_col_std_dev = [
            [
                0.48002447,
                0.38894286,
                0.4819874,
                np.nan,
                0.25671222,
                0.32250231,
                np.nan,
                0.29348932,
            ],
            [
                0.4949243,
                0.46165233,
                0.48568705,
                np.nan,
                0.46761583,
                0.46335822,
                np.nan,
                0.4655109,
            ],
            [
                0.42148452,
                0.49947006,
                0.49363665,
                np.nan,
                0.49695538,
                0.4998946,
                np.nan,
                0.49900958,
            ],
            [
                0.46916094,
                0.46423936,
                0.46550355,
                np.nan,
                0.40251749,
                0.35069396,
                np.nan,
                0.37577601,
            ],
            [
                0.41284167,
                0.37480303,
                0.38563611,
                np.nan,
                0.41249133,
                0.4002662,
                np.nan,
                0.40614819,
            ],
        ]
        expected_col_std_err = [
            [
                0.1028366,
                0.06789606,
                0.06522613,
                np.nan,
                0.03345903,
                0.04066543,
                np.nan,
                0.02659733,
            ],
            [
                0.12475421,
                0.07228711,
                0.06460091,
                np.nan,
                0.0532943,
                0.05249552,
                np.nan,
                0.03740326,
            ],
            [
                0.12056446,
                0.07802173,
                0.06767673,
                np.nan,
                0.05182406,
                0.04935598,
                np.nan,
                0.03577725,
            ],
            [
                0.10249407,
                0.05842565,
                0.05076373,
                np.nan,
                0.03127232,
                0.02435786,
                np.nan,
                0.01945794,
            ],
            [
                0.07421655,
                0.03989992,
                0.03532412,
                np.nan,
                0.03203276,
                0.02927602,
                np.nan,
                0.02162477,
            ],
        ]
        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.columns_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.columns_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(slice_.zscore, expected_zscore)
        np.testing.assert_almost_equal(slice_.pvals, expected_pvals)

    def it_calculates_cat_x_mr_various_measures(self):
        slice_ = Cube(CR.CAT_X_MR_HS).partitions[0]
        expected_zscore = [
            [5.98561407, 2.06703905, -1.98375582, -1.52909314, -0.23349936],
            [0.10667037, -0.40052278, -0.4294405, -2.51725298, 0.84630085],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [-2.65642772, -0.26977425, 1.14820815, -0.24776194, -0.62837448],
            [-1.58344064, -0.52803412, 0.06992955, 2.62630708, 0.05687326],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
        expected_pvals = [
            [
                2.15574980e-09,
                3.87304749e-02,
                4.72830679e-02,
                1.26241357e-01,
                8.15373669e-01,
            ],
            [
                9.15050488e-01,
                6.88771512e-01,
                6.67602683e-01,
                1.18273874e-02,
                3.97384931e-01,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [
                7.89733774e-03,
                7.87333931e-01,
                2.50882647e-01,
                8.04318606e-01,
                5.29758649e-01,
            ],
            [
                1.13321065e-01,
                5.97475651e-01,
                9.44249733e-01,
                8.63169073e-03,
                9.54646158e-01,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
        expected_table_std_dev = [
            [0.26982777, 0.20175242, 0.10614473, 0.17290444, 0.22056401],
            [0.18268971, 0.2363915, 0.26958793, 0.29282782, 0.36225248],
            [0.31735855, 0.30254544, 0.28661105, 0.33136132, 0.40489719],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15203597, 0.32214688, 0.40445594, 0.45318467, 0.44650059],
            [0.20070068, 0.31904263, 0.40822282, 0.48738756, 0.46520183],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.24779961, 0.42250711, 0.49311729, 0.46756128, 0.49151833],
        ]
        expected_table_std_err = [
            [0.02031794, 0.01387539, 0.00674372, 0.00808768, 0.01015302],
            [0.01375648, 0.01625767, 0.01712781, 0.01369714, 0.01667523],
            [0.023897, 0.02080736, 0.01820934, 0.01549956, 0.01863825],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01144826, 0.02215544, 0.02569641, 0.0211979, 0.02055334],
            [0.0151127, 0.02194194, 0.02593574, 0.02279776, 0.0214142],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01865923, 0.02905764, 0.03132936, 0.02187038, 0.0226256],
        ]
        expected_col_std_dev = [
            [0.4964821, 0.33305134, 0.14814455, 0.19222783, 0.24516228],
            [0.39446083, 0.38216178, 0.36232051, 0.32260503, 0.39589898],
            [0.48183561, 0.46026052, 0.38241808, 0.36325841, 0.43799672],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3384968, 0.47745422, 0.49106178, 0.47991786, 0.4751812],
            [0.42365382, 0.47497424, 0.49293283, 0.49973929, 0.48921994],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.48183561, 0.46026052, 0.38241808, 0.36325841, 0.43799672],
        ]
        expected_col_std_err = [
            [0.08827619, 0.03960109, 0.0132104, 0.01003574, 0.01263043],
            [0.07013646, 0.04544051, 0.03230898, 0.01684241, 0.02039618],
            [0.08567199, 0.05472675, 0.03410112, 0.01896482, 0.02256499],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.06018587, 0.05677115, 0.04378914, 0.02505532, 0.02448069],
            [0.07532707, 0.05647627, 0.04395598, 0.02609015, 0.02520394],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.08567199, 0.05472675, 0.03410112, 0.01896482, 0.02256499],
        ]

        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.columns_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.columns_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(slice_.zscore, expected_zscore)
        np.testing.assert_almost_equal(slice_.pvals, expected_pvals)

    def it_calculates_mr_x_mr_row_proportions(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        expected = [
            [1.0, 0.28566937, 0.43456698, 1.0],
            [0.13302403, 1.0, 0.34959546, 1.0],
            [0.12391245, 0.23498805, 1.0, 1.0],
            [0.22804396, 0.47751837, 0.72838875, 1.0],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_knows_mr_x_mr_column_proportions(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_proportions,
            [
                [1.0, 0.13302403, 0.12391245, 0.22804396],
                [0.28566937, 1.0, 0.23498805, 0.47751837],
                [0.43456698, 0.34959546, 1.0, 0.72838875],
                [1.0, 1.0, 1.0, 1.0],
            ],
        )

    def it_reorders_cat_x_cat(self):
        transforms = {
            "rows_dimension": {
                "insertions": {},
                "order": {"type": "explicit", "element_ids": [6, 1, 2, 5, 4, 3]},
            },
            "columns_dimension": {
                "insertions": {},
                "order": {"type": "explicit", "element_ids": [5, 1, 2, 4, 3]},
            },
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [0, 1, 1, 0, 1],
                [1, 28, 20, 0, 10],
                [0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 3, 8, 0, 5],
                [2, 3, 4, 0, 2],
            ]
        )
        np.testing.assert_equal(slice_.base_counts, expected)

    def it_prunes_cat_x_cat_with_hs(self):
        # Pruned - without insertions
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
        expected = np.array(
            [[28, 20, 10, 1], [1, 0, 1, 0], [3, 4, 2, 2], [3, 8, 5, 0], [1, 1, 1, 0]]
        )
        np.testing.assert_equal(slice_.base_counts, expected)

        # Pruned (just rows) - with insertions
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
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
        np.testing.assert_equal(slice_.base_counts, expected)

        # Pruned (just columns) - with insertions
        transforms = {"columns_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
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
        np.testing.assert_equal(slice_.base_counts, expected)

        # Pruned (rows and columns) - with insertions
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
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
        np.testing.assert_equal(slice_.base_counts, expected)

        # Not pruned - with insertions
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
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
        np.testing.assert_equal(slice_.base_counts, expected)

    def it_accommodates_an_all_missing_element_rows_dimension(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_ALL_MISSING_ROW_ELEMENTS), 0, None, None, 0)
        row_proportions = slice_.row_proportions
        np.testing.assert_almost_equal(row_proportions, np.array([]))

    def it_knows_means_with_subtotals_on_cat_x_cat(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_MEAN_SUBTOT), 0, None, None, 0)

        means = slice_.means

        np.testing.assert_almost_equal(
            means, np.array([[38.3333333, np.nan, 65.0, 55.0, 34.0]])
        )

    def it_knows_its_insertions(self):
        slice_ = Cube(CR.CAT_X_CAT_HS_2ROWS_1COL).partitions[0]

        np.testing.assert_array_almost_equal(
            slice_.insertions,
            np.array(
                [
                    [
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 2.9842060e-03, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 1.2085762e-05, np.inf],
                        [
                            1.09954577e-02,
                            1.64231069e-01,
                            7.74991104e-04,
                            4.78920155e-01,
                            1.73194792e-12,
                            2.68565170e-11,
                            2.94880115e-03,
                        ],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 1.1900698e-02, np.inf],
                        [
                            2.87540141e-05,
                            2.72376900e-02,
                            4.27168678e-09,
                            7.46184742e-02,
                            0.0,
                            2.89875191e-09,
                            3.51260516e-01,
                        ],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0, np.inf],
                    ],
                    [
                        [np.inf, np.inf, np.inf, np.inf, np.inf, -2.96936015, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, -4.37603499, np.inf],
                        [
                            2.54284314e00,
                            1.39098139e00,
                            3.36157570e00,
                            -7.08040423e-01,
                            -7.05452463e00,
                            -6.66285184e00,
                            2.97302533e00,
                        ],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, -2.51507523, np.inf],
                        [
                            -4.18311635e00,
                            2.20809445e00,
                            5.87331384e00,
                            1.78280240e00,
                            -8.48620633e00,
                            -5.93723152e00,
                            9.32147088e-01,
                        ],
                        [np.inf, np.inf, np.inf, np.inf, np.inf, 9.70800153, np.inf],
                    ],
                ]
            ),
        )


class Describe_Strand(object):
    """Integration-test suite for `cr.cube.cubepart._Strand` object."""

    def it_provides_values_for_univariate_cat(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]

        assert strand.base_counts == (10, 5)
        assert strand.bases == (15, 15)
        assert strand.counts == (10, 5)
        assert strand.cube_index == 0
        assert strand.cube_is_mr_by_itself is False
        assert strand.dimension_types == (DT.CAT,)
        assert strand.has_means is False
        assert strand.inserted_row_idxs == ()
        assert strand.is_empty is False
        assert strand.means == (np.nan, np.nan)
        np.testing.assert_equal(strand.min_base_size_mask, [False, False])
        assert strand.name == "v7"
        assert strand.ndim == 1
        assert strand.population_counts == (0.0, 0.0)
        np.testing.assert_equal(strand.row_base, [10, 5])
        assert strand.row_count == 2
        assert strand.row_labels == ("C", "E")
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
        assert pytest.approx(strand.var_scale_mean) == 0.8888888
        assert strand.variable_name == "v7"

    def it_provides_values_for_cat_with_means_and_insertions(self):
        strand = Cube(CR.CAT_WITH_MEANS_AND_INSERTIONS).partitions[0]

        assert strand.is_empty is False
        np.testing.assert_almost_equal(
            strand.means, [19.85555556, 13.85416667, 52.78947368, np.nan, np.nan]
        )
        assert strand.title == "Untitled"

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
