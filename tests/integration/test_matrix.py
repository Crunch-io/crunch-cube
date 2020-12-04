# encoding: utf-8

"""Integration-test suite for `cr.cube.cubepart` module."""

import numpy as np

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR


class DescribeAssembler(object):
    """Integration-test suite for `cr.cube.matrix.Assembler`."""

    def it_computes_assembled_ucounts_for_cat_hs_x_cat_hs_hiddens(self):
        """Assembler inserts, hides, prunes, and places in payload order."""
        slice_ = _Slice(
            Cube(CR.CAT_HS_X_CAT_HS_EMPTIES),
            slice_idx=0,
            transforms={
                "rows_dimension": {"elements": {"2": {"hide": True}}, "prune": True},
                "columns_dimension": {"elements": {"2": {"hide": True}}, "prune": True},
            },
            population=None,
            mask_size=0,
        )

        unweighted_counts = slice_.unweighted_counts

        assert np.array_equal(
            unweighted_counts,
            [
                [118, 65, 53, 33, 33],
                [40, 32, 8, 12, 12],
                [168, 66, 102, 82, 82],
                [163, 93, 70, 190, 190],
                [331, 159, 172, 272, 272],
                [100, 49, 51, 23, 23],
                [341, 175, 166, 234, 234],
            ],
        )

    def it_computes_assembled_ucounts_for_ca_subvar_x_ca_cat_hiddens(self):
        """Assembler hides, prunes, and places in payload order.

        This fixture has no insertions, and exercises the "no-insertions" case which
        requires certain subtle special handling.
        """
        slice_ = _Slice(
            Cube(CR.CA_SUBVAR_X_CA_CAT_EMPTIES),
            slice_idx=0,
            transforms={
                "rows_dimension": {"elements": {"5": {"hide": True}}, "prune": True},
                "columns_dimension": {
                    "elements": {"99": {"hide": True}},
                    "prune": True,
                },
            },
            population=None,
            mask_size=0,
        )

        unweighted_counts = slice_.unweighted_counts

        assert np.array_equal(
            unweighted_counts,
            [
                [2734, 5887, 1017],
                [2810, 7000, 474],
                [347, 2577, 4467],
            ],
        )

    def it_computes_assembled_ucounts_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
        """Assembler inserts, hides, prunes, and places in explicit order."""
        slice_ = _Slice(
            Cube(CR.CAT_HS_X_CAT_HS_EMPTIES),
            slice_idx=0,
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
            population=None,
            mask_size=0,
        )

        unweighted_counts = slice_.unweighted_counts

        assert np.array_equal(
            unweighted_counts,
            [
                [118, 33, 53, 65, 33],
                [163, 190, 70, 93, 190],
                [331, 272, 172, 159, 272],
                [40, 12, 8, 32, 12],
                [168, 82, 102, 66, 82],
                [100, 23, 51, 49, 23],
                [341, 234, 166, 175, 234],
            ],
        )

    def it_computes_assembled_ucounts_for_cat_hs_x_mr_hiddens_explicit_order(self):
        """Assembler inserts, hides, prunes, and places in explicit order."""
        slice_ = _Slice(
            Cube(CR.CAT_HS_X_MR),
            slice_idx=0,
            transforms={
                "rows_dimension": {
                    "elements": {"5": {"hide": True}},
                    "prune": True,
                    "order": {"type": "explicit", "element_ids": [6, 1, 5, 2, 4, 0]},
                },
                "columns_dimension": {
                    "elements": {"1": {"hide": True}},
                    "order": {"type": "explicit", "element_ids": [5, 1, 4, 2, 3]},
                },
            },
            population=None,
            mask_size=0,
        )

        unweighted_counts = slice_.unweighted_counts

        assert np.array_equal(
            unweighted_counts,
            [
                [300, 320, 53, 93],
                [27, 13, 7, 4],
                [58, 36, 16, 21],
                [85, 49, 23, 25],
                [134, 130, 26, 39],
            ],
        )

    def it_computes_assembled_ucounts_for_mr_x_cat(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        assert np.array_equal(
            slice_.unweighted_counts,
            [
                [8, 7, 0, 6, 5, 0],
                [7, 16, 0, 26, 27, 0],
                [4, 21, 0, 39, 54, 0],
                [13, 36, 0, 130, 190, 0],
                [27, 58, 0, 134, 166, 0],
            ],
        )

    def it_computes_assembled_ucounts_for_mr_x_mr_slices(self):
        slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_.unweighted_counts, [[1159, 3597], [197, 604], [192, 582]]
        )

        slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[1]
        np.testing.assert_array_equal(
            slice_.unweighted_counts, [[159, 94], [1182, 625], [1142, 623]]
        )

    def it_computes_ca_x_mr_hs_columns_base(self):
        slice_ = Cube(CR.CA_X_MR_WEIGHTED_HS).partitions[0]
        np.testing.assert_array_equal(
            slice_._assembler.columns_base, np.array([504, 215, 224, 76, 8, 439])
        )

    def it_computes_mr_x_mr_columns_base(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_._assembler.columns_base,
            np.array(
                [[12, 18, 26, 44], [7, 29, 20, 45], [10, 22, 34, 53], [12, 29, 34, 61]]
            ),
        )

    def it_computes_cat_x_mr_columns_base(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_._assembler.columns_base, np.array([40, 34, 38])
        )
