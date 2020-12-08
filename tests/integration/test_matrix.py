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
            slice_.columns_base, np.array([504, 215, 224, 76, 8, 439])
        )

    def it_computes_mr_x_mr_columns_base(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_.columns_base,
            np.array(
                [[12, 18, 26, 44], [7, 29, 20, 45], [10, 22, 34, 53], [12, 29, 34, 61]]
            ),
        )

    def it_computes_cat_x_mr_columns_base(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]
        np.testing.assert_array_equal(slice_.columns_base, np.array([40, 34, 38]))

    def it_computes_weighted_counts_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        weighted_counts = slice_.counts

        assert np.array_equal(
            weighted_counts,
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

    def it_computes_cat_x_mr_weighted_counts(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts, np.array([[12, 12, 12], [28, 22, 26]])
        )

    def it_computes_mr_x_cat_weighted_counts(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [13.9429388, 6.0970738, 0.0, 4.1755362, 7.4159721, 0.0],
                [8.9877522, 12.5606144, 0.0, 24.8653747, 24.3169928, 0.0],
                [2.8233988, 19.5475854, 0.0, 51.0432736, 52.3448558, 0.0],
                [14.0988864, 43.2918709, 0.0, 131.9766084, 177.5210258, 0.0],
                [24.1996722, 73.3217774, 0.0, 129.7684193, 149.4757717, 0.0],
            ],
        )

    def it_computes_mr_x_mr_weighted_counts(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.counts,
            [
                [22.96727041, 3.79786399, 8.77385271, 22.96727041],
                [3.79786399, 45.77891654, 12.46883034, 45.77891654],
                [8.77385271, 12.46883034, 86.97282879, 86.97282879],
                [22.96727041, 45.77891654, 86.97282879, 130.67846872],
            ],
        )

    def it_computes_table_margin_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert slice_.table_margin == 877

    def it_computes_cat_x_mr_table_margin_with_explicit_ordering(self):
        transforms = {
            "columns_dimension": {
                "order": {"type": "explicit", "element_ids": [5, 1, 6, 4, 0, 2]}
            }
        }
        slice_ = Cube(CR.CAT_X_MR_2, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_.table_margin,
            np.array([471.9317685, 176.3655518, 457.0509557, 211.4205877, 247.7407379]),
        )

    def it_computes_mr_x_cat_table_margin_with_explicit_ordering(self):
        transforms = {
            "rows_dimension": {
                "order": {"type": "explicit", "element_ids": [5, 1, 6, 4, 0, 2]}
            }
        }
        slice_ = Cube(CR.MR_X_CAT, transforms=transforms).partitions[0]

        np.testing.assert_almost_equal(
            slice_.table_margin,
            np.array([471.9317685, 176.3655516, 457.0509557, 211.4205878, 247.740738]),
        )

    def it_computes_mr_x_mr_table_margin(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_margin,
            [
                [166.0021903, 107.5444392, 126.86878474, 166.0021903],
                [107.5444392, 141.86768069, 100.00460577, 141.86768069],
                [126.86878474, 100.00460577, 180.99361257, 180.99361257],
                [166.0021903, 141.86768069, 180.99361257, 236.5388192],
            ],
        )

    def it_knows_the_column_labels(self):
        transforms = {
            "columns_dimension": {
                "order": {"type": "explicit", "element_ids": [1, 3, 0, 2]}
            }
        }
        slice_ = Cube(CR.CAT_HS_X_CAT_HS, transforms=transforms).partitions[0]
        np.testing.assert_equal(
            slice_.column_labels, ["Bravo", "Delta", "Alpha", "Charlie", "Last 2"]
        )

    def it_computes_rows_base_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert np.array_equal(slice_.rows_base, [151, 353, 603, 52, 250, 123, 575])

    def it_computes_cat_x_mr_rows_base(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_base,
            np.array(
                [
                    [15, 15, 13, 20, 32],
                    [24, 34, 37, 50, 69],
                    [0, 0, 0, 0, 0],
                    [57, 75, 81, 159, 167],
                    [69, 86, 111, 221, 208],
                    [0, 0, 0, 0, 0],
                ]
            ),
        )

    def it_computes_mr_x_cat_rows_base(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_base, np.array([26, 76, 118, 369, 385])
        )

    def it_computes_mr_x_mr_rows_base(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_equal(
            slice_.rows_base,
            [
                [12, 7, 10, 12],
                [18, 29, 22, 29],
                [26, 20, 34, 34],
                [44, 45, 53, 61],
            ],
        )

    def it_computes_table_base_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert np.array_equal(slice_.table_base, 877)

    def it_computes_cat_x_mr_table_base(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_base, np.array([165, 210, 242, 450, 476])
        )

    def it_computes_mr_x_cat_table_base(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_base, np.array([165, 210, 242, 450, 476])
        )

    def it_computes_mr_x_mr_table_base(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_equal(
            slice_.table_base,
            [
                [68, 43, 51, 68],
                [43, 60, 42, 60],
                [51, 42, 72, 72],
                [68, 60, 72, 96],
            ],
        )

    def it_computes_columns_margin_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert np.array_equal(slice_.columns_margin, [549, 328, 276, 273, 328])

    def it_computes_means_cat_x_cat_columns_margin(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.columns_margin, np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

    def it_computes_cat_x_mr_columns_margin(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.columns_margin,
            np.array([31.631521, 70.7307341, 125.7591135, 366.8883914, 376.7656406]),
        )

    def it_computes_mr_x_cat_columns_margin(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.columns_margin,
            np.array(
                [
                    [21.7886996, 32.8157604, 0.0, 58.866254, 62.8948376, 0.0],
                    [15.7386377, 40.7857418, 0.0, 76.986916, 77.9092923, 0.0],
                    [12.2215027, 40.9814885, 0.0, 91.95429, 102.5834568, 0.0],
                    [20.9530004, 63.1359564, 0.0, 165.6720366, 207.2899623, 0.0],
                    [30.9432236, 88.2393316, 0.0, 165.8214891, 186.9277242, 0.0],
                ]
            ),
        )

    def it_computes_mr_x_mr_columns_margin(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.columns_margin,
            [
                [22.9672704, 28.5502092, 70.8068713, 100.7142240],
                [13.2946142, 45.7789165, 53.0615517, 95.8683881],
                [20.1898745, 35.6664538, 86.9728288, 119.4044105],
                [22.9672704, 45.7789165, 86.9728288, 130.6784687],
            ],
        )

    def it_computes_rows_margin_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert np.array_equal(slice_.rows_margin, [151, 353, 603, 52, 250, 123, 575])

    def it_computes_means_cat_x_cat_rows_margin(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

    def it_computes_cat_x_mr_rows_margin(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            np.array(
                [
                    [21.7886997, 15.7386377, 12.2215027, 20.9530003, 30.9432236],
                    [32.8157604, 40.7857418, 40.9814885, 63.1359564, 88.2393316],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [58.8662541, 76.9869160, 91.9542899, 165.6720366, 165.8214891],
                    [62.8948376, 77.9092922, 102.5834568, 207.2899623, 186.9277242],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def it_computes_mr_x_cat_rows_margin(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            np.array([31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406]),
        )

    def it_computes_mr_x_mr_rows_margin(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.rows_margin,
            [
                [22.9672704, 13.2946142, 20.1898745, 22.9672704],
                [28.5502092, 45.7789165, 35.6664538, 45.7789165],
                [70.8068713, 53.0615517, 86.9728288, 86.9728288],
                [100.7142240, 95.8683881, 119.4044105, 130.6784687],
            ],
        )

    def it_computes_column_index_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        np.testing.assert_almost_equal(
            slice_.column_index,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 143.9988976, np.nan, 84.6836779, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 61.9136961, np.nan, 198.3657368, np.nan],
                    [np.nan, 88.0000000, np.nan, 85.0989011, np.nan],
                    [np.nan, 49.3658537, np.nan, 126.3589744, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def it_computes_cat_x_mr_column_index(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_index,
            np.array(
                [
                    [658.3848782, 189.796643, 33.5334594, 57.3979578, 95.9364833],
                    [120.0355512, 110.5888219, 96.7970596, 73.4819881, 121.190955],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [37.2110379, 99.0983551, 114.4138644, 101.4010835, 97.0906317],
                    [56.1256863, 82.3029403, 99.6433204, 115.8322464, 94.9758447],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )

    def it_computes_mr_x_cat_column_index(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_index,
            np.array(
                [
                    [356.7933052, 103.5935382, np.nan, 39.5493673, 65.7425846, np.nan],
                    [170.6958892, 92.0537803, np.nan, 96.542201, 93.2952896, np.nan],
                    [45.5098681, 93.9644011, np.nan, 109.3514342, 100.5204994, np.nan],
                    [83.763346, 85.3581548, np.nan, 99.1660877, 106.6073023, np.nan],
                    [97.80107, 103.9131364, np.nan, 97.8651086, 99.999255, np.nan],
                ]
            ),
        )

    def it_computes_mr_x_mr_column_index(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.column_index,
            np.array(
                [
                    [722.7771838, 96.1467363, 89.5610897, 164.8249708],
                    [88.5281989, 309.8974187, 72.8221888, 147.9817106],
                    [90.4349646, 72.7520838, 208.1036286, 151.580341],
                    [181.0082575, 181.0082575, 181.0082575, 181.0082575],
                ]
            ),
        )
