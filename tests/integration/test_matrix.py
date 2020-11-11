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

    def it_computes_cat_x_mr_aug_columns_base(self):
        slice_ = Cube(CR.EDU_FAV5_FAV5).partitions[0]
        np.testing.assert_array_equal(
            slice_.columns_base, np.array([263, 399, 539, 377, 586])
        )

    def it_computes_cat_x_mr_weighted_counts(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts, np.array([[12, 12, 12], [28, 22, 26]])
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
            slice_.rows_base,
            np.array([26, 76, 118, 369, 385]),
        )

    def it_computes_cat_hs_x_cat_zscore_subtotals(self):
        slice_ = Cube(CR.CAT_HS_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.zscores,
            np.array(
                [
                    [0.7792989, -1.5492141, -0.8592543, 2.15286265],
                    [-0.1357743, -2.7653773, 1.0805490, 2.79462474],
                    [1.0505670, 2.0867802, -2.4387987, -1.45790913],
                    [1.9249375, 1.1177223, -1.9333085, -1.97529876],
                    [0.4413131, 1.4848043, -0.5800550, -2.03544795],
                    [-3.015444, 1.0346507, 2.7183302, -0.49397002],
                    [-2.8136792, 0.3939848, 3.3080922, -0.60893083],
                ]
            ),
        )

    def it_computes_cat_x_cat_hs_zscore_subtotal(self):
        slice_ = Cube(CR.CAT_X_CAT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.zscores,
            np.array(
                [
                    [
                        5.54308205,
                        -1.89272013,
                        3.23600222,
                        -2.01120343,
                        -1.91199323,
                        -3.23600222,
                    ],
                    [
                        -2.45919839,
                        3.7884298,
                        1.1294771,
                        2.78828863,
                        -3.2898189,
                        -1.1294771,
                    ],
                    [
                        -1.84079007,
                        -4.78785492,
                        -5.78644573,
                        -2.28235088,
                        7.83595828,
                        5.78644573,
                    ],
                    [
                        2.05944524,
                        2.79976025,
                        4.24957107,
                        0.78831429,
                        -5.087638,
                        -4.24957107,
                    ],
                ]
            ),
        )

    def it_computes_cat_hs_x_cat_hs_zscore_subtotals(self):
        slice_ = Cube(CR.CAT_HS_X_CAT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.zscores,
            np.array(
                [
                    [0.99449068, 2.61969845, -1.32622266, -2.25262377, -3.12587272],
                    [2.22083515, 6.51340682, -3.27919756, -5.37647385, -7.56121545],
                    [2.59929144, 7.47697854, -3.7679221, -6.21630971, -8.72156546],
                    [-0.59255056, 3.08506796, -3.07336981, 0.58605429, -2.2052394],
                    [0.34718151, -4.19917753, -7.39220009, 11.42371415, 3.38602009],
                    [-2.45091618, -6.33233748, 12.10302862, -3.56290643, 7.59522049],
                    [-2.04805767, -9.23811055, 5.71662907, 5.47872349, 9.80221982],
                ]
            ),
        )

    def it_computes_cat_x_cat_tbl_stderr(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            np.array([[0.12171612, 0.08777075], [0.12171612, 0.10327956]]),
        )

    def it_computes_cat_x_cat_hs_tbl_stderr(self):
        slice_ = Cube(CR.CAT_X_CAT_HS).partitions[0]
        expected = np.array(
            [
                [0.00608635, 0.00308435, 0.00677378, 0.0018939, 0.00376918, 0.00420705],
                [0.00857053, 0.01042568, 0.01282409, 0.0082005, 0.00946240, 0.01199403],
                [0.01001019, 0.00880555, 0.01267413, 0.0073131, 0.01343507, 0.01450913],
                [0.00745845, 0.00760033, 0.01033568, 0.0054010, 0.00518624, 0.00738622],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_std_err, expected)

    def it_computes_cat_hs_mt_x_cat_hs_mt_tbl_stderr(self):
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
        expected = np.array(
            [
                [0.04833892, 0.05219646, 0.0434404, 0.03265951, 0.0, 0.01072157],
                [0.02798766, 0.04247886, 0.03525895, 0.02946806, 0.0, 0.01530728],
                [0.01104452, 0.01104452, 0.0, 0.01072157, 0.0, 0.0],
                [0.01873208, 0.02787891, 0.02140405, 0.01552997, 0.0, 0.01530728],
                [0.01873208, 0.03400224, 0.02946806, 0.02359023, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01072157, 0.01508098, 0.01072157, 0.01072157, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_std_err, expected)

    def it_computes_means_cat_hs_x_cat_hs(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]
        expected = np.array(
            [
                [41.96875, 30.875, 25.66666667, np.nan, 42.0],
                [51.51515152, 47.95555556, 45.44444444, np.nan, 45.0952381],
                [46.17088608, 44.55504587, 48.09090909, np.nan, 50.8],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [44.03030303, 45.21568627, 54.53333333, np.nan, 56.19512195],
                [45.64516129, 47.41428571, 46.89361702, np.nan, 55.27894737],
                [34.20408163, 43.2745098, 41.2, np.nan, 35.26086957],
            ]
        )
        np.testing.assert_almost_equal(slice_.means, expected)

    def it_computes_means_mr_x_cat(self):
        slice_ = Cube(CR.MEANS_MR_X_CAT).partitions[0]
        expected = np.array(
            [
                [
                    38.79868092,
                    37.91146097,
                    21.56682623,
                    28.90316683,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    12.36141735,
                    10.91788449,
                    8.55836344,
                    -9.23336151,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    25.35566536,
                    -1.87323918,
                    -10.45832265,
                    -19.00932593,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    -1.22773321,
                    -7.99671664,
                    -30.95431483,
                    -18.03417097,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    -23.80382413,
                    -26.69728288,
                    -61.23218388,
                    -48.49820981,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    19.6045351,
                    -24.87663078,
                    -52.08108014,
                    7.63833075,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    -26.98268155,
                    -9.66231773,
                    -90.91475189,
                    -46.92610738,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    19.45552783,
                    -27.48308453,
                    -62.33543385,
                    -39.83388919,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    20.59956268,
                    17.49911157,
                    6.29951372,
                    2.28572239,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            ]
        )
        np.testing.assert_almost_equal(slice_.means, expected)

    def it_computes_means_cat_x_mr(self):
        slice_ = Cube(CR.MEANS_CAT_X_MR).partitions[0]
        expected = np.array(
            [
                [29.0, 30.71428571, 48.0, 49.76923077, 37.2962963],
                [35.57142857, 45.0, 61.23809524, 48.0, 45.74137931],
                [32.0, 42.51785714, 59.42253521, 51.95217391, 41.57654723],
                [28.16666667, 50.46153846, 55.28205128, 54.73076923, 44.1641791],
                [32.6, 47.14814815, 62.38888889, 53.90526316, 45.42771084],
                [24.5, 39.09090909, 51.18181818, 44.19444444, 35.96039604],
            ]
        )
        np.testing.assert_almost_equal(slice_.means, expected)
