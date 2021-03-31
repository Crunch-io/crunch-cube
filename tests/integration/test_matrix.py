# encoding: utf-8

"""Integration-test suite for `cr.cube.cubepart` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice
from cr.cube.dimension import AllDimensions
from cr.cube.matrix.assembler import _BaseOrderHelper
from cr.cube.matrix.cubemeasure import _BaseCubeOverlaps

from ..fixtures import CR, OL


class DescribeAssembler(object):
    """Integration-test suite for `cr.cube.matrix.Assembler`."""

    def it_computes_column_proportions_for_cat_x_cat(self):
        slice_ = _Slice(Cube(CR.CAT_4_X_CAT_5), 0, None, None, 0)
        assert np.round(slice_._assembler.column_proportions, 2) == pytest.approx(
            np.array(
                [
                    [0.07, 0.12, 0.04, 0.00, 0.06],
                    [0.19, 0.32, 0.17, 0.17, 0.27],
                    [0.38, 0.10, 0.54, 0.17, 0.43],
                    [0.37, 0.46, 0.25, 0.67, 0.24],
                ]
            )
        )

    def it_computes_column_unweighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.column_unweighted_bases.tolist() == [
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
        ]

    def it_computes_column_unweighted_bases_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.column_unweighted_bases.tolist() == [
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
            [26, 76, 118, 369, 385],
        ]

    def it_computes_column_unweighted_bases_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.column_unweighted_bases.tolist() == [
            [15, 24, 39, 0, 57, 69, 0, 126],
            [15, 34, 49, 0, 75, 86, 0, 161],
            [13, 37, 50, 0, 81, 111, 0, 192],
            [20, 50, 70, 0, 159, 221, 0, 380],
            [32, 69, 101, 0, 167, 208, 0, 375],
        ]

    def it_computes_column_unweighted_bases_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.column_unweighted_bases.tolist() == [
            [12, 18, 26, 44],
            [7, 29, 20, 45],
            [10, 22, 34, 53],
            [12, 29, 34, 61],
        ]

    def it_computes_column_weighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.column_weighted_bases.tolist() == [
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
            [549, 328, 276, 273, 328],
        ]

    def it_computes_column_weighted_bases_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.column_weighted_bases == pytest.approx(
            np.array(
                [
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                    [31.6315209, 70.7307341, 125.7591136, 366.8883915, 376.7656406],
                ]
            )
        )

    def it_computes_column_weighted_bases_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.column_weighted_bases == pytest.approx(
            np.array(
                [
                    [
                        21.7886997,
                        32.8157604,
                        54.6044601,
                        0.0,
                        58.8662541,
                        62.8948376,
                        0.0,
                        121.7610917,
                    ],
                    [
                        15.7386377,
                        40.7857418,
                        56.5243795,
                        0.0,
                        76.986916,
                        77.9092922,
                        0.0,
                        154.8962082,
                    ],
                    [
                        12.2215027,
                        40.9814885,
                        53.2029912,
                        0.0,
                        91.9542899,
                        102.5834568,
                        0.0,
                        194.5377467,
                    ],
                    [
                        20.9530003,
                        63.1359564,
                        84.0889568,
                        0.0,
                        165.6720366,
                        207.2899623,
                        0.0,
                        372.9619989,
                    ],
                    [
                        30.9432236,
                        88.2393316,
                        119.1825552,
                        0.0,
                        165.8214891,
                        186.9277242,
                        0.0,
                        352.7492133,
                    ],
                ]
            )
        )

    def it_computes_column_weighted_bases_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.column_weighted_bases == pytest.approx(
            np.array(
                [
                    [22.9672704, 28.5502092, 70.8068713, 100.714224],
                    [13.2946142, 45.7789165, 53.0615517, 95.8683881],
                    [20.1898745, 35.6664538, 86.9728288, 119.4044105],
                    [22.9672704, 45.7789165, 86.9728288, 130.6784687],
                ]
            )
        )

    def it_computes_row_unweighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.row_unweighted_bases.tolist() == [
            [151, 151, 151, 151, 151],
            [353, 353, 353, 353, 353],
            [603, 603, 603, 603, 603],
            [52, 52, 52, 52, 52],
            [250, 250, 250, 250, 250],
            [123, 123, 123, 123, 123],
            [575, 575, 575, 575, 575],
        ]

    def it_computes_row_unweighted_bases_for_cat_hs_x_mr_for(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.row_unweighted_bases.tolist() == [
            [15, 15, 13, 20, 32],
            [24, 34, 37, 50, 69],
            [39, 49, 50, 70, 101],
            [0, 0, 0, 0, 0],
            [57, 75, 81, 159, 167],
            [69, 86, 111, 221, 208],
            [0, 0, 0, 0, 0],
            [126, 161, 192, 380, 375],
        ]

    def it_computes_row_unweighted_bases_for_mr_x_cat_hs_for(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.row_unweighted_bases.tolist() == [
            [26, 26, 26, 26, 26, 26, 26, 26],
            [76, 76, 76, 76, 76, 76, 76, 76],
            [118, 118, 118, 118, 118, 118, 118, 118],
            [369, 369, 369, 369, 369, 369, 369, 369],
            [385, 385, 385, 385, 385, 385, 385, 385],
        ]

    def it_computes_row_unweighted_bases_for_mr_x_mr_for(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.row_unweighted_bases.tolist() == [
            [12, 7, 10, 12],
            [18, 29, 22, 29],
            [26, 20, 34, 34],
            [44, 45, 53, 61],
        ]

    def it_computes_row_weighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.row_weighted_bases.tolist() == [
            [151, 151, 151, 151, 151],
            [353, 353, 353, 353, 353],
            [603, 603, 603, 603, 603],
            [52, 52, 52, 52, 52],
            [250, 250, 250, 250, 250],
            [123, 123, 123, 123, 123],
            [575, 575, 575, 575, 575],
        ]

    def it_computes_row_weighted_bases_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.row_weighted_bases == pytest.approx(
            np.array(
                [
                    [21.7886996, 15.7386377, 12.2215027, 20.9530004, 30.9432236],
                    [32.8157604, 40.7857418, 40.9814885, 63.1359564, 88.2393316],
                    [54.60446, 56.5243795, 53.2029912, 84.0889568, 119.1825552],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [58.866254, 76.986916, 91.95429, 165.6720366, 165.8214891],
                    [62.8948376, 77.9092923, 102.5834568, 207.2899623, 186.9277242],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [121.7610916, 154.8962083, 194.5377468, 372.9619989, 352.7492133],
                ]
            )
        )

    def it_computes_row_weighted_bases_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.row_weighted_bases == pytest.approx(
            np.array(
                [
                    [
                        31.631521,
                        31.631521,
                        31.631521,
                        31.631521,
                        31.631521,
                        31.631521,
                        31.631521,
                        31.631521,
                    ],
                    [
                        70.7307341,
                        70.7307341,
                        70.7307341,
                        70.7307341,
                        70.7307341,
                        70.7307341,
                        70.7307341,
                        70.7307341,
                    ],
                    [
                        125.7591135,
                        125.7591135,
                        125.7591135,
                        125.7591135,
                        125.7591135,
                        125.7591135,
                        125.7591135,
                        125.7591135,
                    ],
                    [
                        366.8883914,
                        366.8883914,
                        366.8883914,
                        366.8883914,
                        366.8883914,
                        366.8883914,
                        366.8883914,
                        366.8883914,
                    ],
                    [
                        376.7656406,
                        376.7656406,
                        376.7656406,
                        376.7656406,
                        376.7656406,
                        376.7656406,
                        376.7656406,
                        376.7656406,
                    ],
                ]
            )
        )

    def it_computes_row_weighted_bases_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.row_weighted_bases == pytest.approx(
            np.array(
                [
                    [22.9672704, 13.2946142, 20.1898745, 22.9672704],
                    [28.5502092, 45.7789165, 35.6664538, 45.7789165],
                    [70.8068713, 53.0615517, 86.9728288, 86.9728288],
                    [100.714224, 95.8683881, 119.4044105, 130.6784687],
                ]
            )
        )

    def it_computes_table_unweighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.table_unweighted_bases.tolist() == [
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
        ]

    def it_computes_table_unweighted_bases_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.table_unweighted_bases.tolist() == [
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
            [165, 210, 242, 450, 476],
        ]

    def it_computes_table_unweighted_bases_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.table_unweighted_bases.tolist() == [
            [165, 165, 165, 165, 165, 165, 165, 165],
            [210, 210, 210, 210, 210, 210, 210, 210],
            [242, 242, 242, 242, 242, 242, 242, 242],
            [450, 450, 450, 450, 450, 450, 450, 450],
            [476, 476, 476, 476, 476, 476, 476, 476],
        ]

    def it_computes_table_unweighted_bases_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.table_unweighted_bases.tolist() == [
            [68, 43, 51, 68],
            [43, 60, 42, 60],
            [51, 42, 72, 72],
            [68, 60, 72, 96],
        ]

    def it_computes_table_weighted_bases_for_cat_hs_x_cat_hs(self):
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

        assert slice_.table_weighted_bases.tolist() == [
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
            [877, 877, 877, 877, 877],
        ]

    def it_computes_table_weighted_bases_for_cat_hs_x_mr(self):
        slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
        assert slice_.table_weighted_bases == pytest.approx(
            np.array(
                [
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                    [176.3655516, 211.4205878, 247.740738, 457.0509557, 471.9317685],
                ]
            )
        )

    def it_computes_table_weighted_bases_for_mr_x_cat_hs(self):
        slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
        assert slice_.table_weighted_bases == pytest.approx(
            np.array(
                [
                    [
                        176.3655518,
                        176.3655518,
                        176.3655518,
                        176.3655518,
                        176.3655518,
                        176.3655518,
                        176.3655518,
                        176.3655518,
                    ],
                    [
                        211.4205877,
                        211.4205877,
                        211.4205877,
                        211.4205877,
                        211.4205877,
                        211.4205877,
                        211.4205877,
                        211.4205877,
                    ],
                    [
                        247.7407379,
                        247.7407379,
                        247.7407379,
                        247.7407379,
                        247.7407379,
                        247.7407379,
                        247.7407379,
                        247.7407379,
                    ],
                    [
                        457.0509557,
                        457.0509557,
                        457.0509557,
                        457.0509557,
                        457.0509557,
                        457.0509557,
                        457.0509557,
                        457.0509557,
                    ],
                    [
                        471.9317685,
                        471.9317685,
                        471.9317685,
                        471.9317685,
                        471.9317685,
                        471.9317685,
                        471.9317685,
                        471.9317685,
                    ],
                ]
            )
        )

    def it_computes_table_weighted_bases_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.table_weighted_bases == pytest.approx(
            np.array(
                [
                    [166.0021903, 107.5444392, 126.8687847, 166.0021903],
                    [107.5444392, 141.8676807, 100.0046058, 141.8676807],
                    [126.8687847, 100.0046058, 180.9936126, 180.9936126],
                    [166.0021903, 141.8676807, 180.9936126, 236.5388192],
                ]
            )
        )

    def it_computes_unweighted_counts_for_cat_hs_x_cat_hs_hiddens(self):
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

        assert slice_.unweighted_counts.tolist() == [
            [118, 65, 53, 33, 33],
            [40, 32, 8, 12, 12],
            [168, 66, 102, 82, 82],
            [163, 93, 70, 190, 190],
            [331, 159, 172, 272, 272],
            [100, 49, 51, 23, 23],
            [341, 175, 166, 234, 234],
        ]

    @pytest.mark.parametrize(
        "hidden_rows", ({"5": {"hide": True}}, {"00004": {"hide": True}})
    )
    def it_computes_unweighted_counts_for_ca_subvar_x_ca_cat_hiddens(self, hidden_rows):
        """Assembler hides, prunes, and places in payload order.

        This fixture has no insertions, and exercises the "no-insertions" case which
        requires certain subtle special handling.
        """
        slice_ = _Slice(
            Cube(CR.CA_SUBVAR_X_CA_CAT_EMPTIES),
            slice_idx=0,
            transforms={
                "rows_dimension": {"elements": hidden_rows, "prune": True},
                "columns_dimension": {
                    "elements": {"99": {"hide": True}},
                    "prune": True,
                },
            },
            population=None,
            mask_size=0,
        )

        assert slice_.unweighted_counts.tolist() == [
            [2734, 5887, 1017],
            [2810, 7000, 474],
            [347, 2577, 4467],
        ]

    def it_computes_unweighted_counts_for_cat_hs_x_cat_hs_hiddens_explicit_order(self):
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

        assert slice_.unweighted_counts.tolist() == [
            [118, 33, 53, 65, 33],
            [163, 190, 70, 93, 190],
            [331, 272, 172, 159, 272],
            [40, 12, 8, 32, 12],
            [168, 82, 102, 66, 82],
            [100, 23, 51, 49, 23],
            [341, 234, 166, 175, 234],
        ]

    def it_computes_unweighted_counts_for_cat_hs_x_mr_hiddens_explicit_order(self):
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

        assert slice_.unweighted_counts.tolist() == [
            [300, 320, 53, 93],
            [27, 13, 7, 4],
            [58, 36, 16, 21],
            [85, 49, 23, 25],
            [134, 130, 26, 39],
        ]

    def it_computes_unweighted_counts_for_mr_x_cat(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        assert slice_.unweighted_counts.tolist() == [
            [8, 7, 0, 6, 5, 0],
            [7, 16, 0, 26, 27, 0],
            [4, 21, 0, 39, 54, 0],
            [13, 36, 0, 130, 190, 0],
            [27, 58, 0, 134, 166, 0],
        ]

    def it_computes_unweighted_counts_for_mr_x_mr_slices(self):
        slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]
        assert slice_.unweighted_counts.tolist() == [
            [1159, 3597],
            [197, 604],
            [192, 582],
        ]

        slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[1]
        assert slice_.unweighted_counts.tolist() == [
            [159, 94],
            [1182, 625],
            [1142, 623],
        ]

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

        assert slice_.counts.tolist() == [
            [118, 33, 53, 65, 33],
            [163, 190, 70, 93, 190],
            [331, 272, 172, 159, 272],
            [40, 12, 8, 32, 12],
            [168, 82, 102, 66, 82],
            [100, 23, 51, 49, 23],
            [341, 234, 166, 175, 234],
        ]

    def it_computes_weighted_counts_for_cat_x_mr(self):
        slice_ = Cube(CR.CAT_X_MR).partitions[0]
        assert slice_.counts.tolist() == [
            [12, 12, 12],
            [28, 22, 26],
        ]

    def it_computes_weighted_counts_for_mr_x_cat(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        assert slice_.counts == pytest.approx(
            np.array(
                [
                    [13.9429388, 6.0970738, 0.0, 4.1755362, 7.4159721, 0.0],
                    [8.9877522, 12.5606144, 0.0, 24.8653747, 24.3169928, 0.0],
                    [2.8233988, 19.5475854, 0.0, 51.0432736, 52.3448558, 0.0],
                    [14.0988864, 43.2918709, 0.0, 131.9766084, 177.5210258, 0.0],
                    [24.1996722, 73.3217774, 0.0, 129.7684193, 149.4757717, 0.0],
                ]
            )
        )

    def it_computes_weighted_counts_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        assert slice_.counts == pytest.approx(
            np.array(
                [
                    [22.96727041, 3.79786399, 8.77385271, 22.96727041],
                    [3.79786399, 45.77891654, 12.46883034, 45.77891654],
                    [8.77385271, 12.46883034, 86.97282879, 86.97282879],
                    [22.96727041, 45.77891654, 86.97282879, 130.67846872],
                ]
            )
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

        assert slice_.columns_margin.tolist() == [431, 494, 294, 1219, 433]

    def it_computes_cat_x_mr_columns_margin(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]

        assert slice_.columns_margin == pytest.approx(
            [31.63152, 70.730734, 125.759113, 366.888391, 376.765640]
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

        assert slice_.rows_margin.tolist() == [55, 126, 613, 710, 310, 400, 148]

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

    def it_computes_cat_x_mr_zscores(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        assert slice_.zscores == pytest.approx(
            np.array(
                [
                    [5.9856141, 2.067039, -1.9837558, -1.5290931, -0.2334994],
                    [0.1066704, -0.4005228, -0.4294405, -2.517253, 0.8463009],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [-2.6564277, -0.2697743, 1.1482081, -0.2477619, -0.6283745],
                    [-1.5834406, -0.5280341, 0.0699296, 2.6263071, 0.0568733],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
        )

    def it_computes_mr_x_cat_zscores(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.zscores,
            np.array(
                [
                    [5.9856141, 0.1066704, np.nan, -2.6564277, -1.5834406, np.nan],
                    [2.067039, -0.4005228, np.nan, -0.2697743, -0.5280341, np.nan],
                    [-1.9837558, -0.4294405, np.nan, 1.1482081, 0.0699295, np.nan],
                    [-1.5290931, -2.517253, np.nan, -0.2477619, 2.6263071, np.nan],
                    [-0.2334994, 0.8463008, np.nan, -0.6283745, 0.0568733, np.nan],
                ]
            ),
        )

    def it_computes_mr_x_mr_zscores(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.zscores,
            np.array(
                [
                    [12.8841837, 0.1781302, -1.2190176, 4.1568249],
                    [0.1781302, 11.910822, -2.7003378, 5.6947682],
                    [-1.2190176, -2.7003378, 13.4533867, 9.2929498],
                    [4.1568249, 5.6947682, 9.2929498, 15.3798186],
                ]
            ),
        )

    def it_computes_means_cat_hs_x_cat_hs(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]
        np.testing.assert_almost_equal(
            slice_.means,
            np.array(
                [
                    [41.96875, 30.875, 25.66666667, np.nan, 42.0],
                    [51.51515152, 47.95555556, 45.44444444, np.nan, 45.0952381],
                    [46.17088608, 44.55504587, 48.09090909, np.nan, 50.8],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [44.03030303, 45.21568627, 54.53333333, np.nan, 56.19512195],
                    [45.64516129, 47.41428571, 46.89361702, np.nan, 55.27894737],
                    [34.20408163, 43.2745098, 41.2, np.nan, 35.26086957],
                ]
            ),
        )

    def it_computes_means_cat_x_mr(self):
        slice_ = Cube(CR.MEANS_CAT_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.means,
            np.array(
                [
                    [29.0, 30.71428571, 48.0, 49.76923077, 37.2962963],
                    [35.57142857, 45.0, 61.23809524, 48.0, 45.74137931],
                    [32.0, 42.51785714, 59.42253521, 51.95217391, 41.57654723],
                    [28.16666667, 50.46153846, 55.28205128, 54.73076923, 44.1641791],
                    [32.6, 47.14814815, 62.38888889, 53.90526316, 45.42771084],
                    [24.5, 39.09090909, 51.18181818, 44.19444444, 35.96039604],
                ]
            ),
        )

    def it_computes_means_mr_x_cat(self):
        slice_ = Cube(CR.MEANS_MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.means,
            np.array(
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
            ),
        )

    def it_computes_cat_hs_mt_x_cat_hs_mt_tbl_stderr(self):
        slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            np.array(
                [
                    [0.0483389, 0.0521965, 0.0434404, 0.0326595, 0.0, 0.0107216],
                    [0.0279877, 0.0424789, 0.0352589, 0.0294681, 0.0, 0.0153073],
                    [0.0110445, 0.0110445, 0.0, 0.0107216, 0.0, 0.0],
                    [0.0187321, 0.0278789, 0.0214041, 0.01553, 0.0, 0.0153073],
                    [0.0187321, 0.0340022, 0.0294681, 0.0235902, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0107216, 0.015081, 0.0107216, 0.0107216, 0.0, 0.0],
                ]
            ),
        )

    def it_computes_cat_x_mr_table_stderr(self):
        slice_ = Cube(CR.CAT_X_MR_2).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            np.array(
                [
                    [0.0203179, 0.0138754, 0.0067437, 0.0080877, 0.010153],
                    [0.0137565, 0.0162577, 0.0171278, 0.0136971, 0.0166752],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0114483, 0.0221554, 0.0256964, 0.0211979, 0.0205533],
                    [0.0151127, 0.0219419, 0.0259357, 0.0227978, 0.0214142],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def it_computes_mr_x_cat_table_stderr(self):
        slice_ = Cube(CR.MR_X_CAT).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            np.array(
                [
                    [0.0203179, 0.0137565, 0.0, 0.0114483, 0.0151127, 0.0],
                    [0.0138754, 0.0162577, 0.0, 0.0221554, 0.0219419, 0.0],
                    [0.0067437, 0.0171278, 0.0, 0.0256964, 0.0259357, 0.0],
                    [0.0080877, 0.0136971, 0.0, 0.0211979, 0.0227978, 0.0],
                    [0.010153, 0.0166752, 0.0, 0.0205533, 0.0214142, 0.0],
                ]
            ),
        )

    def it_computes_mr_x_mr_table_stderr(self):
        slice_ = Cube(CR.MR_X_MR).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_std_err,
            np.array(
                [
                    [0.0267982, 0.0177981, 0.0225257, 0.0267982],
                    [0.0177981, 0.0392504, 0.0330351, 0.0392504],
                    [0.0225257, 0.0330351, 0.0371372, 0.0371372],
                    [0.0267982, 0.0392504, 0.0371372, 0.0323307],
                ]
            ),
        )

    @pytest.mark.parametrize(
        "fixture, element_id, descending, expected_value",
        (
            (CR.CAT_4_X_CAT_5, 1, False, [0, 1, 3, 2]),
            (CR.CAT_X_MR_2, 1, True, [0, 4, 1, 3, 5, 2]),
            (CR.MR_X_CAT, 2, True, [4, 3, 2, 1, 0]),
            (CR.MR_X_MR, 3, True, [3, 2, 1, 0]),
        ),
    )
    def it_computes_the_sort_by_value_row_order_to_help(
        self, fixture, element_id, descending, expected_value
    ):
        transforms = {
            "rows_dimension": {
                "order": {
                    "type": "opposing_element",
                    "element_id": element_id,
                    "measure": "col_percent",
                    "direction": "descending" if descending else "ascending",
                }
            }
        }
        slice_ = _Slice(Cube(fixture), 0, transforms, None, 0)
        assembler = slice_._assembler

        assert assembler._row_order.tolist() == expected_value

    def it_computes_sum_cat_x_mr(self):
        slice_ = Cube(CR.SUM_CAT_X_MR).partitions[0]

        assert slice_.sums == pytest.approx(
            np.array([[3.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
        )

    def it_computes_sum_mr_x_cat(self):
        slice_ = Cube(CR.SUM_MR_X_CAT).partitions[0]

        assert slice_.sums == pytest.approx(
            np.array([[3.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        )


class DescribeCubemeasure(object):
    def it_provides_overlaps_for_cat_x_mr_sub_x_mr_sel(self):
        cube_dict = OL.CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=0,
        )

        assert overlaps_measure.__class__.__name__ == "_CatXMrOverlaps"
        assert overlaps_measure.overlaps.tolist() == [
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
            # --- Missing categories are not shown by cube by default
            # [
            #     # A, B, C
            #     [2, 1, 1],
            #     [1, 1, 1],
            #     [1, 1, 1],
            # ],  # G[2] == missing
        ]

    def it_provides_valid_overlaps_for_cat_x_mr_sub_x_mr_sel(self):
        cube_dict = OL.CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=0,
        )

        assert overlaps_measure.valid_overlaps.tolist() == [
            [
                [1.0, 1.0, 1.0],
                [1.0, 3.0, 3.0],
                [1.0, 3.0, 3.0],
            ],
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
        ]

    def it_provides_overlaps_for_ca_sub_x_ca_cat_x_mr_sub_x_mr_sel_subvar_0(self):
        # Test subvar X (partitiions[0])
        cube_dict = OL.CA_SUB_X_CA_CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=0,
        )

        assert overlaps_measure.overlaps.tolist() == [
            #
            # --- Missing categories are not shown
            #
            # [
            #     # CA_CATS: No Data
            #     # ----------------
            #     # A, B, C
            #     [2, 1, 1],  # A
            #     [1, 1, 1],  # B
            #     [1, 1, 1],  # C
            # ],
            [
                # CA_CATS: Poor
                # -------------
                # A, B, C
                [1, 0, 0],  # A
                [0, 0, 0],  # B
                [0, 0, 0],  # C
            ],
            [
                # CA_CATS: Fair
                # -------------
                # A, B, C
                [0, 0, 0],  # A
                [0, 1, 1],  # B
                [0, 1, 2],  # C
            ],
            [
                # CA_CATS: Good
                # -------------
                # A, B, C
                [0, 0, 0],  # A
                [0, 0, 0],  # B
                [0, 0, 0],  # C
            ],
            #
            # --- Missing categories are not shown
            #
            # [
            #     # CA_CATS: Not Shown
            #     # ------------------
            #     # A, B, C
            #     [0, 0, 0],  # A
            #     [0, 0, 0],  # B
            #     [0, 0, 0],  # C
            # ],
        ]

    def it_provides_valid_overlaps_for_ca_sub_x_ca_cat_x_mr_sub_x_mr_sel_subvar_0(self):
        # Test subvar X (partitiions[0])
        cube_dict = OL.CA_SUB_X_CA_CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=0,
        )

        assert overlaps_measure.valid_overlaps.tolist() == [
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [2.0, 2.0, 2.0],
                [2.0, 3.0, 3.0],
                [2.0, 3.0, 3.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]

    def it_provides_overlaps_for_ca_sub_x_ca_cat_x_mr_sub_x_mr_sel_subvar_1(self):
        # Test subvar Y (partitions[1])
        cube_dict = OL.CA_SUB_X_CA_CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=1,
        )

        assert overlaps_measure.overlaps.tolist() == [
            #
            # --- Missing categories are not shown
            #
            # [
            #     # CA_CATS: Missing
            #     # ----------------
            #     # A, B, C
            #     [1, 1, 1],  # A
            #     [1, 1, 1],  # B
            #     [1, 1, 1],  # C
            # ],
            [
                # CA_CATS: Poor
                # -------------
                # A, B, C
                [1, 0, 0],  # A
                [0, 1, 1],  # B
                [0, 1, 1],  # C
            ],
            [
                # CA_CATS: Fair
                # -------------
                # A, B, C
                [1, 0, 0],  # A
                [0, 0, 0],  # B
                [0, 0, 0],  # C
            ],
            [
                # CA_CATS: Good
                # -------------
                # A, B, C
                [0, 0, 0],  # A
                [0, 0, 0],  # B
                [0, 0, 1],  # C
            ],
            #
            # --- Missing categories are not shown
            #
            # [
            #     # CA_CATS: Not Shown
            #     # ------------------
            #     # A, B, C
            #     [0, 0, 0],  # A
            #     [0, 0, 0],  # B
            #     [0, 0, 0],  # C
            # ],
        ]

    def it_provides_valid_overlaps_for_ca_sub_x_ca_cat_x_mr_sub_x_mr_sel_subvar_1(self):
        # Test subvar X (partitiions[1])
        cube_dict = OL.CA_SUB_X_CA_CAT_X_MR_SUB_X_MR_SEL
        overlaps_measure = _BaseCubeOverlaps.factory(
            cube=Cube(cube_dict),
            dimensions=AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
            slice_idx=1,
        )

        assert overlaps_measure.valid_overlaps.tolist() == [
            [
                [2.0, 2.0, 2.0],
                [2.0, 3.0, 3.0],
                [2.0, 3.0, 3.0],
            ],
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        ]


class Describe_BaseOrderHelper(object):
    """Integration-test suite for `cr.cube.matrix._BaseOrderHelper`."""

    @pytest.mark.parametrize(
        "fixture, element_ids, expected_value",
        (
            (CR.CAT_4_X_CAT_5, [999, 1, 3, 2], [3, 0, 2, 1]),
            (CR.CAT_X_MR_2, [5, 1, 6, 4, 0, 2], [4, 0, 5, 3, 2, 1]),
            (CR.MR_X_CAT, [3, 5, 1, 4, 2], [2, 4, 0, 3, 1]),
            (CR.MR_X_MR, [2, 0, 3, 1], [1, 3, 2, 0]),
        ),
    )
    def it_can_compute_an_explicit_row_order(
        self, fixture, element_ids, expected_value
    ):
        transforms = {
            "rows_dimension": {
                "order": {
                    "type": "explicit",
                    "element_ids": element_ids,
                }
            }
        }
        assembler = _Slice(Cube(fixture), 0, transforms, None, 0)._assembler
        row_display_order = _BaseOrderHelper.row_display_order(
            assembler._dimensions, assembler._measures
        )

        assert row_display_order.tolist() == expected_value

    @pytest.mark.parametrize(
        "fixture, element_ids, expected_value",
        (
            (CR.CAT_4_X_CAT_5, [3, 1, 2], [2, 0, 1, 3, 4]),
            (CR.CAT_X_MR_2, [5, 1, 4, 2], [4, 0, 3, 1, 2]),
            (CR.MR_X_CAT, [0, 5, 1, 4, 2], [2, 4, 0, 3, 1, 5]),
            (CR.MR_X_MR, [2, 0, 3, 1], [1, 3, 2, 0]),
        ),
    )
    def it_can_compute_an_explicit_column_order(
        self, fixture, element_ids, expected_value
    ):
        transforms = {
            "columns_dimension": {
                "order": {
                    "type": "explicit",
                    "element_ids": element_ids,
                }
            }
        }
        assembler = _Slice(Cube(fixture), 0, transforms, None, 0)._assembler
        column_display_order = _BaseOrderHelper.column_display_order(
            assembler._dimensions, assembler._measures
        )

        assert column_display_order.tolist() == expected_value
