# encoding: utf-8

"""Integration-test suite for `cr.cube.cubepart` module."""

from __future__ import unicode_literals

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
        slice_ = Cube(CR.CAT_X_CAT, population=9001).partitions[0]

        assert slice_.column_labels.tolist() == ["C", "E"]
        assert slice_.column_percentages.tolist() == [[50, 40], [50, 60]]
        assert pytest.approx(slice_.column_proportions) == [[0.5, 0.4], [0.5, 0.6]]
        with pytest.raises(ValueError) as e:
            slice_.column_share_sum
        assert (
            str(e.value)
            == "`.column_share_sum` is undefined for a cube-result without a sum "
            "measure"
        )
        assert slice_.columns_dimension_name == "v7"
        assert slice_.columns_dimension_type == DT.CAT
        assert slice_.columns_margin.tolist() == [10, 5]
        assert slice_.counts.tolist() == [[5, 2], [5, 3]]
        assert slice_.description == "Pet Owners"
        assert pytest.approx(slice_.row_proportions) == [
            [0.7142857, 0.2857142],
            [0.6250000, 0.3750000],
        ]
        assert slice_.has_scale_means is True
        assert slice_.inserted_column_idxs == ()
        assert slice_.inserted_row_idxs == ()
        assert slice_.is_empty is False
        assert slice_.name == "v4"
        with pytest.raises(ValueError) as e:
            slice_.means
        assert (
            str(e.value)
            == "`.means` is undefined for a cube-result without a mean measure"
        )
        assert pytest.approx(slice_.population_counts) == [
            [3000.333, 1200.133],
            [3000.333, 1800.200],
        ]
        assert pytest.approx(slice_.residual_test_stats) == [
            [[0.7143930, 0.71439304], [0.71439304, 0.7143930]],
            [[0.3659625, -0.3659625], [-0.3659625, 0.3659625]],
        ]
        assert slice_.row_labels.tolist() == ["B", "C"]
        assert pytest.approx(slice_.row_proportions) == [
            [0.7142857, 0.2857143],
            [0.6250000, 0.3750000],
        ]
        assert pytest.approx(slice_.row_percentages) == [
            [71.42857, 28.57142],
            [62.50000, 37.50000],
        ]
        with pytest.raises(ValueError) as e:
            slice_.row_share_sum
        assert (
            str(e.value)
            == "`.row_share_sum` is undefined for a cube-result without a sum measure"
        )
        assert slice_.rows_dimension_description == "Pet Owners"
        assert slice_.rows_dimension_fills == (None, None)
        assert slice_.rows_dimension_name == "v4"
        assert slice_.rows_dimension_type == DT.CAT
        assert slice_.rows_margin.tolist() == [7, 8]
        assert slice_.shape == (2, 2)
        with pytest.raises(ValueError) as e:
            slice_.sums
        assert (
            str(e.value)
            == "`.sums` is undefined for a cube-result without a sum measure"
        )
        with pytest.raises(ValueError) as e:
            slice_.stddev
        assert (
            str(e.value)
            == "`.stddev` is undefined for a cube-result without a stddev measure"
        )
        assert slice_.table_margin == 15
        assert slice_.table_name is None
        assert pytest.approx(slice_.table_percentages) == [
            [33.33333, 13.33333],
            [33.33333, 20.00000],
        ]
        assert pytest.approx(slice_.table_proportions) == [
            [0.3333333, 0.1333333],
            [0.3333333, 0.2000000],
        ]
        with pytest.raises(ValueError) as e:
            slice_.total_share_sum
        assert (
            str(e.value)
            == "`.total_share_sum` is undefined for a cube-result without a sum "
            "measure"
        )
        assert slice_.unweighted_counts.tolist() == [[5, 2], [5, 3]]
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

    def it_provides_values_for_cat_x_datetime(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]

        assert slice_.column_labels.tolist() == [
            "1776-07-04T00:00:00",
            "1950-12-24T00:00:00",
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
        ]
        assert slice_.column_proportions.tolist() == [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        assert slice_.columns_margin.tolist() == [1, 1, 1, 1]
        assert slice_.counts.tolist() == [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        assert slice_.row_labels.tolist() == ["red", "green", "blue", "4", "9"]
        np.testing.assert_almost_equal(
            slice_.row_proportions,
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        )
        assert slice_.rows_margin.tolist() == [1, 1, 1, 1, 0]
        assert slice_.table_margin == 4
        assert slice_.table_proportions.tolist() == [
            [0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
            [0.0, 0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]

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

        # This fixture has both cube_counts and cube_means measure, for this reason
        # both measures are available at cubepart level.
        assert slice_.columns_margin.tolist() == [189, 395, 584, 606, 310]
        assert slice_.counts == pytest.approx(np.array([[189, 395, 584, 606, 310]]))
        assert slice_.means == pytest.approx(
            np.array([[24.4393575, 37.3212274, np.nan, 55.4857195, 73.0242765]]),
            nan_ok=True,
        )
        assert slice_.rows_margin.tolist() == [1500.0]

    def it_provides_values_for_mr_x_mr_means(self):
        slice_ = Cube(CR.MR_X_MR_MEANS).partitions[0]

        assert slice_.counts == pytest.approx(
            np.array([[3, 2, 0], [2, 4, 0], [0, 0, 0]])
        )
        assert slice_.means == pytest.approx(
            np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 2.187795, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
            nan_ok=True,
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
            (CR.CAT_X_MR, [2, 1], ["0002", "0001", "0000"], "cat-x-mr-explicit-order"),
            (CR.CAT_X_MR, [2, 1], [3, "0001", "0000"], "cat-x-mr-explicit-order"),
            (CR.CAT_X_MR, [2, 1], ["0002", 2, 1], "cat-x-mr-explicit-order"),
            (CR.MR_X_CAT, [2, 1, 3, 4, 5], [5, 1, 4, 3, 2], "mr-x-cat-explicit-order"),
            (
                CR.MR_X_CAT,
                ["00c1", "00c0", "00c2", "00c3", "00c4"],
                [5, 1, 4, 3, 2],
                "mr-x-cat-explicit-order",
            ),
            (
                CR.MR_X_CAT_HS_MT,
                [5, 1, 4, 2, 3],
                [5, 1, 4, 3, 2],
                "mr-x-cat-hs-explicit-order",
            ),
            (
                CR.MR_X_CAT_HS_MT,
                ["00c4", "00c0", "00c3", "00c1", "00c2"],
                [5, 1, 4, 3, 2],
                "mr-x-cat-hs-explicit-order",
            ),
            (CR.MR_X_MR, [1, 2, 3, 0], [2, 1, 3, 0], "mr-x-mr-explicit-order"),
            (
                CR.MR_X_MR,
                [
                    "da1129bc216d48e9b8ee5229c1b26c79",
                    "8e6c724d95994da1a483b96f39fcd661",
                    "8190dd271bea45f8b2a4204780ce7168",
                    "Any",
                ],
                [
                    "8e6c724d95994da1a483b96f39fcd661",
                    "da1129bc216d48e9b8ee5229c1b26c79",
                    "8190dd271bea45f8b2a4204780ce7168",
                    "Any",
                ],
                "mr-x-mr-explicit-order",
            ),
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

    def it_can_sort_by_column_percent(self):
        """Responds to order:opposing_element sort-by-value.

        So far, this is limited to column-percents (column-proportions) measure, but
        others will follow.
        """
        transforms = {
            "rows_dimension": {
                "order": {
                    "type": "opposing_element",
                    "element_id": 1,
                    "measure": "col_percent",
                    "direction": "ascending",
                    # --- element-ids are 1, 2, 3, 999 ---
                    "fixed": {"top": [999], "bottom": [1]},
                }
            }
        }
        slice_ = _Slice(Cube(CR.CAT_4_X_CAT_5), 0, transforms, None, 0)

        expected = [
            # --- row-element 999 is a top-exclusion, so it appears first ---
            [36.7, 46.2, 25.0, 66.7, 23.9],  # --- 999 - N/A ---
            # --- 2 and 3 appear in ascending order by first col (col-id 1) ---
            [18.9, 31.8, 17.1, 16.7, 26.6],  # --- 2 - Enough ---
            [37.6, 9.8, 53.9, 16.7, 43.1],  # --- 3 - Not Enough ---
            # --- row-element 1 is a bottom-exclusion, so it appears last ---
            [6.8, 12.1, 3.9, 0.0, 6.4],  # --- 1 - Plenty ---
        ]
        actual = np.round(slice_.column_percentages, 1).tolist()
        assert expected == actual, "\n%s\n\n%s" % (expected, actual)

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

    def it_provides_sum_measure_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR_SUM).partitions[0]
        assert slice_.sums == pytest.approx(
            np.array(
                [
                    [2.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 4.0],
                ]
            )
        )

    def it_provides_stddev_measure_for_cat_x_mr(self):
        slice_ = Cube(CR.CAT_X_MR_STDDEV).partitions[0]

        assert slice_.stddev == pytest.approx(
            np.array(
                [
                    [0.70710678, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ),
            nan_ok=True,
        )
        assert slice_.table_base.tolist() == [3, 3, 3]

    def it_provides_share_of_sum_measure_for_mr_x_mr(self):
        slice_ = Cube(CR.MR_X_MR_SUM).partitions[0]

        assert slice_.column_share_sum == pytest.approx(
            np.array(
                [
                    [0.5, 0.2, 0.2222222],
                    [0.25, 0.4, 0.3333333],
                    [0.25, 0.4, 0.4444444],
                ]
            )
        )
        assert slice_.row_share_sum == pytest.approx(
            np.array(
                [
                    [0.4, 0.2, 0.4],
                    [0.1666666, 0.3333333, 0.5],
                    [0.1428571, 0.2857142, 0.5714285],
                ]
            )
        )
        assert slice_.total_share_sum == pytest.approx(
            np.array(
                [
                    [0.11111111, 0.05555556, 0.11111111],
                    [0.05555556, 0.11111111, 0.16666667],
                    [0.05555556, 0.11111111, 0.22222222],
                ]
            )
        )


class Describe_Strand(object):
    """Integration-test suite for `cr.cube.cubepart._Strand` object."""

    def it_provides_values_for_ca_as_0th(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        strand = Cube(
            CR.CA_AS_0TH, cube_idx=0, transforms=transforms, population=100000000
        ).partitions[0]

        assert strand.population_counts == pytest.approx(
            [54523323.5, 24570078.1, 15710358.3, 5072107.3]
        )
        assert strand.table_name == "Level of in: ATP Men's T"
        assert strand.weighted_bases == pytest.approx([27292.0] * 4)

    def it_provides_values_for_univariate_cat(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL, population=1000).partitions[0]

        assert strand.counts.tolist() == [10, 5]
        assert strand.cube_index == 0
        assert strand.dimension_types == (DT.CAT,)
        assert strand.has_scale_means is True
        assert strand.inserted_row_idxs == ()
        assert strand.is_empty is False
        with pytest.raises(ValueError) as e:
            strand.means
        assert str(e.value) == (
            "`.means` is undefined for a cube-result without a mean measure"
        )
        assert strand.min_base_size_mask.tolist() == [False, False]
        assert strand.name == "v7"
        assert strand.ndim == 1
        assert strand.population_counts == pytest.approx([666.6667, 333.3333])
        assert strand.population_counts_moe == pytest.approx([238.5592, 238.5592])
        assert strand.row_count == 2
        assert strand.row_labels.tolist() == ["C", "E"]
        assert strand.rows_base.tolist() == [10, 5]
        assert strand.rows_dimension_fills == (None, None)
        assert strand.rows_dimension_name == "v7"
        assert strand.rows_dimension_type == DT.CAT
        assert strand.rows_margin.tolist() == [10, 5]
        assert strand.scale_mean == pytest.approx(1.666667)
        assert strand.scale_median == pytest.approx(1.0)
        assert strand.scale_std_dev == pytest.approx(0.9428090)
        assert strand.scale_std_err == pytest.approx(0.2434322)
        assert strand.shape == (2,)
        with pytest.raises(ValueError) as e:
            strand.share_sum
        assert str(e.value) == (
            "`.share_sum` is undefined for a cube-result without a sum measure"
        )
        with pytest.raises(ValueError) as e:
            strand.sums
        assert str(e.value) == (
            "`.sums` is undefined for a cube-result without a sum measure"
        )
        with pytest.raises(ValueError) as e:
            strand.stddev
        assert str(e.value) == (
            "`.stddev` is undefined for a cube-result without a stddev measure"
        )
        assert strand.table_base_range.tolist() == [15, 15]
        assert strand.table_margin_range.tolist() == [15, 15]
        assert strand.table_name == "v7: C"
        assert strand.table_percentages == pytest.approx([66.66667, 33.33333])
        assert strand.table_proportion_moes == pytest.approx([0.2385592, 0.2385592])
        assert strand.table_proportion_stddevs == pytest.approx([0.4714045, 0.4714045])
        assert strand.table_proportion_stderrs == pytest.approx([0.1217161, 0.1217161])
        assert strand.table_proportions == pytest.approx([0.6666667, 0.3333333])
        assert strand.title == "Registered Voters"
        assert strand.unweighted_bases.tolist() == [15, 15]
        assert strand.unweighted_counts.tolist() == [10, 5]
        assert strand.variable_name == "v7"
        assert strand.weighted_bases == pytest.approx([15.0, 15.0])

    def it_provides_values_for_univariate_cat_means_hs(self):
        strand = Cube(CR.CAT_MEANS_HS).partitions[0]

        assert strand.is_empty is False
        np.testing.assert_almost_equal(
            strand.means, [19.85555556, 13.85416667, 52.78947368, np.nan, np.nan]
        )
        assert strand.title == "Untitled"
        assert strand.unweighted_counts.tolist() == [409, 113, 139, 409, 252]

    def it_provides_values_for_univariate_cat_means_and_counts(self):
        """The cube_mean and cube_count measures can appear together."""
        # --- prune to avoid NaNs in results and thereby simplify assertions ---
        transforms = {"rows_dimension": {"prune": True}}
        strand = Cube(CR.CAT_MEANS_AND_COUNTS, transforms=transforms).partitions[0]

        assert strand.means == pytest.approx([74.50347, 83.82950, 80, 79.38796])
        assert strand.ndim == 1
        assert strand.rows_base.tolist() == [806, 14, 28, 780]
        assert strand.shape == (4,)
        assert strand.table_base_range.tolist() == [1628, 1628]
        # --- means cube that also has counts has a table-margin ---
        assert strand.table_margin_range == pytest.approx([1500.961, 1500.961])

    def it_provides_values_for_univariate_datetime(self):
        strand = Cube(CR.DATE, population=9001).partitions[0]

        assert strand.counts.tolist() == [1, 1, 1, 1]
        assert strand.population_counts == pytest.approx(
            [2250.25, 2250.25, 2250.25, 2250.25]
        )
        assert strand.table_margin_range.tolist() == [4, 4]
        assert strand.table_percentages == pytest.approx([25.0, 25.0, 25.0, 25.0])
        assert strand.table_proportion_stddevs == pytest.approx(
            [0.4330127, 0.4330127, 0.4330127, 0.4330127]
        )
        assert strand.table_proportion_stderrs == pytest.approx(
            [0.2165064, 0.2165064, 0.2165064, 0.2165064]
        )
        assert strand.table_proportions == pytest.approx([0.25, 0.25, 0.25, 0.25])

    def it_provides_values_for_univariate_mr_hs(self):
        # --- subtotals shouldn't be in the MR variable, but there are cases when they
        # --- are present. H&S should be ignored for univariate MR.
        strand = Cube(CR.UNIV_MR_WITH_HS).partitions[0]

        assert strand.counts == pytest.approx(
            [
                10488.90,
                5051.444,
                1960.495,
                3985.375,
                3073.367,
                2196.709,
                779.4323,
                14859.56,
                713.5478,
            ]
        )
        assert strand.min_base_size_mask.tolist() == [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        assert strand.name == "Paid for ne"
        assert strand.rows_dimension_type == DT.MR
        assert strand.scale_mean is None
        assert strand.table_name == "Paid for ne: A newspaper"
        assert strand.table_proportion_stddevs == pytest.approx(
            [
                0.4642672,
                0.3584419,
                0.2351762,
                0.3243186,
                0.2891897,
                0.2480032,
                0.1510486,
                0.4970073,
                0.1446697,
            ]
        )
        assert strand.table_proportion_stderrs == pytest.approx(
            [
                0.002541725,
                0.001962362,
                0.001287519,
                0.001775547,
                0.001583227,
                0.001357743,
                0.000826946,
                0.002720966,
                0.000792023,
            ]
        )
        assert strand.unweighted_bases.tolist() == [33358] * 9
        assert strand.weighted_bases == pytest.approx([33364.08] * 9)

    def it_provides_values_for_univariate_numeric(self):
        strand = Cube(CR.NUM, population=9001).partitions[0]

        assert strand.counts.tolist() == [885, 105, 10]
        assert strand.population_counts == pytest.approx([7965.885, 945.105, 90.01])
        assert strand.population_counts_moe == pytest.approx(
            [177.9752, 171.0193, 55.50811]
        )
        assert strand.table_margin_range.tolist() == [1000, 1000]
        assert strand.table_percentages == pytest.approx([88.5, 10.5, 1.0])
        assert strand.table_proportion_moes == pytest.approx(
            [0.019772822, 0.019000029, 0.006166883]
        )
        assert strand.table_proportion_stddevs == pytest.approx(
            [0.31902194, 0.30655342, 0.09949874]
        )
        assert strand.table_proportion_stderrs == pytest.approx(
            [0.01008836, 0.00969407, 0.003146427]
        )
        assert strand.table_proportions == pytest.approx([0.885, 0.105, 0.010])
        assert strand.weighted_bases == pytest.approx([1000.0] * 3)

    def it_provides_values_for_univariate_numeric_binned(self):
        strand = Cube(
            CR.NUM_BINNED, transforms={"rows_dimension": {"prune": True}}
        ).partitions[0]
        assert strand.counts == pytest.approx([118504.4, 155261.3, 182924.0])

    def it_provides_values_for_univariate_text(self):
        strand = Cube(CR.TEXT, population=9001).partitions[0]

        assert strand.counts.tolist() == [1, 1, 1, 1, 1, 1]
        assert strand.population_counts == pytest.approx(
            [1500.167, 1500.167, 1500.167, 1500.167, 1500.167, 1500.167]
        )
        assert strand.table_margin_range.tolist() == [6, 6]
        assert strand.table_percentages == pytest.approx(
            [16.66667, 16.66667, 16.66667, 16.66667, 16.66667, 16.66667],
        )
        assert strand.table_proportion_stddevs == pytest.approx(
            [0.372678, 0.372678, 0.372678, 0.372678, 0.372678, 0.372678],
        )
        assert strand.table_proportion_stderrs == pytest.approx(
            [0.1521452, 0.1521452, 0.1521452, 0.1521452, 0.1521452, 0.1521452],
        )
        assert strand.table_proportions == pytest.approx(
            [0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667],
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

        assert strand.row_labels.tolist() == [
            "Sum A-C",
            "C1 & C2",
            "D",
            "Total A-C",
            "E",
            "AB",
            "Total D-E",
        ]
        assert strand.counts.tolist() == [31506, 16275, 3480, 31506, 4262, 15231, 7742]

    def it_knows_when_it_is_empty(self):
        strand = Cube(CR.OM_SGP8334215_VN_2019_SEP_19_STRAND).partitions[0]
        assert strand.is_empty is True

    def it_provides_stddev_measure_for_CAT(self):
        strand = Cube(CR.CAT_STDDEV).partitions[0]

        assert strand.stddev == pytest.approx([22.898325, 7.778174])
        assert strand.table_base_range.tolist() == [5, 5]

    def it_provides_stddev_measure_for_MR(self):
        strand = Cube(CR.MR_STDDEV).partitions[0]

        assert strand.stddev == pytest.approx([3.22398, 1.23444, 9.23452])
        assert strand.table_base_range.tolist() == [3, 3]

    def it_provides_sum_measure_for_CAT(self):
        strand = Cube(CR.CAT_SUM).partitions[0]

        assert strand.sums == pytest.approx([88.0, 77.0])
        assert strand.table_base_range.tolist() == [5, 5]

    def it_provides_sum_measure_for_CAT_HS(self):
        transforms = {
            "rows_dimension": {
                "insertions": [
                    {
                        "anchor": "bottom",
                        "args": [1, 2],
                        "function": "subtotal",
                        "hide": False,
                        "name": "Sub",
                    }
                ]
            }
        }
        strand = Cube(CR.CAT_SUM, transforms=transforms).partitions[0]

        assert strand.sums == pytest.approx([88.0, 77.0, 165.0])

    def it_provides_sum_measure_for_MR(self):
        strand = Cube(CR.MR_SUM).partitions[0]

        assert strand.sums == pytest.approx([3.0, 2.0, 2.0])
        assert strand.table_base_range.tolist() == [3, 3]

    def it_provides_sum_and_mean_measure_for_CAT(self):
        strand = Cube(CR.NUMERIC_MEASURES_X_CAT).partitions[0]

        assert strand.counts == pytest.approx([3, 2])
        assert strand.means == pytest.approx([2.66666667, 3.5])
        assert strand.sums == pytest.approx([8, 7])

    def it_provides_share_of_sum_measure_for_CAT(self):
        strand = Cube(CR.CAT_SUM).partitions[0]

        assert strand.sums == pytest.approx([88.0, 77.0])
        # --- share of sum is the array of sum divided by its sum, so in this case
        # --- [88/165, 77/165]
        assert strand.share_sum.tolist() == [0.5333333333333333, 0.4666666666666667]
        assert strand.table_base_range.tolist() == [5, 5]

    def it_provides_share_of_sum_measure_for_MR(self):
        strand = Cube(CR.MR_SUM).partitions[0]

        assert strand.share_sum.tolist() == [
            0.42857142857142855,
            0.2857142857142857,
            0.2857142857142857,
        ]
        assert strand.table_base_range.tolist() == [3, 3]


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


class Test_Slice(object):
    """Legacy unit tests for _Slice object.

    In general, these need to be consolidated into Describe_Slice above, and there are
    probably redundancies to be eliminated.
    """

    def test_mr_x_mr_table_base_and_margin(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.MR_X_MR_WITH_PRUNING, transforms=transforms).partitions[0]

        # Assert table base
        expected = np.full((10, 10), 6490)
        np.testing.assert_array_equal(slice_.table_base, expected)

        # Assert table margin
        expected = np.full((10, 10), 6456.761929)
        np.testing.assert_almost_equal(slice_.table_margin, expected)

    def test_mr_x_mr_table_base_and_margin_unpruned(self):
        slice_ = Cube(CR.MR_X_MR_WITH_PRUNING).partitions[0]

        # Assert table base
        expected = np.full((12, 12), 6490)
        np.testing.assert_array_equal(slice_.table_base_unpruned, expected)

        # Assert table margin
        expected = np.full((12, 12), 6456.761929)
        np.testing.assert_almost_equal(slice_.table_margin_unpruned, expected)

    def test_filtered_population_counts(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.CAT_X_CAT_FILT, transforms=transforms, population=100000000
        ).partitions[0]
        expected = np.array(
            [
                [300000.0, 1400000.0, 0.0, 0.0, 0.0, 0.0],
                [5900000.0, 13200000.0, 0.0, 0.0, 0.0, 0.0],
                [600000.0, 2900000.0, 0.0, 0.0, 0.0, 0.0],
                [100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0],
                [300000.0, 600000.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.population_counts, expected)

    def test_labels_simple_cat_array_exclude_missing(self):
        slice_ = Cube(CR.SIMPLE_CAT_ARRAY).partitions[0]
        assert slice_.row_labels.tolist() == [
            "ca_subvar_1",
            "ca_subvar_2",
            "ca_subvar_3",
        ]
        assert slice_.column_labels.tolist() == ["a", "b", "c", "d"]

    def test_as_array_simple_cat_array_exclude_missing(self):
        slice_ = Cube(CR.SIMPLE_CAT_ARRAY).partitions[0]
        expected = np.array([[3, 3, 0, 0], [1, 3, 2, 0], [0, 2, 1, 3]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_as_array_cat_x_num_x_datetime(self):
        """Test 3D cube, slicing accross first (numerical) variable."""
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[1, 1], [0, 0], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.counts, [[2, 1], [1, 1], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [2, 3], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [0, 0], [3, 2], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [0, 0], [1, 1], [0, 1]])

    def test_proportions_cat_x_num_datetime(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.4, 0.2], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.0, 0.0], [0.4, 0.6], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.0, 0.0], [0.0, 0.0], [0.6, 0.4], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_almost_equal(
            slice_.table_proportions,
            [[0.0, 0.0], [0.0, 0.0], [0.33333333, 0.33333333], [0.0, 0.33333333]],
        )

    def test_cat_x_num_x_datetime_margin_by_table(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, 2)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.table_margin, 3)

    def test_margin_cat_x_num_x_datetime_axis_0(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.rows_margin, [2, 0, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.rows_margin, [3, 2, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 5, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 0, 5, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 0, 2, 1])

    def test_margin_cat_x_num_x_datetime_axis_1(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 1])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.columns_margin, [3, 2])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.columns_margin, [2, 3])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.columns_margin, [3, 2])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 2])

    def test_as_array_unweighted_gender_x_ideology(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([[32, 85, 171, 114, 70, 13], [40, 97, 205, 106, 40, 27]])
        np.testing.assert_array_equal(slice_.unweighted_counts, expected)

    def test_as_array_weighted_gender_x_ideology(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array(
            [
                [
                    32.98969072,
                    87.62886598,
                    176.28865979,
                    117.5257732,
                    72.16494845,
                    13.40206186,
                ],
                [
                    38.83495146,
                    94.17475728,
                    199.02912621,
                    102.91262136,
                    38.83495146,
                    26.21359223,
                ],
            ]
        )
        np.testing.assert_almost_equal(slice_.counts, expected)

    def test_margin_weighted_gender_x_ideology_axis_0(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array(
            [
                71.82464218,
                181.80362326,
                375.31778601,
                220.43839456,
                110.99989991,
                39.61565409,
            ]
        )
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_margin_unweighted_gender_x_ideology_axis_0(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([72, 182, 376, 220, 110, 40])
        np.testing.assert_array_equal(slice_.columns_base, expected)

    def test_margin_unweighted_gender_x_ideology_axis_1(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([485, 515])
        np.testing.assert_array_equal(slice_.rows_base, expected)

    def test_margin_weighted_gender_x_ideology_axis_1(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([500, 500])
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_calculate_various_measures_axis_0(self):
        """Calculate standard error across columns."""
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected_zscore = np.array(
            [
                [
                    -0.715899626017458,
                    -0.536708837208206,
                    -1.485149675785275,
                    1.114743776586886,
                    3.355236023985286,
                    -2.077040949965248,
                ],
                [
                    0.715899626017462,
                    0.536708837208211,
                    1.485149675785279,
                    -1.114743776586884,
                    -3.355236023985284,
                    2.07704094996525,
                ],
            ]
        )
        expected_table_std_dev = [
            [0.17860955, 0.28275439, 0.38106557, 0.32204575, 0.25876083, 0.1149889],
            [0.19320144, 0.29207169, 0.39927, 0.30384472, 0.19320144, 0.15976996],
        ]
        expected_table_std_err = [
            [0.00564813, 0.00894148, 0.01205035, 0.01018398, 0.00818274, 0.00363627],
            [0.00610957, 0.00923612, 0.01262603, 0.00960841, 0.00610957, 0.00505237],
        ]
        expected_col_std_dev = [
            [0.49834148, 0.4996758, 0.49908137, 0.49890016, 0.47692704, 0.47313192],
            [0.49834148, 0.4996758, 0.49908137, 0.49890016, 0.47692704, 0.47313192],
        ]
        expected_col_std_err = [
            [0.05880176, 0.03705843, 0.02576154, 0.03360238, 0.04526793, 0.07517074],
            [0.05880176, 0.03705843, 0.02576154, 0.03360238, 0.04526793, 0.07517074],
        ]

        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-col-prop-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-row-prop-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.row_std_dev,
            load_python_expression("econ-gender-x-ideology-weighted-row-std-dev"),
        )
        np.testing.assert_almost_equal(
            slice_.table_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-table-prop-moe"),
        )
        np.testing.assert_almost_equal(slice_.zscores, expected_zscore)

    def test_pvals(self):
        expected = np.array(
            [
                [
                    0.1427612835278633,
                    0.0002121209149277,
                    0.0001314694879104,
                    0.1241771485531613,
                    0.0097454103240531,
                    0.0000000000000699,
                    0.8033849539190183,
                    0.4271118723152929,
                ],
                [
                    0.1427612835278667,
                    0.0002121209149277,
                    0.0001314694879104,
                    0.1241771485531642,
                    0.0097454103240533,
                    0.0000000000000699,
                    0.8033849539190183,
                    0.4271118723152956,
                ],
            ]
        )
        # Test without pruning
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning and H&S
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_pvals_stats(self):
        expected = np.array(
            [
                [
                    0.0436818197570077,
                    0.0002697141695955,
                    0.0913940671748992,
                    0.6836420776424197,
                    0.4681291494279529,
                    0.0000013632752629,
                ],
                [
                    0.0436818197570077,
                    0.0002697141695955,
                    0.0913940671748992,
                    0.6836420776424197,
                    0.4681291494279529,
                    0.0000013632752629,
                ],
            ]
        )
        # Test without pruning
        slice_ = Cube(CR.STATS_TEST).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.STATS_TEST, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning and H&S
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.STATS_TEST, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_mean_age_for_blame_x_gender(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(
            CR.ECON_MEAN_AGE_BLAME_X_GENDER, transforms=transforms
        ).partitions[0]
        expected = np.array(
            [
                [52.78205128205122, 49.9069767441861],
                [50.43654822335009, 48.20100502512572],
                [51.5643564356436, 47.602836879432715],
                [58, 29],
                [37.53846153846155, 39.45238095238095],
            ]
        )
        np.testing.assert_almost_equal(slice_.means, expected)
        assert slice_.ndim == 2

    def test_various_measures_admit_by_dept_unweighted_rows(self):
        """see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/
        zvalues-spec.js#L42
        """
        slice_ = Cube(CR.ADMIT_X_DEPT_UNWEIGHTED).partitions[0]
        expected_zscores = np.array(
            [
                [
                    18.04029230689576,
                    13.018394979149804,
                    -2.576739836814013,
                    -3.059526328377408,
                    -7.230244530709987,
                    -19.321410263144653,
                ],
                [
                    -18.040292306895765,
                    -13.018394979149804,
                    2.576739836814013,
                    3.059526328377408,
                    7.230244530709987,
                    19.321410263144653,
                ],
            ]
        )
        expected_table_std_dev = [
            [0.33934583, 0.27398329, 0.25706606, 0.2364359, 0.17726851, 0.10030056],
            [0.26071661, 0.21271283, 0.33814647, 0.31969003, 0.29534847, 0.35469478],
        ]
        expected_table_std_err = [
            [0.00504412, 0.00407255, 0.00382109, 0.00351444, 0.00263496, 0.00149089],
            [0.00387535, 0.00316181, 0.00502629, 0.00475195, 0.00439013, 0.00527227],
        ]
        expected_col_std_dev = [
            [0.47876747, 0.48213008, 0.47720873, 0.47358921, 0.43399681, 0.24550986],
            [0.47876747, 0.48213008, 0.47720873, 0.47358921, 0.43399681, 0.24550986],
        ]
        expected_col_std_err = [
            [0.01567414, 0.01993363, 0.01575024, 0.01682826, 0.01795892, 0.00918798],
            [0.01567414, 0.01993363, 0.01575024, 0.01682826, 0.01795892, 0.00918798],
        ]

        np.testing.assert_almost_equal(slice_.zscores, expected_zscores)
        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("admit-x-dept-unweighted-col-prop-moe"),
        )

    def test_various_measures_admit_by_gender_weighted_rows(self):
        """see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/
        zvalues-spec.js#L67
        """
        slice_ = Cube(CR.ADMIT_X_GENDER_WEIGHTED).partitions[0]
        expected_zscores = np.array(
            [
                [9.42561984520692, -9.425619845206922],
                [-9.425619845206922, 9.42561984520692],
            ]
        )
        expected_table_std_dev = [[0.44013199, 0.32828883], [0.47059018, 0.45061221]]
        expected_table_std_err = [[0.00659641, 0.00492018], [0.0070529, 0.00675348]]
        expected_col_std_dev = [[0.49668253, 0.45933735], [0.49668253, 0.45933735]]
        expected_col_std_err = [[0.00966009, 0.01080163], [0.00966009, 0.01080163]]
        expected_col_proportions_moe = [
            [0.0189334366, 0.0211708092],
            [0.0189334366, 0.0211708092],
        ]

        np.testing.assert_almost_equal(slice_.zscores, expected_zscores)
        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe, expected_col_proportions_moe
        )

    def test_selected_crosstab_as_array(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                [9928.20954289002, 11524.821237084192],
                [9588.843313998908, 9801.254016136965],
                [11697.435357575358, 13095.670425525452],
                [9782.8995547749, 10531.918128023966],
                [4417.596222134318, 3448.380316269752],
                [6179.175512581436, 6490.427474934746],
            ]
        )
        np.testing.assert_almost_equal(slice_.counts, expected)

    def test_selected_crosstab_margin_by_rows(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                21453.03077997421,
                19390.097330135875,
                24793.105783100807,
                20314.817682798865,
                7865.976538404069,
                12669.602987516182,
            ]
        )
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_selected_crosstab_margin_by_cols(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                [14566.261567907562, 15607.301233922663],
                [14456.513325488017, 15450.609903833058],
                [14415.136475733132, 15405.898678070093],
                [11485.661204663904, 11912.588886491172],
                [11664.69933815247, 12110.196347286023],
                [11547.413553551738, 11961.575582997419],
            ]
        )
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_selected_crosstab_margin_total(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                30173.5628018302,
                29907.1232293211,
                29821.0351538032,
                23398.2500911551,
                23774.8956854385,
                23508.9891365492,
            ]
        )
        np.testing.assert_almost_equal(slice_.table_margin, expected)

    def test_selected_crosstab_proportions_by_rows(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                [0.4627882020361299, 0.5372117979638701],
                [0.4945227014975337, 0.5054772985024663],
                [0.47180193800279874, 0.5281980619972013],
                [0.481564723224583, 0.5184352767754171],
                [0.5616081106479636, 0.4383918893520365],
                [0.48771658580541166, 0.5122834141945883],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_selected_crosstab_proportions_by_cols(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                [0.6815894041587091, 0.7384249886863752],
                [0.6632887957217867, 0.6343603312193796],
                [0.8114689290154947, 0.8500426167391849],
                [0.8517489224566737, 0.8840998567462627],
                [0.3787149667617584, 0.28475015741941767],
                [0.535113381358101, 0.5426064007955989],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_selected_crosstab_proportions_by_cell(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
        expected = np.array(
            [
                [0.329036700375595, 0.381950958618156],
                [0.320620717695708, 0.327723062528721],
                [0.392254504152701, 0.439142047148397],
                [0.418103897371069, 0.450115632023491],
                [0.185809278853744, 0.14504292098248],
                [0.262843097025161, 0.27608279697761],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_as_array(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array([[40, 14, 18], [14, 34, 16], [18, 16, 38]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_x_pets_proportions_by_cell(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [0.5, 0.2, 0.2571429],
                [0.2, 0.4303797, 0.2285714],
                [0.2571429, 0.2285714, 0.5428571],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_proportions_by_col(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [1.0, 0.4827586, 0.4736842],
                [0.4117647, 1.0, 0.4210526],
                [0.5294118, 0.5517241, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_pets_proportions_by_row(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [1.0, 0.4117647, 0.5294118],
                [0.4827586, 1.0, 0.5517241],
                [0.4736842, 0.4210526, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_x_fruit_as_array(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array([[12, 28], [12, 22], [12, 26]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_x_fruit_margin_row(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array([40, 34, 38])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_pets_array_as_array(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array([[45, 34], [40, 40], [32, 38]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_array_proportions(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array(
            [[0.5696203, 0.4303797], [0.5000000, 0.500000], [0.4571429, 0.5428571]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_array_margin_by_row(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array([79, 80, 70])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_fruit_x_pets_proportions_by_cell(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array(
            [[0.15, 0.15189873, 0.17142857], [0.35, 0.27848101, 0.37142857]]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_fruit_x_pets_proportions_by_row(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array(
            [[0.4285714, 0.48, 0.5217391], [0.5384615, 0.4074074, 0.5531915]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_fruit_x_pets_proportions_by_col(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array([[0.3, 0.3529412, 0.3157895], [0.7, 0.6470588, 0.6842105]])
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_fruit_proportions_by_cell(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.15, 0.35], [0.15189873, 0.27848101], [0.17142857, 0.37142857]]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_fruit_proportions_by_col(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.4285714, 0.5384615], [0.48, 0.4074074], [0.5217391, 0.5531915]]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_fruit_proportions_by_row(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.3, 0.7], [0.3529412, 0.6470588], [0.3157895, 0.6842105]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_cat_x_cat_array_proportions_by_row(self):
        """Get the proportions for each slice of the 3D cube."""
        slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]
        expected = [[0.52, 0.48], [0.57142857, 0.42857143], [0.47826087, 0.52173913]]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]
        expected = [
            [0.59259259, 0.40740741],
            [0.46153846, 0.53846154],
            [0.44680851, 0.55319149],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_identity_x_period_axis_out_of_bounds(self):
        slice_ = Cube(CR.NUM_X_NUM_EMPTY).partitions[0]
        expected = np.array([94, 0, 248, 210, 102, 0, 0, 0, 286, 60])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_ca_with_single_cat(self):
        slice_ = Cube(CR.CA_SINGLE_CAT).partitions[0]
        expected = np.array([79, 80, 70, 0])
        np.testing.assert_almost_equal(slice_.rows_base, expected)

    def test_pets_array_x_pets_by_col(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = [0.59097127, 0.0, 0.55956679], [0.40902873, 1.0, 0.44043321]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_array_x_pets_row(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = [0.44836533, 0.0, 0.48261546], [0.39084967, 1.0, 0.47843137]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_array_x_pets_cell(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = (
            [0.24992768, 0.00000000, 0.26901938],
            [0.17298235, 0.44258027, 0.21174429],
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_array_percentages(self):
        slice_ = Cube(CR.PETS_X_PETS_ARRAY).partitions[0]
        expected = [
            [0.58823529, 0.41176471],
            [0.00000000, 1.00000000],
            [0.47058824, 0.52941176],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_profiles_percentages_add_up_to_100(self):
        slice_ = Cube(CR.PROFILES_PERCENTS).partitions[0]
        props = slice_.row_percentages
        actual_sum = np.sum(props, axis=1)
        expected_sum = np.ones(props.shape[0]) * 100
        np.testing.assert_almost_equal(actual_sum, expected_sum)

    def test_cat_x_cat_as_array_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
        expected = np.array(
            [
                [2, 2, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 2],
                [0, 2, 0, 0],
                [0, 2, 0, 1],
                [0, 1, 0, 0],
            ]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array([[2, 2, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 1, 0]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_cat_x_cat_props_by_col_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
        expected = np.array(
            [
                [1.0, 0.25, np.nan, 0.25],
                [0.0, 0.0, np.nan, 0.0],
                [0.0, 0.125, np.nan, 0.5],
                [0.0, 0.25, np.nan, 0.0],
                [0.0, 0.25, np.nan, 0.25],
                [0.0, 0.125, np.nan, 0.0],
            ]
        )
        np.testing.assert_array_equal(slice_.column_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [1.0, 0.25, 0.25],
                [0.0, 0.125, 0.5],
                [0.0, 0.25, 0.0],
                [0.0, 0.25, 0.25],
                [0.0, 0.125, 0.0],
            ]
        )
        np.testing.assert_array_equal(slice_.column_proportions, expected)

    def test_cat_x_cat_props_by_row_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
        expected = np.array(
            [
                [0.4, 0.4, 0.0, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.33333333, 0.0, 0.66666667],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.66666667, 0.0, 0.33333333],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [0.4, 0.4, 0.2],
                [0.0, 0.33333333, 0.66666667],
                [0.0, 1.0, 0.0],
                [0.0, 0.66666667, 0.33333333],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_cat_x_cat_props_by_cell_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
        expected = np.array(
            [
                [0.14285714, 0.14285714, 0.0, 0.07142857],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.07142857, 0.0, 0.14285714],
                [0.0, 0.14285714, 0.0, 0.0],
                [0.0, 0.14285714, 0.0, 0.07142857],
                [0.0, 0.07142857, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [0.14285714, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.14285714],
                [0.0, 0.14285714, 0.0],
                [0.0, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_single_col_margin_not_iterable(self):
        slice_ = Cube(CR.SINGLE_COL_MARGIN_NOT_ITERABLE).partitions[0]
        assert slice_.columns_margin == 1634

    def test_3d_percentages_by_col(self):
        # ---CAT x CAT x CAT---
        slice_ = Cube(CR.GENDER_PARTY_RACE).partitions[0]
        expected = [
            [0.17647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.17647059, 0.05882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.23529412, 0.0, 0.0, 0.0, 0.0, 0.05882353, 0.0, 0.0],
            [0.11764706, 0.05882353, 0.0, 0.05882353, 0.0, 0.05882353, 0.0, 0.0],
        ]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

        expected = [
            [0.04761905, 0.0, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0],
            [0.14285714, 0.04761905, 0.0952381, 0.04761905, 0.0, 0.04761905, 0.0, 0.0],
            [0.23809524, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.19047619, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        slice_ = Cube(CR.GENDER_PARTY_RACE).partitions[1]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_ca_with_single_cat_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CA_SINGLE_CAT, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.unweighted_counts, [[79], [80], [70]])

    def test_ca_x_single_cat_counts(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        assert slice_.counts == pytest.approx(np.array([[13], [12]]))
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        assert slice_.counts == pytest.approx(np.array([[16], [12]]))
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        assert slice_.counts == pytest.approx(np.array([[11], [12]]))

    def test_ca_x_single_cat_props_by_col(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        expected = [[0.52], [0.48]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        expected = [[0.57142857], [0.42857143]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        expected = [[0.47826087], [0.52173913]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_ca_x_single_cat_props_by_row(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_ca_x_single_cat_props_by_cell(self):
        expected = [
            [[0.52], [0.48]],
            [[0.57142857], [0.42857143]],
            [[0.47826087], [0.52173913]],
        ]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[2])

    def test_ca_x_single_cat_col_margins(self):
        expected = [25, 28, 23]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.columns_margin, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.columns_margin, expected[2])

    def test_ca_x_single_cat_rows_margins(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.rows_margin, [13, 12])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.rows_margin, [16, 12])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.rows_margin, [11, 12])

    def test_ca_x_single_cat_cell_margins(self):
        expected = [25, 28, 23]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.table_margin, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.table_margin, expected[2])

    def test_ca_subvar_x_cat_hs_counts_prune(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CA_SUBVAR_HS_X_CAT_HS, transforms=transforms).partitions[0]
        expected = np.array([[3, 3, 0, 0, 6], [1, 3, 2, 0, 4], [0, 2, 1, 3, 2]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_values_services(self):
        slice_ = Cube(CR.MR_X_CA_CAT_X_CA_SUBVAR).partitions[0]
        expected = np.array(
            [
                [
                    0.14285714,
                    0.10204082,
                    0.20512821,
                    0.16363636,
                    0.16438356,
                    0.1372549,
                    0.18181818,
                    0.2991453,
                    0.32,
                    0.44776119,
                ],  # noqa
                [
                    0.07142857,
                    0.23469388,
                    0.17948718,
                    0.14545455,
                    0.20547945,
                    0.09803922,
                    0.27272727,
                    0.11111111,
                    0.352,
                    0.23880597,
                ],  # noqa
                [
                    0.12857143,
                    0.19387755,
                    0.1025641,
                    0.16363636,
                    0.1369863,
                    0.15686275,
                    0.25,
                    0.17094017,
                    0.136,
                    0.14925373,
                ],  # noqa
                [
                    0.15714286,
                    0.15306122,
                    0.14102564,
                    0.05454545,
                    0.17808219,
                    0.09803922,
                    0.18181818,
                    0.20512821,
                    0.064,
                    0.05223881,
                ],  # noqa
                [
                    0.12857143,
                    0.12244898,
                    0.1025641,
                    0.05454545,
                    0.15068493,
                    0.07843137,
                    0.06060606,
                    0.1025641,
                    0.064,
                    0.05970149,
                ],  # noqa
                [
                    0.05714286,
                    0.09183673,
                    0.20512821,
                    0.09090909,
                    0.09589041,
                    0.11764706,
                    0.03030303,
                    0.02564103,
                    0.032,
                    0.01492537,
                ],  # noqa
                [
                    0.08571429,
                    0.04081633,
                    0.05128205,
                    0.07272727,
                    0.01369863,
                    0.11764706,
                    0.01515152,
                    0.05128205,
                    0.024,
                    0.02238806,
                ],  # noqa
                [
                    0.17142857,
                    0.04081633,
                    0.01282051,
                    0.03636364,
                    0.02739726,
                    0.01960784,
                    0.00757576,
                    0.00854701,
                    0.008,
                    0.00746269,
                ],  # noqa
                [
                    0.01428571,
                    0.02040816,
                    0.0,
                    0.14545455,
                    0.01369863,
                    0.11764706,
                    0.0,
                    0.0,
                    0.0,
                    0.00746269,
                ],  # noqa
                [
                    0.04285714,
                    0.0,
                    0.0,
                    0.07272727,
                    0.01369863,
                    0.05882353,
                    0.0,
                    0.02564103,
                    0.0,
                    0.0,
                ],  # noqa
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_mr_props_with_hs_by_cell(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
        expected = np.array(
            [
                [
                    0.10769231,
                    0.16923077,
                    0.27692308,
                    0.26153846,
                    0.15384615,
                    0.15384615,
                ],  # noqa
                [
                    0.11111111,
                    0.20634921,
                    0.31746032,
                    0.19047619,
                    0.15873016,
                    0.15873016,
                ],  # noqa
                [
                    0.09090909,
                    0.22727273,
                    0.31818182,
                    0.24242424,
                    0.12121212,
                    0.12121212,
                ],  # noqa
                [
                    0.10447761,
                    0.14925373,
                    0.25373134,
                    0.13432836,
                    0.17910448,
                    0.17910448,
                ],  # noqa
                [
                    0.07462687,
                    0.11940299,
                    0.19402985,
                    0.23880597,
                    0.1641791,
                    0.1641791,
                ],  # noqa
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_mr_props_with_hs_by_row(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
        expected = np.array(
            [
                [0.15555556, 0.24444444, 0.4, 0.37777778, 0.22222222, 0.22222222],
                [
                    0.16666667,
                    0.30952381,
                    0.47619048,
                    0.28571429,
                    0.23809524,
                    0.23809524,
                ],  # noqa
                [
                    0.13333333,
                    0.33333333,
                    0.46666667,
                    0.35555556,
                    0.17777778,
                    0.17777778,
                ],  # noqa
                [
                    0.18421053,
                    0.26315789,
                    0.44736842,
                    0.23684211,
                    0.31578947,
                    0.31578947,
                ],  # noqa
                [0.125, 0.2, 0.325, 0.4, 0.275, 0.275],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_mr_props_with_hs_by_col(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
        expected = np.array(
            [
                [
                    0.53846154,
                    0.6875,
                    0.62068966,
                    0.94444444,
                    0.55555556,
                    0.55555556,
                ],  # noqa
                [
                    0.58333333,
                    0.68421053,
                    0.64516129,
                    0.66666667,
                    0.71428571,
                    0.71428571,
                ],  # noqa
                [0.5, 0.78947368, 0.67741935, 0.76190476, 0.57142857, 0.57142857],
                [0.53846154, 0.58823529, 0.56666667, 0.5, 0.63157895, 0.63157895],
                [
                    0.45454545,
                    0.47058824,
                    0.46428571,
                    0.76190476,
                    0.61111111,
                    0.61111111,
                ],  # noqa
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_3d_pruning_indices(self):
        """Test pruning indices for a simple XYZ cube."""
        # Zeroth slice of the XYZ array:
        #
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  1  | 0  | False
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        #  True False True

        # Both rows and columns get pruned
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[1]])

        # Just columns get pruned
        transforms = {"columns_dimension": {"insertions": {}, "prune": True}}
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts, [[1], [0], [0], [1], [1], [0], [0], [1]]
        )

        # Just rows get pruned
        transforms = {"rows_dimension": {"insertions": {}, "prune": True}}
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[0, 1, 0]])

    def test_mr_x_ca_rows_margin(self):
        slice_ = Cube(CR.MR_X_CA_HS).partitions[0]
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(slice_.rows_margin, expected)
        slice_ = Cube(CR.MR_X_CA_HS).partitions[1]
        expected = np.array([4, 4, 4])
        np.testing.assert_array_equal(slice_.rows_margin, expected)
        slice_ = Cube(CR.MR_X_CA_HS).partitions[2]
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_ca_x_mr_margin(self):
        slice_ = Cube(CR.CA_X_MR_WEIGHTED_HS).partitions[0]
        expected = np.array([504, 215, 224, 76, 8, 439])
        np.testing.assert_array_equal(slice_.columns_base, expected)

    def test_ca_x_mr_margin_prune(self):
        # ---CA x MR---
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CA_X_MR_WEIGHTED_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.columns_base, np.array([504, 215, 224, 76, 8, 439])
        )
        assert slice_.table_name == "q1. Aftensmad: K\xf8d (svin/lam/okse)"

    def test_mr_x_cat_x_mr_pruning(self):
        # No pruning
        slice_ = Cube(CR.MR_X_CAT_X_MR_PRUNE).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [9, 7, 5, 0],
                [0, 5, 2, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        # With pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.MR_X_CAT_X_MR_PRUNE, transforms=transforms).partitions[0]
        # Last column is not pruned, because the not-selected base counts
        # (for that column) are not all zeros.
        np.testing.assert_array_equal(
            slice_.counts,
            [[9, 7, 5, 0], [0, 5, 2, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]],
        )

    def test_gender_x_weight_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.GENDER_X_WEIGHT, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, 208)

    def test_proportions_cat_x_mr_x_cat(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(
            CR.CAT_X_MR_X_CAT["slides"][0]["cube"], transforms=transforms
        ).partitions[0]

        # Test first slice
        expected = np.array(
            [
                [0.3647320622384293, 0.3539601689446188],
                [0.2217369551352075, 0.21179394663351137],
                [0.2892943293978049, 0.2755799995757032],
                [0.30472468951159837, 0.31183395927247587],
                [0.1564932915102434, 0.12122561350397994],
                [0.14341569357975512, 0.16573829062583212],
                [0.3088773171830892, 0.30399159711003093],
                [0.3531835200406305, 0.4001713397700346],
                [0.2572968713520773, 0.24375187975969445],
                [0.25130939319998297, 0.1977549624477041],
                [0.274565755319545, 0.2997170882672239],
                [0.32608867343144654, 0.344478827337916],
                [0.35921238787890847, 0.3513630660099009],
                [0.25634234169007164, 0.16894820580901035],
                [0.22099299650777843, 0.22466833670878553],
                [0.2276649518247445, 0.24565762260105783],
                [0.2643381997593724, 0.1982327504994244],
                [0.41491884119922506, 0.4258666971460735],
                [0.4623019656879477, 0.520868963921971],
                [0.35277296769118416, 0.2813894529707485],
                [0.4003762375617921, 0.42681767440587187],
                [0.25925087940264385, 0.2731916153395818],
                [0.4753330440584336, 0.43648850969829933],
                [0.28148528548474344, 0.24591762645117995],
                [0.49727220036895625, 0.5108530445282087],
                [0.32590772578924143, 0.359683660082846],
                [0.3595152413326164, 0.4049397928654146],
                [0.42108775668830034, 0.3454454870295823],
                [0.4205886117048783, 0.39122538735554757],
                [0.228770284468049, 0.20059146301631123],
                [0.24603034757739972, 0.2735752782805667],
                [0.3065494749862463, 0.32314540506421324],
                [0.27194030884117687, 0.30970380251451973],
                [0.2091262494207975, 0.22920517801957993],
                [0.31769765653105564, 0.28958158962721786],
                [0.3560762345705489, 0.3102687419303191],
                [0.4804715042882989, 0.5011391743289532],
                [0.3811599892254701, 0.4226226669501276],
                [0.41029213392178276, 0.1884401015918774],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

        # Test second slice
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(
            CR.CAT_X_MR_X_CAT["slides"][0]["cube"], transforms=transforms
        ).partitions[1]
        expected = np.array(
            [
                [0.4031214008509537, 0.4056176483118717],
                [0.24070198556407071, 0.25911741489783324],
                [0.3464300357360538, 0.33912070707907394],
                [0.3574620243397532, 0.37758989006965377],
                [0.14907456320910992, 0.15890701660953999],
                [0.20651100193920027, 0.1789634776591901],
                [0.30520247457440536, 0.3270678407142803],
                [0.3810285430516052, 0.3997309090998591],
                [0.34271319381187965, 0.17385655771473044],
                [0.3591305867483556, 0.2685531153107514],
                [0.2996020578719067, 0.29215934221779744],
                [0.3902291806673195, 0.39354067543725346],
                [0.40337866312317217, 0.4250345472210801],
                [0.32114152359818676, 0.30423847092340256],
                [0.2653311867224287, 0.27833063737964403],
                [0.313528046383254, 0.3000437600045656],
                [0.3920027268884396, 0.36933090371694116],
                [0.515781215492543, 0.4851941444303692],
                [0.5427173809468948, 0.5754703450096686],
                [0.3514753251895221, 0.3381463298582681],
                [0.40789566838261765, 0.43525540615386243],
                [0.3595684862225866, 0.38051785122829174],
                [0.556336290160639, 0.47918668411102733],
                [0.3006536550040242, 0.31770376362899333],
                [0.5132046910093269, 0.5548090283383379],
                [0.4409089206826546, 0.36616426510098665],
                [0.40390891699096854, 0.3888593109712533],
                [0.3914326716352874, 0.3346157095319079],
                [0.4423415709934932, 0.47752483308415145],
                [0.33543493750667275, 0.30087121770598385],
                [0.30409560492337334, 0.33096009035672747],
                [0.4028029720384424, 0.4497865293548307],
                [0.37991282964691514, 0.3817002730808065],
                [0.26937198075202085, 0.2530238979016483],
                [0.3367641290249356, 0.3210357156509789],
                [0.4376473666508847, 0.4044796591984694],
                [0.5986306705327854, 0.5886692367162286],
                [0.3493779725965881, 0.3533483607971598],
                [0.08962504168280223, 0.4352830423033842],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_partitions(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[0]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [24992768.29621058, 0.0, 26901938.09661558],
                [17298235.46427536, 44258027.19120625, 21174428.69540066],
            ],
        )
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[1]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [0.0, 19459910.91314028, 24053452.11581288],
                [48106904.23162583, 16648106.90423161, 22216035.63474388],
            ],
        )
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[2]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [21474773.60931439, 18272962.4838292, 0.0],
                [25808538.16300128, 23673997.41267791, 53751617.07632601],
            ],
        )


class Test_Nub(object):
    """Legacy unit-tests for 0D cube."""

    def test_mean_no_dims(self):
        cube = Cube(CR.ECON_MEAN_NO_DIMS)
        assert cube.description is None
        assert cube.name is None
        assert cube.missing == 0

        nub = cube.partitions[0]

        np.testing.assert_almost_equal(nub.means, np.array([49.095]))
        assert nub.ndim == 0
        np.testing.assert_almost_equal(nub.table_base, np.array([49.095]))
        np.testing.assert_array_equal(nub.unweighted_count, 1000)
