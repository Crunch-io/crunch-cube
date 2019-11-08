# encoding: utf-8

import numpy as np

from cr.cube.cubepart import CubePartition, _Slice
from cr.cube.cube import Cube
from cr.cube.enum import DIMENSION_TYPE as DT

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---


class Describe_Slice(object):
    """Integration-test suite for _Slice object."""

    def it_is_not_empty(self):
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        assert slice_.is_empty is False

    def it_is_empty(self):
        slice_ = Cube(CR.OM_SGP8334215_VN_2019_SEP_19).partitions[0]
        assert slice_.is_empty is True

    def it_loads_from_cat_x_cat_cube(self):
        cube = Cube(CR.CAT_X_CAT)
        slice_ = _Slice(cube, 0, None, None, 0)
        expected = np.array([[0.71428571, 0.28571429], [0.625, 0.375]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def it_provides_fills(self):
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        assert slice_.rows_dimension_fills == (None, None, None, None, None, None, None)

    def it_provides_missing(self):
        cube = Cube(CR.CAT_X_CAT)
        assert cube.missing == 5

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

        # With insertions (only row for now)
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        expected = [
            [0.47502103, 0.81547519, 0.34045416, 0.16820858, 0.0, 0.01631623],
            [0.24473593, 0.65688643, 0.4121505, 0.27407663, 0.0, 0.06903693],
            [0.515, 0.515, 0.0, 0.485, 0.0, 0.0],
            [0.27321912, 0.63390442, 0.3606853, 0.18575293, 0.0, 0.18034265],
            [0.19080605, 0.69080605, 0.5, 0.30919395, 0.0, 0.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.33333333, 0.66666667, 0.33333333, 0.33333333, 0.0, 0.0],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        assert slice_.inserted_row_idxs == (1,)
        assert slice_.inserted_column_idxs == (1,)
        assert slice_.name == "MaritalStatus"
        assert slice_.dimension_types == (DT.CAT, DT.CAT)
        assert slice_.ndim == 2
        assert slice_.table_name is None

        # Test zscores
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

        # Test pvals
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

        # Test column index
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

    def it_provides_base_counts(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_PRUNING_HS), 0, None, None, 0)
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
        assert slice_.table_base == 91

    def it_provides_various_names_and_labels(self):
        slice_ = _Slice(Cube(CR.CAT_X_CAT_PRUNING_HS), 0, None, None, 0)
        assert slice_.columns_dimension_name == "ShutdownBlame"
        assert slice_.rows_dimension_description == "What is your marital status?"
        assert slice_.rows_dimension_type == DT.CAT
        assert slice_.columns_dimension_type == DT.CAT

    def it_calculates_cat_x_cat_column_proportions(self):
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
        expected = [
            [0.77796143, 0.69805616, 0.61055807, 0.52882073, np.nan, 0.32659933],
            [0.1953168, 0.27401008, 0.360181, 0.41988366, np.nan, 0.67340067],
            [0.02837466, 0.01483081, 0.0, 0.05129561, np.nan, 0.0],
            [0.08347107, 0.1012239, 0.12066365, 0.10893707, np.nan, 0.67340067],
            [0.08347107, 0.15795536, 0.23951735, 0.25965098, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
            [0.02672176, 0.02793377, 0.02926094, 0.05129561, np.nan, 0.0],
        ]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

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
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
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
        slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
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


class Describe_Strand(object):
    """Integration-test suite for `cr.cube.cubepart._Strand` object."""

    def it_provides_nans_for_means_insertions(self):
        strand = CubePartition.factory(
            Cube(CR.CAT_WITH_MEANS_AND_INSERTIONS), 0, None, None, None, 0
        )
        np.testing.assert_almost_equal(
            strand.means, [19.85555556, 13.85416667, 52.78947368, np.nan, np.nan]
        )

    def it_is_not_empty(self):
        strand = CubePartition.factory(
            Cube(CR.CAT_WITH_MEANS_AND_INSERTIONS), 0, None, None, None, 0
        )
        assert strand.is_empty is False

    def it_is_empty(self):
        strand = CubePartition.factory(
            Cube(CR.OM_SGP8334215_VN_2019_SEP_19_STRAND), 0, None, None, None, 0
        )
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
