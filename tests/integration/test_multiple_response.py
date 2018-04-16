from unittest import TestCase
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube

from .fixtures import CAT_X_MR_SIMPLE
from .fixtures import CAT_X_MR_PRUNED_ROW
from .fixtures import CAT_X_MR_PRUNED_COL
from .fixtures import CAT_X_MR_PRUNED_ROW_COL
from .fixtures import MR_X_CAT_PRUNED_COL
from .fixtures import MR_X_CAT_PRUNED_ROW
from .fixtures import MR_X_CAT_PRUNED_ROW_COL
from .fixtures import SIMPLE_MR
from .fixtures import MR_X_CAT_PROFILES_STATS_WEIGHTED
from .fixtures import ARRAY_X_MR
from .fixtures import MR_X_SINGLE_WAVE
from .fixtures import CAT_X_MR_X_MR
from .fixtures import SELECTED_3_WAY_2
from .fixtures import SELECTED_3_WAY
from .fixtures import PROMPTED_AWARENESS


# pylint: disable=invalid-name
class TestMultipleResponse(TestCase):
    def test_labels_simple_mr_exclude_missing(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = [['Response #1', 'Response #2', 'Response #3']]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_simple_mr_include_missing(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = [
            ['Response #1', 'Response #2', 'Response #3']
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_simple_mr_exclude_missing(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = np.array([3, 4, 0])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_simple_mr_axis_none(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = np.array([5, 6, 6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_simple_mr(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = np.array([0.6, 0.6666667, 0.])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_simple_mr_prune(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = np.array([0.6, 0.6666667])
        actual = np.ma.compressed(cube.proportions(prune=True))
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_mr_exclude_missing(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = [
            ['rambutan', 'satsuma'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_mr_include_missing(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = [
            ['rambutan', 'satsuma', 'No Data'],
            ['dog', 'cat', 'wombat'],
        ]
        actual = cube.labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_as_array_cat_x_mr(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [12, 12, 12],
            [28, 22, 26],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_mr_pruned_row(self):
        cube = CrunchCube(CAT_X_MR_PRUNED_ROW)

        # Not pruned
        expected = np.array([
            [12, 12, 12],
            [0, 0, 0],
        ])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([[12, 12, 12]])
        actual = np.ma.compress_rows(cube.as_array(prune=True))
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_mr_pruned_col(self):
        cube = CrunchCube(CAT_X_MR_PRUNED_COL)

        # Not pruned
        actual = cube.as_array()
        expected = np.array([
            [6, 6, 0],
            [16, 12, 0],
        ])
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([
            [6, 6],
            [16, 12],
        ])
        actual = np.ma.compress_cols(cube.as_array(prune=True))
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_mr_pruned_row_col(self):
        cube = CrunchCube(CAT_X_MR_PRUNED_ROW_COL)

        # Not pruned
        actual = cube.as_array()
        expected = np.array([
            [6, 6, 0],
            [0, 0, 0],
        ])
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([[6, 6]])
        table = cube.as_array(prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_mr_x_cat_pruned_col(self):
        cube = CrunchCube(MR_X_CAT_PRUNED_COL)

        # Not pruned
        actual = cube.as_array()
        expected = np.array([
            [12, 0],
            [12, 0],
            [12, 0],
        ])
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([
            [12],
            [12],
            [12],
        ])
        actual = np.ma.compress_cols(cube.as_array(prune=True))
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_mr_x_cat_pruned_row(self):
        cube = CrunchCube(MR_X_CAT_PRUNED_ROW)

        # Not pruned
        actual = cube.as_array()
        expected = np.array([
            [6, 16],
            [6, 12],
            [0, 0],
        ])
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([
            [6, 16],
            [6, 12],
        ])
        actual = np.ma.compress_rows(cube.as_array(prune=True))
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_mr_x_cat_pruned_row_col(self):
        cube = CrunchCube(MR_X_CAT_PRUNED_ROW_COL)

        # Not pruned
        actual = cube.as_array()
        expected = np.array([
            [6, 0],
            [6, 0],
            [0, 0],
        ])
        np.testing.assert_array_equal(actual, expected)

        # Pruned
        expected = np.array([
            [6],
            [6],
        ])
        table = cube.as_array(prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_axis_none(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([80, 79, 70])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_mr_by_col(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([40, 34, 38])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_mr_by_cell(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [0.15, 0.15189873, 0.17142857],
            [0.35, 0.27848101, 0.37142857],
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_mr_by_col(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [.3, .3529412, .3157895],
            [.7, .6470588, .6842105],
        ])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_mr_by_row(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [0.42857143, 0.48, 0.52173913],
            [0.53846154, 0.40740741, 0.55319149],
        ])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_from_r_row_margin(self):
        cube = CrunchCube(MR_X_CAT_PROFILES_STATS_WEIGHTED)
        expected = np.array([
            [
                -1.465585354569577,
                3.704125875262655,
                3.823689449491973,
                1.53747452587281,
                2.584734165643072,
                -7.488143461076757,
                -0.248968750486873,
                0.794143540856786
            ],
            [
                1.465585354569564,
                -3.704125875262655,
                -3.823689449491981,
                -1.537474525872799,
                -2.584734165643066,
                7.488143461076757,
                0.248968750486873,
                -0.794143540856781
            ]
        ])
        actual = cube.standardized_residuals
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_row(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [.8571429, 1.1152941, .9610984],
            [1.0769231, .9466231, 1.019037],
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_cell(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [.8571429, 1.1152941, .9610984],
            [1.0769231, .9466231, 1.019037],
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_index_by_col(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        expected = np.array([
            [.8571429, 1.1152941, .9610984],
            [1.0769231, .9466231, 1.019037],
        ])
        actual = cube.index()
        np.testing.assert_almost_equal(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.mr_dim_ind', 2)
    def test_cat_x_mr_index_bad_direction(self):
        cube = CrunchCube(CAT_X_MR_SIMPLE)
        with self.assertRaises(ValueError):
            cube.index()

    def test_mr_x_single_wave(self):
        cube = CrunchCube(MR_X_SINGLE_WAVE)
        expected = np.array([
            308.32755712, 187.06825269, 424.82328071, 72.68885079,
            273.15993803, 467.62527785, 62.183386, 442.80441811,
            281.57825919, 0., 237.35065847, 233.19692455, 0., 0., 0., 0.,
            0., 0., 0., 38.05075633, 90.93234493, 123.22747266, 142.42909713,
        ])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_col(self):
        cube = CrunchCube(ARRAY_X_MR)
        expected = np.array([
            [0.5146153267487166, 0.04320534228100489, 0.5933354514113938],
            [0.4853846732512835, 0.9567946577189951, 0.4066645485886063],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_row(self):
        cube = CrunchCube(ARRAY_X_MR)
        expected = np.array([
            [0.41922353375674093, 0.03471395310157275, 0.5832027484767315],
            [0.5143557893611596, 1, 0.5199603338915276],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=2)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_array_x_mr_by_cell(self):
        cube = CrunchCube(ARRAY_X_MR)
        expected = np.array([
            [0.41922353375674093, 0.03471395310157275, 0.5832027484767315],
            [0.5143557893611596, 1, 0.5199603338915276],
        ])
        # Only compare the first slice (parity with whaam tests)
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_simple_mr_margin_by_col(self):
        cube = CrunchCube(SIMPLE_MR)
        expected = np.array([3, 4, 0])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_cat_x_mr_x_mr_proportions_by_row(self):
        cube = CrunchCube(CAT_X_MR_X_MR)
        # Set axis to 2 (and not 1), since 3D cube
        actual = cube.proportions(axis=2)
        expected = np.array([
            [[0.1159, 0.3597],
             [0.0197, 0.0604],
             [0.0192, 0.0582]],

            [[0.0159, 0.0094],
             [0.1182, 0.0625],
             [0.1142, 0.0623]],
        ])
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_mr_counts_pruned(self):
        cube = CrunchCube(CAT_X_MR_X_MR)
        cube.margin()
        # FIXME pruning doesn't work for 3d cubes, these expectations are wrong
        # FIXME (prune indices arrays should be vectors of scalars)
        # pruned_expected = [
        #     np.array([[[False, False], [False, False], [False, False]],
        #               [[False, False], [False, False], [False, False]]]),
        #     np.array([[[False, False], [False, False], [False, False]],
        #               [[False, False], [False, False], [False, False]]])
        # ]
        # pruned = cube.prune_indices()
        # self.assertEqual(len(pruned), len(pruned_expected))
        # for i, actual in enumerate(pruned):
        #     np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_cat_x_mr_x_mr_proportions_by_col(self):
        cube = CrunchCube(CAT_X_MR_X_MR)
        # Set axis to 1 (and not 0), since 3D cube
        actual = cube.proportions(axis=1, weighted=False)
        expected = np.array([
            [[0.166284074605452, 0.516068866571019],
             [0.0280267463366055, 0.0859297197325366],
             [0.0276657060518732, 0.0838616714697406]],

            [[0.0228120516499283, 0.0134863701578192],
             [0.168160478019633, 0.0889173424384692],
             [0.164553314121037, 0.0897694524495677]],
        ])
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_mr_proportions_by_cell(self):
        cube = CrunchCube(CAT_X_MR_X_MR)
        actual = cube.proportions()
        expected = np.array([
            [[0.05795, 0.17985],
             [0.00985, 0.0302],
             [0.0096, 0.0291]],

            [[0.00795, 0.0047],
             [0.0591, 0.03125],
             [0.0571, 0.03115]],
        ])
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_x_cat_x_cat_by_row(self):
        cube = CrunchCube(SELECTED_3_WAY_2)
        # Only compare 0 slice (parity with whaam tests)
        expected = np.array([
            [0.5923110874002918, 0.3758961399306439],
            [0, 0],
            [0.49431928922535223, 0.6091963925363675]
        ])
        actual = cube.proportions(axis=2)[0]
        # Since cube is 3D, row dim is 2 (instead of 1)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_col(self):
        cube = CrunchCube(SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        expected = np.array([[1, 0], [1, 0], [1, 0]])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis 0 represents column direction.
        actual = cube.proportions(axis=0)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_row(self):
        cube = CrunchCube(SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        expected = np.array([
            [0.0997975162008577, np.nan],
            [0.20327963774693497, np.nan],
            [0.3113417143573762, np.nan],
        ])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis 1 represents row direction.
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_mr_x_cat_by_cell(self):
        cube = CrunchCube(SELECTED_3_WAY)
        # Only take first slice (parity with whaam tests).
        # TODO: Check with @jonkeane, since R results are slightly
        # different. (It's using (0, 2) rather than (1, 2) axis).
        expected = np.array([
            [0.03326584, 0],
            [0.06775988, 0],
            [0.10378057, 0],
        ])
        # Since MR is 2nd dim, the cube is considered 2 dimensional,
        # and the axis None represents cell direction.
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_props_pruned(self):
        cube = CrunchCube(PROMPTED_AWARENESS)
        expected = np.array([
            9.70083312e-01, 9.53131845e-01, 9.64703914e-01,
            9.59703205e-01, 9.37891446e-01, 8.84137923e-01,
            7.77056917e-01, 7.15135296e-01, 9.03057657e-01,
            8.67103783e-01, 8.38011719e-01, 8.60897234e-01,
            7.68101070e-01, 7.59030477e-01, 8.66127931e-01,
            6.89111039e-01, 7.39338305e-01, 1.89895586e-01,
            1.95866187e-01, 8.90452848e-01, 6.10278144e-01,
            6.35237428e-01, 6.54874171e-01, 6.89736947e-01,
            2.31607423e-01, 4.44608376e-01, 6.06987388e-01,
            4.16165746e-01, 2.06262071e-01, 2.08512519e-01,
            1.59533129e-01, 1.86245154e-01, 1.01661334e-01,
            1.82235674e-01, 7.30060936e-01, 4.45912391e-01,
            4.87037442e-01, 1.29527814e-01, 4.95486986e-01,
            2.84392427e-01, 3.93962082e-01, 3.91279968e-01,
            8.96639874e-02, 9.50985735e-04, 1.35477929e-01,
            1.86531215e-01,
        ])
        actual = np.ma.compressed(cube.proportions(prune=True))
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_counts_not_pruned(self):
        cube = CrunchCube(PROMPTED_AWARENESS)
        expected = np.array([
            224833, 221990, 223560, 222923, 217586, 206164, 183147, 167720,
            209355, 201847, 193826, 198744, 180015, 174349, 200050, 160769,
            167969, 43193, 44339, 207539, 135973, 146002, 146789, 160692,
            53995, 95741, 135700, 91878, 48465, 48929, 35189, 42764,
            21194, 41422, 167652, 95676, 111961, 26137, 0, 0, 111760, 60761,
            87645, 85306, 18873, 178, 30461, 42843,
        ])
        # Use unweighted, because of the whole numbers (for comparison)
        actual = cube.as_array(weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_mr_counts_pruned(self):
        cube = CrunchCube(PROMPTED_AWARENESS)
        expected = np.array([
            224833, 221990, 223560, 222923, 217586, 206164, 183147, 167720,
            209355, 201847, 193826, 198744, 180015, 174349, 200050, 160769,
            167969, 43193, 44339, 207539, 135973, 146002, 146789, 160692,
            53995, 95741, 135700, 91878, 48465, 48929, 35189, 42764,
            21194, 41422, 167652, 95676, 111961, 26137, 111760, 60761,
            87645, 85306, 18873, 178, 30461, 42843,
        ])
        # Use unweighted, because of the whole numbers (for comparison)
        actual = cube.as_array(weighted=False, prune=True)
        np.testing.assert_array_equal(actual[~actual.mask], expected)

        pruned_expected = [
            np.array(
                [False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False,
                 False, False, True, True, False, False, False, False, False,
                 False, False, False])
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_mr_props_not_pruned(self):
        cube = CrunchCube(PROMPTED_AWARENESS)
        expected = np.array([
            9.70083312e-01, 9.53131845e-01, 9.64703914e-01,
            9.59703205e-01, 9.37891446e-01, 8.84137923e-01,
            7.77056917e-01, 7.15135296e-01, 9.03057657e-01,
            8.67103783e-01, 8.38011719e-01, 8.60897234e-01,
            7.68101070e-01, 7.59030477e-01, 8.66127931e-01,
            6.89111039e-01, 7.39338305e-01, 1.89895586e-01,
            1.95866187e-01, 8.90452848e-01, 6.10278144e-01,
            6.35237428e-01, 6.54874171e-01, 6.89736947e-01,
            2.31607423e-01, 4.44608376e-01, 6.06987388e-01,
            4.16165746e-01, 2.06262071e-01, 2.08512519e-01,
            1.59533129e-01, 1.86245154e-01, 1.01661334e-01,
            1.82235674e-01, 7.30060936e-01, 4.45912391e-01,
            4.87037442e-01, 1.29527814e-01, 0.00000000e+00,
            0.00000000e+00, 4.95486986e-01, 2.84392427e-01,
            3.93962082e-01, 3.91279968e-01, 8.96639874e-02,
            9.50985735e-04, 1.35477929e-01, 1.86531215e-01,
        ])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)
