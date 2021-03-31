# encoding: utf-8

"""Integration-test suite for multiple-response cube behaviors"""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR
from ..util import load_python_expression


def test_labels_simple_mr_exclude_missing():
    strand = Cube(CR.SIMPLE_MR).partitions[0]
    assert strand.row_labels.tolist() == ["Response #1", "Response #2", "Response #3"]


def test_as_array_simple_mr_exclude_missing():
    strand = Cube(CR.SIMPLE_MR).partitions[0]
    counts = strand.counts
    np.testing.assert_array_equal(counts, (3, 4, 0))


def test_margin_simple_mr_axis_none():
    strand = Cube(CR.SIMPLE_MR).partitions[0]
    assert strand.table_margin_range.tolist() == [5, 6]


def test_proportions_simple_mr():
    strand = Cube(CR.SIMPLE_MR).partitions[0]
    table_proportions = strand.table_proportions
    np.testing.assert_almost_equal(table_proportions, (0.6, 0.6666667, 0.0))


def test_std_dev_err_simple_mr():
    strand = Cube(CR.SIMPLE_MR).partitions[0]
    assert strand.table_proportion_stddevs == pytest.approx([0.4898979, 0.4714045, 0.0])
    assert strand.table_proportion_stderrs == pytest.approx([0.219089, 0.1924501, 0.0])


def test_1D_mr_with_means():
    strand = Cube(CR.MR_MEAN_FILT_WGTD).partitions[0]

    assert strand.ndim == 1
    assert strand.means == pytest.approx([3.7240515, 2.5784293, 2.2185933, 1.8653349])
    assert strand.unweighted_counts.tolist() == [336, 136, 248, 5216]
    assert strand.unweighted_bases.tolist() == [5837, 5837, 5837, 5837]
    assert strand.table_base_range.tolist() == [5837, 5837]


def test_deck_with_means():
    strand = Cube(CR.DECK_WITH_MEAN).partitions[0]

    assert strand.ndim == 1
    assert strand.table_base_range.tolist() == [588, 588]
    assert strand.table_margin_range == pytest.approx([585.086, 585.086])


def test_3D_with_means():
    slice_ = Cube(CR.MEANS_3D).partitions[0]
    np.testing.assert_almost_equal(
        slice_.means,
        [
            [-2.0, np.nan, np.nan, -1.0, 1.0],
            [-2.0, np.nan, np.nan, np.nan, 0.0],
            [-2.0, np.nan, 1.0, np.nan, -1.0],
        ],
    )
    assert slice_.table_base == 24
    np.testing.assert_array_equal(slice_.columns_base, [9, 0, 3, 3, 9])
    np.testing.assert_array_equal(slice_.rows_base, [9, 6, 9])


def test_mr_x_cat_with_means():
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_WITH_MEANS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.means,
        [
            [38.79868092, 37.91146097, 21.56682623, 28.90316683] + [np.nan] * 4,
            [12.36141735, 10.91788449, 8.55836344, -9.23336151] + [np.nan] * 4,
            [25.35566536, -1.87323918, -10.45832265, -19.00932593] + [np.nan] * 4,
            [-1.22773321, -7.99671664, -30.95431483, -18.03417097] + [np.nan] * 4,
            [-23.80382413, -26.69728288, -61.23218388, -48.49820981] + [np.nan] * 4,
            [19.6045351, -24.87663078, -52.08108014, 7.63833075] + [np.nan] * 4,
            [-26.98268155, -9.66231773, -90.91475189, -46.92610738] + [np.nan] * 4,
            [19.45552783, -27.48308453, -62.33543385, -39.83388919] + [np.nan] * 4,
            [20.59956268, 17.49911157, 6.29951372, 2.28572239] + [np.nan] * 4,
        ],
    )


def test_proportions_simple_mr_prune():
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    strand = Cube(CR.SIMPLE_MR, transforms=transforms).partitions[0]

    table_proportions = strand.table_proportions

    # Nothing actually gets pruned because not-selected are not 0
    np.testing.assert_almost_equal(table_proportions, (0.6, 0.6666667, 0.0))


def test_labels_cat_x_mr_exclude_missing():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    assert tuple(slice_.row_labels) == ("rambutan", "satsuma")
    assert tuple(slice_.column_labels) == ("dog", "cat", "wombat")


def test_as_array_cat_x_mr():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = [[12, 12, 12], [28, 22, 26]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_as_array_cat_x_mr_pruned_row():
    # Not pruned, with H&S
    slice_ = Cube(CR.CAT_X_MR_PRUNED_ROW).partitions[0]
    expected = [[12, 12, 12], [12, 12, 12], [0, 0, 0], [0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)

    # Pruned, with H&S
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_MR_PRUNED_ROW, transforms=transforms).partitions[0]
    expected = [[12, 12, 12], [12, 12, 12], [0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_as_array_cat_x_mr_pruned_col():
    # Not pruned
    slice_ = Cube(CR.CAT_X_MR_PRUNED_COL).partitions[0]
    expected = [[6, 6, 0], [6, 6, 0], [16, 12, 0], [16, 12, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_MR_PRUNED_COL, transforms=transforms).partitions[0]
    expected = [[6, 6, 0], [6, 6, 0], [16, 12, 0], [16, 12, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_as_array_cat_x_mr_pruned_row_col():
    # Not pruned
    slice_ = Cube(CR.CAT_X_MR_PRUNED_ROW_COL).partitions[0]
    expected = [[6, 6, 0], [6, 6, 0], [0, 0, 0], [0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_MR_PRUNED_ROW_COL, transforms=transforms).partitions[0]
    expected = [[6, 6, 0], [6, 6, 0], [0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_as_array_mr_x_cat_pruned_col():
    # Pruned and no H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_PRUNED_COL, transforms=transforms).partitions[0]
    np.testing.assert_array_equal(slice_.counts, [[12], [12], [12]])

    # Pruned with H&S
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_PRUNED_COL, transforms=transforms).partitions[0]
    np.testing.assert_array_equal(
        slice_.counts, [[12, 12, 0], [12, 12, 0], [12, 12, 0]]
    )


def test_as_array_mr_x_cat_pruned_row():
    # Not pruned
    slice_ = Cube(CR.MR_X_CAT_PRUNED_ROW).partitions[0]
    expected = [[6, 6, 16, 16], [6, 6, 12, 12], [0, 0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_PRUNED_ROW, transforms=transforms).partitions[0]
    expected = [[6, 6, 16, 16], [6, 6, 12, 12], [0, 0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_as_array_mr_x_cat_pruned_row_col():
    # Not pruned
    slice_ = Cube(CR.MR_X_CAT_PRUNED_ROW_COL).partitions[0]
    expected = [[6, 6, 0, 0], [6, 6, 0, 0], [0, 0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_PRUNED_ROW_COL, transforms=transforms).partitions[0]
    expected = [[6, 6, 0], [6, 6, 0], [0, 0, 0]]
    np.testing.assert_array_equal(slice_.counts, expected)


def test_margin_cat_x_mr_axis_none():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = np.array([80, 79, 70])
    np.testing.assert_array_equal(slice_.table_margin, expected)


def test_margin_cat_x_mr_by_col():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = np.array([40, 34, 38])
    np.testing.assert_array_equal(slice_.columns_margin, expected)


def test_proportions_cat_x_mr_by_cell():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = [[0.15, 0.15189873, 0.17142857], [0.35, 0.27848101, 0.37142857]]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_proportions_cat_x_mr_by_col():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = [[0.3, 0.3529412, 0.3157895], [0.7, 0.6470588, 0.6842105]]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_proportions_cat_x_mr_by_row():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = [[0.42857143, 0.48, 0.52173913], [0.53846154, 0.40740741, 0.55319149]]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_table_base_unpruned_cat_x_mr():
    slice_ = Cube(CR.CAT_X_MR).partitions[0]
    expected = np.array([80, 79, 70])
    np.testing.assert_array_equal(slice_.table_base_unpruned, expected)


def test_mr_x_cat_various_measures():
    slice_ = Cube(CR.MR_X_CAT_PROFILES_STATS_WEIGHTED).partitions[0]
    np.testing.assert_almost_equal(
        slice_.zscores, load_python_expression("mr-x-cat-weighted-zscores")
    )
    np.testing.assert_almost_equal(
        slice_.table_std_err, load_python_expression("mr-x-cat-weighted-table-std-err")
    )
    np.testing.assert_almost_equal(
        slice_.table_std_dev,
        load_python_expression("mr-x-cat-weighted-table-std-dev"),
    )
    np.testing.assert_almost_equal(
        slice_.column_std_dev, load_python_expression("mr-x-cat-weighted-col-std-dev")
    )
    np.testing.assert_almost_equal(
        slice_.column_std_err, load_python_expression("mr-x-cat-weighted-col-std-err")
    )
    np.testing.assert_almost_equal(
        slice_.column_proportions_moe,
        load_python_expression("mr-x-cat-weighted-col-moe"),
    )
    np.testing.assert_almost_equal(
        slice_.table_proportions_moe,
        load_python_expression("mr-x-cat-weighted-table-moe"),
    )


def test_mr_x_single_wave():
    slice_ = Cube(CR.MR_X_SINGLE_WAVE).partitions[0]
    expected = [
        308.32755712,
        187.06825269,
        424.82328071,
        72.68885079,
        273.15993803,
        467.62527785,
        62.183386,
        442.80441811,
        281.57825919,
        0.0,
        237.35065847,
        233.19692455,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        38.05075633,
        90.93234493,
        123.22747266,
        142.42909713,
    ]
    np.testing.assert_almost_equal(slice_.rows_margin, expected)


@pytest.mark.parametrize(
    "transforms, expectation",
    (
        (
            None,
            [[0.514615, 0.0432053, 0.593335], [0.4853846, 0.9567946, 0.4066645]],
        ),
        (
            # Hide MR col using positional element identifier
            {"columns_dimension": {"elements": {"1": {"hide": True}}}},
            [[0.0432053, 0.593335], [0.9567946, 0.4066645]],
        ),
        (
            # Hide MR col using subvaribale element identifier
            {"columns_dimension": {"elements": {"0007": {"hide": True}}}},
            [[0.0432053, 0.593335], [0.9567946, 0.4066645]],
        ),
    ),
)
def test_array_x_mr_by_col(transforms, expectation):
    slice_ = Cube(CR.CA_SUBVAR_X_CA_CAT_X_MR, transforms=transforms).partitions[0]

    assert slice_.column_proportions == pytest.approx(np.array(expectation))


def test_array_x_mr_by_row():
    slice_ = Cube(CR.CA_SUBVAR_X_CA_CAT_X_MR).partitions[0]
    expected = [
        [0.41922353375674093, 0.03471395310157275, 0.5832027484767315],
        [0.5143557893611596, 1, 0.5199603338915276],
    ]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_std_deviation_std_error_array_x_mr_by_row():
    slice_ = Cube(CR.CA_SUBVAR_X_CA_CAT_X_MR).partitions[0]

    np.testing.assert_array_almost_equal(
        slice_.table_std_dev,
        [[0.42525266, 0.13871218, 0.47011351], [0.41662646, 0.49570823, 0.41823255]],
    )
    np.testing.assert_array_almost_equal(
        slice_.table_std_err,
        [[0.02978762, 0.00971635, 0.03292998], [0.02918338, 0.03472281, 0.02929588]],
    )
    np.testing.assert_array_almost_equal(
        slice_.table_proportions_moe,
        [[0.05838266, 0.01904371, 0.06454158], [0.05719838, 0.06805546, 0.05741888]],
    )
    np.testing.assert_array_almost_equal(
        slice_.column_std_dev,
        [[0.49978635, 0.20331906, 0.49121125], [0.49978635, 0.20331906, 0.49121125]],
    )
    np.testing.assert_array_almost_equal(
        slice_.column_std_err,
        [[0.05158518, 0.02113084, 0.04615627], [0.05158518, 0.02113084, 0.04615627]],
    )
    np.testing.assert_array_almost_equal(
        slice_.column_proportions_moe,
        [[0.1011051, 0.04141569, 0.09046463], [0.1011051, 0.04141569, 0.09046463]],
    )


def test_array_x_mr_by_cell():
    slice_ = Cube(CR.CA_SUBVAR_X_CA_CAT_X_MR).partitions[0]
    expected = [[0.23701678, 0.01962626, 0.32972586], [0.223554, 0.43462911, 0.2259899]]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_simple_mr_margin_by_col():
    slice_ = Cube(CR.SIMPLE_MR).partitions[0]
    np.testing.assert_array_equal(slice_.rows_margin, [3, 4, 0])


def test_cat_x_mr_x_mr_proportions_by_row():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.row_proportions,
        [[0.19169699, 0.5949388], [0.19543651, 0.59920635], [0.19712526, 0.59753593]],
    )
    np.testing.assert_array_equal(
        slice_.unweighted_counts, [[1159, 3597], [197, 604], [192, 582]]
    )
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.row_proportions,
        [[0.17207792, 0.1017316], [0.1963129, 0.10380335], [0.19141804, 0.10442508]],
    )
    np.testing.assert_array_equal(
        slice_.unweighted_counts, [[159, 94], [1182, 625], [1142, 623]]
    )


def test_cat_x_mr_x_mr_row_std_err():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]

    row_std_err = slice_.row_std_err

    assert row_std_err.tolist() == [
        [0.005062449141597438, 0.006313386789860505],
        [0.012489721142934502, 0.015435416534978305],
        [0.012747222182198531, 0.015713252542485612],
    ]


def test_cat_x_mr_x_mr_column_std_err():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]

    column_std_err = slice_.column_std_err

    assert column_std_err.tolist() == [
        [0.011171263445095775, 0.006336762995064969],
        [0.006945529504027231, 0.0039102714263034456],
        [0.006866798249349935, 0.0038462753714309968],
    ]


def test_cat_x_mr_x_mr_table_std_err():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]

    table_std_err = slice_.table_std_err

    assert table_std_err.tolist() == [
        [0.003201049671592117, 0.004799123982561817],
        [0.0013896729831150924, 0.0023822644689454613],
        [0.0013722740251130602, 0.0023412125063735673],
    ]


def test_cat_x_mr_x_mr_pruned_rows():
    # Not pruned
    slice_ = Cube(CR.CAT_X_MR_X_MR_PRUNED_ROWS).partitions[0]
    np.testing.assert_array_equal(slice_.counts, [[0, 2, 2], [1, 3, 2], [0, 0, 0]])
    slice_ = Cube(CR.CAT_X_MR_X_MR_PRUNED_ROWS).partitions[1]
    np.testing.assert_array_equal(slice_.counts, [[3, 3, 6], [0, 3, 4], [0, 0, 0]])

    # Pruned
    transforms = {"rows_dimension": {"prune": True}}
    slice_ = Cube(CR.CAT_X_MR_X_MR_PRUNED_ROWS, transforms=transforms).partitions[0]
    np.testing.assert_array_equal(slice_.counts, [[0, 2, 2], [1, 3, 2]])
    slice_ = Cube(CR.CAT_X_MR_X_MR_PRUNED_ROWS, transforms=transforms).partitions[1]
    np.testing.assert_array_equal(slice_.counts, [[3, 3, 6], [0, 3, 4]])


def test_cat_x_mr_x_mr_pruned_cols():
    # Not pruned
    slice_ = Cube(CR.CAT_X_MR_X_MR_EMPTY_COLS).partitions[0]
    expected = [
        [1.42180119, 5.67259693, 0.00000000, 0.0],
        [5.96105631, 1.46479350, 22.51724162, 0.0],
        [1.79182255, 1.19879791, 23.60391651, 0.0],
        [4.67364825, 8.02124010, 93.65643860, 0.0],
        [16.0107376, 13.06260544, 206.93592858, 0.0],
    ]
    np.testing.assert_almost_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_MR_X_MR_EMPTY_COLS, transforms=transforms).partitions[0]
    expected = [
        [1.42180119, 5.67259693, 0.0],
        [5.96105631, 1.46479350, 22.51724162],
        [1.79182255, 1.19879791, 23.60391651],
        [4.67364825, 8.02124010, 93.65643860],
        [16.0107376, 13.06260544, 206.93592858],
    ]
    np.testing.assert_almost_equal(slice_.counts, expected)


def test_cat_x_mr_x_mr_proportions_by_col():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.column_proportions,
        [[0.60553814, 0.60372608], [0.10292581, 0.1013763], [0.10031348, 0.09768379]],
    )
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.column_proportions,
        [[0.08141321, 0.09003831], [0.60522273, 0.598659], [0.58474142, 0.5967433]],
    )


def test_cat_x_mr_x_mr_proportions_by_cell():
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.table_proportions, [[0.1159, 0.3597], [0.0197, 0.0604], [0.0192, 0.0582]]
    )
    slice_ = Cube(CR.CAT_X_MR_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.table_proportions, [[0.0159, 0.0094], [0.1182, 0.0625], [0.1142, 0.0623]]
    )


def test_mr_x_cat_x_cat_by_col():
    # TODO: Check expectations with Mike and Jon
    slice_ = Cube(CR.SELECTED_3WAY_2_FILLEDMISSING).partitions[0]
    expected = [
        [0.5923110874002918, 0.3758961399306439],
        [0, 0],
        [0.49431928922535223, 0.6091963925363675],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_cat_x_mr_x_cat_by_col():
    slice_ = Cube(CR.SELECTED_3WAY_FILLEDMISSING).partitions[0]
    expected = [
        [0.0997975162008577, np.nan],
        [0.20327963774693497, np.nan],
        [0.3113417143573762, np.nan],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_cat_x_mr_x_cat_by_row():
    slice_ = Cube(CR.SELECTED_3WAY_FILLEDMISSING).partitions[0]
    expected = [[1, 0], [1, 0], [1, 0]]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_cat_x_mr_x_cat_by_cell():
    slice_ = Cube(CR.SELECTED_3WAY_FILLEDMISSING).partitions[0]
    expected = [
        [0.0997975162008577, 0],
        [0.20327963774693497, 0],
        [0.3113417143573762, 0],
    ]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_mr_props_pruned():
    expected = (
        9.70083312e-01,
        9.53131845e-01,
        9.64703914e-01,
        9.59703205e-01,
        9.37891446e-01,
        8.84137923e-01,
        7.77056917e-01,
        7.15135296e-01,
        9.03057657e-01,
        8.67103783e-01,
        8.38011719e-01,
        8.60897234e-01,
        7.68101070e-01,
        7.59030477e-01,
        8.66127931e-01,
        6.89111039e-01,
        7.39338305e-01,
        1.89895586e-01,
        1.95866187e-01,
        8.90452848e-01,
        6.10278144e-01,
        6.35237428e-01,
        6.54874171e-01,
        6.89736947e-01,
        2.31607423e-01,
        4.44608376e-01,
        6.06987388e-01,
        4.16165746e-01,
        2.06262071e-01,
        2.08512519e-01,
        1.59533129e-01,
        1.86245154e-01,
        1.01661334e-01,
        1.82235674e-01,
        7.30060936e-01,
        4.45912391e-01,
        4.87037442e-01,
        1.29527814e-01,
        0.00000000e00,
        0.00000000e00,
        4.95486986e-01,
        2.84392427e-01,
        3.93962082e-01,
        3.91279968e-01,
        8.96639874e-02,
        9.50985735e-04,
        1.35477929e-01,
        1.86531215e-01,
    )
    # Not pruned
    strand = Cube(CR.PROMPTED_AWARENESS).partitions[0]
    np.testing.assert_almost_equal(strand.table_proportions, expected)
    # Pruned (nothing gets pruned because of not-selected != 0)
    transforms = {"columns_dimension": {"prune": True}}
    strand = Cube(CR.PROMPTED_AWARENESS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(strand.table_proportions, expected)


def test_mr_counts_not_pruned():
    expected = [
        224833,
        221990,
        223560,
        222923,
        217586,
        206164,
        183147,
        167720,
        209355,
        201847,
        193826,
        198744,
        180015,
        174349,
        200050,
        160769,
        167969,
        43193,
        44339,
        207539,
        135973,
        146002,
        146789,
        160692,
        53995,
        95741,
        135700,
        91878,
        48465,
        48929,
        35189,
        42764,
        21194,
        41422,
        167652,
        95676,
        111961,
        26137,
        0,
        0,
        111760,
        60761,
        87645,
        85306,
        18873,
        178,
        30461,
        42843,
    ]
    # Not pruned
    slice_ = Cube(CR.PROMPTED_AWARENESS).partitions[0]
    np.testing.assert_array_equal(slice_.unweighted_counts, expected)
    # Pruned (actuall nothing is pruned because not-selected != 0)
    transforms = {"columns_dimension": {"prune": True}}
    slice_ = Cube(CR.PROMPTED_AWARENESS, transforms=transforms).partitions[0]
    np.testing.assert_array_equal(slice_.unweighted_counts, expected)


def test_mr_x_cat_x_mr_pruned_rows():
    # Not pruned
    slice_ = Cube(CR.MR_X_CAT_X_MR).partitions[3]
    expected = [
        [4.67364825, 18.28952353, 8.07855047, 14.86987594],
        [8.02124010, 17.29617716, 7.15665312, 15.44355489],
        [93.6564386, 150.46443416, 96.56536588, 188.31770695],
        [0.0, 0.0, 0.0, 0.0],
    ]
    np.testing.assert_almost_equal(slice_.counts, expected)

    # Pruned
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.MR_X_CAT_X_MR, transforms=transforms).partitions[3]
    expected = [
        [4.67364825, 18.28952353, 8.07855047, 14.86987594],
        [8.02124010, 17.29617716, 7.15665312, 15.44355489],
        [93.6564386, 150.46443416, 96.56536588, 188.31770695],
    ]
    np.testing.assert_almost_equal(slice_.counts, expected)


def test_mr_x_num_with_means_pruned():
    transforms = {"columns_dimension": {"prune": True}}
    slice_ = Cube(CR.MEANS_MR_X_CAT, transforms=transforms).partitions[0]
    expected = [
        [
            38.79868092168848,
            37.911460968343384,
            21.566826228784073,
            28.903166828677023,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            12.361417346615546,
            10.917884486901682,
            8.55836343660059,
            -9.233361511954936,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            25.355665361847333,
            -1.8732391808022093,
            -10.458322652141625,
            -19.009325927555476,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            -1.227733208698524,
            -7.996716641982194,
            -30.95431482676578,
            -18.03417096792156,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            -23.803824133140722,
            -26.697282878138527,
            -61.23218387646827,
            -48.49820981263205,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            19.60453509722427,
            -24.876630784866492,
            -52.08108013616227,
            7.638330747500843,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            -26.98268155304967,
            -9.662317734860315,
            -90.91475189122735,
            -46.92610737983078,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            19.45552783488183,
            -27.48308452819968,
            -62.33543385309548,
            -39.83388919377415,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            20.599562677755262,
            17.499111571449458,
            6.299513722727599,
            2.285722391598358,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    ]
    np.testing.assert_almost_equal(slice_.means, expected)


def test_mr_x_num_with_means_not_pruned():
    slice_ = Cube(CR.MEANS_MR_X_CAT).partitions[0]
    expected = [
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
    np.testing.assert_almost_equal(slice_.means, expected)


def test_mr_x_num_rows_margin():
    slice_ = Cube(CR.MEANS_MR_X_CAT).partitions[0]
    expected = [4805, 3614, 1156, 1200, 644, 258, 167, 170, 11419]
    np.testing.assert_array_equal(slice_.rows_base, expected)


def test_mr_x_num_cols_margin_not_pruned_unweighted():
    slice_ = Cube(CR.MEANS_MR_X_CAT).partitions[0]
    expected = [1728, 1523, 1570, 1434, 1459, 1429, 1461, 1432, 0, 0, 0, 0]
    np.testing.assert_array_equal(slice_.columns_base[0], expected)


def test_mr_x_num_cols_margin_pruned_unweighted():
    transforms = {"columns_dimension": {"prune": True}}
    slice_ = Cube(CR.MEANS_MR_X_CAT, transforms=transforms).partitions[0]
    expected = [1728, 1523, 1570, 1434, 1459, 1429, 1461, 1432]
    np.testing.assert_array_equal(slice_.columns_base[0], expected)


def test_num_x_mr_props_by_row():
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.AGE_X_ACCRPIPE, transforms=transforms).partitions[0]
    expected = [
        [0.33333333, 0.44444444, 0.18518519, 0.03703704],
        [0.512, 0.24, 0.208, 0.04],
        [0.37234043, 0.35106383, 0.21276596, 0.06382979],
        [0.36956522, 0.36141304, 0.22826087, 0.04076087],
        [0.32666667, 0.40888889, 0.22666667, 0.03777778],
        [0.29718004, 0.42516269, 0.23210412, 0.04555315],
        [0.2633833, 0.41541756, 0.27623126, 0.04496788],
        [0.23193916, 0.42205323, 0.26996198, 0.07604563],
        [0.25373134, 0.4212272, 0.24378109, 0.08126036],
        [0.26359833, 0.41004184, 0.28242678, 0.04393305],
        [0.27687296, 0.44625407, 0.21824104, 0.05863192],
        [0.28571429, 0.4516129, 0.21198157, 0.05069124],
        [0.29166667, 0.39583333, 0.23958333, 0.07291667],
        [0.25925926, 0.40740741, 0.18518519, 0.14814815],
        [0.5, 0.33333333, 0.16666667, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_num_x_mr_props_by_col():
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.AGE_X_ACCRPIPE, transforms=transforms).partitions[0]
    expected = [
        [0.00707547, 0.00676437, 0.00472144, 0.00413223],
        [0.05031447, 0.01691094, 0.02455146, 0.02066116],
        [0.05503145, 0.03720406, 0.03777148, 0.04958678],
        [0.10691824, 0.07497182, 0.07932011, 0.06198347],
        [0.11556604, 0.10372041, 0.09631728, 0.07024793],
        [0.1077044, 0.11048478, 0.10103872, 0.08677686],
        [0.09669811, 0.10935738, 0.12181303, 0.08677686],
        [0.09591195, 0.12514092, 0.13408876, 0.16528926],
        [0.12028302, 0.14317926, 0.1388102, 0.20247934],
        [0.0990566, 0.11048478, 0.12747875, 0.08677686],
        [0.0668239, 0.07722661, 0.06326723, 0.07438017],
        [0.04874214, 0.05524239, 0.0434372, 0.04545455],
        [0.02201258, 0.02142052, 0.0217186, 0.02892562],
        [0.00550314, 0.00620068, 0.00472144, 0.01652893],
        [0.00235849, 0.0011274, 0.00094429, 0.0],
        [0.0, 0.0005637, 0.0, 0.0],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_num_x_mr_props_by_cell():
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.AGE_X_ACCRPIPE, transforms=transforms).partitions[0]
    expected = [
        [0.00207039, 0.00276052, 0.00115022, 0.00023004],
        [0.0147228, 0.00690131, 0.00598114, 0.00115022],
        [0.01610306, 0.01518288, 0.00920175, 0.00276052],
        [0.03128594, 0.03059581, 0.01932367, 0.00345066],
        [0.03381643, 0.04232804, 0.02346446, 0.00391074],
        [0.03151599, 0.04508857, 0.02461468, 0.00483092],
        [0.02829538, 0.04462848, 0.02967564, 0.00483092],
        [0.02806533, 0.0510697, 0.03266621, 0.00920175],
        [0.03519669, 0.0584311, 0.03381643, 0.01127214],
        [0.02898551, 0.04508857, 0.0310559, 0.00483092],
        [0.01955372, 0.03151599, 0.01541293, 0.00414079],
        [0.01426271, 0.02254428, 0.01058201, 0.00253048],
        [0.00644122, 0.00874166, 0.00529101, 0.00161031],
        [0.00161031, 0.00253048, 0.00115022, 0.00092017],
        [0.00069013, 0.00046009, 0.00023004, 0.0],
        [0.0, 0.00023004, 0.0, 0.0],
    ]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_cat_x_mr_x_cat_missing_proportions_by_cell():
    slice_ = Cube(CR.CAT_X_MR_X_CAT_MISSING).partitions[0]
    expected = [
        [
            0.07211986,
            0.00000000,
            0.03422948,
            0.10268843,
            0.00000000,
            0.07184337,
            0.03435395,
            0.10659828,
            0.07013931,
            0.03422948,
            0.07171890,
            0.03578536,
            0.07224433,
        ],
        [
            0.09633168,
            0.07100362,
            0.00000000,
            0.06757349,
            0.03352949,
            0.06731444,
            0.03218831,
            0.06425999,
            0.06705898,
            0.09976180,
            0.03512613,
            0.03352949,
            0.06757349,
        ],
        [
            0.03223990,
            0.03555871,
            0.03212309,
            0.06768180,
            0.10100556,
            0.03223990,
            0.03223990,
            0.03223990,
            0.09794622,
            0.03223990,
            0.03212309,
            0.03358323,
            0.03212309,
        ],
        [
            0.03084476,
            0.06828734,
            0.06462713,
            0.09583320,
            0.09698609,
            0.09569622,
            0.03095693,
            0.06498843,
            0.06309156,
            0.06510060,
            0.06462713,
            0.06602916,
            0.03224680,
        ],
        [
            0.06891722,
            0.03807479,
            0.03767189,
            0.11014275,
            0.03452115,
            0.06904229,
            0.03452115,
            0.03452115,
            0.10631513,
            0.03452115,
            0.03767189,
            0.07363141,
            0.03439607,
        ],
    ]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_cat_x_mr_x_cat_missing_proportions_by_row():
    slice_ = Cube(CR.CAT_X_MR_X_CAT_MISSING).partitions[0]
    expected = [
        [
            0.10215990,
            0.00000000,
            0.04848706,
            0.14546118,
            0.00000000,
            0.10176825,
            0.04866338,
            0.15099960,
            0.09935439,
            0.04848706,
            0.10159194,
            0.05069102,
            0.10233622,
        ],
        [
            0.13101878,
            0.09657060,
            0.00000000,
            0.09190535,
            0.04560278,
            0.09155302,
            0.04377867,
            0.08739873,
            0.09120557,
            0.13568403,
            0.04777434,
            0.04560278,
            0.09190535,
        ],
        [
            0.05433591,
            0.05992931,
            0.05413904,
            0.11406835,
            0.17023095,
            0.05433591,
            0.05433591,
            0.05433591,
            0.16507485,
            0.05433591,
            0.05413904,
            0.05659990,
            0.05413904,
        ],
        [
            0.03674991,
            0.08136077,
            0.07699981,
            0.11418021,
            0.11555381,
            0.11401700,
            0.03688355,
            0.07743029,
            0.07517027,
            0.07756393,
            0.07699981,
            0.07867027,
            0.03842036,
        ],
        [
            0.09652974,
            0.05332992,
            0.05276559,
            0.15427279,
            0.04835246,
            0.09670493,
            0.04835246,
            0.04835246,
            0.14891158,
            0.04835246,
            0.05276559,
            0.10313274,
            0.04817727,
        ],
    ]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_cat_x_mr_x_cat_missing_proportions_by_col():
    slice_ = Cube(CR.CAT_X_MR_X_CAT_MISSING).partitions[0]
    expected = [
        [
            0.67814114,
            np.nan,
            0.47727273,
            0.57668021,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.47461929,
            1.0,
            1.0,
            0.66874513,
        ],
        [
            1.0,
            1.0,
            0.0,
            0.50139377,
            1.0,
            1.0,
            1.0,
            0.64413423,
            0.67567568,
            1.0,
            0.52272727,
            1.0,
            0.50696106,
        ],
        [
            0.24415882,
            1.0,
            0.47727273,
            0.50139377,
            1.0,
            0.5,
            1.0,
            0.47552448,
            1.0,
            1.0,
            0.47727273,
            1.0,
            0.2406135,
        ],
        [
            0.49909256,
            1.0,
            1.0,
            0.73936493,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.25155048,
        ],
        [
            0.64413423,
            1.0,
            1.0,
            0.61554653,
            0.47817837,
            1.0,
            1.0,
            0.5,
            1.0,
            0.47552448,
            0.52272727,
            1.0,
            0.49909256,
        ],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_cat_by_mr_hs_col_percentage():
    slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
    expected = [
        [
            0.44079255048452126,
            0.12706996944659374,
            0.02245084839681615,
            0.03842827050449689,
            0.06423004017489264,
        ],
        [
            0.19275310260052075,
            0.17758354354978947,
            0.15543673007827843,
            0.11799738543815302,
            0.19460845018411604,
        ],
        [
            0.6335456530850421,
            0.3046535129963832,
            0.1778875784750946,
            0.1564256559426499,
            0.2588384903590087,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.13200554712733142,
            0.35154979021049976,
            0.40588130853850274,
            0.35971868142125785,
            0.3444274246321915,
        ],
        [
            0.23444879978762653,
            0.3437966967931169,
            0.41623111298640264,
            0.48385566263609225,
            0.3967340850087999,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.36645434691495793,
            0.6953464870036167,
            0.8221124215249054,
            0.8435743440573501,
            0.7411615096409914,
        ],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_cat_by_mr_hs_row_percentage():
    slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
    expected = [
        [
            0.6399160598631669,
            0.5710629087411291,
            0.23101895600507977,
            0.6728815025053523,
            0.7820669403643355,
        ],
        [
            0.18579712140082952,
            0.3079658200088574,
            0.47698573451339377,
            0.6856928028950228,
            0.8309421217908072,
        ],
        [
            0.3670032197169821,
            0.38122252445013116,
            0.4204835796699004,
            0.6825005266236387,
            0.8182527171506743,
        ],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [
            0.07093259641440403,
            0.3229818264527783,
            0.5550939883964581,
            0.7966136657386232,
            0.7825790252579282,
        ],
        [
            0.1179106652004235,
            0.31211928726996235,
            0.510266054902361,
            0.8563898792610921,
            0.799644741373273,
        ],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [
            0.09519878824704209,
            0.3175182147280742,
            0.5314553657872426,
            0.8298369139048698,
            0.791622434443332,
        ],
    ]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_cat_by_mr_hs_cell_percentage():
    slice_ = Cube(CR.CAT_HS_X_MR).partitions[0]
    expected = [
        [
            0.07905704201278009,
            0.042511244168882356,
            0.011396586674371037,
            0.03084751530784915,
            0.05127790466367509,
        ],
        [
            0.03457066167210144,
            0.05941055477622783,
            0.07890339533757364,
            0.09472000966485485,
            0.15536520805707787,
        ],
        [
            0.11362770368488154,
            0.10192179894511019,
            0.09029998201194467,
            0.125567524972704,
            0.20664311272075297,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.023675463829173602,
            0.11761094327986822,
            0.20603504288604813,
            0.28875688096249497,
            0.2749728413021995,
        ],
        [
            0.042048869914617856,
            0.11501714673796347,
            0.21128885073190257,
            0.388405326703676,
            0.3167317431612616,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.06572433374379145,
            0.2326280900178317,
            0.4173238936179507,
            0.677162207666171,
            0.5917045844634612,
        ],
    ]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_mr_by_cat_hs_col_percentage():
    slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
    expected = [
        [
            0.6399160598631669,
            0.18579712140082952,
            0.3670032197169821,
            np.nan,
            0.07093259641440403,
            0.1179106652004235,
            np.nan,
            0.09519878824704209,
        ],
        [
            0.5710629087411291,
            0.3079658200088574,
            0.38122252445013116,
            np.nan,
            0.3229818264527783,
            0.31211928726996235,
            np.nan,
            0.3175182147280742,
        ],
        [
            0.23101895600507977,
            0.47698573451339377,
            0.4204835796699004,
            np.nan,
            0.5550939883964581,
            0.510266054902361,
            np.nan,
            0.5314553657872426,
        ],
        [
            0.6728815025053523,
            0.6856928028950228,
            0.6825005266236387,
            np.nan,
            0.7966136657386232,
            0.8563898792610921,
            np.nan,
            0.8298369139048698,
        ],
        [
            0.7820669403643355,
            0.8309421217908072,
            0.8182527171506743,
            np.nan,
            0.7825790252579282,
            0.799644741373273,
            np.nan,
            0.791622434443332,
        ],
    ]
    np.testing.assert_almost_equal(slice_.column_proportions, expected)


def test_mr_by_cat_hs_row_percentage():
    slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
    expected = [
        [
            0.44079255048452126,
            0.19275310260052075,
            0.633545653085042,
            0.0,
            0.13200554712733142,
            0.23444879978762653,
            0.0,
            0.366454346914958,
        ],
        [
            0.12706996944659374,
            0.17758354354978947,
            0.3046535129963832,
            0.0,
            0.35154979021049976,
            0.3437966967931169,
            0.0,
            0.6953464870036167,
        ],
        [
            0.02245084839681615,
            0.15543673007827843,
            0.17788757847509457,
            0.0,
            0.40588130853850274,
            0.41623111298640264,
            0.0,
            0.8221124215249053,
        ],
        [
            0.03842827050449689,
            0.11799738543815302,
            0.1564256559426499,
            0.0,
            0.35971868142125785,
            0.48385566263609225,
            0.0,
            0.8435743440573501,
        ],
        [
            0.06423004017489264,
            0.19460845018411604,
            0.2588384903590087,
            0.0,
            0.3444274246321915,
            0.3967340850087999,
            0.0,
            0.7411615096409915,
        ],
    ]
    np.testing.assert_almost_equal(slice_.row_proportions, expected)


def test_mr_by_cat_hs_cell_percentage():
    slice_ = Cube(CR.MR_X_CAT_HS_MT).partitions[0]
    expected = [
        [
            0.07905704201278009,
            0.03457066167210144,
            0.11362770368488152,
            0.0,
            0.023675463829173602,
            0.042048869914617856,
            0.0,
            0.06572433374379145,
        ],
        [
            0.042511244168882356,
            0.05941055477622783,
            0.10192179894511019,
            0.0,
            0.11761094327986822,
            0.11501714673796347,
            0.0,
            0.2326280900178317,
        ],
        [
            0.011396586674371037,
            0.07890339533757364,
            0.09029998201194467,
            0.0,
            0.20603504288604813,
            0.21128885073190257,
            0.0,
            0.4173238936179507,
        ],
        [
            0.03084751530784915,
            0.09472000966485485,
            0.125567524972704,
            0.0,
            0.28875688096249497,
            0.388405326703676,
            0.0,
            0.677162207666171,
        ],
        [
            0.05127790466367509,
            0.15536520805707787,
            0.20664311272075295,
            0.0,
            0.2749728413021995,
            0.3167317431612616,
            0.0,
            0.5917045844634611,
        ],
    ]
    np.testing.assert_almost_equal(slice_.table_proportions, expected)


def test_mr_x_cat_min_base_size_mask():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }

    # Table margin evaluates to:
    #
    # array([176.36555176, 211.42058767, 247.74073787, 457.05095566, 471.93176847])
    #
    # We thus choose the min base size to be 220, and expeect it to broadcast across
    # columns (in the row direction, i.e. axis=1), sincee the MR is what won't be
    # collapsed after doing the base calculation in the table direction.
    slice_ = Cube(CR.MR_X_CAT_HS_MT, transforms=transforms, mask_size=220).partitions[0]

    expected = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(slice_.min_base_size_mask.table_mask, expected)

    # Column margin evaluates to:
    #
    # np.array(
    #     [
    #         [15, 24, 0, 57, 69, 0],
    #         [15, 34, 0, 75, 86, 0],
    #         [13, 37, 0, 81, 111, 0],
    #         [20, 50, 0, 159, 221, 0],
    #         [32, 69, 0, 167, 208, 0],
    #     ]
    # )
    #
    # We thus choose the min base size to be 30, and expeect it to not be broadcast.
    slice_ = Cube(CR.MR_X_CAT_HS_MT, transforms=transforms, mask_size=30).partitions[0]

    expected_column_mask = np.array(
        [
            [True, True, True, False, False, True],
            [True, False, True, False, False, True],
            [True, False, True, False, False, True],
            [True, False, True, False, False, True],
            [False, False, True, False, False, True],
        ]
    )
    np.testing.assert_array_equal(
        slice_.min_base_size_mask.column_mask, expected_column_mask
    )

    # Row margin evaluates to:
    #
    # np.array([31.63152104, 70.73073413, 125.75911351, 366.88839144, 376.76564059])
    #
    # We thus choose the min base size to be 80, and expeect it to broadcast across
    # columns (in the row direction, i.e. axis=1), sincee the MR is what won't be
    # collapsed after doing the base calculation in the row direction.
    slice_ = Cube(CR.MR_X_CAT_HS_MT, transforms=transforms, mask_size=80).partitions[0]

    expected_row_mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(slice_.min_base_size_mask.row_mask, expected_row_mask)


def test_cat_x_mr_min_base_size_mask():
    # Table margin evaluates to:
    #
    # array([80, 79, 70])
    #
    # We thus choose the min base size to be 75, and expeect it to broadcast across
    # rows (in the col direction, i.e. axis=0), sincee the MR is what won't be
    # collapsed after doing the base calculation in the table direction.
    expected = np.array([[False, False, True], [False, False, True]])
    slice_ = Cube(CR.CAT_X_MR, mask_size=75).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.table_mask, expected)

    # Column margin evaluates to:
    #
    # np.array([40, 34, 38])
    #
    # We thus choose the min base size to be 35, and expeect it to broadcast across
    # rows (in the col direction, i.e. axis=0), sincee the MR is what won't be
    # collapsed after doing the base calculation in the table direction.
    expected = np.array([[False, True, False], [False, True, False]])
    slice_ = Cube(CR.CAT_X_MR, mask_size=35).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.column_mask, expected)

    # Row margin evaluates to:
    #
    # np.array([[28, 25, 23], [52, 54, 47]])
    #
    # We thus choose the min base size to be 25, and expeect it to not be broadcast
    expected_row_mask = np.array([[False, False, True], [False, False, False]])
    slice_ = Cube(CR.CAT_X_MR, mask_size=25).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.row_mask, expected_row_mask)


def test_mr_x_mr_min_base_size_mask():
    # Table margin evaluates to:
    #
    # array([[10000, 10000],
    #        [10000, 10000],
    #        [10000, 10000]])
    #
    # We thus choose the min base size to be 11000, and expeect it to be broadcast
    # across all values
    expected = np.array([[True, True], [True, True], [True, True]])
    slice_ = Cube(CR.CAT_X_MR_X_MR, mask_size=11000).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.table_mask, expected)

    # Column margin evaluates to:
    #
    # array([[1914, 5958],
    #        [1914, 5958],
    #        [1914, 5958]])
    #
    # We thus choose the min base size to be 2000, and expeect it to broadcast across
    # rows (in the col direction, i.e. axis=0), sincee the MR is what won't be
    # collapsed after doing the base calculation in the table direction.
    expected = np.array([[True, False], [True, False], [True, False]])
    slice_ = Cube(CR.CAT_X_MR_X_MR, mask_size=2000).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.column_mask, expected)

    # Row margin evaluates to:
    #
    # array([[6046, 6046],
    #        [1008, 1008],
    #        [ 974,  974]])
    #
    # We thus choose the min base size to be 1000, and expeect it to broadcast across
    # rows (in the col direction, i.e. axis=0), sincee the MR is what won't be
    # collapsed after doing the base calculation in the table direction.
    expected = np.array([[False, False], [False, False], [True, True]])
    slice_ = Cube(CR.CAT_X_MR_X_MR, mask_size=1000).partitions[0]
    np.testing.assert_array_equal(slice_.min_base_size_mask.row_mask, expected)
