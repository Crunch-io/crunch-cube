import numpy as np

from cr.cube.crunch_cube import CrunchCube
from cr.cube.slices import FrozenSlice

from ..fixtures import CR, SM  # ---mnemonic: SM = 'scale means'---


def test_ca_cat_x_items():
    slice_ = FrozenSlice(CrunchCube(SM.CA_CAT_X_ITEMS))
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.scale_means_column is None
    assert slice_.scale_means_column_margin is None


def test_ca_items_x_cat():
    slice_ = FrozenSlice(CrunchCube(SM.CA_ITEMS_X_CAT))
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.scale_means_row_margin is None


def test_ca_x_mr():
    slice_ = FrozenSlice(CrunchCube(SM.CA_X_MR), slice_idx=0)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.29787234, 1.8, 1.48730964, np.nan]
    )
    assert slice_.scale_means_column is None
    assert slice_.scale_means_column_margin is None
    assert slice_.scale_means_row_margin == 1.504548211036992

    slice_ = FrozenSlice(CrunchCube(SM.CA_X_MR), slice_idx=1)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.31746032, 3.10743802, 3.09976976, np.nan]
    )
    assert slice_.scale_means_column is None

    slice_ = FrozenSlice(CrunchCube(SM.CA_X_MR), slice_idx=2)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.31205674, 3.23913043, 3.37745455, np.nan]
    )
    assert slice_.scale_means_column is None

    slice_ = FrozenSlice(CrunchCube(SM.CA_X_MR), slice_idx=3)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.53676471, 3.34814815, 3.3147877, np.nan]
    )
    assert slice_.scale_means_column is None


def test_cat_x_ca_cat_x_items():
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=0)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.34545455, 2.46938776, 2.7037037, 2.65454545]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=1)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.41935484, 3.25663717, 3.48, 3.58536585]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=2)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.49429038, 3.44905009, 3.59344262, 3.53630363]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=3)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.43365696, 3.02816901, 3.37987013, 3.32107023]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=4)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.22670025, 2.49473684, 2.79848866, 2.78987342]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CA_CAT_X_ITEMS), slice_idx=5)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.53061224, 3.68421053, 3.9862069, 4.03472222]
    )
    assert slice_.scale_means_column is None


def test_cat_x_cat():
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CAT))
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.6009281, 2.3522267, 2.3197279, 3.3949192]
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.43636364, 2.45238095, 2.4730832, 2.68387097, 2.8375, 2.15540541],
    )

    # Test ScaleMeans marginal
    assert slice_.scale_means_column_margin == 2.536319612590799
    assert slice_.scale_means_row_margin == 2.6846246973365617


def test_cat_x_mr():
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_MR))
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )
    assert slice_.scale_means_column is None

    assert slice_.scale_means_column_margin is None
    assert slice_.scale_means_row_margin == 2.5323565323565322


def test_mr_x_cat():
    slice_ = FrozenSlice(CrunchCube(SM.MR_X_CAT))
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )

    assert slice_.scale_means_column_margin == 2.5323565323565322
    assert slice_.scale_means_row_margin is None


def test_univariate_cat():
    slice_ = FrozenSlice(CrunchCube(SM.UNIVARIATE_CAT))
    np.testing.assert_almost_equal(slice_.scale_means_row, [2.6865854])


def test_cat_x_cat_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = FrozenSlice(
        CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS), transforms=transforms
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294],
    )

    # Test with H&S
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
    slice_.scale_means_row
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.0851689, 3.38429752, 3.66666667, 4.13235294],
    )


def test_univariate_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS), transforms)
    np.testing.assert_almost_equal(slice_.scale_means_row, [2.17352056])

    # Test with H&S
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS))
    np.testing.assert_almost_equal(slice_.scale_means_row, [2.17352056])


def test_cat_x_cat_with_hs_on_both_dims():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = FrozenSlice(
        CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS), transforms=transforms
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294],
    )

    # Test with H&S
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS))
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 2.2423698, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.0851689, 3.38429752, 3.66666667, 4.13235294],
    )


def test_ca_x_mr_with_hs_and_pruning():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=0, transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.50818336, 2.56844883, 2.90251939, np.nan]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=1, transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.78385708, 2.69292009, 3.11594714, np.nan]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=2, transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [np.nan, np.nan, np.nan, np.nan]
    )
    assert slice_.scale_means_column is None

    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=0, transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.50818336, 2.56844883, 2.90251939]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=1, transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.78385708, 2.69292009, 3.11594714]
    )
    assert slice_.scale_means_column is None
    slice_ = FrozenSlice(CrunchCube(CR.CA_X_MR_HS), slice_idx=2, transforms=transforms)
    np.testing.assert_almost_equal(slice_.scale_means_row, [])
    assert slice_.scale_means_column is None


def test_cat_x_cat_pruning_and_hs():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = FrozenSlice(CrunchCube(CR.CAT_X_CAT_PRUNING_HS), transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.57933884, 2.10618401, 2.30460074, np.nan, 2.34680135]
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.74213625, 1.97, 2.45356177, 2.11838791, np.nan, 2.0],
    )

    # Just H&S
    slice_ = FrozenSlice(CrunchCube(CR.CAT_X_CAT_PRUNING_HS))
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [1.57933884, 1.8308135, 2.10618401, 2.30460074, np.nan, 2.34680135],
    ),
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.74213625, 2.2364515, 1.97, 2.45356177, 2.11838791, np.nan, 2.0],
    )

    # Just pruning
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = FrozenSlice(CrunchCube(CR.CAT_X_CAT_PRUNING_HS), transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.57933884, 2.10618401, 2.30460074, 2.34680135]
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.74213625, 1.97, 2.45356177, 2.11838791, 2.0]
    )

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = FrozenSlice(CrunchCube(CR.CAT_X_CAT_PRUNING_HS), transforms=transforms)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.57933884, 2.106184, 2.3046007, 2.34680135]
    ),
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.74213625, 1.97, 2.45356177, 2.11838791, 2.0]
    )


def test_cat_x_cat_scale_means_margin():
    slice_ = FrozenSlice(CrunchCube(SM.CAT_X_CAT_SM_MARGIN))
    assert slice_.scale_means_row_margin == 2.6846246973365617
    assert slice_.scale_means_column_margin == 2.536319612590799


def test_cat_single_element_x_cat():
    slice_ = FrozenSlice(CrunchCube(SM.CAT_SINGLE_ELEMENT_X_CAT))
    np.testing.assert_equal(slice_.scale_means_row, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(slice_.scale_means_column, [np.nan])


def test_means_univariate_cat():
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_WITH_HS))
    np.testing.assert_almost_equal(slice_.scale_means_row, [2.1735205616850553])


def test_means_bivariate_cat():
    slice_ = FrozenSlice(CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS))
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )


def test_means_cat_x_mr():
    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS))
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.7, 1.6470588, 1.6842105])
    assert slice_.scale_means_column is None


def test_means_mr_x_cat():
    slice_ = FrozenSlice(CrunchCube(CR.PETS_X_FRUIT))
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.7, 1.6470588, 1.6842105]
    )


def test_means_cat_array_cat_dim_first():
    slice_ = FrozenSlice(CrunchCube(CR.PETS_ARRAY_CAT_FIRST))
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.44333002, 1.48049069, 1.57881177]
    )


def test_means_cat_array_subvar_dim_first():
    slice_ = FrozenSlice(CrunchCube(CR.PETS_ARRAY_SUBVAR_FIRST))
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.44333002, 1.48049069, 1.57881177]
    )
    assert slice_.scale_means_column is None


def test_means_cat_x_cat_arr_fruit_first():
    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY), slice_idx=0)
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.48, 1.4285714, 1.5217391]
    )
    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY), slice_idx=1)
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.40740741, 1.53846154, 1.55319149]
    )


def test_means_cat_x_cat_arr_subvars_first():
    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST), slice_idx=0)
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.71111111, 1.6, 1.65625])
    assert slice_.scale_means_column is None

    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST), slice_idx=1)
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.64705882, 1.7, 1.68421053]
    )
    assert slice_.scale_means_column is None


def test_means_cat_x_cat_arr_pets_first():
    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST), slice_idx=0)
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.48, 1.40740741])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.71111111, 1.64705882])

    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST), slice_idx=1)
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.42857143, 1.53846154])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.6, 1.7])

    slice_ = FrozenSlice(CrunchCube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST), slice_idx=2)
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.52173913, 1.55319149])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.65625, 1.68421053])


def test_means_with_null_values():
    slice_ = FrozenSlice(CrunchCube(CR.SCALE_WITH_NULL_VALUES))
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.2060688, 1.0669344, 1.023199]
    )
    assert slice_.scale_means_column is None
