import numpy as np

from cr.cube.cube import Cube

from ..fixtures import CR, SM  # ---mnemonic: SM = 'scale means'---


def test_ca_cat_x_items():
    slice_ = Cube(SM.CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.scale_means_column is None
    assert slice_.scale_means_columns_margin is None


def test_ca_items_x_cat():
    slice_ = Cube(SM.CA_ITEMS_X_CAT).partitions[0]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.scale_means_rows_margin is None


def test_ca_x_mr():
    slice_ = Cube(SM.CA_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.29787234, 1.8, 1.48730964, np.nan]
    )
    assert slice_.scale_means_column is None
    assert slice_.scale_means_columns_margin is None
    assert slice_.scale_means_rows_margin == 1.504548211036992

    slice_ = Cube(SM.CA_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.31746032, 3.10743802, 3.09976976, np.nan]
    )
    assert slice_.scale_means_column is None

    slice_ = Cube(SM.CA_X_MR).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.31205674, 3.23913043, 3.37745455, np.nan]
    )
    assert slice_.scale_means_column is None

    slice_ = Cube(SM.CA_X_MR).partitions[3]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [3.53676471, 3.34814815, 3.3147877, np.nan]
    )
    assert slice_.scale_means_column is None


def test_cat_x_ca_cat_x_items():
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.34545455, 2.46938776, 2.7037037, 2.65454545]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.41935484, 3.25663717, 3.48, 3.58536585]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.49429038, 3.44905009, 3.59344262, 3.53630363]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[3]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.43365696, 3.02816901, 3.37987013, 3.32107023]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[4]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.22670025, 2.49473684, 2.79848866, 2.78987342]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[5]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.53061224, 3.68421053, 3.9862069, 4.03472222]
    )
    assert slice_.scale_means_column is None


def test_cat_x_cat():
    slice_ = Cube(SM.CAT_X_CAT).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.6009281, 2.3522267, 2.3197279, 3.3949192]
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.43636364, 2.45238095, 2.4730832, 2.68387097, 2.8375, 2.15540541],
    )

    # Test ScaleMeans marginal
    assert slice_.scale_means_columns_margin == 2.536319612590799
    assert slice_.scale_means_rows_margin == 2.6846246973365617


def test_cat_x_mr():
    slice_ = Cube(SM.CAT_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )
    assert slice_.scale_means_column is None

    assert slice_.scale_means_columns_margin is None
    assert slice_.scale_means_rows_margin == 2.5323565323565322


def test_mr_x_cat():
    slice_ = Cube(SM.MR_X_CAT).partitions[0]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )

    assert slice_.scale_means_columns_margin == 2.5323565323565322
    assert slice_.scale_means_rows_margin is None


def test_univariate_cat():
    strand = Cube(SM.UNIVARIATE_CAT).partitions[0]
    np.testing.assert_almost_equal(strand.scale_mean, [2.6865854])


def test_univariate_cat_with_hiding():
    strand_ = Cube(SM.BOLSHEVIK_HAIR).partitions[0]
    np.testing.assert_almost_equal(strand_.scale_mean, [1.504548211])

    # Appling hiding transforms
    transforms = {
        "rows_dimension": {"elements": {"5": {"hide": True}, "4": {"hide": True}}}
    }
    strand_with_hiding_ = Cube(SM.BOLSHEVIK_HAIR, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(strand_.scale_mean, strand_with_hiding_.scale_mean)


def test_cat_x_cat_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294],
    )

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
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
    strand = Cube(CR.ECON_BLAME_WITH_HS, transforms).partitions[0]
    np.testing.assert_almost_equal(strand.scale_mean, [2.17352056])

    # Test with H&S
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    np.testing.assert_almost_equal(strand.scale_mean, [2.17352056])


def test_cat_x_cat_with_hs_on_both_dims():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(
        CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, transforms=transforms
    ).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294],
    )

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
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
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.50818336, 2.56844883, 2.90251939, np.nan]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.78385708, 2.69292009, 3.11594714, np.nan]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [np.nan, np.nan, np.nan, np.nan]
    )
    assert slice_.scale_means_column is None

    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.50818336, 2.56844883, 2.90251939]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [2.78385708, 2.69292009, 3.11594714]
    )
    assert slice_.scale_means_column is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(slice_.scale_means_row, [])
    assert slice_.scale_means_column is None


def test_cat_x_cat_pruning_and_hs():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.57933884, 2.10618401, 2.30460074, np.nan, 2.34680135]
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.74213625, 1.97, 2.45356177, 2.11838791, np.nan, 2.0],
    )

    # Just H&S
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
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
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [1.57933884, 1.83081353, 2.10618401, 2.30460074, 2.34680135],
    )
    np.testing.assert_almost_equal(
        slice_.scale_means_column,
        [1.74213625, 2.2364515, 1.97, 2.45356177, 2.11838791, 2.0],
    )

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.57933884, 2.106184, 2.3046007, 2.34680135]
    ),
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.74213625, 1.97, 2.45356177, 2.11838791, 2.0]
    )


def test_cat_x_cat_scale_means_margin():
    slice_ = Cube(SM.CAT_X_CAT_SM_MARGIN).partitions[0]
    assert slice_.scale_means_rows_margin == 2.6846246973365617
    assert slice_.scale_means_columns_margin == 2.536319612590799


def test_cat_single_element_x_cat():
    slice_ = Cube(SM.CAT_SINGLE_ELEMENT_X_CAT).partitions[0]
    np.testing.assert_equal(slice_.scale_means_row, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(slice_.scale_means_column, [np.nan])


def test_means_univariate_cat():
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    np.testing.assert_almost_equal(strand.scale_mean, [2.1735205616850553])


def test_means_bivariate_cat():
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )


def test_means_cat_x_mr():
    slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.7, 1.6470588, 1.6842105])
    assert slice_.scale_means_column is None


def test_means_mr_x_cat():
    slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.7, 1.6470588, 1.6842105]
    )


def test_means_cat_array_cat_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_CAT_FIRST).partitions[0]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.44333002, 1.48049069, 1.57881177]
    )


def test_means_cat_array_subvar_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_SUBVAR_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.44333002, 1.48049069, 1.57881177]
    )
    assert slice_.scale_means_column is None


def test_means_cat_x_cat_arr_fruit_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.48, 1.4285714, 1.5217391]
    )
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]
    assert slice_.scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.scale_means_column, [1.40740741, 1.53846154, 1.55319149]
    )


def test_means_cat_x_cat_arr_subvars_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[0]
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.71111111, 1.6, 1.65625])
    assert slice_.scale_means_column is None

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.64705882, 1.7, 1.68421053]
    )
    assert slice_.scale_means_column is None


def test_means_cat_x_cat_arr_pets_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[0]
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.48, 1.40740741])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.71111111, 1.64705882])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[1]
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.42857143, 1.53846154])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.6, 1.7])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[2]
    np.testing.assert_almost_equal(slice_.scale_means_row, [1.52173913, 1.55319149])
    np.testing.assert_almost_equal(slice_.scale_means_column, [1.65625, 1.68421053])


def test_means_with_null_values():
    slice_ = Cube(CR.SCALE_WITH_NULL_VALUES).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_means_row, [1.2060688, 1.0669344, 1.023199]
    )
    assert slice_.scale_means_column is None


def test_var_scale_mean_univariate_cat():
    # Test nonmissing with no null numeric values
    strand_ = Cube(SM.UNIVARIATE_CAT).partitions[0]
    is_a_number_mask = ~np.isnan(strand_._numeric_values)

    np.testing.assert_almost_equal(strand_.var_scale_mean, 5.4322590719809645)
    np.testing.assert_array_equal(
        strand_._counts_as_array[is_a_number_mask], [54, 124, 610, 306, 396, 150]
    )

    # Test nonmissing with null numeric value
    strand_ = Cube(SM.UNIVARIATE_CAT_WITH_NULL_NUMERIC_VALUE).partitions[0]
    is_a_number_mask = ~np.isnan(strand_._numeric_values)

    np.testing.assert_almost_equal(strand_.var_scale_mean, 5.517066895232401)
    np.testing.assert_array_equal(
        strand_._counts_as_array[is_a_number_mask], [124, 610, 306, 396, 150]
    )

    # Test with all null numeric value
    strand_ = Cube(SM.UNIVARIATE_CAT_WITH_ALL_NULL_NUMERIC_VALUE).partitions[0]
    is_a_number_mask = ~np.isnan(strand_._numeric_values)

    assert strand_.var_scale_mean is None
    np.testing.assert_array_equal(
        is_a_number_mask, np.array([False] * len(strand_._numeric_values))
    )
    np.testing.assert_array_equal(strand_._counts_as_array[is_a_number_mask], [])


def test_var_scale_means_for_ca_itmes_x_cat():
    # These 2 fixtures represent 1 dataset and its transpose version
    slice_ = Cube(SM.CA_ITEMS_X_CAT).partitions[0]
    slice2_ = Cube(SM.CA_CAT_X_ITEMS).partitions[0]

    # Testing that the scale means (row and col) are equal on the 2 diverse
    # datasets
    np.testing.assert_array_equal(
        slice_.var_scale_means_column, slice2_.var_scale_means_row
    )

    np.testing.assert_almost_equal(
        slice2_.var_scale_means_row, [2.56410909, 5.17893869, 4.75445248, 4.81611278]
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column, [2.56410909, 5.17893869, 4.75445248, 4.81611278]
    )

    assert slice2_.var_scale_means_column is None
    assert slice_.var_scale_means_row is None


def test_var_scale_means_for_fruit_x_pets_array():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]

    assert slice_.var_scale_means_row is None
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column, np.array([0.2496, 0.24489796, 0.24952741])
    )

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]

    np.testing.assert_almost_equal(
        slice_.var_scale_means_column, np.array([0.24142661, 0.24852071, 0.24717067])
    )


def test_var_scale_means_for_econ_blame_x_ideology_row_and_col_hs():
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]

    assert slice_.var_scale_means_column is not None
    assert slice_.var_scale_means_row is not None

    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.88930748, 0.93655622, 1.36425875, 0.96388566, 3.55555556, 2.55601211],
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [
            0.51774691,
            0.51796281,
            0.99555556,
            0.84071826,
            1.12549449,
            1.19867769,
            2.4775,
        ],
    )


def test_var_scale_means_cat_x_cat_with_hs_on_both_dims():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(
        CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS, transforms=transforms
    ).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [0.51774691, 0.51796281, 0.99555556, 1.12549449, 1.19867769, 2.4775],
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.88930748, 0.93655622, 0.96388566, 3.55555556, 2.55601211],
    )

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [
            0.51774691,
            0.51796281,
            0.99555556,
            0.84071826,
            1.12549449,
            1.19867769,
            2.4775,
        ],
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.88930748, 0.93655622, 1.36425875, 0.96388566, 3.55555556, 2.55601211],
    )


def test_var_scale_means_for_univariate_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    strand = Cube(CR.ECON_BLAME_WITH_HS, transforms).partitions[0]
    np.testing.assert_almost_equal(strand.var_scale_mean, [1.1363901131679894])

    # Test with H&S
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    np.testing.assert_almost_equal(strand.var_scale_mean, [1.1363901131679894])


def test_var_scale_means_cat_x_cat_arr_subvars_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row, [0.2054321, 0.24, 0.22558594]
    )
    assert slice_.var_scale_means_column is None

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[1]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row, [0.2283737, 0.21, 0.21606648]
    )
    assert slice_.scale_means_column is None


def test_var_scale_means_cat_x_cat_pruning_and_hs():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [1.4459092, 2.14619102, 2.40430987, np.nan, 0.87972883],
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.72358198, 0.9991, 1.87633763, 0.4859843, np.nan, 0.66666667],
    )

    # Just H&S
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [1.4459092, 1.8494177, 2.14619102, 2.40430987, np.nan, 0.87972883],
    ),
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.72358198, 1.08423566, 0.9991, 1.87633763, 0.4859843, np.nan, 0.66666667],
    )

    # Just pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [1.4459092, 1.8494177, 2.14619102, 2.40430987, 0.87972883],
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.72358198, 1.08423566, 0.9991, 1.87633763, 0.4859843, 0.66666667],
    )

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row, [1.4459092, 2.14619102, 2.40430987, 0.87972883]
    ),
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [0.72358198, 0.9991, 1.87633763, 0.4859843, 0.66666667],
    )


def test_var_scale_means_nps_type():
    slice_ = Cube(SM.FACEBOOK_APPS_X_AGE).partitions[0]
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row,
        [1905.11600238, 2111.67820069, 1655.65636907, 981.86821176],
    )
    assert slice_.var_scale_means_column is None
