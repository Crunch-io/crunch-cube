# encoding: utf-8

"""Integration tests for scale-mean measures and marginals."""

import numpy as np
import pytest

from cr.cube.cube import Cube

from ..fixtures import CR, SM


def test_ca_cat_x_items():
    slice_ = Cube(SM.CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.rows_scale_mean is None
    assert slice_.rows_scale_mean_margin is None


def test_ca_items_x_cat():
    slice_ = Cube(SM.CA_ITEMS_X_CAT).partitions[0]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean, [1.50454821, 3.11233766, 3.35788192, 3.33271833]
    )
    assert slice_.columns_scale_mean_margin is None


def test_ca_itmes_x_cat_var_scale_means():
    # These 2 fixtures represent 1 dataset and its transpose version
    slice_ = Cube(SM.CA_ITEMS_X_CAT).partitions[0]
    slice2_ = Cube(SM.CA_CAT_X_ITEMS).partitions[0]

    # Testing that the scale means (row and col) are equal on the 2 diverse
    # datasets
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        slice2_.columns_scale_mean_stddev
    )

    assert slice2_._columns_scale_mean_variance == pytest.approx(
        [2.56410909, 5.17893869, 4.75445248, 4.81611278],
    )

    assert slice2_.rows_scale_mean_stddev is None
    assert slice_.columns_scale_mean_stddev is None


def test_ca_x_mr():
    slice_ = Cube(SM.CA_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.29787234, 1.8, 1.48730964, np.nan]
    )
    assert slice_.rows_scale_mean is None
    assert slice_.rows_scale_mean_margin is None
    assert slice_.columns_scale_mean_margin == 1.504548211036992

    slice_ = Cube(SM.CA_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [3.31746032, 3.10743802, 3.09976976, np.nan]
    )
    assert slice_.rows_scale_mean is None

    slice_ = Cube(SM.CA_X_MR).partitions[2]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [3.31205674, 3.23913043, 3.37745455, np.nan]
    )
    assert slice_.rows_scale_mean is None

    slice_ = Cube(SM.CA_X_MR).partitions[3]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [3.53676471, 3.34814815, 3.3147877, np.nan]
    )
    assert slice_.rows_scale_mean is None


def test_cat_x_ca_cat_x_items():
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.34545455, 2.46938776, 2.7037037, 2.65454545]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[1]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.41935484, 3.25663717, 3.48, 3.58536585]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[2]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.49429038, 3.44905009, 3.59344262, 3.53630363]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[3]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.43365696, 3.02816901, 3.37987013, 3.32107023]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[4]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.22670025, 2.49473684, 2.79848866, 2.78987342]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[5]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.53061224, 3.68421053, 3.9862069, 4.03472222]
    )
    assert slice_.rows_scale_mean is None


def test_cat_x_cat():
    slice_ = Cube(SM.CAT_X_CAT).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.6009281, 2.3522267, 2.3197279, 3.3949192]
    )
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [1.43636364, 2.45238095, 2.4730832, 2.68387097, 2.8375, 2.15540541],
    )

    # Test ScaleMeans marginal
    assert slice_.rows_scale_mean_margin == 2.536319612590799
    assert slice_.columns_scale_mean_margin == 2.6846246973365617


def test_cat_hs_x_cat_hs_var_scale_means():
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]

    assert slice_.rows_scale_mean_stddev is not None
    assert slice_.columns_scale_mean_stddev is not None

    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.943031, 0.9677583, 1.1680149, 0.9817768, 1.8856181, 1.5987533]
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [0.7195463, 0.7196963, 0.9977753, 0.9169069, 1.0608933, 1.0948414, 1.5740076]
    )
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [0.51774691, 0.51796281, 0.99555556, 0.84071826, 1.12549449, 1.19867769, 2.4775]
    )


def test_cat_x_mr():
    slice_ = Cube(SM.CAT_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )
    assert slice_.rows_scale_mean is None

    assert slice_.rows_scale_mean_margin is None
    assert slice_.columns_scale_mean_margin == 2.5323565323565322


def test_cat_x_cat_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [3.87368421, 2.51767677, 3.38429752, 3.66666667, 4.13235294],
    )

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [3.87368421, 2.51767677, 3.0851689, 3.38429752, 3.66666667, 4.13235294],
    )


def test_ca_x_mr_with_hs_and_pruning():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.50818336, 2.56844883, 2.90251939, np.nan]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.78385708, 2.69292009, 3.11594714, np.nan]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [np.nan, np.nan, np.nan, np.nan]
    )
    assert slice_.rows_scale_mean is None

    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.50818336, 2.56844883, 2.90251939]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [2.78385708, 2.69292009, 3.11594714]
    )
    assert slice_.rows_scale_mean is None
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(slice_.columns_scale_mean, [])
    assert slice_.rows_scale_mean is None


def test_cat_x_cat_pruning_and_hs():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [1.57933884, 2.10618401, 2.30460074, np.nan, 2.34680135],
    )
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [1.74213625, 1.97, 2.45356177, 2.11838791, np.nan, 2.0],
    )

    # Just H&S
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [1.57933884, 1.8308135, 2.10618401, 2.30460074, np.nan, 2.34680135],
    ),
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [1.74213625, 2.2364515, 1.97, 2.45356177, 2.11838791, np.nan, 2.0],
    )

    # Just pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [1.57933884, 1.83081353, 2.10618401, 2.30460074, 2.34680135],
    )
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean,
        [1.74213625, 2.2364515, 1.97, 2.45356177, 2.11838791, 2.0],
    )

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.57933884, 2.106184, 2.3046007, 2.34680135]
    ),
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean, [1.74213625, 1.97, 2.45356177, 2.11838791, 2.0]
    )


def test_cat_x_cat_scale_means_margin():
    slice_ = Cube(SM.CAT_X_CAT_SM_MARGIN).partitions[0]
    assert slice_.columns_scale_mean_margin == 2.6846246973365617
    assert slice_.rows_scale_mean_margin == 2.536319612590799


def test_cat_x_ca_subvar_scale_means():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [0.2054321, 0.24, 0.22558594]
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [0.4532462, 0.4898979, 0.4749589]
    )
    assert slice_.rows_scale_mean_stddev is None

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[1]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [0.2283737, 0.21, 0.21606648]
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [0.4778846, 0.4582576, 0.4648295]
    )
    assert slice_.rows_scale_mean is None


def test_cat_x_cat_pruning_and_hs_var_scale_means():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [1.4459092, 2.14619102, 2.40430987, np.nan, 0.87972883], nan_ok=True
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [1.2024596, 1.4649884, 1.5505837, np.nan, 0.9379386], nan_ok=True
    )
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.8506362, 0.9995499, 1.3697947, 0.6971257, np.nan, 0.8164966], nan_ok=True
    )

    # Just H&S
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [1.4459092, 1.8494177, 2.14619102, 2.40430987, np.nan, 0.87972883], nan_ok=True
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [1.2024596, 1.359933, 1.4649884, 1.5505837, np.nan, 0.9379386], nan_ok=True
    )
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.8506362, 1.0412664, 0.9995499, 1.3697947, 0.6971257, np.nan, 0.8164966],
        nan_ok=True,
    )

    # Just pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [1.4459092, 1.8494177, 2.14619102, 2.40430987, 0.87972883]
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [1.2024596, 1.359933, 1.4649884, 1.5505837, 0.9379386]
    )
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.8506362, 1.0412664, 0.9995499, 1.3697947, 0.6971257, 0.8164966]
    )

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.CAT_HS_MT_X_CAT_HS_MT, transforms=transforms).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [1.4459092, 2.14619102, 2.40430987, 0.87972883]
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [1.2024596, 1.4649884, 1.5505837, 0.9379386]
    )
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.8506362, 0.9995499, 1.3697947, 0.6971257, 0.8164966]
    )


def test_cat_nps_numval_x_cat_var_scale_means():
    slice_ = Cube(SM.CAT_NPS_NUMVAL_X_CAT).partitions[0]
    assert slice_._columns_scale_mean_variance == pytest.approx(
        [1905.11600238, 2111.67820069, 1655.65636907, 981.86821176],
    )
    assert slice_.columns_scale_mean_stddev == pytest.approx(
        [43.6476346, 45.9529999, 40.6897575, 31.3347764],
    )
    assert slice_.rows_scale_mean_stddev is None


def test_cat_single_element_x_cat():
    slice_ = Cube(SM.CAT_SINGLE_ELEMENT_X_CAT).partitions[0]
    np.testing.assert_equal(slice_.columns_scale_mean, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(slice_.rows_scale_mean, [np.nan])


def test_means_univariate_cat():
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    np.testing.assert_almost_equal(strand.scale_mean, [2.1735205616850553])


def test_means_bivariate_cat():
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean,
        [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85],
    )


def test_means_cat_x_mr():
    slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.7, 1.6470588, 1.6842105]
    )
    assert slice_.rows_scale_mean is None


def test_means_mr_x_cat():
    slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(slice_.rows_scale_mean, [1.7, 1.6470588, 1.6842105])


def test_means_cat_array_cat_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_CAT_FIRST).partitions[0]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean, [1.44333002, 1.48049069, 1.57881177]
    )


def test_means_cat_array_subvar_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_SUBVAR_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.44333002, 1.48049069, 1.57881177]
    )
    assert slice_.rows_scale_mean is None


def test_means_cat_x_cat_arr_fruit_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(slice_.rows_scale_mean, [1.48, 1.4285714, 1.5217391])
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean, [1.40740741, 1.53846154, 1.55319149]
    )


def test_means_cat_x_cat_arr_subvars_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.71111111, 1.6, 1.65625]
    )
    assert slice_.rows_scale_mean is None

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[1]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.64705882, 1.7, 1.68421053]
    )
    assert slice_.rows_scale_mean is None


def test_means_cat_x_cat_arr_pets_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[0]
    np.testing.assert_almost_equal(slice_.columns_scale_mean, [1.48, 1.40740741])
    np.testing.assert_almost_equal(slice_.rows_scale_mean, [1.71111111, 1.64705882])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[1]
    np.testing.assert_almost_equal(slice_.columns_scale_mean, [1.42857143, 1.53846154])
    np.testing.assert_almost_equal(slice_.rows_scale_mean, [1.6, 1.7])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[2]
    np.testing.assert_almost_equal(slice_.columns_scale_mean, [1.52173913, 1.55319149])
    np.testing.assert_almost_equal(slice_.rows_scale_mean, [1.65625, 1.68421053])


def test_means_with_null_values():
    slice_ = Cube(CR.SCALE_WITH_NULL_VALUES).partitions[0]
    np.testing.assert_almost_equal(
        slice_.columns_scale_mean, [1.2060688, 1.0669344, 1.023199]
    )
    assert slice_.rows_scale_mean is None


def test_mean_univariate_cat_var_scale_mean():
    # Test nonmissing with no null numeric values
    strand = Cube(SM.UNIVARIATE_CAT).partitions[0]
    assert strand.scale_mean == pytest.approx(2.686585)

    # Test nonmissing with null numeric value
    strand = Cube(SM.UNIVARIATE_CAT_WITH_NULL_NUMERIC_VALUE).partitions[0]
    assert strand.scale_mean == pytest.approx(2.744010)

    # Test with all null numeric value
    strand = Cube(SM.UNIVARIATE_CAT_WITH_ALL_NULL_NUMERIC_VALUE).partitions[0]
    assert strand.scale_mean is None


def test_mr_x_cat():
    slice_ = Cube(SM.MR_X_CAT).partitions[0]
    assert slice_.columns_scale_mean is None
    np.testing.assert_almost_equal(
        slice_.rows_scale_mean, [2.45070423, 2.54471545, 2.54263006, np.nan]
    )

    assert slice_.rows_scale_mean_margin == 2.5323565323565322
    assert slice_.columns_scale_mean_margin is None


def test_rows_and_new_rows_scale_mean_stddev_for_fruit_x_pets_array():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]

    assert slice_._columns_scale_mean_variance is None
    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.4995998, 0.4948717, 0.4995272]
    )

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]

    assert slice_.rows_scale_mean_stddev == pytest.approx(
        [0.4913518, 0.4985185, 0.4971626]
    )


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


def test_univariate_with_hs_scale_means_row():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    strand = Cube(CR.ECON_BLAME_WITH_HS, transforms).partitions[0]
    assert strand.scale_mean == pytest.approx(2.1735206)

    # Test with H&S
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    assert strand.scale_mean == pytest.approx(2.1735206)


def test_univariate_ca_subvar_with_empty_total_counts():
    strand = Cube(SM.UNIVARIATE_CA_SUBVAR).partitions[0]

    # --- scale_meanm, scale_std_dev and scale_std_err can be None when
    # --- _total_weighted_count is 0.
    assert strand.scale_mean is None
    assert strand.scale_std_dev is None
    assert strand.scale_std_err is None
