import numpy as np

from cr.cube.cube import Cube

from ..fixtures import CR, SM  # ---mnemonic: SM = 'scale means'---


def test_ca_cat_x_items():
    slice_ = Cube(SM.CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.60128358, 2.27572817, 2.1804707, 2.19456437]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.95359134, 1.35523445, 1.29850702, 1.30690004]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 5, 5, 5])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_ca_items_x_cat():
    slice_ = Cube(SM.CA_ITEMS_X_CAT).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column, [1.60128358, 2.27572817, 2.1804707, 2.19456437]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column, [0.95359134, 1.35523445, 1.29850702, 1.30690004]
    )
    np.testing.assert_almost_equal(slice_.scale_median_column, [1, 5, 5, 5])
    assert slice_.scale_std_dev_row is None
    assert slice_.scale_std_err_row is None
    assert slice_.scale_median_row is None


def test_ca_x_mr():
    slice_ = Cube(SM.CA_X_MR).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.43286986, 1.75336411, 1.59143099, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.16816367, 1.4294503, 1.29743245, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CA_X_MR).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [2.23837666, 2.32059287, 2.2735136, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.2687902, 1.31539322, 1.28870705, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [5.0, 5.0, 4.0, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CA_X_MR).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [2.24756685, 2.18544045, 2.17207083, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.2265349, 1.19263148, 1.18533545, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [5.0, 5.0, 5.0, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CA_X_MR).partitions[3]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [2.1892387, 2.2020816, 2.1932015, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.199206, 1.206241, 1.2013768, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [5.0, 5.0, 5.0, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_cat_x_ca_cat_x_items():
    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.6092707, 2.241556, 2.22469, 2.2660994]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.0642764, 1.4824324, 1.4712783, 1.498664]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 4, 1])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.661212, 2.2292455, 2.1188676, 2.095007]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.9708492, 1.3028205, 1.2383133, 1.2243686]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 5, 5, 5])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.5301377, 2.172702, 2.1050968, 2.1110668]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8819616, 1.2523315, 1.2133643, 1.2168054]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 5, 5, 5])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[3]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.5764139, 2.3345044, 2.1969246, 2.2629614]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.9453286, 1.3999329, 1.3174304, 1.3570307]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 5, 5, 5])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[4]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.5302137, 2.4273602, 2.3953561, 2.3938526]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.0035434, 1.5919092, 1.5709202, 1.5699343]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 4, 4, 4])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(SM.CAT_X_CA_CAT_X_ITEMS).partitions[5]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.6549498, 1.656014, 1.3338366, 1.3250429]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8781279, 0.8786926, 0.707743, 0.7030771]
    )
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_cat_x_cat():
    slice_ = Cube(SM.CAT_X_CAT).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [1.83663364, 1.85424071, 1.85212099, 1.78435798, 1.78496323, 1.8552804],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [2.35648861, 2.32601576, 2.3064134, 2.14673544]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [1.15324225, 1.16429792, 1.16296692, 1.12041779, 1.12079784, 1.16495076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.43821414, 1.41961592, 1.4076522, 1.3101974]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 4])
    np.testing.assert_almost_equal(slice_.scale_median_column, [0, 2, 2, 2, 4, 2])
    # Test Scale Variance
    np.testing.assert_almost_equal(
        slice_.var_scale_means_row, [5.55303858, 5.41034929, 5.31954278, 4.60847303]
    )
    np.testing.assert_almost_equal(
        slice_.var_scale_means_column,
        [3.37322314, 3.43820862, 3.43035216, 3.1839334, 3.18609375, 3.44206538],
    )


def test_cat_x_mr():
    slice_ = Cube(SM.CAT_X_MR).partitions[0]

    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.84458654, 1.92650335, 1.83760589, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [1.15914191, 1.2106186, 1.15475526, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2.0, 2.0, 2.0, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_mr_x_cat():
    slice_ = Cube(SM.MR_X_CAT).partitions[0]

    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column, [1.84458654, 1.92650335, 1.83760589, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column, [1.15914191, 1.2106186, 1.15475526, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_column, [2.0, 2.0, 2.0, np.nan])
    assert slice_.scale_std_dev_row is None
    assert slice_.scale_std_err_row is None
    assert slice_.scale_median_row is None


def test_univariate_cat():
    strand_ = Cube(SM.UNIVARIATE_CAT).partitions[0]
    np.testing.assert_almost_equal(strand_.scale_std_dev, [2.3307207194301434])
    np.testing.assert_almost_equal(strand_.scale_std_err, [0.05755304590263942])
    np.testing.assert_almost_equal(strand_.scale_median, [4.0])


def test_univariate_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    strand_ = Cube(CR.ECON_BLAME_WITH_HS, transforms).partitions[0]
    np.testing.assert_almost_equal(strand_.scale_std_dev, [1.066016])
    np.testing.assert_almost_equal(strand_.scale_std_err, [0.0337611])
    np.testing.assert_almost_equal(strand_.scale_median, [2.0])

    # Test with H&S
    strand = Cube(CR.ECON_BLAME_WITH_HS).partitions[0]
    np.testing.assert_almost_equal(strand.scale_std_dev, [1.066016])
    np.testing.assert_almost_equal(strand.scale_std_err, [0.0337611])
    np.testing.assert_almost_equal(strand_.scale_median, [2.0])


def test_univariate_cat_with_hiding():
    strand_ = Cube(SM.BOLSHEVIK_HAIR).partitions[0]
    np.testing.assert_almost_equal(strand_.scale_std_dev, [1.6012836])
    np.testing.assert_almost_equal(strand_.scale_std_err, [0.0394328])
    np.testing.assert_almost_equal(strand_.scale_median, [1.0])

    # Appling hiding transforms
    transforms = {
        "rows_dimension": {"elements": {"5": {"hide": True}, "4": {"hide": True}}}
    }
    strand_with_hiding_ = Cube(SM.BOLSHEVIK_HAIR, transforms=transforms).partitions[0]

    np.testing.assert_almost_equal(
        strand_.scale_std_dev, strand_with_hiding_.scale_std_dev
    )
    np.testing.assert_almost_equal(
        strand_.scale_std_err, strand_with_hiding_.scale_std_err
    )
    np.testing.assert_almost_equal(strand_.scale_median, [1.0])


def test_cat_x_cat_with_hs():
    # Test without H&S
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.943031, 0.9677583, 0.9817768, 1.8856181, 1.5987533],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [0.7195463, 0.7196963, 0.9977753, 1.0608932, 1.0948414, 1.5740076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.5244975, 0.5382504, 0.5460472, 1.048748, 0.8891988],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.4880641, 0.4881659, 0.6767852, 0.7195978, 0.7426246, 1.0676403],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 1, 1, 5])
    np.testing.assert_almost_equal(slice_.scale_median_column, [4, 3, 3, 4, 4])

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.943031, 0.9677583, 1.1680149, 0.9817768, 1.8856181, 1.5987533],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [0.7195463, 0.7196963, 0.9977753, 1.0608932, 1.0948414, 1.5740076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.5244975, 0.5382504, 0.6496296, 0.5460472, 1.048748, 0.8891988],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.4880641, 0.4881659, 0.6767852, 0.7195978, 0.7426246, 1.0676403],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 1, 1, 5])
    np.testing.assert_almost_equal(slice_.scale_median_column, [4, 3, 3, 3, 4, 4])


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
        slice_.scale_std_dev_column,
        [0.943031, 0.9677583, 0.9817768, 1.8856181, 1.5987533],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [0.7195463, 0.7196963, 0.9977753, 1.0608932, 1.0948414, 1.5740076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.5244975, 0.5382504, 0.5460472, 1.048748, 0.8891988],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.4880641, 0.4881659, 0.6767852, 0.7195978, 0.7426246, 1.0676403],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 1, 1, 5])
    np.testing.assert_almost_equal(slice_.scale_median_column, [4, 3, 3, 4, 4])

    # Test with H&S
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.943031, 0.9677583, 1.1680149, 0.9817768, 1.8856181, 1.5987533],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [0.7195463, 0.7196963, 0.9977753, 0.9169069, 1.0608932, 1.0948414, 1.5740076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.5244975, 0.5382504, 0.6496296, 0.5460472, 1.048748, 0.8891988],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.4880641, 0.4881659, 0.6767852, 0.6219327, 0.7195978, 0.7426246, 1.0676403],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 2, 1, 1, 5])
    np.testing.assert_almost_equal(slice_.scale_median_column, [4, 3, 3, 3, 4, 4])


def test_ca_x_mr_with_hs_and_pruning():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.4689114, 1.4509704, 1.5892758, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8822918, 0.8715157, 0.9545879, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [3.0, 3.0, 3.0, np.nan])

    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.3729408, 1.4601345, 1.4346546, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.7995152, 0.8502914, 0.8354534, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [3.0, 3.0, 4.0, np.nan])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_almost_equal(
        slice_.scale_median_row, [np.nan, np.nan, np.nan, np.nan]
    )
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.4689114, 1.4509704, 1.5892758]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8822918, 0.8715157, 0.9545879]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [3, 3, 3])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.3729408, 1.4601345, 1.4346546]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.7995152, 0.8502914, 0.8354534]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [3, 3, 4])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(CR.CA_X_MR_HS, transforms=transforms).partitions[2]
    np.testing.assert_almost_equal(slice_.scale_std_dev_row, [])
    np.testing.assert_almost_equal(slice_.scale_std_err_row, [])
    np.testing.assert_almost_equal(slice_.scale_median_row, [])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_cat_x_cat_pruning_and_hs():
    transforms = {
        "columns_dimension": {"insertions": {}},
        "rows_dimension": {"insertions": {}},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.2024596, 1.4649884, 1.5505837, np.nan, 0.9379386]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.8506362, 0.9995499, 1.3697947, 0.6971257, np.nan, 0.8164966],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8620521, 1.0502609, 1.1116248, np.nan, 0.6724151]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.6159585, 0.7237891, 0.991889, 0.5047993, np.nan, 0.5912375],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1, np.nan, 3])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 1, 2, 2, np.nan, 2])

    # Just H&S
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [1.2024596, 1.359933, 1.4649884, 1.5505837, np.nan, 0.9379386],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.8506362, 1.0412664, 0.9995499, 1.3697947, 0.6971257, np.nan, 0.8164966],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.8620521, 0.9749459, 1.0502609, 1.1116248, np.nan, 0.6724151],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.6159585, 0.7539967, 0.7237891, 0.991889, 0.5047993, np.nan, 0.5912375],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1, 1, np.nan, 3])
    np.testing.assert_almost_equal(
        slice_.scale_median_column, [2, 2, 1, 2, 2, np.nan, 2]
    )

    # Just pruning
    transforms = {
        "rows_dimension": {"prune": True},
        "columns_dimension": {"prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.2024596, 1.359933, 1.4649884, 1.5505837, 0.9379386]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.8506362, 1.0412664, 0.9995499, 1.3697947, 0.6971257, 0.8164966],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.8620521, 0.9749459, 1.0502609, 1.1116248, 0.6724151],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.6159585, 0.7539967, 0.7237891, 0.991889, 0.5047993, 0.5912375],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1, 1, 3])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 2, 3, 2, 2, 2])

    # Pruning and H&S
    transforms = {
        "rows_dimension": {"insertions": {}, "prune": True},
        "columns_dimension": {"insertions": {}, "prune": True},
    }
    slice_ = Cube(CR.CAT_X_CAT_PRUNING_HS, transforms=transforms).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [1.2024596, 1.4649884, 1.5505837, 0.9379386]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.8506362, 0.9995499, 1.3697947, 0.6971257, 0.8164966],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8620521, 1.0502609, 1.1116248, 0.6724151]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.6159585, 0.7237891, 0.991889, 0.5047993, 0.5912375],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1, 3])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 3, 2, 2, 2])


def test_cat_single_element_x_cat():
    slice_ = Cube(SM.CAT_SINGLE_ELEMENT_X_CAT).partitions[0]
    np.testing.assert_equal(slice_.scale_std_dev_row, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(slice_.scale_std_dev_column, [np.nan])
    np.testing.assert_equal(slice_.scale_std_err_row, [np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(slice_.scale_std_err_column, [np.nan])
    np.testing.assert_almost_equal(
        slice_.scale_median_row, [np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_almost_equal(slice_.scale_median_column, [np.nan])


def test_bivariate_cat():
    slice_ = Cube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row,
        [0.7195463, 0.7196963, 0.9977753, 1.0608932, 1.0948414, 1.5740076],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row,
        [0.4880641, 0.4881659, 0.6767852, 0.7195978, 0.7426246, 1.0676403],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column,
        [0.94303101, 0.96775835, 1.16801487, 0.98177679, 1.88561808, 1.5987533],
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column,
        [0.52449747, 0.53825039, 0.64962959, 0.5460472, 1.04874804, 0.88919882],
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2, 1, 1, 5])
    np.testing.assert_almost_equal(slice_.scale_median_column, [4, 3, 3, 3, 4, 4])


def test_cat_array_cat_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_CAT_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column, [0.4967781, 0.4996192, 0.4937496]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column, [0.4058676, 0.4081887, 0.4033933]
    )
    assert slice_.scale_std_dev_row is None
    assert slice_.scale_std_err_row is None


def test_cat_array_subvar_dim_first():
    slice_ = Cube(CR.PETS_ARRAY_SUBVAR_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [0.4967781, 0.4996192, 0.4937496]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.4058676, 0.4081887, 0.4033933]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 2])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_cat_x_cat_arr_fruit_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]
    assert slice_.scale_std_dev_row is None
    assert slice_.scale_std_err_row is None
    assert slice_.scale_median_row is None
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column, [0.4995998, 0.4948717, 0.4995272]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column, [0.4115476, 0.4076527, 0.4114878]
    )
    np.testing.assert_almost_equal(slice_.scale_median_column, [1, 1, 2])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]
    assert slice_.scale_std_dev_row is None
    assert slice_.scale_std_err_row is None
    assert slice_.scale_median_row is None
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_column, [0.4913518, 0.4985185, 0.4971626]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_column, [0.4016248, 0.4074828, 0.4063745]
    )
    np.testing.assert_almost_equal(slice_.scale_median_column, [1, 2, 2])


def test_cat_x_cat_arr_subvars_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [0.4532462, 0.4898979, 0.4749589]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.3519867, 0.3804501, 0.3688485]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST).partitions[1]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [0.4778846, 0.4582576, 0.4648295]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.3688528, 0.3537038, 0.3587763]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2, 2])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None


def test_cat_x_cat_arr_pets_first():
    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[0]
    np.testing.assert_almost_equal(slice_.scale_std_dev_row, [0.4995998, 0.4913518])
    np.testing.assert_almost_equal(slice_.scale_std_dev_column, [0.4532462, 0.4778846])
    np.testing.assert_almost_equal(slice_.scale_std_err_row, [0.4177309, 0.4108345])
    np.testing.assert_almost_equal(slice_.scale_std_err_column, [0.3493187, 0.3683077])
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 2])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[1]
    np.testing.assert_almost_equal(slice_.scale_std_dev_row, [0.4948717, 0.4985185])
    np.testing.assert_almost_equal(slice_.scale_std_dev_column, [0.4898979, 0.4582576])
    np.testing.assert_almost_equal(slice_.scale_std_err_row, [0.404061, 0.4070387])
    np.testing.assert_almost_equal(slice_.scale_std_err_column, [0.381385, 0.356753])
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 2])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 2])

    slice_ = Cube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST).partitions[2]
    np.testing.assert_almost_equal(slice_.scale_std_dev_row, [0.4995272, 0.4971626])
    np.testing.assert_almost_equal(slice_.scale_std_dev_column, [0.4749589, 0.4648295])
    np.testing.assert_almost_equal(slice_.scale_std_err_row, [0.4021576, 0.4002539])
    np.testing.assert_almost_equal(slice_.scale_std_err_column, [0.3673771, 0.3595421])
    np.testing.assert_almost_equal(slice_.scale_median_row, [2, 2])
    np.testing.assert_almost_equal(slice_.scale_median_column, [2, 2])


def test_with_null_values():
    slice_ = Cube(CR.SCALE_WITH_NULL_VALUES).partitions[0]
    np.testing.assert_almost_equal(
        slice_.scale_std_dev_row, [0.911757, 0.9089713, 0.9419575]
    )
    np.testing.assert_almost_equal(
        slice_.scale_std_err_row, [0.8749051, 0.8722319, 0.9038849]
    )
    np.testing.assert_almost_equal(slice_.scale_median_row, [1, 1, 1])
    assert slice_.scale_std_dev_column is None
    assert slice_.scale_std_err_column is None
    assert slice_.scale_median_column is None
