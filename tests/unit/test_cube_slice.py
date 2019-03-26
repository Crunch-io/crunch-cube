# encoding: utf-8

"""Unit test suite for cr.cube.cube_slice module."""

from mock import Mock, patch
import numpy as np
import pytest

from cr.cube.crunch_cube import CrunchCube
from cr.cube.cube_slice import CubeSlice
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.dimension import Dimension

from ..unitutil import instance_mock, method_mock, property_mock


class DescribeCubeSlice(object):
    def it_can_calculate_correct_axis_for_cube(self, axis_fixture, cube_):
        axis, ndim, ca_as_0th, expected_value = axis_fixture
        cube_.ndim = ndim
        slice_ = CubeSlice(cube_, None)
        slice_.ca_as_0th = ca_as_0th

        updated_axis = slice_._calculate_correct_axis_for_cube(axis)

        assert updated_axis == expected_value

    def it_can_calculate_std_res(self, std_res_fixture, cube_, dim_types_prop_):
        dim_types, expected_value = std_res_fixture
        dim_types_prop_.return_value = dim_types
        slice_ = CubeSlice(cube_, None)
        counts, total, colsum, rowsum = Mock(), Mock(), Mock(), Mock()
        slice_._calculate_std_res(counts, total, colsum, rowsum)

        # Assert correct methods are invoked
        expected_value.assert_called_once_with(slice_, counts, total, colsum, rowsum)

    def it_can_extract_slice_from_cube_result(self, extract_fixture, cube_):
        result, ndim, ca_as_0th, index, expected_value = extract_fixture
        cube_.ndim = ndim
        slice_ = CubeSlice(cube_, None)
        slice_.ca_as_0th = ca_as_0th
        slice_._index = index

        extracted_result = slice_._extract_slice_result_from_cube(result)

        np.testing.assert_array_equal(extracted_result, expected_value)

    def it_provides_a_default_repr(self):
        slice_ = CubeSlice(None, None)
        repr_ = repr(slice_)
        assert repr_.startswith("<cr.cube.cube_slice.CubeSlice object at 0x")

    def it_knows_if_it_is_univariate_ca(self, cube_):
        result_mock = "I am a fake univariate CA result"
        cube_.is_univariate_ca = result_mock
        slice_ = CubeSlice(cube_, None)
        assert slice_.is_univariate_ca == result_mock

    def it_knows_if_it_is_weighted(self, cube_):
        result_mock = "I am a fake weighted result"
        cube_.is_weighted = result_mock
        slice_ = CubeSlice(cube_, None)
        assert slice_.is_weighted == result_mock

    def it_knows_its_dimension_types(self, dim_types_fixture, cube_):
        cube_dim_types, expected_value = dim_types_fixture
        cube_.dim_types = cube_dim_types
        slice_ = CubeSlice(cube_, None)

        dim_types = slice_.dim_types

        assert dim_types == expected_value

    def it_knows_its_name(self, name_fixture, dimensions_prop_):
        dimensions, expected_name = name_fixture
        dimensions_prop_.return_value = dimensions
        assert CubeSlice(None, None).name == expected_name

    def it_knows_its_population_fraction(self, cube_):
        result_mock = "I am a fake population fraction result"
        cube_.population_fraction = result_mock
        slice_ = CubeSlice(cube_, None)
        assert slice_.population_fraction == result_mock

    def it_knows_its_scalar_type_std_res(self, scalar_std_res_fixture, cube_):
        counts, total, colsum, rowsum, expected_value = scalar_std_res_fixture
        slice_ = CubeSlice(cube_, None)
        std_res = slice_._scalar_type_std_res(counts, total, colsum, rowsum)
        np.testing.assert_almost_equal(std_res, expected_value)

    def it_knows_its_univariate_ca_main_axis(self, cube_):
        result_mock = "I am a fake univariate CA main axis result"
        cube_.univariate_ca_main_axis = result_mock
        slice_ = CubeSlice(cube_, None)
        assert slice_.univariate_ca_main_axis == result_mock

    def it_knows_whether_its_a_double_mr(self, is_double_mr_fixture, dim_types_prop_):
        dim_types, expected_value = is_double_mr_fixture
        dim_types_prop_.return_value = dim_types
        slice_ = CubeSlice(None, None)

        is_double_mr = slice_.is_double_mr

        assert is_double_mr is expected_value

    def it_can_compare_pairwise(
        self, cube_, dim_types_prop_, ndim_prop_, pairwise_comparisons_fixture
    ):
        dim_types, slice_can_show = pairwise_comparisons_fixture
        dim_types_prop_.return_value = dim_types
        ndim_prop_.return_value = len(dim_types)
        slice_ = CubeSlice(cube_, None)

        assert slice_.can_compare_pairwise == slice_can_show

    def it_updates_hs_dims_arguments(self, hs_fixture, cube_):
        hs_dims, ndim, expected_value = hs_fixture
        cube_.ndim = ndim
        slice_ = CubeSlice(cube_, None)

        updated_hs_dims = slice_._hs_dims_for_cube(hs_dims)

        assert updated_hs_dims == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ((), None),
            (("fake slice name",), "fake slice name"),
            (("fake slice name", "this is not the slice name"), "fake slice name"),
        ]
    )
    def name_fixture(self, request):
        dim_names, expected_name = request.param
        dimensions = [instance_mock(request, Dimension) for _ in range(len(dim_names))]
        for dimension, name in zip(dimensions, dim_names):
            dimension.name = name
        return dimensions, expected_name

    @pytest.fixture(
        params=[
            ((DT.CAT,), False),
            ((DT.MR,), False),
            ((DT.BINNED_NUMERIC,), False),
            ((DT.DATETIME,), False),
            ((DT.LOGICAL,), False),
            ((DT.TEXT,), False),
            ((DT.CA_CAT,), False),
            ((DT.CA_SUBVAR, DT.CA_CAT), False),
            ((DT.CA_SUBVAR, DT.MR, DT.CA_CAT), False),
            ((DT.MR, DT.CAT), False),
            ((DT.LOGICAL, DT.CAT), False),
            ((DT.CA_CAT, DT.CAT), True),
            ((DT.CAT, DT.CA_CAT), True),
            ((DT.CAT, DT.CAT), True),
            ((DT.BINNED_NUMERIC, DT.CAT), True),
            ((DT.CAT, DT.BINNED_NUMERIC), True),
            ((DT.DATETIME, DT.CAT), True),
            ((DT.CAT, DT.DATETIME), True),
            ((DT.CAT, DT.TEXT), True),
            ((DT.TEXT, DT.CAT), True),
        ]
    )
    def pairwise_comparisons_fixture(self, request):
        dim_types, slice_can_show = request.param
        return dim_types, slice_can_show

    @pytest.fixture(
        params=[
            (0, 1, False, 0),
            (1, 1, False, 1),
            (None, 1, False, None),
            (0, 2, False, 0),
            (1, 2, False, 1),
            (None, 2, False, None),
            (0, 3, False, 1),
            (1, 3, False, 2),
            (None, 3, False, None),
            (0, 0, False, 0),
            (None, 2, True, 1),
        ]
    )
    def axis_fixture(self, request):
        axis, ndim, ca_as_0th, expected_value = request.param
        return axis, ndim, ca_as_0th, expected_value

    @pytest.fixture(
        params=[
            # Expect same value as result, since cube has < 3 dimensions
            (1, 1, False, 0, 1),
            (1, 2, False, 1, 1),
            # Expect same value as result, since there's nothing to index, even
            # though the ndim is == 3
            ([1], 3, False, 1, 1),
            # Expect slicing to take place
            ([0, 1, 2], 3, False, 0, 0),
            ([0, 1, 2], 3, False, 1, 1),
            ([0, 1, 2], 3, False, 2, 2),
            # Return entire result if not capable to index correctly
            ([0, 1, 2], 3, False, 3, [0, 1, 2]),
            # Check that it converts extracted tuples
            ((0, 1, (1, 2)), 3, False, 2, [1, 2]),
        ]
    )
    def extract_fixture(self, request):
        result, ndim, ca_as_0th, index, expected = request.param
        expected_value = np.array(expected)
        return result, ndim, ca_as_0th, index, expected_value

    @pytest.fixture(
        params=[
            ([0, 1], 1, [0, 1]),
            ([0, 1], 2, [0, 1]),
            ([0, 1], 3, [1, 2]),
            (None, 2, None),
            (None, 3, None),
        ]
    )
    def hs_fixture(self, request):
        axis, ndim, expected_value = request.param
        return axis, ndim, expected_value

    @pytest.fixture(
        params=[
            ((DT.CAT,), (DT.CAT,)),
            ((DT.CA_SUBVAR, DT.CA_CAT), (DT.CA_SUBVAR, DT.CA_CAT)),
            ((DT.CA_SUBVAR, DT.MR, DT.CA_CAT), (DT.MR, DT.CA_CAT)),
        ]
    )
    def dim_types_fixture(self, request):
        cube_dim_types, expected_value = request.param
        return cube_dim_types, expected_value

    @pytest.fixture(
        params=[
            ((DT.MR,), False),
            ((DT.CAT, DT.CAT), False),
            ((DT.MR, DT.CAT), False),
            ((DT.CAT, DT.MR), False),
            ((DT.MR, DT.MR), True),
        ]
    )
    def is_double_mr_fixture(self, request):
        dim_types, expected_value = request.param
        return dim_types, expected_value

    @pytest.fixture(
        params=[
            (
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
                ],
                1000,
                [
                    71.82464218,
                    181.80362326,
                    375.31778601,
                    220.43839456,
                    110.99989991,
                    39.61565409,
                ],
                [500, 500],
                [
                    [
                        -0.71589963,
                        -0.53670884,
                        -1.48514968,
                        1.11474378,
                        3.35523602,
                        -2.07704095,
                    ],
                    [
                        0.71589963,
                        0.53670884,
                        1.48514968,
                        -1.11474378,
                        -3.35523602,
                        2.07704095,
                    ],
                ],
            ),
            (
                [
                    [
                        3702.47166525,
                        3643.75050424,
                        1997.62192817,
                        918.59066509,
                        771.46715459,
                        3320.71410532,
                        166.44054077,
                        278.10407838,
                    ],
                    [
                        4114.85193946,
                        3647.97800893,
                        1924.89008796,
                        925.51974172,
                        731.59384577,
                        4172.16468577,
                        184.56963554,
                        281.00640173,
                    ],
                ],
                30781.73498867,
                [
                    7817.32360471,
                    7291.72851317,
                    3922.51201613,
                    1844.11040681,
                    1503.06100036,
                    7492.87879108,
                    351.0101763,
                    559.11048011,
                ],
                [14799.1606418, 15982.57434687],
                [
                    [
                        -1.46558535,
                        3.70412588,
                        3.82368945,
                        1.53747453,
                        2.58473417,
                        -7.48814346,
                        -0.24896875,
                        0.79414354,
                    ],
                    [
                        1.46558535,
                        -3.70412588,
                        -3.82368945,
                        -1.53747453,
                        -2.58473417,
                        7.48814346,
                        0.24896875,
                        -0.79414354,
                    ],
                ],
            ),
        ]
    )
    def scalar_std_res_fixture(self, request):
        counts, total, colsum, rowsum, expected = request.param
        counts = np.array(counts)
        total = np.array(total)
        rowsum = np.array(rowsum)
        colsum = np.array(colsum)
        expected_value = np.array(expected)
        return counts, total, colsum, rowsum, expected_value

    @pytest.fixture(params=[((DT.CAT, DT.CAT), False), ((DT.MR, DT.CAT), True)])
    def std_res_fixture(self, request, _array_type_std_res, _scalar_type_std_res):
        dim_types, is_array = request.param
        expected_value = _array_type_std_res if is_array else _scalar_type_std_res
        return dim_types, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, CrunchCube)

    @pytest.fixture
    def dimensions_prop_(self, request):
        return property_mock(request, CubeSlice, "dimensions")

    @pytest.fixture
    def dim_types_prop_(self, request):
        return property_mock(request, CubeSlice, "dim_types")

    @pytest.fixture
    def ndim_prop_(self, request):
        return property_mock(request, CubeSlice, "ndim")

    @pytest.fixture
    def _array_type_std_res(self, request):
        return method_mock(request, CubeSlice, "_array_type_std_res")

    @pytest.fixture
    def _scalar_type_std_res(self, request):
        return method_mock(request, CubeSlice, "_scalar_type_std_res")


# pylint: disable=invalid-name, no-self-use, protected-access
# pylint: disable=too-many-public-methods, missing-docstring
class TestCubeSlice(object):
    """Test class for the CubeSlice unit tests."""

    def it_can_apply_pruning_mask(self, mask_fixture):
        cube_, res, expected_type = mask_fixture
        slice_ = CubeSlice(cube_, None)

        return_type = type(slice_._apply_pruning_mask(res))

        assert return_type == expected_type

    def test_init(self):
        """Test that init correctly invoked cube construction and sets index."""
        cube = Mock()
        index = Mock()
        cs = CubeSlice(cube, index)
        assert cs._cube == cube
        assert cs._index == index

    def test_init_ca_as_0th(self):
        """Test creation of the 0th CA slice."""
        cube = Mock()
        cube.dim_types = (DT.CA_SUBVAR, DT.CAT)
        assert CubeSlice(cube, 0, ca_as_0th=True)

        cube.dim_types = (DT.CAT, DT.CAT)
        with pytest.raises(ValueError):
            CubeSlice(cube, 0, ca_as_0th=True)

    def test_ndim_invokes_ndim_from_cube(self):
        """Test if ndim calls corresponding cube's method."""
        cube = Mock(ndim=3)
        cs = CubeSlice(cube, 1)
        assert cs.ndim == 2

    def test_table_name(self):
        """Test correct name is returned.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        """
        # Assert name for <3D
        fake_title = "Cube Title"
        cube = Mock()
        cube.ndim = 2
        cube.dimensions = [Mock(), Mock()]
        cube.dimensions[0].name = fake_title
        cs = CubeSlice(cube, 1)
        assert cs.table_name is None
        assert cs.name == fake_title

        # Assert name for 3D
        fake_labels = [[Mock(), "Analysis Slice XY", Mock()]]
        cube.labels.return_value = fake_labels
        cube.ndim = 3
        cube.name = fake_title
        cs = CubeSlice(cube, 1)
        assert cs.table_name == "Cube Title: Analysis Slice XY"
        assert cs.name == "Cube Title"

    def test_proportions(self):
        """Test that proportions method delegetes its call to CrunchCube.

        When the number of dimensions is equal to 3, the
        correct slice needs to be returned. Axis needs to be increased by 1,
        for row and column directions.
        """
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.proportions.return_value = array

        # Assert correct slice is returned when index is set
        cs = CubeSlice(cube, index=1)
        assert cs.proportions() == array[1]

    def test_margin(self):
        """Test that margin method delegetes its call to CrunchCube.

        When the number of dimensions is equal to 3, the
        correct slice needs to be returned. Axis needs to be increased by 1
        for row and column directions.
        """
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.margin.return_value = array
        cs = CubeSlice(cube, 1)

        # Assert correct slice is returned when index is set
        assert cs.margin() == array[1]

    def test_as_array(self):
        """Test that as_array method delegetes its call to CrunchCube.

        When the number of dimensions is smaller than 3, all the arguments
        sould just be passed to the corresponding cube method, and the
        result returned. When the number of dimensions is equal to 3, the
        correct slice needs to be returned.
        """
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.as_array.return_value = array

        # Assert correct slice is returned when index is set
        cs = CubeSlice(cube, index=1)
        assert cs.as_array() == array[1]

    def test_cube_slice_labels(self):
        """Test correct labels are returned for row and col dimensions."""
        cube = Mock()
        cube.ndim = 3
        all_labels = [Mock(), Mock(), Mock()]
        cube.labels.return_value = all_labels
        cs = CubeSlice(cube, 1)
        assert cs.labels() == all_labels[-2:]

        cube.ndim = 2
        cube.dim_types = (DT.CA_SUBVAR, Mock())
        cs = CubeSlice(cube, 1, ca_as_0th=True)
        assert cs.labels() == all_labels[1:]

    def test_prune_indices(self):
        """Assert that correct prune indices are extracted from 3D cube."""
        cube = Mock()
        cube.ndim = 3
        all_prune_inds = [Mock(), (1, 2), Mock()]
        cube.prune_indices.return_value = all_prune_inds
        cs = CubeSlice(cube, 1)
        # Assert extracted indices tuple is converted to list
        actual = cs._prune_indices()
        expected = np.array([1, 2])
        np.testing.assert_array_equal(actual, expected)

    def test_has_means(self):
        """Test that has_means invokes same method on CrunchCube."""
        cube = Mock()
        expected = "Test if has means"
        cube.has_means = expected
        actual = CubeSlice(cube, 1).has_means
        assert actual == expected

    def test_pruning_2d_labels(self):
        """Test that 2D labels are fetched from cr.cube, and pruned."""
        cube = Mock()
        cube.ndim = 2
        cube.prune_indices.return_value = [
            np.array([True, False]),
            np.array([False, False, True]),
        ]
        cube.labels.return_value = [
            [Mock(), "fake_lbl_1"],
            ["fake_lbl_2", "fake_lbl_3", Mock()],
        ]
        actual = CubeSlice(cube, 0).labels(prune=True)
        expected = [["fake_lbl_1"], ["fake_lbl_2", "fake_lbl_3"]]
        assert actual == expected

    def test_pruning_3d_labels(self):
        """Test that 2D labels are fetched from cr.cube, and pruned."""
        cube = Mock()
        cube.ndim = 3
        cube.prune_indices.return_value = [
            Mock(),
            (np.array([True, False]), np.array([False, False, True])),
            Mock(),
        ]
        cube.labels.return_value = [
            Mock(),
            [Mock(), "fake_lbl_1"],
            ["fake_lbl_2", "fake_lbl_3", Mock()],
        ]
        actual = CubeSlice(cube, 1).labels(prune=True)
        expected = [["fake_lbl_1"], ["fake_lbl_2", "fake_lbl_3"]]
        assert actual == expected

    def test_col_dim_ind(self):
        """Test column dimension index for normal slice vs CA as 0th."""
        cube = Mock()
        cube.dim_types = (DT.CA_SUBVAR, Mock())
        cs = CubeSlice(cube, 0, ca_as_0th=False)
        assert cs.col_dim_ind == 1

        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.col_dim_ind == 0

    def test_axis_for_ca_as_0th(self):
        """Test if the axis parameter is updated correctly for the CA as 0th."""
        cube = Mock()
        cube.dim_types = (DT.CA_SUBVAR, Mock())
        cube.ndim = 2
        cube.margin.return_value = np.array([0, 1, 2])
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        cs.margin(axis=None)
        cube.margin.assert_called_once_with(
            axis=1,
            include_missing=False,
            include_transforms_for_dims=None,
            prune=False,
            weighted=True,
            include_mr_cat=False,
        )

    def test_inserted_hs_indices(self):
        """Test H&S indices for different slices."""
        cube = Mock()
        cube.ndim = 3
        cube.inserted_hs_indices.return_value = [1, 2, 3]
        cs = CubeSlice(cube, 0)
        assert cs.inserted_hs_indices() == [2, 3]

        cube.dim_types = (DT.CA_SUBVAR, Mock())
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.inserted_hs_indices() == [1, 2, 3]

    def test_has_ca(self):
        """Test if slice has CA."""
        cube = Mock()
        cube.ndim = 2
        cube.dim_types = (DT.CA_SUBVAR, Mock())

        cs = CubeSlice(cube, 0)
        assert cs.has_ca

        cube.ndim = 3
        cube.dim_types = (DT.CA_SUBVAR, Mock(), Mock())
        cs = CubeSlice(cube, 0)
        assert not cs.has_ca

    def test_mr_dim_ind(self):
        """Test MR dimension index(indices)."""
        cube = Mock()
        cube.ndim = 2
        cube.mr_dim_ind = 0

        cs = CubeSlice(cube, 0)
        assert cs.mr_dim_ind == 0

        cube.mr_dim_ind = 1
        cs = CubeSlice(cube, 0)
        assert cs.mr_dim_ind == 1

        cube.ndim = 3
        cube.mr_dim_ind = 1
        cs = CubeSlice(cube, 0)
        assert cs.mr_dim_ind == 0
        cube.mr_dim_ind = 0
        cs = CubeSlice(cube, 0)
        assert cs.mr_dim_ind is None
        cube.mr_dim_ind = (1, 2)
        assert cs.mr_dim_ind == (0, 1)
        cube.mr_dim_ind = (0, 2)
        cs = CubeSlice(cube, 0)
        assert cs.mr_dim_ind == 1

    def test_ca_main_axis(self):
        """Test interpretation of the main axis for CA cube."""
        cube = Mock()
        cube.dim_types = (DT.CA_SUBVAR, Mock())
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis == 1
        cube.dim_types = (Mock(), DT.CA_SUBVAR)
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis == 0
        cube.dim_types = (Mock(), Mock())
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis is None

    def test_has_mr(self):
        """Test if slice has MR dimension(s)."""
        cube = Mock()
        cube.dim_types = (DT.MR, Mock())
        cs = CubeSlice(cube, 0)
        assert cs.has_mr
        cube.dim_types = (Mock(), DT.MR)
        cs = CubeSlice(cube, 0)
        assert cs.has_mr
        cube.dim_types = (Mock(), Mock())
        cs = CubeSlice(cube, 0)
        assert not cs.has_mr

    @patch("cr.cube.measures.scale_means.ScaleMeans.margin")
    @patch("cr.cube.measures.scale_means.ScaleMeans.__init__")
    def test_scale_means_marginal(self, mock_sm_init, mock_sm_margin):
        """Test if slice method invokes cube method."""
        mock_sm_init.return_value = None

        cs = CubeSlice({}, 0)
        fake_axis = Mock()
        cs.scale_means_margin(fake_axis)
        assert mock_sm_margin.called_once_with(fake_axis)

    def test_scale_means_for_ca_as_0th(self):
        """Test that CA as 0th slice always returns empty scale means.

        If we observe the slice as the 0th dimension of the Tabbook, we need it's 1st
        dimension in the context of scale means (because the 0th dimension is CA items,
        and can't have numerical values, and is used only for slicing)
        """
        cube = Mock()
        cube.dim_types = (DT.CA_SUBVAR,)
        scale_means_value = Mock()

        cube.scale_means.return_value = [[None, [scale_means_value, Mock(), Mock()]]]
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.scale_means() == [scale_means_value]

        cube.scale_means.return_value = [[None, None]]
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.scale_means() == [None]

    def test_shape_property_deprecated(self):
        cube = Mock()

        cube.ndim = 2
        cube.as_array.return_value = np.zeros((3, 2))
        cs = CubeSlice(cube, 0)
        with pytest.warns(DeprecationWarning):
            # TODO: Remove once 'shape' is removed
            assert cs.shape == (3, 2)

    def test_get_shape(self, shape_fixture):
        """Test shape based on 'as_array' and pruning."""
        slice_, prune, expected = shape_fixture
        actual = slice_.get_shape(prune=prune)
        assert actual == expected

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (None, [0, 1], np.ndarray),
            (None, [0, 1], np.ndarray),
            ([True, False], [0, 1], np.ma.core.MaskedArray),
            ([False, False], [0, 1], np.ma.core.MaskedArray),
        ]
    )
    def mask_fixture(self, request):
        mask, values, expected_type = request.param
        res = np.array(values)
        array = (
            np.zeros((2,))
            if mask is None
            else np.ma.masked_array(np.zeros((2,)), np.array(mask))
        )
        cube_ = instance_mock(request, CrunchCube)
        cube_.as_array.return_value = array
        cube_.ndim = 2
        return cube_, res, expected_type

    @pytest.fixture(
        params=[
            (False, None, (3, 2)),
            (True, [[True, False], [True, False], [True, False]], (3,)),
            (True, [[False, False], [True, True], [True, True]], (2,)),
            (True, [[False, False], [True, True], [False, False]], (2, 2)),
            (True, [[True, True], [True, True], [True, True]], ()),
        ]
    )
    def shape_fixture(self, request):
        prune, mask, expected = request.param
        array = np.zeros((3, 2))
        cube = Mock()
        cube.ndim = 2
        if mask is not None:
            array = np.ma.masked_array(array, np.array(mask))
        cube.as_array.return_value = array
        cs = CubeSlice(cube, 0)
        return cs, prune, expected
