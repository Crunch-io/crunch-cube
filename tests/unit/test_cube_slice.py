'''Unit tests for the CubeSlice class.'''
from unittest import TestCase
from mock import Mock
import numpy as np

from cr.cube.cube_slice import CubeSlice


# pylint: disable=invalid-name, no-self-use, protected-access
class TestCubeSlice(TestCase):
    '''Test class for the CubeSlice unit tests.'''

    def test_init(self):
        '''Test that init correctly invoked cube construction and sets index.'''
        cube = Mock()
        index = Mock()
        cs = CubeSlice(cube, index)
        assert cs._cube == cube
        assert cs._index == index

    def test_init_ca_as_0th(self):
        '''Test creation of the 0th CA slice.'''
        cube = Mock()
        cube.dim_types = ['categorical_array', 'categorical']
        assert CubeSlice(cube, 0, ca_as_0th=True)

        cube.dim_types = ['categorical', 'categorical']
        with self.assertRaises(ValueError):
            CubeSlice(cube, 0, ca_as_0th=True)

    def test_ndim_invokes_ndim_from_cube(self):
        '''Test if ndim calls corresponding cube's method.'''
        cube = Mock(ndim=3)
        cs = CubeSlice(cube, 1)
        assert cs.ndim == 2

    def test_table_name(self):
        '''Test correct name is returned.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        '''
        # Assert name for <3D
        fake_title = 'Cube Title'
        cube = Mock()
        cube.ndim = 2
        cube.name = fake_title
        cs = CubeSlice(cube, 1)
        assert cs.table_name is None
        assert cs.name == fake_title

        # Assert name for 3D
        fake_labels = [[Mock(), 'Analysis Slice XY', Mock()]]
        cube.labels.return_value = fake_labels
        cube.ndim = 3
        cs = CubeSlice(cube, 1)
        assert cs.table_name == 'Cube Title: Analysis Slice XY'
        assert cs.name == 'Cube Title'

    def test_proportions(self):
        '''Test that proportions method delegetes its call to CrunchCube.

        When the number of dimensions is equal to 3, the
        correct slice needs to be returned. Axis needs to be increased by 1,
        for row and column directions.
        '''
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.proportions.return_value = array

        # Assert arguments are passed correctly
        cs = CubeSlice(cube, 1)
        cs.proportions(axis=0)
        # Expect axis to be increased by 1, because 3D
        cs._cube.proportions.assert_called_once_with(axis=1)

        # Assert correct slice is returned when index is set
        cs = CubeSlice(cube, index=1)
        assert cs.proportions() == array[1]

    def test_margin(self):
        '''Test that margin method delegetes its call to CrunchCube.

        When the number of dimensions is equal to 3, the
        correct slice needs to be returned. Axis needs to be increased by 1
        for row and column directions.
        '''
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.margin.return_value = array
        cs = CubeSlice(cube, 1)

        # Assert arguments are passed correctly
        cs.margin(axis=0)
        # Expect axis to be increased by 1, because 3D
        cs._cube.margin.assert_called_once_with(axis=1)

        # Assert correct slice is returned when index is set
        assert cs.margin() == array[1]

    def test_as_array(self):
        '''Test that as_array method delegetes its call to CrunchCube.

        When the number of dimensions is smaller than 3, all the arguments
        sould just be passed to the corresponding cube method, and the
        result returned. When the number of dimensions is equal to 3, the
        correct slice needs to be returned.
        '''
        cube = Mock()
        cube.ndim = 3
        array = [Mock(), Mock(), Mock()]
        cube.as_array.return_value = array

        # Assert arguments are passed correctly
        cs = CubeSlice(cube, 1)
        arg = Mock()
        kw_arg = Mock()
        cs.as_array(arg, kw_arg=kw_arg)
        cs._cube.as_array.assert_called_once_with(arg, kw_arg=kw_arg)

        # Assert correct slice is returned when index is set
        cs = CubeSlice(cube, index=1)
        assert cs.as_array() == array[1]

    def test_cube_slice_labels(self):
        '''Test correct labels are returned for row and col dimensions.'''
        cube = Mock()
        cube.ndim = 3
        all_labels = [Mock(), Mock(), Mock()]
        cube.labels.return_value = all_labels
        cs = CubeSlice(cube, 1)
        assert cs.labels() == all_labels[-2:]

        cube.ndim = 2
        cube.dim_types = ['categorical_array', Mock()]
        cs = CubeSlice(cube, 1, ca_as_0th=True)
        assert cs.labels() == all_labels[1:]

    def test_prune_indices(self):
        '''Assert that correct prune indices are extracted from 3D cube.'''
        cube = Mock()
        cube.ndim = 3
        all_prune_inds = [Mock(), (1, 2), Mock()]
        cube.prune_indices.return_value = all_prune_inds
        cs = CubeSlice(cube, 1)
        # Assert extracted indices tuple is converted to list
        actual = cs.prune_indices()
        expected = np.array([1, 2])
        np.testing.assert_array_equal(actual, expected)

    def test_has_means(self):
        '''Test that has_means invokes same method on CrunchCube.'''
        cube = Mock()
        expected = 'Test if has means'
        cube.has_means = expected
        actual = CubeSlice(cube, 1).has_means
        assert actual == expected

    def test_dim_types(self):
        '''Test only last 2 dim types are returned.'''
        cube = Mock()
        all_dim_types = [Mock(), Mock(), Mock()]
        expected = all_dim_types[-2:]
        cube.dim_types = all_dim_types
        cube.ndim = 3
        actual = CubeSlice(cube, 0).dim_types
        assert actual == expected

    def test_pruning_2d_labels(self):
        '''Test that 2D labels are fetched from cr.cube, and pruned.'''
        cube = Mock()
        cube.ndim = 2
        cube.prune_indices.return_value = [
            np.array([True, False]), np.array([False, False, True]),
        ]
        cube.labels.return_value = [
            [Mock(), 'fake_lbl_1'], ['fake_lbl_2', 'fake_lbl_3', Mock()],
        ]
        actual = CubeSlice(cube, 0).labels(prune=True)
        expected = [['fake_lbl_1'], ['fake_lbl_2', 'fake_lbl_3']]
        assert actual == expected

    def test_pruning_3d_labels(self):
        '''Test that 2D labels are fetched from cr.cube, and pruned.'''
        cube = Mock()
        cube.ndim = 3
        cube.prune_indices.return_value = [
            Mock(),
            (np.array([True, False]), np.array([False, False, True])),
            Mock(),
        ]
        cube.labels.return_value = [
            Mock(),
            [Mock(), 'fake_lbl_1'],
            ['fake_lbl_2', 'fake_lbl_3', Mock()],
        ]
        actual = CubeSlice(cube, 1).labels(prune=True)
        expected = [['fake_lbl_1'], ['fake_lbl_2', 'fake_lbl_3']]
        assert actual == expected

    def test_col_dim_ind(self):
        '''Test column dimension index for normal slice vs CA as 0th.'''
        cube = Mock()
        cube.dim_types = ['categorical_array', Mock()]
        cs = CubeSlice(cube, 0, ca_as_0th=False)
        assert cs.col_dim_ind == 1

        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.col_dim_ind == 0

    def test_axis_for_ca_as_0th(self):
        '''Test if the axis parameter is updated correctly for the CA as 0th.'''
        cube = Mock()
        cube.dim_types = ['categorical_array', Mock()]
        cube.ndim = 2
        cube.margin.return_value = np.array([0, 1, 2])
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        cs.margin(axis=None)
        cube.margin.assert_called_once_with(axis=1)

    def test_update_hs_dims(self):
        '''Test if H&S dims are updated for 3D cubes.'''
        cube = Mock()
        cube.ndim = 3
        cs = CubeSlice(cube, 0)
        expected = {'include_transforms_for_dims': [1, 2]}
        actual = cs._update_args({'include_transforms_for_dims': [0, 1]})
        assert actual == expected

    def test_inserted_hs_indices(self):
        '''Test H&S indices for different slices.'''
        cube = Mock()
        cube.ndim = 3
        cube.inserted_hs_indices.return_value = [1, 2, 3]
        cs = CubeSlice(cube, 0)
        assert cs.inserted_hs_indices() == [2, 3]

        cube.dim_types = ['categorical_array', Mock()]
        cs = CubeSlice(cube, 0, ca_as_0th=True)
        assert cs.inserted_hs_indices() == [1, 2, 3]

    def test_has_ca(self):
        '''Test if slice has CA.'''
        cube = Mock()
        cube.ndim = 2
        cube.dim_types = ['categorical_array', Mock()]

        cs = CubeSlice(cube, 0)
        assert cs.has_ca

        cube.ndim = 3
        cube.dim_types = ['categorical_array', Mock(), Mock()]
        cs = CubeSlice(cube, 0)
        assert not cs.has_ca

    def test_mr_dim_ind(self):
        '''Test MR dimension index(indices).'''
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
        '''Test interpretation of the main axis for CA cube.'''
        cube = Mock()
        cube.dim_types = ['categorical_array', Mock()]
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis == 1
        cube.dim_types = [Mock(), 'categorical_array']
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis == 0
        cube.dim_types = [Mock(), Mock()]
        cs = CubeSlice(cube, 0)
        assert cs.ca_main_axis is None

    def test_has_mr(self):
        '''Test if slice has MR dimension(s).'''
        cube = Mock()
        cube.dim_types = ['multiple_response', Mock()]
        cs = CubeSlice(cube, 0)
        assert cs.has_mr
        cube.dim_types = [Mock(), 'multiple_response']
        cs = CubeSlice(cube, 0)
        assert cs.has_mr
        cube.dim_types = [Mock(), Mock()]
        cs = CubeSlice(cube, 0)
        assert not cs.has_mr

    def test_is_double_mr(self):
        '''Test if slice are double MRs.'''
        cube = Mock()
        cube.dim_types = ['multiple_response', Mock()]
        cs = CubeSlice(cube, 0)
        assert not cs.is_double_mr

        cube.dim_types = [Mock(), 'multiple_response']
        cs = CubeSlice(cube, 0)
        assert not cs.is_double_mr

        cube.dim_types = [Mock(), Mock()]
        cs = CubeSlice(cube, 0)
        assert not cs.is_double_mr

        cube.dim_types = ['multiple_response'] * 2
        cs = CubeSlice(cube, 0)
        assert cs.is_double_mr

        cube.ndim = 3
        cube.dim_types = ['multiple_response'] * 3
        cs = CubeSlice(cube, 0)
        # It is double MR because the last two are MRs
        assert cs.is_double_mr

        cube.ndim = 3
        cube.dim_types = [Mock()] + ['multiple_response'] * 2
        cs = CubeSlice(cube, 0)
        assert cs.is_double_mr

        cube.ndim = 3
        cube.dim_types = ['multiple_response', Mock(), 'multiple_response']
        cs = CubeSlice(cube, 0)
        # Not double MR because the 0th dims is 'just' tabs
        assert not cs.is_double_mr

        cube.ndim = 3
        cube.dim_types = ['multiple_response', 'multiple_response', Mock()]
        cs = CubeSlice(cube, 0)
        # Not double MR because the 0th dims is 'just' tabs
        assert not cs.is_double_mr

    def test_ca_dim_ind(self):
        '''Test if slice are double MRs.'''
        cube = Mock()

        cube.ca_dim_ind = None
        cs = CubeSlice(cube, 0)
        assert cs.ca_dim_ind == None

        cube.ca_dim_ind = 0
        cs = CubeSlice(cube, 0)
        assert cs.ca_dim_ind == 0

        cube.ca_dim_ind = 1
        cs = CubeSlice(cube, 0)
        assert cs.ca_dim_ind == 1

        cube.ndim = 3
        cube.ca_dim_ind = 1
        cs = CubeSlice(cube, 0)
        assert cs.ca_dim_ind == 0

        cube.ndim = 3
        cube.ca_dim_ind = 0
        cs = CubeSlice(cube, 0)
        assert cs.ca_dim_ind == None
