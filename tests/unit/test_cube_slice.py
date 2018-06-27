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
        assert cs.ndim == 3

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
