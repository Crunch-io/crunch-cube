'''Unit tests for the CubeSlice class.'''
from unittest import TestCase
from mock import Mock

from cr.cube.cube_slice import CubeSlice


# pylint: disable=invalid-name, no-self-use, protected-access
class TestCubeSlice(TestCase):
    '''Test class for the CubeSlice unit tests.'''

    def test_init_no_index(self):
        '''Test that init correctly invoked cube construction.'''
        cube = Mock()
        cs = CubeSlice(cube)
        assert cs._cube == cube
        assert cs._index is None

    def test_init_with_index(self):
        '''Test that init correctly invoked cube construction and sets index.'''
        cube = Mock()
        index = Mock()
        cs = CubeSlice(cube, index)
        assert cs._cube == cube
        assert cs._index == index

    def test_ndim_invokes_ndim_from_cube(self):
        '''Test if ndim calls corresponding cube's method.'''
        fake_ndim = Mock()
        cube = Mock()
        cube.ndim = fake_ndim

        cs = CubeSlice(cube)
        assert cs.ndim == fake_ndim

    def test_name(self):
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
        cs = CubeSlice(cube)
        assert cs.name == fake_title

        # Assert name for 3D
        fake_labels = [[Mock(), 'Analysis Slice XY', Mock()]]
        cube.labels.return_value = fake_labels
        cube.ndim = 3
        cs = CubeSlice(cube, 1)
        assert cs.name == 'Cube Title: Analysis Slice XY'

    def test_as_array(self):
        '''Test that as_array method delegetes its call to CrunchCube.

        When the number of dimensions is smaller than 3, all the arguments
        sould just be passed to the corresponding cube method, and the
        result returned. When the number of dimensions is equal to 3, the
        correct slice needs to be returned.
        '''
        cube = Mock()

        # Assert arguments are passed correctly
        cs = CubeSlice(cube)
        arg = Mock()
        kw_arg = Mock()
        cs.as_array(arg, kw_arg=kw_arg)
        cs._cube.as_array.assert_called_once_with(arg, kw_arg=kw_arg)

        # Assert entire array is returned when index is not set
        array = Mock()
        cube.as_array.return_value = array
        cs = CubeSlice(cube)
        assert cs.as_array() == array

        # Assert correct slice is returned when index is set
        array = [Mock(), Mock(), Mock()]
        cube.as_array.return_value = array
        cs = CubeSlice(cube, index=1)
        assert cs.as_array() == array[1]

    def test_rows_title(self):
        '''Assert correct rows title is returned.'''
        cube = Mock()
        cube.ndim = 3
        cube.dimensions = [Mock(), Mock(), Mock()]
        cube.dimensions[1].name = '1st Dimension Name'
        cs = CubeSlice(cube, 1)
        assert cs.rows_title == '1st Dimension Name'

    def test_inserted_rows_indices(self):
        '''Assert correct inserted row indices are returned.
        '''
        fake_indices = Mock()
        cube = Mock()
        cube.inserted_hs_indices.return_value = [
            Mock(), (fake_indices, Mock()), Mock(),
        ]
        cs = CubeSlice(cube, 1)
        assert cs.inserted_rows_indices == fake_indices
