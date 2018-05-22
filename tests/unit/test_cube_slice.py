'''Unit tests for the CubeSlice class.'''
from unittest import TestCase
from mock import Mock
from mock import patch

from cr.cube.cube_slice import CubeSlice


# pylint: disable=invalid-name, no-self-use, protected-access
class TestCrunchCube(TestCase):
    '''Test class for the CubeSlice unit tests.'''

    @patch('cr.cube.crunch_cube.CrunchCube.__init__')
    def test_init_no_index(self, mock_init):
        '''Test that init correctly invoked cube construction.'''
        response = Mock()
        cs = CubeSlice(response)
        mock_init.assert_called_once_with(response)
        assert cs._index is None

    @patch('cr.cube.crunch_cube.CrunchCube.__init__')
    def test_init_with_index(self, mock_init):
        '''Test that init correctly invoked cube construction and sets index.'''
        response = Mock()
        index = Mock()
        cs = CubeSlice(response, index)
        mock_init.assert_called_once_with(response)
        assert cs._index == index
