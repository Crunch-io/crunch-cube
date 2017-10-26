from mock import Mock
from mock import patch
from unittest import TestCase

from cr.cube.crunch_cube import CrunchCube


class TestCrunchCube(TestCase):

    def test_init_raises_value_type_on_initialization(self):
        with self.assertRaises(TypeError) as ctx:
            CrunchCube(Mock())
        expected = (
            'Unsupported type provided: {}. '
            'A `cube` must be JSON or `dict`.'
        ).format(type(Mock()))
        self.assertEqual(str(ctx.exception), expected)

    @patch('cr.cube.crunch_cube.json.loads')
    def test_init_invokes_json_loads_for_string_input(self, mock_loads):
        mock_loads.return_value = {'value': {'result': {'dimensions': []}}}
        fake_json = 'fake cube json'
        CrunchCube(fake_json)
        mock_loads.assert_called_once_with(fake_json)

    @patch('cr.cube.crunch_cube.CrunchCube._get_dimensions')
    def test_init_invokes_get_dimensions(self, mock_get_dimensions):
        mock_get_dimensions.return_value = None
        fake_cube_dict = {'value': Mock()}
        CrunchCube(fake_cube_dict)
        mock_get_dimensions.assert_called_once_with(fake_cube_dict['value'])
